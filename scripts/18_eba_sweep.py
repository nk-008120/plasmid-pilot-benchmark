#!/usr/bin/env python3
"""Validate the EBA positional scorer on the 23 hand-audited constructs, and sweep its one tuned number.

    python scripts/18_eba_sweep.py --embedder blosum
    python scripts/18_eba_sweep.py --embedder facebook/esm2_t33_650M_UR50D --n-decoys 4000 --precompute no

this project's working rules has two rules this script exists to obey. Any rule that will be applied to the corpus or the
synthetic tier is first scored against the 23 hand-audited constructs, where the answer comes from the
depositor rather than from alignment. And when a tuned threshold is involved, the sweep and its plateau go
in the record, not the best value.

**The tuned number is `shift`.** The substitution score is `(cosine - shift) * scale`, so `shift` sets
where "unrelated" starts. At 0 almost every residue pair looks like a match and the local alignment runs
to the full length of both sequences; too high and nothing aligns at all. What it buys is not a bigger
score but a bigger *ratio* between a true remote homolog and noise, so the sweep is scored on retrieval,
not on score.

**The reference set is reduced on purpose, and the control is re-run on the same reduction.** Exhaustive
dynamic programming against all 32,387 twilight proteins costs about 35 s per query per shift value, which
is three hours for one sweep. This builds a smaller catalogue instead, every homolog of the 16 hand-tier
payloads, the payloads themselves, and a seeded random sample of decoys, and re-runs exact Smith-Waterman
on *that* catalogue too, so the EBA and align numbers are comparable to each other even though neither is
comparable to the full-reference numbers in the working record. The reduced catalogue is written out, so the
reduction is reproducible rather than a detail of one run.

**A two-stage scorer has two ways to fail.** With `--top-k` set, a prefilter shortlists before the dynamic
programming reranks, and a payload the prefilter drops can never be recovered. Prefilter recall is
therefore reported separately from top-1 accuracy. `--top-k 0` runs exhaustive dynamic programming, which
is the default here because the reduced catalogue makes it affordable and because a prefilter would make
the result hard to attribute.

**Prefilter recall is reported per arm, and as a curve over `top_k` rather than at one value** (the working record, 33.8, open item 32). Both changes came from the same measurement. Reporting the `whole` arm alone
hid the finding that decomposing the query buys more shortlist recall than a twenty-five-fold larger
shortlist: on `synth_random` below f = 0.50, `auto_per_segment` at `top_k` 200 reaches 52.0% where
`whole` at `top_k` 5,000 reaches 48.0%. And reporting one `top_k` cannot show where recall saturates,
which is what this project's working rules means by deriving a limit instead of choosing it. The curve is free: the
prefilter's *rank* of the true answer is recorded per construct, and recall at every `k` follows from it.

**`--precompute` is a flag because the default is a trap.** `get_eba_scorer` defaults `precompute=True`
for any non-BLOSUM embedder, which on the 40,750-protein catalogue under ESM-2 650M is 95.5 GB of
per-residue vectors. The docstring example above used to advertise that checkpoint with no way to turn
it off.

Writes `eba_sweep_<embedder>.csv`, `_summary.json` and `figures/fig_d6_eba_sweep.png` / `.pdf`.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from pilot.align import get_scorer
from pilot.cosine_floor import background_pair_stats
from pilot.arms import build_arms
from pilot.eba import EbaScorer
from pilot.homology import Homolog
from pilot.positional import get_positional_embedder
from pilot.reference import Entry, iter_fasta, write_fasta

SHIFTS = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)
HOLDOUTS = (0.50, 0.25)
ARMS = ("whole", "oracle")
# The grid the prefilter recall curve is reported on. Not a setting to tune: it is the x-axis of the
# measurement that decides whether `top_k` can be small enough for the two-stage design to be worth it.
RECALL_AT = (200, 500, 2000, 5000, 10000)


def read_faa(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for e in iter_fasta(path):
        oid = e.header.split()[0]
        out[oid.split("|")[0]][oid] = e.seq
    return out


def load_homologs(path: Path) -> dict[str, list[Homolog]]:
    out: dict[str, list[Homolog]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            out[r["payload_accession"]].append(
                Homolog(r["homolog_accession"], float(r["score"]), float(r["identity"]),
                        float(r["coverage"]), int(r["length"])))
    return out


def reduced_reference(full: list[Entry], payloads: set[str], homologs: dict[str, list[Homolog]],
                      n_decoys: int, seed: int) -> list[Entry]:
    """Payloads, every homolog of a payload, and a seeded decoy sample. Order is stable across runs."""
    wanted = set(payloads)
    for acc in payloads:
        wanted.update(h.accession for h in homologs.get(acc, []))
    keep = [e for e in full if e.accession in wanted]
    rest = [e for e in full if e.accession not in wanted]
    rng = random.Random(seed)
    decoys = rng.sample(rest, min(n_decoys, len(rest)))
    return sorted(keep + decoys, key=lambda e: e.accession)


def best_target_rank(row: np.ndarray, target_idx: np.ndarray, mask_idx: np.ndarray) -> int | None:
    """Same rule as `scripts/10_twilight.py`, so the two scripts' ranks mean the same thing."""
    if target_idx.size == 0:
        return None
    r = row.astype(np.float64, copy=True)
    r[mask_idx] = -np.inf
    best = float(np.max(r[target_idx]))
    if not np.isfinite(best):
        return None
    return int((r >= best).sum())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results/baseline2"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--reference", type=Path, default=Path("data/reference/reference_twilight.fasta"))
    ap.add_argument("--homologs", type=Path, default=Path("data/reference/homologs.csv"))
    ap.add_argument("--embedder", default="blosum", help="blosum, kmerpos, or an ESM-2 checkpoint name")
    ap.add_argument("--n-decoys", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--top-k", type=int, default=0, help="0 = exhaustive dynamic programming, no prefilter")
    ap.add_argument("--prefilter", default="kmer")
    ap.add_argument("--gap-open", type=float, default=0.5)
    ap.add_argument("--gap-extend", type=float, default=0.1)
    ap.add_argument("--shifts", type=float, nargs="*", default=list(SHIFTS))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-control", action="store_true", help="skip the Smith-Waterman control re-run")
    ap.add_argument("--precompute", choices=("auto", "yes", "no"), default="auto",
                    help="embed the whole catalogue's residues once ('yes'), per chunk ('no'), or let "
                         "the embedder decide ('auto', the historical behaviour: everything but BLOSUM "
                         "precomputes). 'no' is the only safe setting on a large catalogue under a "
                         "650M checkpoint: see the module docstring.")
    ap.add_argument("--recall-at", type=int, nargs="*", default=list(RECALL_AT),
                    help="the top_k grid the prefilter recall curve is reported on. Costs nothing: the "
                         "whole curve comes from one recorded rank per construct.")
    args = ap.parse_args(argv)

    for p in (args.results_dir / "orfs.faa", args.reference, args.homologs, args.manifest):
        if not p.is_file():
            print(f"not found: {p}", file=sys.stderr)
            return 2

    with open(args.manifest, newline="", encoding="utf-8-sig", errors="replace") as fh:
        man = {r["filename"].strip(): r for r in csv.DictReader(fh) if (r.get("filename") or "").strip()}
    with open(args.results_dir / "pilot_summary.csv", newline="", encoding="utf-8-sig") as fh:
        summary = list(csv.DictReader(fh))
    orfs_by_plasmid = read_faa(args.results_dir / "orfs.faa")
    homologs = load_homologs(args.homologs)
    full = list(iter_fasta(args.reference))

    payloads = {(man.get(r["source_file"], {}).get("payload_accession") or "").strip()
                for r in summary}
    payloads = {p for p in payloads if p}
    refs = reduced_reference(full, payloads, homologs, args.n_decoys, args.seed)
    ref_acc = [e.accession for e in refs]
    ref_seqs = [e.seq for e in refs]
    idx_of = {a: i for i, a in enumerate(ref_acc)}
    ref_by_acc = {e.accession: e.seq for e in refs}
    out_fa = args.results_dir / "reference_reduced.fasta"
    write_fasta(refs, out_fa)
    print(f"reduced reference: {len(refs):,} proteins "
          f"({len(refs) - min(args.n_decoys, len(full) - len(refs)):,} payload-related + decoys) -> {out_fa.name}")
    print(f"    {len(payloads)} hand-tier payload accessions, "
          f"{sum(1 for a in payloads if a in idx_of)} of them present")

    # Queries are built once and scored by every scorer, so a change in the numbers can only come from the
    # scorer. `build_arms` is the same call `06_decompose.py` and `10_twilight.py` make.
    queries: list[tuple[str, str, dict[str, str]]] = []
    for srow in summary:
        pid = srow["plasmid_id"]
        acc = (man.get(srow["source_file"], {}).get("payload_accession") or "").strip()
        if not acc or acc not in ref_by_acc:
            continue
        arms = build_arms(pid, orfs_by_plasmid.get(pid, {}), ref_by_acc[acc])
        if arms is None:
            continue
        q = {a: s for a, s in (("whole", arms.whole), ("oracle", arms.oracle)) if s}
        if q:
            queries.append((pid, acc, q))
    print(f"    {len(queries)} scorable constructs\n")

    def evaluate(scorer, label: str) -> list[dict]:
        rows = []
        t0 = time.time()
        for pid, acc, qs in queries:
            scored = {a: scorer.score_all(s) for a, s in qs.items()}
            # Per arm, not just `whole`. The prefilter sees a different query in every arm, so one arm's
            # shortlist recall says nothing about another's, which is the whole of 33.8.
            pf_rows = ({a: scorer.prefilter_scores(s) for a, s in qs.items()}
                       if isinstance(scorer, EbaScorer) else {})
            hs = homologs.get(acc, [])
            for thr in HOLDOUTS:
                mask = np.array([idx_of[h.accession] for h in hs
                                 if h.identity > thr and h.accession in idx_of], dtype=int)
                tgt = np.array([idx_of[h.accession] for h in hs
                                if h.identity <= thr and h.accession in idx_of], dtype=int)
                for arm, row in scored.items():
                    r = best_target_rank(row, tgt, mask)
                    pf = pf_rows.get(arm)
                    # The prefilter's own rank of the best surviving target. Recall at any `top_k`
                    # follows from it, so the curve needs no extra run and no extra prefilter pass.
                    pr = None if pf is None else best_target_rank(pf, tgt, mask)
                    reachable = ("" if pr is None or scorer.top_k >= len(ref_seqs)
                                 else int(pr <= scorer.top_k))
                    rows.append(dict(scorer=label, threshold=thr, plasmid_id=pid, payload_accession=acc,
                                     arm=arm, n_homologs_available=int(tgt.size),
                                     n_held_out=int(mask.size),
                                     best_target_rank="" if r is None else r,
                                     prefilter_rank="" if pr is None else pr,
                                     target_in_shortlist=reachable))
        print(f"        {label:<34}{time.time() - t0:>7.1f}s", flush=True)
        return rows

    rows: list[dict] = []
    print("scoring:")
    if not args.no_control:
        rows += evaluate(get_scorer("align", ref_seqs), "align")

    emb_kw = {} if args.embedder in ("blosum", "blosum62") or args.embedder.startswith("kmerpos") else {
        "batch_size": args.batch_size}
    embedder = get_positional_embedder(args.embedder, **emb_kw)

    # Which of the swept shifts admit a local alignment at all. this project's working rules asks for the binding count of
    # every limit; this is the limit that binds on the whole scoring system. A shift at or below the
    # background cosine makes `cosine - shift` non-negative in expectation, so the optimal local
    # alignment never terminates and the score measures length rather than similarity, which is a
    # candidate explanation for 19.3's flat, low plateau. This is a WARNING and not a
    # refusal on purpose: sweeping into the degenerate range is legitimate and 19.3 did it: what was
    # missing was any statement of where the boundary is.
    pair_stats = background_pair_stats(embedder, [e.seq for e in refs])
    degenerate = [sh for sh in args.shifts if pair_stats.expected_pair_score(sh) >= 0]
    print(f"background cosine ({pair_stats.embedder_name or args.embedder}): "
          f"{pair_stats.mean_cosine:.4f} from {pair_stats.n_pairs:,} cross-protein residue pairs over "
          f"{pair_stats.n_proteins} proteins, clustered SE {pair_stats.se_clustered:.5f}")
    print(f"  minimum viable shift (E[pair score] < 0): {pair_stats.min_viable_shift:.4f}; "
          f"recommended range {pair_stats.recommended_shift_range()[0]:.4f} to "
          f"{pair_stats.recommended_shift_range()[1]:.4f}; BLOSUM62-analogous point "
          f"{pair_stats.blosum62_analogous_shift():.4f}")
    if degenerate:
        print(f"  *** {len(degenerate)} of {len(args.shifts)} swept shifts are DEGENERATE "
              f"({', '.join(f'{sh:g}' for sh in degenerate)}): E[pair score] >= 0, so the local "
              f"alignment does not terminate and the score measures length, not similarity. Their "
              f"rows are still computed and written: read them as a null, not as a tuning curve.")
    else:
        print("  all swept shifts admit a local alignment (E[pair score] < 0)")
    print(flush=True)
    pf = None
    if args.top_k and args.prefilter not in ("none", ""):
        pf = get_scorer(args.prefilter, ref_seqs)
    for shift in args.shifts:
        pre = (args.embedder not in ("blosum", "blosum62") if args.precompute == "auto"
               else args.precompute == "yes")
        sc = EbaScorer(embedder, ref_seqs, prefilter=pf, shift=shift, gap_open=args.gap_open,
                       gap_extend=args.gap_extend, top_k=args.top_k, precompute=pre)
        rows += evaluate(sc, f"eba shift={shift:g}")

    stem = args.embedder.replace("/", "_")
    out_csv = args.results_dir / f"eba_sweep_{stem}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    agg: dict = {"embedder": args.embedder, "n_reference": len(refs), "n_decoys": args.n_decoys,
                 "seed": args.seed, "top_k": args.top_k, "prefilter": args.prefilter if args.top_k else None,
                 "gap_open": args.gap_open, "gap_extend": args.gap_extend,
                 "shifts": list(args.shifts), "holdouts": list(HOLDOUTS), "by_scorer": {}}
    labels = list(dict.fromkeys(r["scorer"] for r in rows))
    for thr in HOLDOUTS:
        print(f"\n{'':<20}" + f"holdout {thr:.0%}".center(34, " "))
        print(f"{'scorer':<20}" + "".join(f"{a + ' top1':>13}" for a in ARMS)
              + f"{'oracle - whole':>16}{'prefilter recall':>18}")
        for lab in labels:
            at = [r for r in rows if r["scorer"] == lab and r["threshold"] == thr]
            d: dict = {}
            for arm in ARMS:
                got = [r for r in at if r["arm"] == arm and r["best_target_rank"] != ""]
                d[f"{arm}_top1"] = sum(1 for r in got if int(r["best_target_rank"]) == 1)
                d[f"{arm}_n"] = len(got)
                per: dict[str, list[int]] = defaultdict(list)
                for r in got:
                    per[r["payload_accession"]].append(1 if int(r["best_target_rank"]) == 1 else 0)
                d[f"{arm}_top1_macro"] = (round(100 * sum(sum(v) / len(v) for v in per.values())
                                                / len(per), 1) if per else None)
            for arm in ARMS:
                reach = [r["target_in_shortlist"] for r in at
                         if r["arm"] == arm and r["target_in_shortlist"] != ""
                         and r["n_homologs_available"]]
                d[f"{arm}_prefilter_recall"] = (round(100 * sum(reach) / len(reach), 1)
                                                if reach else None)
                # The curve. `prefilter_rank` is the rank the prefilter gave the best surviving target,
                # so recall at k is just the share of those ranks at or below k, and the binding count
                # this project's working rules asks for is its complement.
                pranks = [int(r["prefilter_rank"]) for r in at
                          if r["arm"] == arm and r["prefilter_rank"] != ""
                          and r["n_homologs_available"]]
                d[f"{arm}_prefilter_recall_at"] = (
                    {str(k): round(100 * sum(1 for x in pranks if x <= k) / len(pranks), 1)
                     for k in args.recall_at} if pranks else None)
                d[f"{arm}_prefilter_n"] = len(pranks)
                d[f"{arm}_prefilter_dropped_at_top_k"] = (
                    None if not pranks or not args.top_k
                    else sum(1 for x in pranks if x > args.top_k))
            d["prefilter_recall"] = d.get("whole_prefilter_recall")
            d["gap"] = d["oracle_top1"] - d["whole_top1"]
            agg["by_scorer"].setdefault(lab, {})[f"{thr:.2f}"] = d
            cells = "".join(f"{d[a + '_top1']:>8}/{d[a + '_n']:<4}" for a in ARMS)
            rec = "-" if d["prefilter_recall"] is None else f"{d['prefilter_recall']:.1f}%"
            print(f"{lab:<20}{cells}{d['gap']:>+16}{rec:>18}")

    # The prefilter recall curve, per arm, per holdout. This is the table that decides whether the
    # two-stage design can be cheap: `top_k` belongs where recall saturates on the arm being scored, and
    # the binding count beside it is what shows the limit is clear (this project's working rules, "Report binding, not
    # settings"). A row of dashes means this run had no prefilter, `--top-k 0`, which is the default.
    curves = [(lab, thr, arm, agg["by_scorer"][lab][f"{thr:.2f}"])
              for lab in labels for thr in HOLDOUTS for arm in ARMS]
    if any(d.get(f"{arm}_prefilter_recall_at") for _, _, arm, d in curves):
        print("\nprefilter recall curve (share of constructs whose best surviving target the prefilter "
              "ranks at or above k)")
        print(f"{'scorer':<20}{'holdout':>8}{'arm':>10}{'n':>6}"
              + "".join(f"{'k=' + str(k):>10}" for k in args.recall_at)
              + f"{'dropped at top_k':>19}")
        for lab, thr, arm, d in curves:
            at = d.get(f"{arm}_prefilter_recall_at")
            if not at:
                continue
            drop = d.get(f"{arm}_prefilter_dropped_at_top_k")
            print(f"{lab:<20}{thr:>7.0%}{arm:>10}{d[f'{arm}_prefilter_n']:>6}"
                  + "".join(f"{at[str(k)]:>9.1f}%" for k in args.recall_at)
                  + f"{'-' if drop is None else f'{drop} of ' + str(d[f'{arm}_prefilter_n']):>19}")
        print("A prefilter recall below 100% at the top_k in use is an upper bound on the arm: a "
              "reference the\nprefilter never shortlisted cannot reach rank 1, whatever the reranker "
              "does.")
    agg["recall_at_grid"] = list(args.recall_at)
    agg["precompute"] = args.precompute
    agg["background_pair_stats"] = vars(pair_stats)
    agg["degenerate_shifts"] = [float(x) for x in degenerate]

    (args.results_dir / f"eba_sweep_{stem}_summary.json").write_text(json.dumps(agg, indent=2),
                                                                     encoding="utf-8")
    draw(agg, args.results_dir / "figures", stem)
    print(f"\nwrote {out_csv.name} + _summary.json")
    return 0


def draw(agg: dict, out_dir: Path, stem: str) -> None:
    try:
        import matplotlib  # noqa: PLC0415
    except ImportError:
        print("matplotlib not installed; skipping the figure")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    shifts = agg["shifts"]
    fig, axes = plt.subplots(1, len(agg["holdouts"]), figsize=(9.6, 4.2), sharey=True)
    for ax, thr in zip(np.atleast_1d(axes), agg["holdouts"]):
        key = f"{thr:.2f}"
        for arm, colour in (("whole", "#b4b4b4"), ("oracle", "#b3402f")):
            ys = []
            for sh in shifts:
                d = agg["by_scorer"].get(f"eba shift={sh:g}", {}).get(key)
                ys.append(100 * d[f"{arm}_top1"] / d[f"{arm}_n"] if d and d[f"{arm}_n"] else np.nan)
            ax.plot([s * 100 for s in shifts], ys, "-o", color=colour, ms=4,
                    label=f"EBA {'ORF as called' if arm == 'whole' else 'oracle span'}")
        ctl = agg["by_scorer"].get("align", {}).get(key)
        if ctl:
            for arm, style in (("whole", ":"), ("oracle", "--")):
                if ctl[f"{arm}_n"]:
                    ax.axhline(100 * ctl[f"{arm}_top1"] / ctl[f"{arm}_n"], ls=style, lw=1.2,
                               color="#2f6fb3",
                               label=f"Smith-Waterman {'ORF as called' if arm == 'whole' else 'oracle span'}")
        ax.set_title(f"{thr:.0%} holdout", fontsize=10, loc="left")
        ax.set_xlabel("cosine shift (x100)")
        ax.grid(alpha=0.25, lw=0.6)
    np.atleast_1d(axes)[0].set_ylabel("payload family retrieved at rank 1 (%)")
    np.atleast_1d(axes)[0].set_ylim(-4, 104)
    np.atleast_1d(axes)[-1].legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.suptitle(f"EBA positional scorer, {agg['embedder']}: the shift plateau on the 23 hand-audited "
                 f"constructs", fontsize=11, x=0.02, ha="left")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_d6_eba_sweep_{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir}/fig_d6_eba_sweep_{stem}.png / .pdf")


if __name__ == "__main__":
    raise SystemExit(main())
