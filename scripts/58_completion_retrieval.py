#!/usr/bin/env python3
"""Does extending a short fragment with a 650M completion lift its payload's remote family? (R15)

    python scripts/58_completion_retrieval.py --ceiling-only            # the ThinkPad pre-check
    python scripts/58_completion_retrieval.py --scorer align            # the full alignment run
    python scripts/58_completion_retrieval.py --scorer facebook/esm2_t33_650M_UR50D

**The question, narrowed.** this project measured what `facebook/esm2_t33_650M_UR50D` can complete and
found a prior that appears **above 33 aa** and is absent below it: identity to truth runs -1.3 SE of the
matched baseline at 8 and 16 aa, then +2.1 / +2.2 / +4.3 / +3.2 at 33 / 66 / 150 / 300. So R15 is not
"does the model carry a prior", it does, but "does a completion at 33 to 66 aa **lift a remote
relative's rank**, and is any lift inference about the family or recall of a memorised sequence".
Below 33 aa there is nothing to measure and this script refuses those lengths by default.

**Why four queries and not two.** Appending a completion changes three things at once: the query gets
longer, it gains residues of some composition, and it gains residues in a particular *order*. Only the
third is the model's contribution, and a raw local-alignment score rises with query length for free.
`pilot.complete.extension_arms` therefore builds `fragment` (the control, the same string
`51_fragment_probe.py` queries), `extended` (the arm), `shuffled` (the model's own completion permuted:
same length, same composition, no order) and `truth` (the real continuation).

**`truth` is the ceiling, and it is why this script has a `--ceiling-only` mode.** It needs no model and
no GPU, and it answers the one question that decides whether the run is worth making: if appending the
*real* 12 residues does not move the rank, then nothing a model could emit will either, the manipulation
is too small for the instrument, and no statement about the 650M is available from the full run. That is
the falsifying case named before the measurement, and unlike R14's, this one is
checkable in advance and for free.

**The contamination gate travels with every row.** `identity(completion, truth)` is a column here, not a
footnote: a lift at 30% identity to the real continuation is inference about the family; one at 95% is
retrieval of a sequence the weights memorised: still operationally useful, and a different claim
. ESM-2 trained on UniRef and all 173 payloads are Swiss-Prot, so "completed it" and
"memorised it" predict the same rank and only this column separates them.

**What is shared with the rest of the benchmark, because it has to be.** The catalogue, the gated
homolog table, the holdout construction and `pilot.align.best_target_rank`, so a rank here means what
a rank in `twilight_*.csv` and `fragment_probe_*.csv` means. What is *not* shared is arms: this script
has four because the manipulation needs them, where `51_fragment_probe.py` has none because a fragment
is one query.

**Limits this run is bound by, stated rather than discovered later.** The completions exist only as
`results/completion_650m/completion_sweep_*.csv`, which carries `n_completed = 12` and nothing else, so
the extension length is 12 residues and is not swept: +36% of a 33 aa fragment and +18% of a 66 aa
one. The sweep has no `start` column either, so the prefix-cut convention is re-derived against the
catalogue at startup and the run **refuses** rather than assuming it (this project's habit).

Writes `completion_retrieval_<scorer>.csv` and `completion_retrieval_<scorer>_summary.json`.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.align import best_target_rank, get_scorer, model_kwargs  # noqa: E402
from pilot.complete import (  # noqa: E402
    EXTENSION_ARMS,
    GENERIC_ARMS,
    check_prefix_cut,
    donor_completions,
    extension_arms,
    fixed_tails,
    identity,
    read_completion_sweep,
)
from pilot.homology import MAX_LEN, MIN_LEN, Homolog, score_floor  # noqa: E402
from pilot.reference import iter_fasta  # noqa: E402

#: The same sweep every other retrieval script walks, so a row here is comparable at the same holdout.
THRESHOLDS = (1.00, 0.90, 0.70, 0.50, 0.35, 0.25)

#: The lengths 69.2 leaves open. Not a tunable: below 33 aa both checkpoints sit on the floor and
#: `--lengths 8` is a run that cannot inform anything.
DEFAULT_LENGTHS = (33, 66)

#: Fixed by the user 19 Oct.
RECALL_AT = 10


def load_homologs(path: Path) -> dict[str, list[Homolog]]:
    """Payload accession -> its homolog rows, read exactly as `51_fragment_probe.py` reads them."""
    out: dict[str, list[Homolog]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            out[r["payload_accession"]].append(
                Homolog(r["homolog_accession"], float(r["score"]), float(r["identity"]),
                        float(r["coverage"]), int(r["length"])))
    return out


def paired(rows: list[dict], arm: str, against: str = "fragment") -> dict:
    """Better / equal / worse over the constructs where both arms produced a rank.

    A mean of ranks across payloads is dominated by whichever payload happens to sit at 20,000, so the
    headline is the paired count and the net is reported beside it rather than instead of it. An
    unranked cell (no target survives the holdout) is excluded from both sides and counted, because
    dropping it silently would make the denominators of the two arms differ.
    """
    better = worse = equal = unpaired = 0
    net = 0
    for r in rows:
        a, b = r[f"rank_{against}"], r[f"rank_{arm}"]
        if a == "" or b == "":
            unpaired += 1
            continue
        a, b = int(a), int(b)
        net += a - b
        if b < a:
            better += 1
        elif b > a:
            worse += 1
        else:
            equal += 1
    return dict(better=better, equal=equal, worse=worse, unpaired=unpaired, net_rank_points=net)


def summarise(rows: list[dict]) -> list[dict]:
    """One row per (threshold, fragment length): the metrics the reporting rules allow.

    The cell is the stratum. Each payload contributes exactly one row to each cell, so micro and macro
    coincide here and 27.4's distinction does not bite: said explicitly, because it does bite
    everywhere else in this repository and a reader is entitled to know which case this is.
    """
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["threshold"], r["frag_len_aa"])].append(r)
    out = []
    for (thr, m), rs in sorted(by.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        row = dict(threshold=thr, frag_len_aa=m, n_payloads=len({r["payload_accession"] for r in rs}),
                   n_queries=len(rs))
        for arm in EXTENSION_ARMS:
            ranked = [int(r[f"rank_{arm}"]) for r in rs if r[f"rank_{arm}"] != ""]
            row[f"recall_at_{RECALL_AT}_{arm}"] = round(
                sum(1 for x in ranked if x <= RECALL_AT) / len(rs), 4) if rs else 0.0
            row[f"mrr_{arm}"] = round(sum(1.0 / x for x in ranked) / len(rs), 4) if rs else 0.0
            row[f"median_rank_{arm}"] = "" if not ranked else int(np.median(ranked))
            row[f"n_rank_ge_1e4_{arm}"] = sum(1 for x in ranked if x >= 10_000)
        for arm in EXTENSION_ARMS[1:]:          # every arm except the `fragment` baseline
            p = paired(rs, arm)
            row[f"{arm}_better"], row[f"{arm}_equal"], row[f"{arm}_worse"] = (
                p["better"], p["equal"], p["worse"])
            row[f"{arm}_net"] = p["net_rank_points"]
        # The decisive pairing for the fixed-tail control is against `extended`, not against
        # `fragment`. this project does not predict that a generic tail helps: `extended` already
        # does, and the open question is whether the model's own output beats a string that knows
        # nothing about the query. Against `fragment` both arms would simply look like wins.
        for arm in (*GENERIC_ARMS, "misconditioned"):
            p = paired(rs, arm, against="extended")
            row[f"{arm}_vs_extended_better"] = p["better"]
            row[f"{arm}_vs_extended_equal"] = p["equal"]
            row[f"{arm}_vs_extended_worse"] = p["worse"]
            row[f"{arm}_vs_extended_net"] = p["net_rank_points"]
        ids = [r["identity_to_truth"] for r in rs if r["identity_to_truth"] != ""]
        row["mean_identity_to_truth"] = round(float(np.mean(ids)), 4) if ids else ""
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path,
                    default=Path("results/gate_all173/reference_twilight.fasta"))
    ap.add_argument("--homologs", type=Path, default=Path("results/gate_all173/homologs.csv"))
    ap.add_argument("--completions", type=Path,
                    default=Path("results/completion_650m/completion_sweep_esm2_t33_650M_UR50D.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/completion_retrieval"))
    ap.add_argument("--scorer", default="align",
                    help="bare checkpoint name for ESM-2; there is no esm2: prefix")
    ap.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS),
                    help="fragment lengths in aa. 69.2 leaves 33 and 66 open and nothing below 33")
    ap.add_argument("--mode", default="single", choices=("single", "iterative"),
                    help="which completion mode's rows to use; both are in the sweep")
    ap.add_argument("--ceiling-only", action="store_true",
                    help="score the `fragment` and `truth` arms only. No model is loaded and no "
                         "completion is read for its content, so this is the free pre-check: if the "
                         "real continuation does not move the rank, the run cannot inform anything")
    ap.add_argument("--max-payloads", type=int, default=0,
                    help="0 = every eligible payload. The binding count is reported either way")
    ap.add_argument("--max-evalue", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=1, help="alignment arm only")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/embed_cache"))
    args = ap.parse_args(argv)

    for p in (args.reference, args.homologs, args.completions):
        if not p.is_file():
            print(f"not found: {p}", file=sys.stderr)
            return 2
    if min(args.lengths) < 33:
        print("69.2 puts both checkpoints on the floor below 33 aa; pass --lengths 33 66 or longer",
              file=sys.stderr)
        return 2

    refs = list(iter_fasta(args.reference))
    ref_acc = [e.accession for e in refs]
    idx_of = {a: i for i, a in enumerate(ref_acc)}
    ref_by_acc = {e.accession: e.seq for e in refs}
    homologs = load_homologs(args.homologs)
    in_window = [e for e in refs if MIN_LEN <= len(e.seq) <= MAX_LEN]
    db_residues = sum(len(e.seq) for e in in_window)

    comps = read_completion_sweep(args.completions, lengths=args.lengths, mode=args.mode)
    if not comps:
        print(f"no rows at lengths {args.lengths} mode {args.mode} in {args.completions}",
              file=sys.stderr)
        return 2

    # ---- provenance: the sweep records no cut position, so derive it rather than assume it ---------
    confirmed, bad = check_prefix_cut(comps, ref_by_acc.get)
    if bad:
        print(f"the prefix-cut convention does not hold for {len(bad)} of {len(comps)} rows; "
              f"every query would be built on a guess. First three: {bad[:3]}", file=sys.stderr)
        return 2
    print(f"provenance: prefix cut confirmed against the catalogue on {confirmed} of {len(comps)}  "
          f"completion rows, 0 disagreements")

    # ---- eligibility, with every exclusion counted (this project's working rules: report binding, not settings) ------
    loosest = max(t for t in THRESHOLDS if t < 1.0)
    keep, not_in_cat, no_relative = [], 0, 0
    for c in comps:
        if c.payload_accession not in ref_by_acc:
            not_in_cat += 1
            continue
        if not [h for h in homologs.get(c.payload_accession, [])
                if h.identity <= loosest and h.accession != c.payload_accession
                and h.accession in idx_of]:
            no_relative += 1
            continue
        keep.append(c)
    if not keep:
        print("no completion row is eligible; nothing to score", file=sys.stderr)
        return 2
    if args.max_payloads:
        chosen = sorted({c.payload_accession for c in keep})[:args.max_payloads]
        keep = [c for c in keep if c.payload_accession in chosen]

    n_pay = len({c.payload_accession for c in keep})
    print(f"catalogue: {len(refs):,} entries, {len(in_window):,} in the {MIN_LEN}-{MAX_LEN} aa window, "
          f"{db_residues:,} residues")
    print(f"completions: {len(keep)} rows over {n_pay} payloads at lengths {sorted(args.lengths)}, "
          f"mode {args.mode}, {keep[0].n_completed} residues appended (excluded: {not_in_cat} absent "
          f"from the catalogue, {no_relative} with no relative in it below {loosest:.2f})")

    # One tail length for the whole run, so one pair of fixed tails serves every query, which is
    # what makes them a control. The line above has always printed `keep[0].n_completed` as if the
    # sweep were uniform and never checked it; print the count and refuse instead.
    widths = sorted({c.n_completed for c in keep})
    if len(widths) != 1:
        print(f"the sweep appends {len(widths)} different numbers of residues {widths}; a fixed "
              f"tail has one length, so filter with --lengths or --mode first", file=sys.stderr)
        return 2
    donors = donor_completions(keep)
    # The assignment is a derangement by SLOT, so no row is handed its own row back. It can still
    # be handed a string equal to its own, because two payloads can produce the same completion:
    # 13 of these collapse to `L`. The first run refused on that, and the refusal was wrong: this
    # repository already decided the case when a degenerate completion shuffles to itself, and the
    # right behaviour is to produce it rather than perturb it, because forcing a difference would
    # make the control a different query. So this is a binding count, printed, not a refusal.
    same = [c for c in keep
            if donors[(c.payload_accession, c.fragment_len, c.mode)] == c.completed]
    print(f"donor completions: {len(donors)} assigned within (length, mode) groups, derangement by "
          f"slot on all of them; {len(same)} received a string equal to their own because two "
          f"payloads completed identically, so `misconditioned` == `extended` on those cells")
    tails = fixed_tails([e.seq for e in refs], widths[0])
    print(f"fixed tails: {widths[0]} residues each, derived from all {len(refs):,} catalogue entries "
          f"and identical for every query: "
          + ", ".join(f"{a}={tails[a]}" for a in GENERIC_ARMS))

    arms = ("fragment", "truth") if args.ceiling_only else EXTENSION_ARMS
    if args.ceiling_only:
        print("ceiling-only: scoring `fragment` and `truth`. If `truth` does not beat `fragment` the "
              "manipulation is below the instrument and the full run says nothing about the model.")

    is_align = args.scorer in ("align", "sw", "alignment")
    t0 = time.time()
    scorer = get_scorer(args.scorer, [e.seq for e in refs], cache_dir=(args.cache_dir or None),
                        threads=args.threads,
                        **model_kwargs(args.scorer, batch_size=args.batch_size,
                                       max_length=args.max_length))
    print(f"scorer ready in {time.time() - t0:.0f}s ({scorer.name})", flush=True)
    if not is_align:
        print("  an embedding arm: read recall@10 and MRR, and a mean rank only beside its "
              "rank >= 10,000 count ", flush=True)

    rows: list[dict] = []
    t0, n_scored = time.time(), 0
    for n, c in enumerate(keep, 1):
        seq = ref_by_acc[c.payload_accession]
        key = f"{c.payload_accession}:{c.fragment_len}:{c.mode}"
        q = extension_arms(seq[:c.fragment_len], c.completed, c.truth, key, tails,
                           donors[(c.payload_accession, c.fragment_len, c.mode)])
        hs = homologs.get(c.payload_accession, [])
        scored = {}
        for arm in arms:
            scored[arm] = scorer.score_all(q[arm])
            n_scored += 1
        for thr in THRESHOLDS:
            mask = np.array([idx_of[h.accession] for h in hs
                             if h.identity > thr and h.accession in idx_of], dtype=int)
            tgt = np.array([idx_of[h.accession] for h in hs
                            if h.identity <= thr and h.accession in idx_of], dtype=int)
            if thr >= 1.0:
                mask = np.array([], dtype=int)
                tgt = np.array([idx_of[c.payload_accession]], dtype=int)
            row = dict(
                scorer=scorer.name, payload_accession=c.payload_accession,
                payload_len_aa=len(seq), frag_len_aa=c.fragment_len,
                frag_len_nt=c.fragment_len * 3, n_completed=c.n_completed, mode=c.mode,
                threshold=thr, n_homologs_available=int(tgt.size), n_held_out=int(mask.size),
                identity_to_truth=round(c.identity_to_truth, 4),
                degenerate=int(len(set(c.completed)) == 1),
            )
            for arm in EXTENSION_ARMS:
                if arm not in arms:
                    row[f"rank_{arm}"] = ""
                    row[f"len_{arm}"] = ""
                    continue
                r = best_target_rank(scored[arm], tgt, mask)
                row[f"rank_{arm}"] = "" if r is None else r
                row[f"len_{arm}"] = len(q[arm])
                if is_align:
                    floor = score_floor(args.max_evalue, len(q[arm]), db_residues)
                    ts = float(np.max(scored[arm][tgt])) if tgt.size else None
                    row[f"target_clears_floor_{arm}"] = "" if ts is None else int(ts >= floor)
            rows.append(row)
        if n % 10 == 0 or n == len(keep):
            print(f"    {n}/{len(keep)} completions, {n_scored} queries, {time.time() - t0:.0f}s",
                  flush=True)
    scoring_seconds = time.time() - t0

    # ---- output ----------------------------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = scorer.name.replace("/", "_").replace(":", "_")
    if args.ceiling_only:
        tag += "_ceiling"
    csv_path = args.out_dir / f"completion_retrieval_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary_rows = summarise(rows)
    cols = ["threshold", "frag_len_aa", "n_payloads"]
    cols += [f"recall_at_{RECALL_AT}_{a}" for a in arms] + [f"mrr_{a}" for a in arms]
    cols += [f"{a}_better" for a in arms if a != "fragment"]
    cols += [f"{a}_worse" for a in arms if a != "fragment"]
    print()
    print("  ".join(f"{c:>22}" for c in cols))
    for s in summary_rows:
        print("  ".join(f"{str(s.get(c, '')):>22}" for c in cols))

    # ---- the verdict the ceiling arm exists to deliver --------------------------------------------
    ceiling = {}
    for m in sorted({r["frag_len_aa"] for r in rows}):
        at = [r for r in rows if r["frag_len_aa"] == m and r["threshold"] < 1.0]
        ceiling[m] = paired(at, "truth")
    print()
    print("the ceiling, over the five remote holdouts: does appending the REAL continuation move the "
          "rank at all?")
    for m, p in ceiling.items():
        verdict = ("the manipulation is below the instrument at this length: a model cannot beat a "
                   "perfect completion, so a full run here would measure nothing"
                   if p["better"] == 0 else
                   f"yes on {p['better']} paired cells, so there is room for a completion to act in")
        print(f"    {m} aa: truth better on {p['better']}, equal on {p['equal']}, worse on "
              f"{p['worse']} -> {verdict}")

    if not args.ceiling_only:
        print()
        print("the contamination gate: a lift is only inference if the completion is "
              "NOT the real sequence.")
        for m in sorted({r["frag_len_aa"] for r in rows}):
            at = [r for r in rows if r["frag_len_aa"] == m and r["threshold"] < 1.0]
            gain = [r for r in at if r["rank_extended"] != "" and r["rank_fragment"] != ""
                    and int(r["rank_extended"]) < int(r["rank_fragment"])]
            ids = [r["identity_to_truth"] for r in gain]
            print(f"    {m} aa: {len(gain)} cells improved; identity to truth on those cells "
                  f"{'n/a' if not ids else f'mean {np.mean(ids):.3f}, max {max(ids):.3f}'}")

    summary = dict(
        scorer=scorer.name, scorer_spec=args.scorer, is_alignment=is_align,
        ceiling_only=args.ceiling_only, arms=list(arms),
        reference=str(args.reference), homologs=str(args.homologs),
        completions=str(args.completions), completion_mode=args.mode,
        n_completed_residues=keep[0].n_completed,
        prefix_cut_confirmed=confirmed, prefix_cut_disagreements=0,
        catalogue_entries=len(refs), catalogue_in_window=len(in_window), db_residues=db_residues,
        thresholds=list(THRESHOLDS), lengths_aa=sorted(args.lengths),
        payloads=sorted({c.payload_accession for c in keep}), n_payloads=n_pay,
        excluded=dict(absent_from_catalogue=not_in_cat,
                      no_relative_below_the_loosest_holdout=no_relative),
        recall_at=RECALL_AT, by_threshold_and_length=summary_rows,
        ceiling_over_remote_holdouts={str(k): v for k, v in ceiling.items()},
        shuffle="deterministic, sha256-seeded on payload:length:mode, stable across machines",
        max_evalue=args.max_evalue, batch_size=args.batch_size, max_length=args.max_length,
        threads=args.threads, n_queries_scored=n_scored,
        scoring_seconds=round(scoring_seconds, 1),
    )
    with open(args.out_dir / f"completion_retrieval_{tag}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {csv_path} ({len(rows):,} rows) and the summary JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
