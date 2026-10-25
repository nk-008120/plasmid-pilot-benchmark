#!/usr/bin/env python3
"""Can a short fragment of a payload still retrieve that payload's remote family?

    python scripts/51_fragment_probe.py --scorer align --max-payloads 12     # the control, ThinkPad
    python scripts/51_fragment_probe.py --scorer facebook/esm2_t33_650M_UR50D --max-payloads 12

**What this measures, and what it does not.** The proposal's first extension is short fragments, 25 to
200 nt. `scripts/50_fragment_bound.py` already says what alignment statistics *permit* there: the
significance gate inverts to a score floor, and on the 40,750-protein catalogue no fragment under
**18 aa / 54 nt** can reach `E <= 1e-3` even against a perfect copy of itself. That is a ceiling derived
from arithmetic. This script is the measurement the ceiling predicts about: it cuts fragments at lengths
spanning the bound, queries the catalogue with each one, and reports the rank of the payload's surviving
homologs at every holdout. The bound says the curve must be flat on the floor below 18 aa; anything above
that is unmeasured until this runs.

**It is not a decomposition experiment**, so it does not go through `build_arms` and has no arms. A
fragment is already a single query, there is nothing to segment, and `pick_payload_orf`'s 50 aa minimum
would silently drop every fragment shorter than itself, which is most of the range in question. What is
shared with the main benchmark is the part that has to be: the catalogue, the homolog table, the holdout
construction, and `pilot.align.best_target_rank`, so a rank here means exactly what a rank in
`twilight_*.csv` means.

**Five things it fixes in place, each of which is a way to produce a meaningless number:**

1. **Lengths span the bound**, so the curve crosses it rather than sitting on one side. The default grid
   is 8 to 66 aa (24 to 198 nt), which covers the proposal's range and brackets 18.
2. **Fragments are cut from the payload's own catalogue sequence**, so a right answer exists to be found,
   and from the *payload* rather than from a construct, so this is not a dilution measurement wearing a
   fragment's clothes. Dilution is what `10_twilight.py` measures.
3. **One frame, stated: the payload's own.** Cutting a peptide out of a protein is the generous case, a
   nucleotide fragment in unknown frame yields a shorter clean peptide than its length in codons. This
   error points the same way as the bound's, so both are ceilings and neither flatters the extension.
4. **The holdout is the same gated homolog table split by threshold**, not a new mask. Building a mask by
   identity alone would hold out chance alignments and make retrieval look harder than specified
; that trap is exactly this shape and is available here.
5. **`recall@10` and MRR are primary**, per the working record and, for an embedding scorer, 32.2, which is
   why mean rank is printed only beside its count of rank >= 10,000 failures.

**Offsets are deterministic and there is no seed.** For `--n-offsets 3` the cuts start at 25%, 50% and
75% of the available span, so one unlucky window cannot be the result and re-running cannot change it.

**Threshold 1.00 is the bound's own condition, not an afterthought.** The bound assumes the fragment
aligns to a perfect copy of itself, and at a 1.00 holdout the payload's own entry is the target and
nothing is masked, so that row is where the predicted crossing is testable. The remote thresholds are the
question a screen actually asks.

Writes `fragment_probe_<scorer>.csv`, `fragment_probe_<scorer>_summary.json`.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.align import best_target_rank, get_scorer, model_kwargs  # noqa: E402
from pilot.homology import MAX_LEN, MIN_LEN, Homolog, score_floor  # noqa: E402
from pilot.reference import iter_fasta  # noqa: E402

#: The same sweep `10_twilight.py` walks, so a fragment row and a whole-ORF row are comparable at the
#: same holdout. 1.00 is kept for the reason the module docstring gives.
THRESHOLDS = (1.00, 0.90, 0.70, 0.50, 0.35, 0.25)

#: Fragment lengths in amino acids. 18 is T5's bound on the 40,750-protein catalogue and the rest
#: bracket it: three below, where significance is provably unreachable, and five above, spanning the
#: proposal's range to its 200 nt top (66 aa). Not a tunable: it is the x-axis of the measurement.
DEFAULT_LENGTHS = (8, 12, 16, 18, 20, 25, 33, 50, 66)

#: `recall@k` depth, fixed by the user 19 Oct and not swept here.
RECALL_AT = 10


def load_homologs(path: Path) -> dict[str, list[Homolog]]:
    """Payload accession -> its homolog rows. A third copy of this six-line reader (`10_twilight.py` and
    `18_eba_sweep.py` have the others) rather than a refactor of two measurement scripts on the night
    before a GPU window. What must not drift is the ranking rule, and that is one function in
    `pilot.align`."""
    out: dict[str, list[Homolog]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            out[r["payload_accession"]].append(
                Homolog(r["homolog_accession"], float(r["score"]), float(r["identity"]),
                        float(r["coverage"]), int(r["length"])))
    return out


def cut_starts(payload_len: int, frag_len: int, n_offsets: int) -> list[int]:
    """Deterministic 0-based start positions, evenly spaced over the available span.

    No RNG and no seed: for `n_offsets = 3` the cuts begin at 25%, 50% and 75% of `payload_len - frag_len`,
    so the measurement cannot move between runs and no single window can be the whole result. A payload
    exactly as long as the fragment yields one cut at 0, and duplicates are collapsed, which happens
    for short payloads and is why the emitted row carries its own `start`.
    """
    if frag_len > payload_len:
        return []
    span = payload_len - frag_len
    if span == 0:
        return [0]
    return sorted({int(round(span * (i + 1) / (n_offsets + 1))) for i in range(n_offsets)})


def _bound_prediction(reference: Path, db_residues: int, max_evalue: float) -> dict:
    """T5's bound for this catalogue, computed by loading `50_fragment_bound.py` rather than restating it.

    The probe prints the prediction it is about to test, from the same code that derived it. A number
    retyped here could drift from the script that owns it, and then the probe would be checking itself
    against a stale ceiling.
    """
    p = Path(__file__).with_name("50_fragment_bound.py")
    spec = importlib.util.spec_from_file_location("_fragment_bound", p)
    if spec is None or spec.loader is None:  # pragma: no cover (only if the file is gone)
        return {}
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    freq, freq_src = mod.composition(reference)
    s_comp = sum(freq[a] * mod.BLOSUM62_DIAGONAL[a] for a in mod.BLOSUM62_DIAGONAL)
    s_max = float(max(mod.BLOSUM62_DIAGONAL.values()))
    return {
        "min_aa_composition": mod.min_length(s_comp, db_residues, max_evalue),
        "min_aa_most_generous": mod.min_length(s_max, db_residues, max_evalue),
        "s_diag_composition": round(s_comp, 4),
        "composition_source": freq_src,
    }


def summarise(rows: list[dict], scorer_is_alignment: bool) -> list[dict]:
    """One row per (threshold, length): the metrics the reporting rules allow, and nothing else.

    `recall@10` and MRR first because an embedding arm's rank distribution is bimodal and its mean does
    not describe the centre (32.2). The mean is still emitted, and so is the count of rank >= 10,000 that
    has to be quoted beside it, so the caveat travels with the number instead of living in prose.
    """
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["threshold"], r["frag_len_aa"], r["cut"])].append(r)
    out = []
    for (thr, m, cut), rs in sorted(by.items(), key=lambda kv: (-kv[0][0], kv[0][2], kv[0][1])):
        ranked = [int(r["best_target_rank"]) for r in rs if r["best_target_rank"] != ""]
        n_hopeless = sum(1 for r in rs if r["best_target_rank"] == "")
        rec = sum(1 for x in ranked if x <= RECALL_AT) / len(rs) if rs else 0.0
        mrr = sum(1.0 / x for x in ranked) / len(rs) if rs else 0.0
        row = dict(
            threshold=thr, frag_len_aa=m, frag_len_nt=m * 3 if cut == "fragment" else "", cut=cut,
            n_queries=len(rs), n_payloads=len({r["payload_accession"] for r in rs}),
            n_unscorable=n_hopeless,
            **{f"recall_at_{RECALL_AT}": round(rec, 4)},
            mrr=round(mrr, 4),
            mean_rank="" if not ranked else round(sum(ranked) / len(ranked), 2),
            median_rank="" if not ranked else int(np.median(ranked)),
            n_rank_ge_1e4=sum(1 for x in ranked if x >= 10_000),
        )
        # The bound's own quantity, and only where it has units. Under a cosine scorer a score row is not
        # in BLOSUM62 points and comparing it to a floor near 100 would be this project's error again.
        if scorer_is_alignment:
            cl = [r for r in rs if r["best_score_clears_floor"] != ""]
            row["frac_best_clears_floor"] = (
                "" if not cl else round(sum(1 for r in cl if r["best_score_clears_floor"] == 1) / len(cl), 4))
            tc = [r for r in rs if r["target_score_clears_floor"] != ""]
            row["frac_target_clears_floor"] = (
                "" if not tc else round(sum(1 for r in tc if r["target_score_clears_floor"] == 1) / len(tc), 4))
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path,
                    default=Path("results/gate_all173/reference_twilight.fasta"))
    ap.add_argument("--homologs", type=Path, default=Path("results/gate_all173/homologs.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/fragment_probe"))
    ap.add_argument("--scorer", default="align",
                    help="bare checkpoint name for ESM-2; there is no esm2: prefix")
    ap.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS),
                    help="fragment lengths in amino acids")
    ap.add_argument("--n-offsets", type=int, default=3,
                    help="cuts per payload per length, evenly spaced and deterministic")
    ap.add_argument("--max-payloads", type=int, default=12,
                    help="payloads to probe, taken evenly spaced from the sorted eligible list. The "
                         "binding count is reported")
    ap.add_argument("--no-full-control", action="store_true",
                    help="skip the whole-payload query. It is the control: a full-length 'fragment' must "
                         "reproduce the rank the payload already has in the twilight benchmark")
    ap.add_argument("--max-evalue", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048,
                    help="in the reference-matrix cache key; 2048 is what every other "
                         "ESM-2 run in this project used")
    ap.add_argument("--threads", type=int, default=1, help="alignment arm only")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/embed_cache"))
    args = ap.parse_args(argv)

    for p in (args.reference, args.homologs):
        if not p.is_file():
            print(f"not found: {p}", file=sys.stderr)
            return 2
    if min(args.lengths) < 1:
        print("fragment lengths must be positive", file=sys.stderr)
        return 2

    refs = list(iter_fasta(args.reference))
    ref_acc = [e.accession for e in refs]
    idx_of = {a: i for i, a in enumerate(ref_acc)}
    ref_by_acc = {e.accession: e.seq for e in refs}
    homologs = load_homologs(args.homologs)

    # The floor's `N` is the residue count of the database actually searched, under the same 50-5000 aa
    # window the catalogue's own reader applies. The scorer still scores every entry.
    in_window = [e for e in refs if MIN_LEN <= len(e.seq) <= MAX_LEN]
    db_residues = sum(len(e.seq) for e in in_window)

    # ---- payload selection, with every exclusion counted ---------------------------------------
    # Per this project's working rules: a limit that binds is reported, and a limit nothing reaches is free. Each count
    # below is a reason a payload cannot answer the question, and they are different reasons.
    loosest = max(t for t in THRESHOLDS if t < 1.0)
    longest = max(args.lengths)
    eligible, not_in_cat, too_short, no_relative = [], 0, 0, 0
    for acc in sorted(homologs):
        if acc not in ref_by_acc:
            not_in_cat += 1
            continue
        if len(ref_by_acc[acc]) < longest:
            too_short += 1
            continue
        if not [h for h in homologs[acc]
                if h.identity <= loosest and h.accession != acc and h.accession in idx_of]:
            # The payload has nothing in the catalogue but itself, so at every holdout below 1.00 it is
            # a decided miss before a fragment is cut, and such a row says nothing about fragment
            # length. The gate is the LOOSEST holdout on purpose: requiring a sub-25% relative here
            # would quietly shrink the sample at 0.90 and 0.70 as well, and the per-threshold attrition
            # belongs in `n_unscorable`, which is reported, rather than in the selection.
            no_relative += 1
            continue
        eligible.append(acc)

    if not eligible:
        print("no payload is eligible; nothing to probe", file=sys.stderr)
        return 2
    # Evenly spaced over the sorted list rather than the first N, so the sample is not an artefact of
    # where the alphabet starts, and deterministic so a re-run probes the same payloads.
    if args.max_payloads and len(eligible) > args.max_payloads:
        k = args.max_payloads
        chosen = [eligible[int(round(i * (len(eligible) - 1) / (k - 1)))] for i in range(k)] if k > 1 \
            else [eligible[len(eligible) // 2]]
        chosen = sorted(dict.fromkeys(chosen))
    else:
        chosen = eligible

    pred = _bound_prediction(args.reference, db_residues, args.max_evalue)
    print(f"catalogue: {len(refs):,} entries, {len(in_window):,} in the {MIN_LEN}-{MAX_LEN} aa window, "
          f"{db_residues:,} residues")
    print(f"payloads: {len(chosen)} probed of {len(eligible)} eligible of {len(homologs)} in the table "
          f"(excluded: {not_in_cat} absent from the catalogue, {too_short} shorter than "
          f"{longest} aa, {no_relative} with no relative in the catalogue but themselves)")
    if pred:
        print(f"the bound this probe tests (scripts/50_fragment_bound.py, {pred['composition_source']}): "
              f"significance is unreachable below {pred['min_aa_composition']} aa / "
              f"{pred['min_aa_composition'] * 3} nt, and below {pred['min_aa_most_generous']} aa even at "
              f"the most generous residue")
    print(f"lengths (aa): {args.lengths}   -> nt: {[m * 3 for m in args.lengths]}")
    print(f"frame: the payload's own protein, so every fragment is a clean peptide. This is the "
          f"generous case and it is a ceiling in the same direction as the bound.")

    t0 = time.time()
    scorer = get_scorer(args.scorer, [e.seq for e in refs], cache_dir=(args.cache_dir or None),
                        threads=args.threads,
                        **model_kwargs(args.scorer, batch_size=args.batch_size,
                                       max_length=args.max_length))
    is_align = args.scorer in ("align", "sw", "alignment")
    print(f"scorer ready in {time.time() - t0:.0f}s ({scorer.name})", flush=True)
    if not is_align:
        print("  an embedding arm: read recall@10 and MRR, and a mean rank only beside its "
              "rank >= 10,000 count ", flush=True)

    # ---- the queries ---------------------------------------------------------------------------
    queries: list[tuple[str, int, int, str]] = []  # (payload, frag_len, start, cut)
    for acc in chosen:
        L = len(ref_by_acc[acc])
        for m in sorted(args.lengths):
            for s in cut_starts(L, m, args.n_offsets):
                queries.append((acc, m, s, "fragment"))
        if not args.no_full_control:
            queries.append((acc, L, 0, "full"))
    print(f"{len(queries):,} queries ({len(chosen)} payloads x {len(args.lengths)} lengths x up to "
          f"{args.n_offsets} offsets"
          f"{'' if args.no_full_control else ', plus one full-payload control each'})", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    for n, (acc, m, start, cut) in enumerate(queries, 1):
        q = ref_by_acc[acc][start:start + m]
        row = scorer.score_all(q)
        top = int(np.argmax(row))
        floor = score_floor(args.max_evalue, len(q), db_residues) if is_align else None
        hs = homologs.get(acc, [])
        for thr in THRESHOLDS:
            mask = np.array([idx_of[h.accession] for h in hs
                             if h.identity > thr and h.accession in idx_of], dtype=int)
            tgt = np.array([idx_of[h.accession] for h in hs
                            if h.identity <= thr and h.accession in idx_of], dtype=int)
            if thr >= 1.0:
                # No holdout: the payload's own entry is the answer, exactly as in `10_twilight.py`.
                # This is the row the bound speaks to, because it is the perfect self-match it assumes.
                mask, tgt = np.array([], dtype=int), np.array([idx_of[acc]], dtype=int)
            r = best_target_rank(row, tgt, mask)
            masked_row = row.astype(np.float64, copy=True)
            masked_row[mask] = -np.inf
            best_avail = float(np.max(masked_row)) if np.isfinite(masked_row).any() else float("-inf")
            tgt_score = float(np.max(row[tgt])) if tgt.size else None
            rows.append(dict(
                scorer=scorer.name, cut=cut, frag_len_aa=m,
                frag_len_nt=m * 3 if cut == "fragment" else "",
                payload_accession=acc, payload_len_aa=len(ref_by_acc[acc]), start_0based=start,
                frame="payload_protein", threshold=thr,
                n_homologs_available=int(tgt.size), n_held_out=int(mask.size),
                best_target_rank="" if r is None else r,
                target_score="" if tgt_score is None else round(tgt_score, 4),
                top1_accession=ref_acc[top], top1_score=round(float(row[top]), 4),
                score_floor="" if floor is None else round(floor, 2),
                best_score_clears_floor="" if floor is None
                else int(np.isfinite(best_avail) and best_avail >= floor),
                target_score_clears_floor="" if (floor is None or tgt_score is None)
                else int(tgt_score >= floor),
            ))
        if n % 25 == 0 or n == len(queries):
            print(f"    {n:,}/{len(queries):,} queries, {time.time() - t0:.0f}s", flush=True)

    scoring_seconds = time.time() - t0

    # ---- output -------------------------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = scorer.name.replace("/", "_").replace(":", "_")
    csv_path = args.out_dir / f"fragment_probe_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary_rows = summarise(rows, is_align)
    print()
    cols = ["threshold", "cut", "frag_len_aa", "frag_len_nt", "n_queries",
            f"recall_at_{RECALL_AT}", "mrr", "mean_rank", "n_rank_ge_1e4"]
    if is_align:
        cols += ["frac_best_clears_floor"]
    print("  ".join(f"{c:>18}" for c in cols))
    for s in summary_rows:
        print("  ".join(f"{str(s.get(c, '')):>18}" for c in cols))

    # ---- the two crossings, which are NOT the same question ------------------------------------
    # The bound is a statement about *significance*: no fragment under m aa can reach E <= 1e-3 even
    # against a perfect copy of itself. It says nothing about *rank*. A short fragment can sit at rank 1
    # on a score far below the floor, because ranking only needs to beat the other entries. Reporting one
    # crossing and calling it "the bound checked" would be this repository's own recurring error,
    # testing a quantity against a condition derived for a different one, so both are emitted and only
    # the significance crossing is compared with the prediction.
    def _crossing(key, test) -> dict:
        out = {}
        for thr in THRESHOLDS:
            at = [s for s in summary_rows if s["threshold"] == thr and s["cut"] == "fragment"]
            hit = [s["frag_len_aa"] for s in at if test(s)]
            out[f"{thr:.2f}"] = min(hit) if hit else None
        return out

    def _fmt(v):
        return "none in this grid" if v is None else f"{v} aa / {v * 3} nt"

    cross_rank = _crossing("rank", lambda s: s[f"recall_at_{RECALL_AT}"] > 0)
    cross_sig = (_crossing("sig", lambda s: (s.get("frac_best_clears_floor") or 0) > 0)
                 if is_align else {})
    print()
    print(f"shortest fragment reaching recall@{RECALL_AT} > 0 (a RANK statement), by holdout:")
    for k, v in cross_rank.items():
        print(f"  holdout {k}: {_fmt(v)}")
    if is_align:
        print()
        print("shortest fragment whose best available hit clears the significance floor (what the bound "
              "is about):")
        for k, v in cross_sig.items():
            print(f"  holdout {k}: {_fmt(v)}")
    if pred and is_align:
        # The bound is a BRACKET and must be read as one. `min_aa_composition` weights the BLOSUM62
        # diagonal by the catalogue's average composition, so it describes a *typical* fragment;
        # `min_aa_most_generous` uses the highest diagonal entry and is the floor over *every* possible
        # fragment. A real 16-mer rich in W, C, H, Y or F scores above the average and can therefore
        # clear the floor shorter than the composition value, with no contradiction anywhere. Only a
        # crossing below the most-generous value would say the arithmetic is wrong.
        typ, hard = pred["min_aa_composition"], pred["min_aa_most_generous"]
        got = cross_sig.get("1.00")
        if got is None:
            verdict = ("no fragment in this grid reached significance even against a perfect copy of "
                       f"itself: consistent if the grid tops out below {typ} aa")
        elif got < hard:
            verdict = (f"BELOW the most generous bound ({hard} aa), which is a floor over every "
                       f"possible fragment. The arithmetic or this run is wrong; do not report the "
                       f"extension until it is resolved")
        elif got < typ:
            verdict = (f"inside the bracket [{hard}, {typ}] aa, as a real fragment should be: the "
                       f"composition value is a median, not a floor")
        else:
            verdict = f"at or above the composition value ({typ} aa), the conservative end"
        print()
        print(f"at the 1.00 holdout, where the bound's perfect self-match is the one unmasked entry: "
              f"significance first reached at {_fmt(got)} -> {verdict}")
        # Where the median fragment crosses is the number the composition bound actually predicts, so it
        # is printed beside it rather than left for a reader to derive from the table.
        med = [s["frag_len_aa"] for s in summary_rows
               if s["threshold"] == 1.0 and s["cut"] == "fragment"
               and (s.get("frac_best_clears_floor") or 0) >= 0.5]
        if med:
            print(f"  half or more of the fragments first clear the floor at {_fmt(min(med))}, against "
                  f"a composition-weighted prediction of {typ} aa. That is what the composition value "
                  f"predicts and it is the honest way to quote it.")
        if (cross_rank.get("1.00") is not None and got is not None
                and cross_rank["1.00"] < got):
            print(f"  and note the gap: fragments of {_fmt(cross_rank['1.00'])} already rank their own "
                  f"family inside the top {RECALL_AT} while scoring below the floor. That is not a "
                  f"contradiction: it is a fragment that a screen applying an E-value gate would "
                  f"discard despite its rank, and it is the honest form of the extension's claim.")

    summary = dict(
        scorer=scorer.name, scorer_spec=args.scorer, is_alignment=is_align,
        reference=str(args.reference), homologs=str(args.homologs),
        catalogue_entries=len(refs), catalogue_in_window=len(in_window), db_residues=db_residues,
        max_evalue=args.max_evalue, thresholds=list(THRESHOLDS), lengths_aa=list(args.lengths),
        n_offsets=args.n_offsets, offsets="deterministic, evenly spaced, no seed",
        frame="payload_protein (clean peptide; a ceiling, like the bound)",
        payloads_probed=chosen, n_payloads_eligible=len(eligible),
        n_payloads_in_table=len(homologs),
        excluded=dict(absent_from_catalogue=not_in_cat, shorter_than_longest_fragment=too_short,
                      no_relative_below_the_loosest_holdout=no_relative),
        full_payload_control=not args.no_full_control,
        n_queries=len(queries), recall_at=RECALL_AT,
        bound_prediction=pred,
        crossings_rank_aa=cross_rank, crossings_significance_aa=cross_sig,
        crossing_note=("rank and significance are different claims: a fragment can rank its family "
                       "first on a score below the E-value floor. Only the significance crossing is "
                       "comparable with the derived bound."),
        by_threshold_and_length=summary_rows,
        batch_size=args.batch_size, max_length=args.max_length,
        # Added 22 Oct, after R13. That run existed to settle an alignment
        # rate, a rate is a function of the thread count, and this summary recorded nine
        # settings and not that one, so its own answer had to be quoted as 0.327 s/query
        # *if it was run as written*, with no artefact able to confirm it. A setting that a
        # reported number is a function of belongs beside the number. scoring_seconds covers
        # the catalogue-pass loop alone, so scoring_seconds / n_queries is the rate and does
        # not include writing the output.
        threads=args.threads, scoring_seconds=round(scoring_seconds, 1),
    )
    with open(args.out_dir / f"fragment_probe_{tag}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {csv_path} ({len(rows):,} rows) and the summary JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
