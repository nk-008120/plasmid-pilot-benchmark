#!/usr/bin/env python3
"""Report the best-hit segmenter against the existing baseline, paired, with the spans beside it.

    python scripts/25_bhs_report.py \
        --bhs      results/gate173_bhs_hand/twilight_sw-blosum62-11-1_hand.csv \
        --baseline results/gate173_control_hand/twilight_sw-blosum62-11-1_hand.csv \
        --spans    results/gate173_bhs_hand/bhs_spans_sw-blosum62-11-1_hand.csv

Three things, in the order they should be read:

1. **The paired table.** Restricted to the constructs scorable in *both* files, with that n stated. The
   unpaired form sums a sample change and an effect change into one number, which is the error this project's working rules
   records twice (21.3, and the cap probe in "Name the falsifying case").
2. **Rank improvement and MRR beside top-1.** Top-1 is brittle here: this project measured the top-1 gap
   rattling +5/+6/+5/+6/+3 across a sixteenfold catalogue range while the MRR gap held between 0.189 and
   0.256. A 28 -> 6 improvement is real and top-1 cannot see it.
3. **The per-construct spans against the oracle span**, which is the named falsifying case made
   checkable: a miss is either "claimed the wrong thing" (no segment near the payload) or "claimed a
   chimera" (one span straddling payload and distractor, leaving neither clean). Those need opposite
   responses, so they are never collapsed into a single miss count.

Prints only; writes nothing, so it can be re-run against any pair of result files.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from pilot.decompose import span_rows_at

# The six constructs where `oracle` beats `whole` at the 25% holdout: the places the new arm has room.
# Written down in the working record before the run, so the prediction cannot be read back onto the result.
PREDICTED = {
    "P46108": ("108653", 49, 2),
    "P04517": ("171782", 28, 1),
    "Q9R1E0": ("17551", 3, 1),
    "Q13137": ("187828", 3, 2),
    "Q99ZW2": ("63155/85400", 2, 1),
}


def read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def ranks_at(rows: list[dict], thr: str, arm: str) -> dict[str, int]:
    """plasmid_id -> rank, for the rows that are scorable (a blank rank means unscorable)."""
    return {r["plasmid_id"]: int(r["best_target_rank"]) for r in rows
            if r["threshold"] == thr and r["arm"] == arm and r["best_target_rank"] != ""}


def stats(ranks: dict[str, int], keys: list[str]) -> tuple[int, float, float]:
    """(top-1 count, MRR, mean rank) over exactly `keys`, which must all be present."""
    vals = [ranks[k] for k in keys]
    return (sum(1 for v in vals if v == 1),
            sum(1 / v for v in vals) / len(vals),
            sum(vals) / len(vals))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bhs", type=Path, required=True, help="the run carrying the bhs arm")
    ap.add_argument("--baseline", type=Path, required=True, help="the existing whole/auto/oracle run")
    ap.add_argument("--spans", type=Path, default=None, help="bhs_spans_*.csv from the same run")
    ap.add_argument("--spans-threshold", type=float, default=None,
                    help="which holdout's segmentation to read out of --spans. Required for a span "
                         "file written on or after 20 Oct, which carries one segmentation per "
                         "(construct, threshold); a file without a `threshold` column holds one "
                         "segmentation and this is then ignored.")
    ap.add_argument("--threshold", default="0.25")
    args = ap.parse_args(argv)

    new, base = read(args.bhs), read(args.baseline)
    thr = args.threshold

    new_arms = sorted({r["arm"] for r in new})
    base_arms = sorted({r["arm"] for r in base})
    print(f"bhs file: {args.bhs}\n                arms {new_arms}")
    print(f"baseline file: {args.baseline}\n                arms {base_arms}")
    print(f"holdout: {thr}\n")

    # ---- 1. the paired table -------------------------------------------------------------------
    # Pair on the constructs scorable under every arm in both files. A construct unscorable in one arm
    # cannot contribute to a comparison of that arm, and dropping it from only one side is how an
    # unpaired table turns a sample change into an apparent effect.
    per_arm_new = {a: ranks_at(new, thr, a) for a in new_arms}
    per_arm_base = {a: ranks_at(base, thr, a) for a in base_arms}
    shared_arms = [a for a in ("whole", "auto", "oracle") if a in per_arm_new and a in per_arm_base]

    common = set.intersection(*[set(per_arm_base[a]) for a in shared_arms],
                              *[set(per_arm_new[a]) for a in new_arms])
    keys = sorted(common)
    n = len(keys)
    print(f"PAIRED on the {n} constructs scorable under every arm in both files.")
    unscorable = (set().union(*[set(per_arm_new[a]) for a in new_arms],
                              *[set(per_arm_base[a]) for a in shared_arms]) - common)
    if unscorable:
        print(f"  excluded as not scorable everywhere ({len(unscorable)}): {', '.join(sorted(unscorable))}")
    print()

    print(f"{'arm':<22}{'source':<10}{'top-1':>8}{'MRR':>9}{'mean rank':>12}"
          f"{'d top-1':>10}{'d MRR':>9}{'d mean':>9}")
    w_t1, w_mrr, w_mean = stats(per_arm_base["whole"], keys)
    ref = None
    for label, ranks, src in ([(a, per_arm_base[a], "baseline") for a in shared_arms]
                              + [(a, per_arm_new[a], "new run") for a in new_arms]):
        t1, mrr, mean = stats(ranks, keys)
        if label == "whole" and src == "baseline":
            ref = (t1, mrr, mean)
        print(f"{label:<22}{src:<10}{t1:>5}/{n:<2}{mrr:>9.3f}{mean:>12.2f}"
              f"{t1 - ref[0]:>+10}{mrr - ref[1]:>+9.3f}{mean - ref[2]:>+9.2f}")

    # whole must be identical in both files: same ORFs, same catalogue, same scorer. If it is not, the
    # two runs are not comparable and nothing below means anything.
    if "whole" in per_arm_new:
        diff = {k for k in keys if per_arm_new["whole"][k] != per_arm_base["whole"][k]}
        print(f"\nsanity: `whole` identical across the two runs on all {n} paired constructs: "
              f"{'YES' if not diff else 'NO: ' + str(sorted(diff))}")

    # ---- 2. per-construct ranks ----------------------------------------------------------------
    print(f"\n{'construct':<42}{'payload':<9}" + "".join(f"{a:>9}" for a in
          ["whole", "auto", *([a for a in new_arms if a == 'bhs']), "oracle"]))
    acc_of = {r["plasmid_id"]: r["payload_accession"] for r in new}
    improved, worsened = [], []
    for k in keys:
        cells = []
        for a in ["whole", "auto", "bhs", "oracle"]:
            src = per_arm_new if a in per_arm_new else per_arm_base
            cells.append(f"{src[a][k]:>9}" if a in src else f"{'-':>9}")
        print(f"{k[:40]:<42}{acc_of.get(k, ''):<9}" + "".join(cells))
        if "bhs" in per_arm_new:
            b, w = per_arm_new["bhs"][k], per_arm_base["whole"][k]
            (improved if b < w else worsened if b > w else []).append((k, w, b))

    if "bhs" in per_arm_new:
        print(f"\nbhs vs whole: {len(improved)} improved, {len(worsened)} worsened, "
              f"{n - len(improved) - len(worsened)} unchanged")
        for k, w, b in improved:
            print(f"  improved  {k[:40]:<42}{acc_of.get(k, ''):<9}{w:>6} -> {b}")
        for k, w, b in worsened:
            print(f"  WORSENED  {k[:40]:<42}{acc_of.get(k, ''):<9}{w:>6} -> {b}")

        # ---- 3. did the prediction hold? -------------------------------------------------------
        print("\nTHE PREDICTION: the six constructs where oracle beats whole.")
        print(f"{'construct':<42}{'payload':<9}{'whole':>7}{'oracle':>8}{'bhs':>7}   verdict")
        recovered = partial = missed = 0
        for k in keys:
            acc = acc_of.get(k, "")
            if acc not in PREDICTED:
                continue
            w = per_arm_base["whole"][k]
            o = per_arm_base["oracle"][k]
            b = per_arm_new["bhs"][k]
            if o >= w:
                continue  # not one of the six at this holdout after all
            if b <= o:
                verdict, recovered = "RECOVERED (reaches the ceiling)", recovered + 1
            elif b < w:
                verdict, partial = f"partial ({w} -> {b}, ceiling {o})", partial + 1
            else:
                verdict, missed = "missed", missed + 1
            print(f"{k[:40]:<42}{acc:<9}{w:>7}{o:>8}{b:>7}      {verdict}")
        print(f"\n  of the six: {recovered} recovered to the ceiling, {partial} partially improved, "
              f"{missed} missed")
        nine = [k for k in keys if per_arm_base["oracle"][k] >= per_arm_base["whole"][k]]
        harmed = [k for k in nine if per_arm_new["bhs"][k] > per_arm_base["whole"][k]]
        print(f"  of the {len(nine)} where whole already equals or beats oracle: "
              f"{len(harmed)} harmed by bhs{' (' + ', '.join(k[:28] for k in harmed) + ')' if harmed else ''}")

    # ---- 4. the spans, which separate a wrong claim from a chimera ------------------------------
    if args.spans and args.spans.is_file():
        spans = span_rows_at(read(args.spans), args.spans_threshold)
        by_pid: dict[str, list[dict]] = defaultdict(list)
        for r in spans:
            by_pid[r["plasmid_id"]].append(r)
        print("\n\nPER-CONSTRUCT SPANS vs THE ORACLE SPAN")
        print("'chimera' = one claimed span covers the oracle span AND at least as much outside it, so "
              "neither\npayload nor distractor comes out clean. 'clean' = a span mostly inside the "
              "oracle span.\n")
        print(f"{'construct':<40}{'orf':>5}{'oracle':>12}{'it':>3}{'claimed':>12}{'hit':<9}"
              f"{'id':>6}{'ofOrac':>8}{'inOrac':>8}  shape")
        for k in sorted(by_pid):
            rs = sorted(by_pid[k], key=lambda r: int(r["iteration"]))
            for i, r in enumerate(rs):
                fo = float(r["frac_of_oracle_covered"] or 0)
                fi = float(r["frac_of_segment_in_oracle"] or 0)
                shape = ("clean" if fi >= 0.80 and fo >= 0.50 else
                         "chimera" if fo >= 0.50 and fi < 0.60 else
                         "partial" if fo >= 0.20 else "elsewhere")
                if not int(r["emitted"]):
                    shape += " (<min, not emitted)"
                print(f"{(k[:38] if i == 0 else ''):<40}{(r['orf_len'] if i == 0 else ''):>5}"
                      f"{(r['oracle_start'] + '-' + r['oracle_end'] if i == 0 else ''):>12}"
                      f"{r['iteration']:>3}{r['seg_start'] + '-' + r['seg_end']:>12}"
                      f"{r['hit_accession']:<9}{r['identity']:>6}{fo:>8.2f}{fi:>8.2f}    {shape}")
        # The aggregate the falsifying case asks for. A chimera is a SPAN-level property, one span
        # covering a lot of the oracle span and a lot outside it, so it is counted over spans, and
        # separately over constructs, rather than being folded into a single per-construct verdict.
        n_chimera = sum(1 for k in by_pid for r in by_pid[k]
                        if float(r["frac_of_oracle_covered"] or 0) >= 0.5
                        and float(r["frac_of_segment_in_oracle"] or 0) < 0.6)
        n_spans = sum(len(v) for v in by_pid.values())
        shapes = defaultdict(int)
        for k in by_pid:
            fo = [float(r["frac_of_oracle_covered"] or 0) for r in by_pid[k]]
            fi = [float(r["frac_of_segment_in_oracle"] or 0) for r in by_pid[k]]
            best = max(range(len(fo)), key=lambda i: fo[i])
            shapes["clean (a span mostly inside the oracle span)" if fo[best] >= 0.5 and fi[best] >= 0.8
                   else "chimera (straddles payload and distractor)" if fo[best] >= 0.5
                   else "partial (a span inside it, under half of it)" if fo[best] >= 0.20
                   else "no span on the payload at all"] += 1
        print(f"\n  CHIMERAS, the named falsifying case, {n_chimera} of {n_spans} claimed spans")
        print("  per construct, the span covering most of the oracle span:")
        for k, v in sorted(shapes.items(), key=lambda kv: -kv[1]):
            print(f"        {v:>3}    {k}")
        wholly_in = sum(1 for k in by_pid for r in by_pid[k]
                        if float(r["frac_of_oracle_covered"] or 0) > 0
                        and float(r["frac_of_segment_in_oracle"] or 0) >= 0.999)
        touching = sum(1 for k in by_pid for r in by_pid[k]
                       if float(r["frac_of_oracle_covered"] or 0) > 0)
        print(f"  spans touching the oracle span that lie ENTIRELY inside it: {wholly_in} of {touching}")
        iters = [int(by_pid[k][0]["n_iterations"]) for k in by_pid]
        print(f"  iterations: n={len(iters)} mean={sum(iters) / len(iters):.2f} max={max(iters)}  "
              f"distribution={ {v: iters.count(v) for v in sorted(set(iters))} }")
        print(f"  stop reasons: { {s: sum(1 for k in by_pid if by_pid[k][0]['stop_reason'] == s) for s in sorted({by_pid[k][0]['stop_reason'] for k in by_pid})} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
