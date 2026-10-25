#!/usr/bin/env python3
"""Does `10_twilight.py`'s per-segment arm agree with `26_bhs_protocol.py`, construct for construct?

    python scripts/34_crosscheck_per_segment.py \
        --twilight results/persegment_corpus/twilight_sw-blosum62-11-1_corpus.csv \
        --detail results/bhs_corpus_full/detail_corpus_25.csv --threshold 0.25

**Why this exists.** `26_bhs_protocol.py` is the reference implementation of per-segment ranking
("Layout"), and `10_twilight.py` reimplemented the aggregation inside the main benchmark so
that every run carries it. Two implementations of one quantity is exactly the situation this project's coverage-denominator bug arose from: they agree on the tier that was checked and drift on the one that
was not. The hand tier matched on 75 of 75 cells; the corpus and the synthetic tiers had never been
compared, and they are the tiers the proposal's n comes from.

So this compares **per construct**, not in aggregate. An aggregate match is weaker than it looks: two
implementations can produce the same mean rank from different per-construct values, and a mean over 286
constructs hides a single swapped pair completely. The aggregates are printed too, but the exit code is
decided by the cell-by-cell comparison.

Column mapping between the two files, which is the one place this can silently go wrong:

| `10_twilight.py` arm | `26_bhs_protocol.py` column |
|---|---|
| `whole` | `rank_whole` |
| `auto` | `rank_auto_pooled` |
| `auto_per_segment` | `rank_auto_per_segment` |
| `bhs` | `rank_pooled` |
| `bhs_per_segment` | `rank_per_segment` |
| `oracle` | `rank_oracle` |

`rank_oracle_selected` has no counterpart: it uses the label to pick a segment and is a diagnostic, not
an arm, so script 10 does not emit it.

Exit code 0 when every shared construct agrees on every shared arm, 1 otherwise. A disagreement is
printed with both values, because the question "which implementation is wrong" needs the pair.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

#: arm name in the twilight CSV -> column name in the protocol detail CSV
ARM_TO_DETAIL = {
    "whole": "rank_whole",
    "auto": "rank_auto_pooled",
    "auto_per_segment": "rank_auto_per_segment",
    "bhs": "rank_pooled",
    "bhs_per_segment": "rank_per_segment",
    "oracle": "rank_oracle",
}


def read_twilight(path: Path, threshold: float) -> dict[str, dict[str, int]]:
    """{plasmid_id: {arm: rank}} at one holdout, skipping the unscorable cells."""
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if abs(float(r["threshold"]) - threshold) > 1e-9 or r["best_target_rank"] == "":
                continue
            out[r["plasmid_id"]][r["arm"]] = int(r["best_target_rank"])
    return out


def read_detail(path: Path) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, str]]]:
    """{plasmid_id: {arm: rank}} from the reference implementation, plus its other columns."""
    ranks: dict[str, dict[str, int]] = {}
    extra: dict[str, dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            pid = r["plasmid_id"]
            ranks[pid] = {arm: int(r[col]) for arm, col in ARM_TO_DETAIL.items()
                          if r.get(col) not in (None, "", "-")}
            extra[pid] = r
    return ranks, extra


def stats(values: list[int]) -> tuple[int, float, float, int]:
    return (sum(1 for v in values if v == 1), sum(1 / v for v in values) / len(values),
            sum(values) / len(values), sum(1 for v in values if v <= 10))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twilight", type=Path, required=True, help="twilight_<scorer>_<tier>.csv")
    ap.add_argument("--detail", type=Path, required=True, help="26_bhs_protocol.py's detail CSV")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--label", default="", help="tier name for the report")
    args = ap.parse_args(argv)

    tw = read_twilight(args.twilight, args.threshold)
    ref, extra = read_detail(args.detail)
    label = args.label or args.twilight.stem

    shared = sorted(set(tw) & set(ref))
    only_tw, only_ref = sorted(set(tw) - set(ref)), sorted(set(ref) - set(tw))
    print(f"\n{'=' * 78}\n{label}: {args.twilight.name} vs {args.detail.name} at the "
          f"{args.threshold:.0%} holdout\n{'=' * 78}")
    print(f"constructs: {len(tw)} in the twilight CSV, {len(ref)} in the detail file, "
          f"{len(shared)} shared")
    # A frame difference is not a mismatch, but it changes what the comparison covers, so it is named
    # rather than silently intersected away ("name the falsifying case").
    for name, ids in (("only in the twilight CSV", only_tw), ("only in the detail file", only_ref)):
        if ids:
            print(f"    {len(ids)}  {name}: {', '.join(ids[:6])}{'...' if len(ids) > 6 else ''}")
    if not shared:
        print("nothing to compare")
        return 1

    arms = [a for a in ARM_TO_DETAIL if all(a in tw[p] and a in ref[p] for p in shared)]
    missing = [a for a in ARM_TO_DETAIL if a not in arms]
    if missing:
        print(f"  arms not present in both files, skipped: {', '.join(missing)}")

    bad = []
    for pid in shared:
        for arm in arms:
            if tw[pid][arm] != ref[pid][arm]:
                bad.append((pid, arm, tw[pid][arm], ref[pid][arm]))
    print(f"\ncell-by-cell: {len(shared)} constructs x {len(arms)} arms = "
          f"{len(shared) * len(arms)} cells, {len(bad)} mismatches")
    for pid, arm, got, want in bad[:40]:
        f = extra[pid].get("f", "")
        print(f"  MISMATCH {pid[:38]:<40}{arm:<18}10_twilight={got:<8}26_bhs_protocol={want:<8}"
              f"{('f=' + f) if f else ''}")
    if len(bad) > 40:
        print(f"  ... and {len(bad) - 40} more")

    print(f"\n{'arm':<20}{'top-1':>10}{'MRR':>9}{'mean rank':>12}{'recall@10':>12}      "
          f"(reference implementation)")
    for arm in arms:
        a = stats([tw[p][arm] for p in shared])
        b = stats([ref[p][arm] for p in shared])
        print(f"{arm:<20}{a[0]:>6}/{len(shared):<3}{a[1]:>9.3f}{a[2]:>12.2f}"
              f"{a[3]:>8}/{len(shared):<3}      {b[0]:>6}/{len(shared):<3}{b[1]:>9.3f}"
              f"{b[2]:>12.2f}{b[3]:>8}/{len(shared):<3}")

    print(f"\n{'AGREE' if not bad else 'DISAGREE'}: "
          f"{len(shared) * len(arms) - len(bad)} of {len(shared) * len(arms)} cells")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
