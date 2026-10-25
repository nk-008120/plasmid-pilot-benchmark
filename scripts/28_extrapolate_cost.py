#!/usr/bin/env python3
"""From a measured cost probe to a range for the whole frame, with the model stated rather than assumed.

    python scripts/28_extrapolate_cost.py \
        --cost-csv results/bhs_corpus_probe/cost_sample30.csv \
        --frame-lengths results/bhs_corpus_probe/frame_orf_lengths.csv

**What is being extrapolated, and why a point estimate would be wrong.** The per-segment best-hit
protocol costs one exact Smith-Waterman pass over the catalogue per query, and a pass costs time
proportional to the query's length. The number of queries per construct is not fixed either: it is
`1 (whole) + k_auto + (iters + 1) (the segmenter's own passes) + k_bhs + 1 (oracle)`, and `iters` is
decided at run time by the significance floor. So the cost of a construct is a product of two things the
probe measures and neither of which is constant, which is exactly the situation where a single
seconds-per-construct figure carries a confidence it has not earned.

Three estimators are therefore reported, and the spread between them is the honest range:

* **per construct**: the probe's mean seconds times the frame size. Assumes the frame's constructs
  cost what the sample's did, which is false to the extent that ORF lengths differ.
* **per ORF residue**: the probe's seconds per residue of *chosen ORF*, times the frame's total ORF
  residues. This is the length-adjusted estimator, and it is the one to prefer: it absorbs a frame whose
  ORFs are longer than the sample's, which is the failure mode named in `27_frame_orf_lengths.py`.
* **bootstrap**: resampling the probe's constructs with replacement, to put sampling error on the
  per-construct figure. Reported as a 10th-90th percentile interval, not a standard error, because the
  per-construct cost distribution is strongly right-skewed (one long ORF with four iterations costs what
  a dozen short ones do) and a symmetric interval around the mean would misdescribe it.

The bootstrap interval is sampling error on the sample *mean* alone. It does not cover the length
mismatch, machine load, or a frame whose iteration counts run higher, so the reported range is the union
of the bootstrap interval and the length-adjusted point: never the bootstrap interval by itself.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def pct(v: list[float], f: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, int(f * (len(s) - 1) + 0.5))]


def hms(seconds: float) -> str:
    h, rem = divmod(int(seconds + 0.5), 3600)
    return f"{h}h {rem // 60:02d}m" if h else f"{rem // 60}m {rem % 60:02d}s"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cost-csv", type=Path, required=True, help="from 26_bhs_protocol.py --cost-csv")
    ap.add_argument("--frame-lengths", type=Path, required=True, help="from 27_frame_orf_lengths.py")
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cost = read_csv(args.cost_csv)
    frame = read_csv(args.frame_lengths)
    sec = [float(r["seconds"]) for r in cost]
    orf = [int(r["orf_len"]) for r in cost]
    qres = [int(r["query_residues"]) for r in cost]
    calls = [int(r["score_all_calls"]) for r in cost]
    n_f = len(frame)
    orf_f = sum(int(r["orf_len"]) for r in frame)

    print(f"probe: {len(sec)} constructs, {sum(sec):.1f}s measured, {sum(calls)} catalogue passes, "
          f"{sum(qres):,} query residues over {sum(orf):,} ORF residues")
    print(f"frame: {n_f} constructs, {orf_f:,} ORF residues "
          f"({orf_f / sum(orf):.2f}x the probe's, on {n_f / len(sec):.2f}x the constructs)")
    print(f"       queries per construct: mean {sum(calls) / len(calls):.2f}, "
          f"min {min(calls)}, max {max(calls)}")
    print(f"       query residues per ORF residue: mean {sum(qres) / sum(orf):.2f}  "
          f"(the protocol scores each ORF about this many times over)\n")

    per_con = sum(sec) / len(sec)
    per_res = sum(sec) / sum(orf)
    e_con, e_res = per_con * n_f, per_res * orf_f
    print(f"per construct    {per_con:>8.2f} s/construct  x {n_f} constructs      "
          f"= {e_con:>9.0f}s  ({hms(e_con)})")
    print(f"per ORF residue  {per_res:>8.4f} s/residue    x {orf_f:,} residues  "
          f"= {e_res:>9.0f}s  ({hms(e_res)})   <- length-adjusted, prefer this one")

    rng = random.Random(args.seed)
    boots = [sum(rng.choices(sec, k=len(sec))) / len(sec) * n_f for _ in range(args.draws)]
    lo, hi = pct(boots, 0.10), pct(boots, 0.90)
    print(f"bootstrap        10th-90th percentile of the per-construct estimate: "
          f"{lo:.0f}-{hi:.0f}s  ({hms(lo)} - {hms(hi)})")

    low, high = min(lo, e_res, e_con), max(hi, e_res, e_con)
    print(f"\nRANGE for the full frame of {n_f}: {low:.0f}-{high:.0f}s = {hms(low)} to {hms(high)}")
    print("  the union of the bootstrap interval and the length-adjusted point. Sampling error on the "
          "mean only;\n  it does not cover machine load, a frame with higher iteration counts, or any "
          "construct the\n  probe's sample could not contain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
