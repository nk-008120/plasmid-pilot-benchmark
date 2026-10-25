#!/usr/bin/env python3
"""Sweep the E-value gate over an existing deep, ungated homolog run, and show the plateau.

    python scripts/24_evalue_sweep.py --homologs results/homologs_audited_deep/homologs.csv

**Why a third gate.** `min_coverage` is a fraction of payload length, so it asks for 78 aligned columns
from a 196 aa payload and 1,222 from a 3,054 aa one, while `min_identity` is flat. Statistical
significance does not scale that way: 78 columns at 20% identity is about 16 identical residues and is
indistinguishable from chance, whereas 1,222 columns at 20% identity is overwhelming evidence. The result
is that scan depth buys thousands of marginal hits for short payloads and nothing at all for long ones.
A search-space E-value is the gate that makes stringency comparable across the length range.

**Swept, not picked.** this project's working rules requires a tuned threshold to be swept with its plateau shown. This
applies each candidate threshold to the *same* ungated run, so the only thing varying is the gate; the
raw Smith-Waterman score is recorded per row, so the E-value is recomputed rather than re-searched.

Writes `evalue_sweep.csv` and a filtered `homologs_e<threshold>.csv` per threshold, ready to hand to
`10_twilight.py --homologs`.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

from pilot.homology import log_evalue_search

THRESHOLDS = (1e-10, 1e-5, 1e-3, 0.01, 0.1, 1.0)
HOLDOUTS = (0.90, 0.70, 0.50, 0.35, 0.25)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--homologs", type=Path,
                    default=Path("results/homologs_audited_deep/homologs.csv"))
    ap.add_argument("--payload-lengths", type=Path,
                    default=Path("results/homologs_audited_deep/payload_lengths.csv"))
    ap.add_argument("--db-residues", type=int, default=207_099_657,
                    help="summed residues of the database that was searched (Swiss-Prot, 50-5,000 aa "
                         "window). This is the search space; it must be the scored database, not the "
                         "top-n that reached traceback, or significance becomes a function of depth.")
    ap.add_argument("--thresholds", type=float, nargs="*", default=list(THRESHOLDS))
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.homologs.is_file():
        print(f"not found: {args.homologs}", file=sys.stderr)
        return 2
    out_dir = args.out_dir or args.homologs.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.payload_lengths, newline="", encoding="utf-8-sig") as fh:
        plen = {r["payload_accession"]: int(r["payload_aa"]) for r in csv.DictReader(fh)}
    with open(args.homologs, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    print(f"{len(rows):,} ungated homolog rows for {len(plen)} payloads; "
          f"search space {args.db_residues:,} residues\n")

    for r in rows:
        r["_lnE"] = log_evalue_search(float(r["score"]), plen[r["payload_accession"]], args.db_residues)

    summary: list[dict] = []
    print(f"{'threshold':>10}{'rows':>9}{'kept %':>8}      " +
          "".join(f"{'<=' + str(int(t * 100)) + '%':>9}" for t in HOLDOUTS) +
          f"{'payloads':>10}{'median/payload':>16}")
    print("-" * 104)
    for thr in sorted(args.thresholds, reverse=True):
        ln_thr = math.log(thr)
        kept = [r for r in rows if r["_lnE"] <= ln_thr]
        per: dict[str, int] = defaultdict(int)
        for r in kept:
            per[r["payload_accession"]] += 1
        avail = {t: sum(1 for r in kept if float(r["identity"]) <= t) for t in HOLDOUTS}
        med = sorted(per.values())[len(per) // 2] if per else 0
        print(f"{thr:>10.0e}{len(kept):>9,}{100 * len(kept) / len(rows):>7.1f}%   "
              + "".join(f"{avail[t]:>9,}" for t in HOLDOUTS)
              + f"{len(per):>10}{med:>16,}")
        summary.append({"threshold": thr, "rows": len(kept), "payloads_with_any": len(per),
                        "median_per_payload": med,
                        **{f"available_at_{int(t * 100)}": avail[t] for t in HOLDOUTS}})
        name = out_dir / f"homologs_e{thr:.0e}.csv"
        with open(name, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=[c for c in rows[0] if not c.startswith("_")])
            w.writeheader()
            w.writerows({k: v for k, v in r.items() if not k.startswith("_")} for r in kept)

    print(f"\n{'payload':<9}{'len':>6}{'ungated':>9}" +
          "".join(f"{t:>10.0e}" for t in sorted(args.thresholds, reverse=True)))
    print("-" * 80)
    for acc in sorted(plen, key=lambda a: -plen[a]):
        mine = [r for r in rows if r["payload_accession"] == acc]
        line = f"{acc:<9}{plen[acc]:>6}{len(mine):>9,}"
        for thr in sorted(args.thresholds, reverse=True):
            line += f"{sum(1 for r in mine if r['_lnE'] <= math.log(thr)):>10,}"
        print(line)

    with open(out_dir / "evalue_sweep.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"\nwrote {out_dir}/evalue_sweep.csv and one homologs_e*.csv per threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
