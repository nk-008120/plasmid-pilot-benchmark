#!/usr/bin/env python3
"""How often do the decomposition motifs fire on proteins that are *not* fusions?

    python scripts/08_motif_specificity.py --reference data/reference/reference.fasta

Every reference protein is a single, real, reviewed protein, so any cut `decompose()` makes in one is a
false positive. This script reports the per-motif hit rate and, more importantly, the **false-split rate**:
the fraction of real single proteins the rules would wrongly carve up. Decomposition that splits genuine
proteins would manufacture exactly the dilution problem it exists to remove, so this number governs how far
the motif catalogue can be extended. Re-run it after touching `CLEAVAGE`, `TAGS`, `LINKERS` or
`INTERIOR_JUNCTION`, and record the result in the working record.

Writes `motif_specificity.csv` next to the reference file.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from pilot.decompose import CLEAVAGE, INTERIOR_JUNCTION, LINKERS, TAGS, decompose, find_motifs
from pilot.reference import iter_fasta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, default=Path("data/reference/reference.fasta"))
    ap.add_argument("--min-len", type=int, default=50)
    args = ap.parse_args(argv)

    refs = [e.seq for e in iter_fasta(args.reference)]
    if not refs:
        print(f"no sequences in {args.reference}")
        return 2
    total_aa = sum(len(s) for s in refs)
    names = ([(n, "cleave") for n, _, _ in CLEAVAGE] + [(n, "tag") for n, _ in TAGS]
             + [(n, "linker") for n, _ in LINKERS])

    per_protein, per_hit = Counter(), Counter()
    for s in refs:
        seen = set()
        for h in find_motifs(s):
            per_hit[h.name] += 1
            seen.add(h.name)
        per_protein.update(seen)

    print(f"{len(refs):,} single real proteins, {total_aa:,} residues\n")
    print(f"{'motif':<16}{'kind':<9}{'junction?':<11}{'proteins':>10}{'% of set':>10}{'per 100k aa':>13}")
    rows = []
    for n, k in names:
        pct = 100 * per_protein[n] / len(refs)
        rate = 1e5 * per_hit[n] / total_aa
        junc = "yes" if n in INTERIOR_JUNCTION else "trim only"
        print(f"{n:<16}{k:<9}{junc:<11}{per_protein[n]:>10}{pct:>9.2f}%{rate:>13.2f}")
        rows.append(dict(motif=n, kind=k, interior_junction=junc, proteins_hit=per_protein[n],
                         pct_of_set=round(pct, 4), hits=per_hit[n], per_100k_aa=round(rate, 3)))

    for label, kwargs in (("rules as shipped", {}), ("interior-junction rule off", {"cut_at_internal_tags": False})):
        split = sum(1 for s in refs if len(decompose(s, min_len=args.min_len, **kwargs)) > 1)
        empty = sum(1 for s in refs if not decompose(s, min_len=args.min_len, **kwargs))
        print(f"\n{label}:")
        print(f"  false-split rate: {split}/{len(refs)} ({100 * split / len(refs):.2f}%)")
        print(f"  queries dropped: {empty}/{len(refs)}  (must be 0)")
        rows.append(dict(motif=f"[{label}]", kind="summary", interior_junction="",
                         proteins_hit=split, pct_of_set=round(100 * split / len(refs), 4),
                         hits=empty, per_100k_aa=""))

    out = args.reference.parent / "motif_specificity.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
