#!/usr/bin/env python3
"""Subset the all-173 exact-alignment homolog catalogue down to the hand-audited payloads.

    python scripts/19_subset_catalogue.py

The all-173 rebuild searched every corpus payload against Swiss-Prot with exact
Smith-Waterman and wrote one large catalogue. The audited-23 twilight benchmark only needs the 16
payload accessions in `data/manifest.csv`, so this filters that catalogue rather than re-searching
Swiss-Prot: same alignment evidence, a fraction of the size, and reproducible from files on disk.

The decoy set is carried over whole, so the catalogue stays a real search rather than a choice among
the payloads' own relatives. Writes `homologs.csv` and `reference_twilight.fasta` into --out-dir,
the two files `scripts/10_twilight.py` reads.

This script derives nothing and decides nothing: every row it keeps was produced by the exact search.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pilot.reference import iter_fasta, write_fasta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--homologs", type=Path,
                    default=Path("results/homologs_all173/homologs.csv"))
    ap.add_argument("--catalogue", type=Path,
                    default=Path("results/homologs_all173/reference_twilight.fasta"))
    ap.add_argument("--decoys", type=Path, default=Path("data/reference/reference.fasta"),
                    help="the standard 5,016-protein reference; carried over whole")
    ap.add_argument("--out-dir", type=Path, default=Path("results/homologs_audited"))
    args = ap.parse_args(argv)

    for p in (args.manifest, args.homologs, args.catalogue, args.decoys):
        if not p.is_file():
            print(f"not found: {p}")
            return 2

    with open(args.manifest, newline="", encoding="utf-8-sig", errors="replace") as fh:
        want = {(r.get("payload_accession") or "").strip() for r in csv.DictReader(fh)}
    want.discard("")
    print(f"audited payload accessions: {len(want)}")

    rows: list[dict[str, str]] = []
    keep: set[str] = set()
    with open(args.homologs, newline="", encoding="utf-8-sig") as fh:
        rdr = csv.DictReader(fh)
        fields = rdr.fieldnames or []
        for r in rdr:
            if r["payload_accession"] in want:
                rows.append(r)
                keep.add(r["homolog_accession"])
    keep |= want
    per_payload = {r["payload_accession"] for r in rows}
    print(f"kept {len(rows):,} homolog rows over {len(per_payload)} payloads "
          f"({len(keep):,} distinct accessions)")
    for acc in sorted(want - per_payload):
        print(f"  no homolog rows for {acc}")

    entries, seen = [], set()
    for src, take_all in ((args.catalogue, False), (args.decoys, True)):
        n = 0
        for e in iter_fasta(src):
            if e.accession in seen or not (take_all or e.accession in keep):
                continue
            entries.append(e)
            seen.add(e.accession)
            n += 1
        print(f"    {src.name}: +{n:,} entries")

    missing = sorted(want - seen)
    if missing:
        print(f"WARNING: payload sequence absent from the catalogue: {', '.join(missing)}")
        print("  10_twilight.py will skip those constructs; fix before trusting the counts.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "homologs.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    write_fasta(entries, args.out_dir / "reference_twilight.fasta")
    print(f"wrote {args.out_dir}: {len(entries):,} entries, {len(rows):,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
