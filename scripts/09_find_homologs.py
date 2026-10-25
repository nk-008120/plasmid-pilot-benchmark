#!/usr/bin/env python3
"""Align every payload against all of Swiss-Prot and record its homologs with percent identity.

    python scripts/09_find_homologs.py --swissprot ~/Downloads/uniprot_sprot.fasta.gz

This is the slow, cacheable step behind the twilight-zone benchmark (see `pilot.homology`). It writes
`data/reference/homologs.csv`, one row per (payload, homolog) with identity and coverage, and
`data/reference/reference_twilight.fasta`, which is the existing decoy set plus every homolog found. The
benchmark then runs on that focused set and simply masks rows, so this sweep is done once, not once per
holdout threshold.

Needs `parasail` (`pip install parasail`). Roughly a minute per payload on one core.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from pilot.homology import find_homologs, iter_swissprot
from pilot.reference import Entry, iter_fasta, write_fasta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swissprot", type=Path, required=True, help="uniprot_sprot.fasta(.gz)")
    ap.add_argument("--reference", type=Path, default=Path("data/reference/reference.fasta"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/reference"))
    ap.add_argument("--top-n", type=int, default=400, help="how many top-scoring hits get a traceback")
    ap.add_argument("--min-identity", type=float, default=0.20)
    ap.add_argument("--min-coverage", type=float, default=0.40)
    args = ap.parse_args(argv)

    if not args.swissprot.is_file():
        print(f"not found: {args.swissprot}", file=sys.stderr)
        return 2
    with open(args.manifest, newline="", encoding="utf-8-sig", errors="replace") as fh:
        payload_accs = sorted({(r.get("payload_accession") or "").strip() for r in csv.DictReader(fh)} - {""})
    base = list(iter_fasta(args.reference))
    ref = {e.accession: e.seq for e in base}
    missing = [a for a in payload_accs if a not in ref]
    if missing:
        print(f"payload accessions absent from the reference set: {missing}", file=sys.stderr)
        return 2

    t0 = time.time()
    print(f"reading {args.swissprot.name}...", flush=True)
    entries = list(iter_swissprot(args.swissprot))
    print(f"    {len(entries):,} entries in {time.time() - t0:.0f}s")

    # the twilight reference starts as the existing set and grows with every homolog found
    rows: list[dict] = []
    keep: dict[str, Entry] = {e.accession: e for e in base}
    sp = dict(entries)
    for i, acc in enumerate(payload_accs, 1):
        t = time.time()
        hits = find_homologs(ref[acc], entries, top_n=args.top_n,
                            min_identity=args.min_identity, min_coverage=args.min_coverage)
        bins = {f">{lo:.0%}": sum(1 for h in hits if h.identity > lo) for lo in (0.9, 0.7, 0.5, 0.35)}
        print(f"  [{i}/{len(payload_accs)}] {acc}    {len(ref[acc]):>5}aa  {len(hits):>4} homologs  "
              f"{bins}    {time.time() - t:.0f}s", flush=True)
        for h in hits:
            rows.append(dict(payload_accession=acc, homolog_accession=h.accession, score=h.score,
                             identity=h.identity, coverage=h.coverage, length=h.length))
            if h.accession in sp:
                keep.setdefault(h.accession, Entry(h.accession, f"{h.accession} homolog_of:{acc}", sp[h.accession]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "homologs.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out_fa = args.out_dir / "reference_twilight.fasta"
    write_fasta(list(keep.values()), out_fa)
    print(f"\n{len(rows):,} homolog rows for {len(payload_accs)} payloads -> {out_csv}")
    print(f"twilight reference: {len(keep):,} proteins -> {out_fa}")
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
