#!/usr/bin/env python3
"""Build the retrieval reference set: the manifest's payload proteins plus seeded Swiss-Prot decoys.

    python scripts/04_build_reference.py --swissprot ~/Downloads/uniprot_sprot.fasta.gz

Writes `data/reference/reference.fasta` and `reference_index.csv`. Fails loudly if a payload accession
named in the manifest is not present in the Swiss-Prot release, rather than scoring against an absent target.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from pilot.reference import build, write_fasta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swissprot", type=Path, required=True, help="uniprot_sprot.fasta(.gz)")
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/reference"))
    ap.add_argument("--n-decoys", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if not args.swissprot.is_file():
        print(f"not found: {args.swissprot}", file=sys.stderr)
        return 2
    with open(args.manifest, newline="", encoding="utf-8-sig", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "payload_accession" not in rows[0]:
        print("manifest has no 'payload_accession' column", file=sys.stderr)
        return 2
    wanted = {(r.get("payload_accession") or "").strip() for r in rows}
    wanted.discard("")
    print(f"{len(wanted)} distinct payload accessions in the manifest; sampling {args.n_decoys} decoys "
          f"(seed {args.seed}) from {args.swissprot.name}")

    entries, missing = build(args.swissprot, wanted, args.n_decoys, args.seed)
    if missing:
        print(f"\nERROR: payload accessions absent from this Swiss-Prot release: {', '.join(missing)}",
              file=sys.stderr)
        print("Fix the manifest or use a release that contains them; refusing to build a reference set "
              "that cannot score those plasmids.", file=sys.stderr)
        return 1

    write_fasta(entries, args.out_dir / "reference.fasta")
    with open(args.out_dir / "reference_index.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["accession", "length_aa", "is_payload", "description"])
        for e in entries:
            w.writerow([e.accession, len(e.seq), e.accession in wanted, e.description])
    n_pay = sum(1 for e in entries if e.accession in wanted)
    print(f"\nWrote {len(entries)} entries ({n_pay} payloads + {len(entries) - n_pay} decoys) to {args.out_dir}/")
    print("Payload proteins in the reference set:")
    for e in entries:
        if e.accession in wanted:
            print(f"      {e.accession}    {len(e.seq):>5}aa  {e.description[:78]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
