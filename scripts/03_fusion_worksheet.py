#!/usr/bin/env python3
"""Lay out what sits inside each plasmid's insert-matching ORF, ready for the manual audit.

    python scripts/03_fusion_worksheet.py --manifest data/manifest.csv --out-dir results/baseline2

Writes `fusion_worksheet.csv` (one row per feature found inside a payload ORF, with blank `piece_is` and
`notes` columns for you to fill) and prints a readable layout per plasmid. Needs `insert_size_bp` in the
manifest. Restrict to the plasmids worth auditing with --only.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from pilot.fusion import features_inside, payload_orf
from pilot.genbank import GENBANK_SUFFIXES, load_plasmids
from pilot.insert import parse_insert_size
from pilot.orfs import find_orfs

PIECE_KINDS = ("payload", "tag", "linker", "second_gene", "readthrough", "backbone_partner", "domain", "other")
COLUMNS = [
    "plasmid_id", "stated_insert_bp", "orf_nt", "orf_aa", "orf_minus_insert_bp", "insert_verdict",
    "feature_label", "feature_type", "same_strand", "start_aa", "end_aa", "length_aa", "covered_fraction",
    "n_segments", "piece_is", "notes",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/baseline2"))
    ap.add_argument("--min-aa", type=int, default=50)
    ap.add_argument("--insert-tolerance", type=float, default=0.10)
    ap.add_argument("--only", default="", help="comma-separated Addgene IDs, e.g. 187828,17551 (default: all)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing worksheet (destroys your notes)")
    args = ap.parse_args(argv)

    if not args.manifest.is_file():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    with open(args.manifest, newline="", encoding="utf-8-sig", errors="replace") as fh:
        man = {r["filename"].strip(): r for r in csv.DictReader(fh) if (r.get("filename") or "").strip()}

    out = args.out_dir / "fusion_worksheet.csv"
    if out.exists() and not args.force:
        print(f"refusing to overwrite {out}: it may contain your notes. Use --force to replace it.", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    rows = []
    for path in sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in GENBANK_SUFFIXES):
        if wanted and not any(w in path.name for w in wanted):
            continue
        size = parse_insert_size(man.get(path.name, {}).get("insert_size_bp"))
        for pl in load_plasmids(path):
            orfs = find_orfs(pl.seq, circular=pl.circular, min_aa=args.min_aa)
            orf, chk = payload_orf(pl, orfs, size, args.insert_tolerance)
            if orf is None:
                print(f"\n=== {pl.id}: no insert size in the manifest, skipped")
                continue
            diff = orf.length_nt - chk.insert_size_bp
            print(f"\n=== {pl.id}")
            print(f"    stated insert {chk.insert_size_bp} bp | payload ORF {orf.length_nt} nt "
                  f"({orf.length_aa} aa) | difference {diff:+d} bp | {chk.verdict}")
            hits = features_inside(pl, orf)
            if not hits:
                print("    (no annotated feature overlaps this ORF - the payload is unannotated)")
            for h in hits:
                mark = " " if h.same_strand else "~"  # ~ = annotated on the opposite strand to the ORF
                seg = f" seg={h.n_segments}" if h.n_segments > 1 else ""
                print(f"      {mark}{h.start_aa:>6}..{h.end_aa:<6}aa  {h.label[:32]:<34}{h.type:<14}"
                      f"cov={h.covered_fraction}{seg}")
                rows.append({
                    "plasmid_id": pl.id, "stated_insert_bp": chk.insert_size_bp, "orf_nt": orf.length_nt,
                    "orf_aa": orf.length_aa, "orf_minus_insert_bp": diff, "insert_verdict": chk.verdict,
                    "feature_label": h.label, "feature_type": h.type, "same_strand": h.same_strand,
                    "start_aa": h.start_aa, "end_aa": h.end_aa, "length_aa": h.length_aa,
                    "covered_fraction": h.covered_fraction, "n_segments": h.n_segments,
                    "piece_is": "", "notes": "",
                })

    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out}")
    print("Fill 'piece_is' with one of: " + ", ".join(PIECE_KINDS))
    print("A payload ORF with no features listed means the insert is unannotated - name it from the repository page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
