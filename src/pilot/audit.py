"""Pre-fill the manual audit worksheet from cds_recovery.csv.

Every annotated CDS that was not an exact match becomes a row to review by hand, with the verdict columns blank.
It refuses to overwrite an existing worksheet, because by then it holds your hand-written judgements.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

VERDICTS = (
    "annotation_wrong", "caller_wrong", "tag", "linker", "second_gene", "readthrough", "splice", "other",
)
COLUMNS = [
    "plasmid_id", "cds_label", "location_1based", "strand", "length_aa", "status", "invalid_reason",
    "orf_id", "n_extra_aa", "c_extra_aa", "orf_minus_cds_aa", "reason_for_review",
    "verdict", "extra_residues_are", "notes",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results/baseline"))
    ap.add_argument("--out", type=Path, default=None, help="default: <results-dir>/audit.csv")
    ap.add_argument(
        "--include-exact", type=int, default=0,
        help="also add this many randomly chosen exact rows (seeded), to catch annotations the caller agrees with but that are wrong",
    )
    ap.add_argument("--force", action="store_true", help="overwrite an existing worksheet (destroys your notes)")
    args = ap.parse_args(argv)

    src = args.results_dir / "cds_recovery.csv"
    if not src.is_file():
        print(f"not found: {src} (run scripts/01_validate_pilot.py first, with the same --out-dir)", file=sys.stderr)
        return 2
    out = args.out or args.results_dir / "audit.csv"
    if out.exists() and not args.force:
        print(f"refusing to overwrite {out}: it may contain your notes. Use --force to replace it.", file=sys.stderr)
        return 2

    with open(src, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    review = [dict(r, reason_for_review=r["status"]) for r in rows if r["status"] not in ("exact", "below_min_aa")]
    exact = [r for r in rows if r["status"] == "exact"]
    sample = random.Random(0).sample(exact, min(args.include_exact, len(exact)))
    review += [dict(r, reason_for_review="exact_sample") for r in sample]
    review.sort(key=lambda r: (r["plasmid_id"], r["location_1based"]))

    with open(out, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(review)

    skipped = sum(1 for r in rows if r["status"] == "below_min_aa")
    print(f"Wrote {len(review)} rows to {out}  ({len(sample)} exact samples; {skipped} below_min_aa rows skipped)")
    print("Fill in 'verdict' with one of: " + ", ".join(VERDICTS))
    print("Fill in 'extra_residues_are' for contained rows (what is the extra sequence?), and 'notes' freely.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
