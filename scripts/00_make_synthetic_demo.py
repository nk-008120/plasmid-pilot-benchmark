#!/usr/bin/env python3
"""Write four synthetic GenBank plasmids for a no-download smoke test of the whole step.

    python scripts/00_make_synthetic_demo.py --out-dir data/synthetic_demo
    python scripts/01_validate_pilot.py --input-dir data/synthetic_demo --out-dir results/synthetic_demo

Expected: every annotated CDS 'exact', except the two parts of demo_fusion, which are 'contained' in one long ORF.
"""
import argparse
from pathlib import Path

from pilot.synthetic import write_demo_set

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("data/synthetic_demo"))
    args = ap.parse_args()
    for path in write_demo_set(args.out_dir):
        print("wrote", path)
