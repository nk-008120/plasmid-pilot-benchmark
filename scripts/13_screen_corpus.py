#!/usr/bin/env python3
"""Screen the fetched NCBI records down to a non-redundant corpus, and report the attrition honestly.

    python scripts/13_screen_corpus.py --in-dir data/corpus --out-dir data/corpus_kept

Reads every `.gbk` written by `11_fetch_corpus.py`, applies the filters in `pilot.corpus`, copies the
survivors to `--out-dir` and writes `corpus_screen.csv` (per record, with the reason it was dropped) plus
a printed summary.

The attrition table is a result, not bookkeeping. "2,063 records returned, N after screening" with the
reasons itemised is what lets a reviewer judge whether the corpus is a corpus or a pile, and the
near-duplicate count in particular is the number that decides whether an n of several hundred means
anything.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

from pilot.corpus import MAX_AMBIGUOUS, MAX_BP, MIN_BP, NEAR_DUP_JACCARD, Record, screen


def load(in_dir: Path) -> list[Record]:
    from Bio import SeqIO

    out = []
    for p in sorted(in_dir.glob("*.gbk")):
        try:
            rec = next(SeqIO.parse(p, "genbank"))
        except (StopIteration, ValueError) as exc:
            print(f"  unreadable, skipped: {p.name} ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue
        out.append(Record(
            accession=rec.id, filename=p.name, seq=str(rec.seq),
            n_cds=sum(1 for f in rec.features if f.type == "CDS"),
            topology=rec.annotations.get("topology", ""), description=rec.description[:200],
        ))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=Path("data/corpus"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/corpus_kept"))
    ap.add_argument("--min-bp", type=int, default=MIN_BP)
    ap.add_argument("--max-bp", type=int, default=MAX_BP)
    ap.add_argument("--keep-uncds", action="store_true", help="keep records with no CDS annotation")
    ap.add_argument("--max-ambiguous", type=float, default=MAX_AMBIGUOUS)
    ap.add_argument("--near-dup", type=float, default=NEAR_DUP_JACCARD,
                    help="sketch Jaccard at or above which two records count as the same construct")
    ap.add_argument("--no-copy", action="store_true", help="report only, copy nothing")
    args = ap.parse_args(argv)

    if not args.in_dir.is_dir():
        print(f"not found: {args.in_dir}, run scripts/11_fetch_corpus.py first", file=sys.stderr)
        return 2
    t0 = time.time()
    print(f"reading {args.in_dir}/...", flush=True)
    records = load(args.in_dir)
    if not records:
        print("no readable records", file=sys.stderr)
        return 2
    print(f"    {len(records):,} records read in {time.time() - t0:.0f}s\nscreening...", flush=True)

    res = screen(records, min_bp=args.min_bp, max_bp=args.max_bp, require_cds=not args.keep_uncds,
                 max_ambiguous=args.max_ambiguous, near_dup_threshold=args.near_dup)
    reason_of = {acc: why for why, accs in res.rejected.items() for acc in accs}

    rows = [dict(accession=r.accession, filename=r.filename, length_bp=r.length, n_cds=r.n_cds,
                 topology=r.topology, ambiguous_fraction=round(r.ambiguous_fraction, 5),
                 outcome="kept" if r.accession not in reason_of else "rejected",
                 reason=reason_of.get(r.accession, ""), description=r.description)
            for r in records]
    with open(args.in_dir.parent / "corpus_screen.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    if not args.no_copy:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for r in res.kept:
            shutil.copy2(args.in_dir / r.filename, args.out_dir / r.filename)

    n = len(records)
    print(f"\n{'outcome':<22}{'count':>8}{'% of input':>12}")
    print(f"    {'kept':<20}{len(res.kept):>8}{100 * len(res.kept) / n:>11.1f}%")
    for why, accs in res.rejected.items():
        if accs:
            print(f"    {why:<20}{len(accs):>8}{100 * len(accs) / n:>11.1f}%")
    assert len(res.kept) + sum(len(v) for v in res.rejected.values()) == n, "records lost in screening"

    kept = res.kept
    if kept:
        lens = sorted(r.length for r in kept)
        cds = sorted(r.n_cds for r in kept)
        print(f"\nkept corpus: {len(kept)} constructs")
        print(f"  length bp   median {lens[len(lens) // 2]:,}   range {lens[0]:,}-{lens[-1]:,}")
        print(f"  CDS count   median {cds[len(cds) // 2]}   range {cds[0]}-{cds[-1]}")
        print(f"  circular    {sum(1 for r in kept if r.topology == 'circular')}/{len(kept)}")
    print(f"\nwrote corpus_screen.csv" + ("" if args.no_copy else f" and {len(res.kept)} files to {args.out_dir}/"))
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
