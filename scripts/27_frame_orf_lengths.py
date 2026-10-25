#!/usr/bin/env python3
"""The ORF length of every construct in a benchmark frame, so a cost probe can be extrapolated honestly.

    python scripts/27_frame_orf_lengths.py \
        --results-dir results/gate173_control_corpus --manifest data/manifest_auto.csv \
        --reference results/gate_all173/reference_twilight.fasta \
        --frame results/gate173_control_corpus/twilight_sw-blosum62-11-1_corpus.csv \
        --threshold 0.25 --out results/bhs_corpus_probe/frame_orf_lengths.csv

**Why this exists.** The per-segment best-hit protocol costs one exact Smith-Waterman pass over the whole
catalogue per query, and the cost of a pass is proportional to the query's length. So "seconds per
construct", measured on a sample, extrapolates to a different set of constructs only if the two have the
same ORF length distribution, and nothing guarantees that. this project's working rules asks for the falsifying case to be
named before a subset is trusted: here it is a frame whose ORFs are systematically longer than the
sample's, which a per-construct mean would understate without leaving any trace.

This script measures the frame's ORF lengths directly, which costs no catalogue passes at all: it calls
`build_arms` **without** a scorer or reference sequences, so the `bhs` arm is not built and the only
alignments are the payload-against-ORF ones that choose the payload-bearing ORF. That is a few hundred
short alignments rather than a few hundred passes over 20.8 million residues.

Writes one row per scorable construct: the chosen ORF's length, the motif arm's segment count, and the
oracle span's length. Reports the distribution, and the sample's share of it when `--sample` is given.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
from collections import defaultdict
from pathlib import Path

from pilot.arms import build_arms
from pilot.reference import iter_fasta

SCRIPTS = Path(__file__).resolve().parent


def _protocol_module():
    """`scripts/26_bhs_protocol.py`, loaded by path: `scripts/` is not an importable package.

    The frame definition and the sampling rule are imported rather than restated: a probe extrapolated
    with a second, independently written copy of "which constructs are in the frame" would be comparing
    two different populations and could not say so.
    """
    spec = importlib.util.spec_from_file_location("bhs_protocol", SCRIPTS / "26_bhs_protocol.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read_faa(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for e in iter_fasta(path):
        oid = e.header.split()[0]
        out[oid.split("|")[0]][oid] = e.seq
    return out


def quantiles(v: list[int]) -> dict[str, float]:
    s = sorted(v)
    q = lambda f: s[min(len(s) - 1, int(f * (len(s) - 1) + 0.5))]  # noqa: E731
    return {"min": s[0], "p25": q(0.25), "median": q(0.50), "p75": q(0.75), "p90": q(0.90), "max": s[-1],
            "mean": round(sum(s) / len(s), 1), "total": sum(s)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--sample", type=int, default=None, help="also report the seeded sample's share")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    m = _protocol_module()
    refs = {e.accession: e.seq for e in iter_fasta(args.reference)}
    with open(args.manifest, newline="", encoding="utf-8-sig", errors="replace") as fh:
        man = {r["filename"].strip(): r for r in csv.DictReader(fh) if (r.get("filename") or "").strip()}
    with open(args.results_dir / "pilot_summary.csv", newline="", encoding="utf-8-sig") as fh:
        summary = list(csv.DictReader(fh))
    orfs_by_plasmid = read_faa(args.results_dir / "orfs.faa")

    frame = m.scorable_frame(args.frame, args.threshold)
    available = [r["plasmid_id"] for r in summary]
    in_frame = m.select_constructs(available, frame=frame)
    sample = set(m.select_constructs(available, frame=frame, sample=args.sample, seed=args.seed)
                 if args.sample else [])
    print(f"frame: {len(frame)} constructs scorable under every arm at {args.threshold:.0%} in "
          f"{args.frame.name}; {len(in_frame)} of them present in {args.results_dir.name}")

    rows = []
    for srow in summary:
        pid = srow["plasmid_id"]
        if pid not in set(in_frame):
            continue
        acc = (man.get(srow["source_file"], {}).get("payload_accession") or "").strip()
        if not acc or acc not in refs:
            continue
        arms = build_arms(pid, orfs_by_plasmid.get(pid, {}), refs[acc])
        if arms is None:
            continue
        rows.append(dict(plasmid_id=pid, payload_accession=acc, orf_id=arms.orf_id,
                         orf_len=len(arms.orf_seq), k_auto=len([s for s in arms.auto if s.seq]),
                         oracle_len=(arms.span.length if arms.span else ""),
                         in_sample=int(pid in sample)))

    print(f"built arms for {len(rows)} of {len(in_frame)} frame constructs "
          f"({len(in_frame) - len(rows)} had no ORF carrying the payload)")
    lens = [r["orf_len"] for r in rows]
    print(f"\nORF length, all {len(lens)} frame constructs: {quantiles(lens)}")
    if sample:
        sl = [r["orf_len"] for r in rows if r["in_sample"]]
        rest = [r["orf_len"] for r in rows if not r["in_sample"]]
        print(f"ORF length, the {len(sl)} sampled:          {quantiles(sl)}")
        print(f"ORF length, the {len(rest)} not sampled:      {quantiles(rest)}")
        print(f"\nthe sample carries {sum(sl):,} of {sum(lens):,} frame ORF residues "
              f"({100 * sum(sl) / sum(lens):.1f}%) on {100 * len(sl) / len(lens):.1f}% of the constructs, "
              f"so a per-construct mean would scale the frame by "
              f"{(sum(lens) / len(lens)) / (sum(sl) / len(sl)):.3f}x against a per-residue one")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
