#!/usr/bin/env python3
"""Build one shared twilight reference set covering every tier's payloads, via DIAMOND.

Two phases. First, collect the payload accessions from any number of manifests and write them as a FASTA:

    python scripts/16_build_twilight_refs.py --manifest data/manifest.csv \\
        --manifest data/manifest_auto.csv --manifest data/synthetic/synthetic_manifest.csv \\
        --swissprot %USERPROFILE%\\Downloads\\uniprot_sprot.fasta.gz --out-dir data/reference

Then run the DIAMOND command it prints, and call it again with `--hits` to turn those alignments into
`homologs.csv` and `reference_twilight.fasta`.

**Why DIAMOND and not `09_find_homologs.py`.** That script uses exact Smith-Waterman via parasail: 565 s
for 16 payloads, which is fine at pilot scale and several hours once the corpus brings a hundred-odd more.
DIAMOND does the sweep in minutes.

**It must be run with `--ultra-sensitive`.** DIAMOND's default mode is tuned for close matches. Measured
against parasail on the same 16 payloads, default mode found 922 homologs to parasail's 3,396, and at
<=35% identity, the regime this benchmark exists to measure, 508 against 3,092, missing 87% of them.
GFP returned 255 remote homologs by exact alignment and **zero** in default mode. Using default mode here
would have made the twilight sweep look impossible at low holdouts for a reason that had nothing to do
with the data. `scripts/payload_homologs.sh` sets the flag; `09_find_homologs.py` remains the exact
reference implementation to validate against.

**One reference set for all three tiers**, deliberately: scoring the hand-audited, corpus and synthetic
constructs against different catalogues would make their numbers incomparable, which is the one thing a
three-tier design cannot afford.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from pilot.homology import Homolog, coverage_of, iter_swissprot
from pilot.reference import Entry, write_fasta

MIN_IDENTITY, MIN_COVERAGE = 0.20, 0.40


def payload_accessions(manifests: list[Path]) -> dict[str, set[str]]:
    """{accession: {manifest names that use it}}, the provenance matters when tiers overlap."""
    out: dict[str, set[str]] = defaultdict(set)
    for m in manifests:
        if not m.is_file():
            print(f"  skipped, not found: {m}", file=sys.stderr)
            continue
        with open(m, newline="", encoding="utf-8-sig", errors="replace") as fh:
            for r in csv.DictReader(fh):
                acc = (r.get("payload_accession") or "").strip()
                if acc:
                    out[acc].add(m.name)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, action="append", required=True)
    ap.add_argument("--swissprot", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/reference"))
    ap.add_argument("--hits", type=Path, help="DIAMOND output from phase 1's command; triggers phase 2")
    ap.add_argument("--min-identity", type=float, default=MIN_IDENTITY)
    ap.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    ap.add_argument("--decoys", type=Path, default=Path("data/reference/reference.fasta"),
                    help="existing decoy set to fold into the twilight reference")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    accs = payload_accessions(args.manifest)
    print(f"{len(accs)} distinct payload accessions across {len(args.manifest)} manifests")
    for m in args.manifest:
        n = sum(1 for v in accs.values() if m.name in v)
        print(f"    {m.name:<34}{n:>5}")

    print(f"\nreading {args.swissprot.name} ...", flush=True)
    sprot = dict(iter_swissprot(args.swissprot, min_len=1, max_len=10_000))
    print(f"    {len(sprot):,} entries")
    missing = sorted(a for a in accs if a not in sprot)
    if missing:
        print(f"    {len(missing)} payload accessions absent from this release: {missing[:8]}", file=sys.stderr)

    q = args.out_dir / "payloads.fasta"
    write_fasta([Entry(a, a, sprot[a]) for a in sorted(accs) if a in sprot], q)
    print(f"\nwrote {q} ({sum(1 for a in accs if a in sprot)} sequences)")

    if not args.hits:
        print("\nPhase 1 done. Now run, in WSL:\n")
        print("  bash /mnt/c/NK/plasmid-pilot/scripts/payload_homologs.sh\n")
        print("then re-run this script with --hits data/reference/payload_hits.tsv")
        return 0

    if not args.hits.is_file():
        print(f"not found: {args.hits}", file=sys.stderr)
        return 2
    print(f"\nreading {args.hits.name} ...", flush=True)
    homologs: dict[str, list[Homolog]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    with open(args.hits, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 7:
                continue
            try:
                payload, pid_, length = f[0], float(f[2]), int(f[3])
                qlen, slen, bits = int(f[4]), int(f[5]), float(f[6])
            except ValueError:
                continue
            sub = f[1].split("|")[1] if f[1].count("|") >= 2 else f[1]
            if slen <= 0 or qlen <= 0 or (payload, sub) in seen:
                continue
            seen.add((payload, sub))
            # coverage is measured against the payload, as in pilot.homology.find_homologs. Dividing by
            # the subject length instead is what made the two routes disagree.
            ident, cov = pid_ / 100.0, coverage_of(length, qlen)
            if ident >= args.min_identity and cov >= args.min_coverage:
                homologs[payload].append(Homolog(sub, bits, round(ident, 4), round(cov, 4), slen))

    rows = [dict(payload_accession=p, homolog_accession=h.accession, score=h.score,
                 identity=h.identity, coverage=h.coverage, length=h.length)
            for p in sorted(homologs) for h in sorted(homologs[p], key=lambda x: -x.identity)]
    out_csv = args.out_dir / "homologs.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    keep: dict[str, Entry] = {}
    if args.decoys.is_file():
        from pilot.reference import iter_fasta

        for e in iter_fasta(args.decoys):
            keep[e.accession] = e
    for a in accs:
        if a in sprot:
            keep.setdefault(a, Entry(a, f"{a} payload", sprot[a]))
    for p, hs in homologs.items():
        for h in hs:
            if h.accession in sprot:
                keep.setdefault(h.accession, Entry(h.accession, f"{h.accession} homolog_of:{p}",
                                                   sprot[h.accession]))
    out_fa = args.out_dir / "reference_twilight.fasta"
    write_fasta(list(keep.values()), out_fa)

    import statistics as st

    counts = [len(v) for v in homologs.values()]
    print(f"\n{len(rows):,} homolog rows for {len(homologs)} payloads -> {out_csv}")
    print(f"  homologs per payload: median {int(st.median(counts))}, range {min(counts)}-{max(counts)}")
    print(f"  payloads with no homolog at all: {sum(1 for a in accs if a not in homologs)}")
    for t in (0.90, 0.70, 0.50, 0.35, 0.25):
        n = sum(1 for p in homologs if any(h.identity <= t for h in homologs[p]))
        print(f"  payloads still scorable at a {t:.0%} holdout: {n}/{len(accs)}")
    print(f"\ntwilight reference: {len(keep):,} proteins -> {out_fa}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
