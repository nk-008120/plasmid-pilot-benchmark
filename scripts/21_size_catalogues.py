#!/usr/bin/env python3
"""Build twilight catalogues of a given size, growing them with *random* decoys rather than families.

    python scripts/21_size_catalogues.py --sizes 10000 20000 40000 80000 160000 \\
        --swissprot %USERPROFILE%\\Downloads\\uniprot_sprot.fasta.gz

**Why the decoys have to be random.** The two catalogue sizes measured so far, 15,205 and 100,685, differ
by more than their size: the larger one was grown by adding *other payloads' homolog families*, so it is
denser in clustered near-relatives, not merely bigger. Any gap-versus-size curve drawn through those two
points confounds "more proteins" with "more family structure". Here every catalogue holds exactly the same
must-keep core, the audited payloads and their own exact-alignment homologs, which are the only proteins
the benchmark can score against, and differs only in how many uniformly sampled Swiss-Prot decoys sit
around it. Decoys are drawn from entries that are *not* homologs of any audited payload, so growth adds
noise rather than answers.

The smallest size is clamped upward where the must-keep core already exceeds it; the real size is written
into the filename and reported, because a catalogue labelled 10k that is actually 10,301 would quietly
mislabel a point on the curve.

Writes `reference_<actual>.fasta` into --out-dir, ready for `10_twilight.py --reference`.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

from pilot.homology import iter_swissprot
from pilot.reference import Entry, iter_fasta, write_fasta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--homologs", type=Path, default=Path("results/homologs_audited/homologs.csv"))
    ap.add_argument("--catalogue", type=Path,
                    default=Path("results/homologs_audited/reference_twilight.fasta"),
                    help="source of the must-keep sequences, so they are byte-identical across sizes")
    ap.add_argument("--swissprot", type=Path, required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10_000, 20_000, 40_000, 80_000, 160_000])
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--out-dir", type=Path, default=Path("results/size_series"))
    args = ap.parse_args(argv)

    for p in (args.manifest, args.homologs, args.catalogue, args.swissprot):
        if not p.is_file():
            print(f"not found: {p}", file=sys.stderr)
            return 2

    t0 = time.time()
    with open(args.manifest, newline="", encoding="utf-8-sig", errors="replace") as fh:
        payloads = {(r.get("payload_accession") or "").strip() for r in csv.DictReader(fh)} - {""}
    with open(args.homologs, newline="", encoding="utf-8-sig") as fh:
        homologs = {r["homolog_accession"] for r in csv.DictReader(fh)}
    must = payloads | homologs
    have = {e.accession: e for e in iter_fasta(args.catalogue) if e.accession in must}
    print(f"must-keep core: {len(payloads)} payloads + {len(homologs):,} homologs "
          f"= {len(have):,} sequences available")

    print(f"reading {args.swissprot.name} ...", flush=True)
    sprot = [(a, s) for a, s in iter_swissprot(args.swissprot) if a not in must]
    print(f"    {len(sprot):,} candidate decoys (Swiss-Prot minus the audited payloads' own families) "
          f"in {time.time() - t0:.0f}s", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    built = []
    for target in sorted(args.sizes):
        n_decoys = max(0, target - len(have))
        if n_decoys > len(sprot):
            print(f"    {target:,}: only {len(sprot):,} decoys exist; capping", file=sys.stderr)
            n_decoys = len(sprot)
        # one generator per size, seeded the same way, so the smaller catalogues are prefixes in
        # distribution rather than unrelated draws; the core is identical in every one either way
        rng = random.Random(args.seed + target)
        picked = rng.sample(sprot, n_decoys)
        entries = list(have.values()) + [Entry(a, f"{a} decoy", s) for a, s in picked]
        entries.sort(key=lambda e: e.accession)
        actual = len(entries)
        out = args.out_dir / f"reference_{actual}.fasta"
        write_fasta(entries, out)
        built.append({"requested": target, "actual": actual, "core": len(have),
                      "decoys": n_decoys, "path": str(out),
                      "residues": sum(len(e.seq) for e in entries)})
        note = "  (core already exceeds the request)" if n_decoys == 0 else ""
        print(f"    {target:>7,} requested -> {actual:>7,} proteins "
              f"({len(have):,} core + {n_decoys:,} random decoys){note}")

    (args.out_dir / "size_series.json").write_text(
        json.dumps({"seed": args.seed, "catalogues": built}, indent=2), encoding="utf-8")
    print(f"\nwrote {len(built)} catalogues to {args.out_dir}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
