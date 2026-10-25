#!/usr/bin/env python3
"""Each filler's own relatives in the catalogue: what condition B holds out, and the inversion residual.

    python scripts/32_filler_homologs.py \
        --manifest data/synthetic_real/synthetic_manifest.csv \
        --reference results/gate_all173/reference_twilight.fasta \
        --payload-homologs results/gate_all173/homologs.csv \
        --out results/synthetic_real/filler_homologs.csv --threads 4

Condition B is "real fillers, with each filler's own relatives held out of the search database for that
construct at the same identity threshold as the payload". That needs, per filler, the same thing
`homologs.csv` holds per payload: every catalogue entry above the derived score floor, with its identity.
This script produces it, under the identical gate: `E <= 1e-3` inverted to a per-query score floor
against **the residue count of the database actually searched** (this project's working rules, "derive the limit instead of
choosing it"). No `--top-n`, so truncation is structurally impossible.

It is cheap only because the filler assignment is one accession per (payload, slot), fixed across
archetypes and dilution levels: 48 distinct fillers for 336 constructs, so 48 searches rather than 351.

**The second job is the one that matters more.** `pilot.fillers` forbids a filler that appears under its
construct's payload in `homologs.csv`, and that file is the E <= 1e-3 **gated** set, so a sub-threshold
relative of the payload can still be drawn. Once each filler's own relatives are known, the residual is
measurable: the **intersection** of a filler's catalogue relatives with its construct's payload's
relatives. A non-empty intersection means some catalogue entry is a relative of both, so retrieving it
could score as success while being the filler's family: the inversion the benchmark must not contain.
This script reports that intersection per construct, and it is the number to check before any result from
this tier is believed.
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

from pilot.homology import MAX_LEN, MIN_LEN, find_homologs, score_floor
from pilot.reference import iter_fasta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--payload-homologs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-evalue", type=float, default=1e-3)
    ap.add_argument("--min-identity", type=float, default=0.20)
    ap.add_argument("--min-coverage", type=float, default=0.40)
    ap.add_argument("--threads", type=int, default=1,
                    help="parasail releases the GIL, so this is real parallelism on the score pass")
    args = ap.parse_args(argv)

    refs = list(iter_fasta(args.reference))
    entries = [(e.accession, e.seq) for e in refs]
    by_acc = dict(entries)
    in_window = [(a, s) for a, s in entries if MIN_LEN <= len(s) <= MAX_LEN]
    db_residues = sum(len(s) for _, s in in_window)
    print(f"CATALOGUE: {args.reference}, {len(refs):,} proteins, {db_residues:,} residues in the "
          f"{MIN_LEN}-{MAX_LEN} aa window (file total {sum(len(s) for _, s in entries):,})")
    print(f"gate E <= {args.max_evalue:g}; the derived floor spans "
          f"{score_floor(args.max_evalue, 100, db_residues):.1f} (m=100) to "
          f"{score_floor(args.max_evalue, 3000, db_residues):.1f} (m=3000); no --top-n")

    rows = list(csv.DictReader(open(args.manifest, newline="", encoding="utf-8-sig")))
    fillers = sorted({a for r in rows for a in r["filler_accessions"].split("|") if a})
    missing = [a for a in fillers if a not in by_acc]
    print(f"\n{len(fillers)} distinct filler accessions over {len(rows)} constructs; "
          f"{len(missing)} not in the catalogue{': ' + ', '.join(missing) if missing else ''}")
    if missing:
        print("  a filler absent from the catalogue has no family to hold out, which would make "
              "condition B identical to condition C for that construct: reported, not silently kept")

    payload_rel: dict[str, set[str]] = defaultdict(set)
    with open(args.payload_homologs, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            payload_rel[r["payload_accession"]].add(r["homolog_accession"])

    out_rows = []
    t0 = time.time()
    found: dict[str, list] = {}
    for i, acc in enumerate(fillers, 1):
        if acc not in by_acc:
            continue
        t1 = time.time()
        hs = find_homologs(by_acc[acc], in_window, min_identity=args.min_identity,
                           min_coverage=args.min_coverage, threads=args.threads,
                           max_evalue=args.max_evalue)
        found[acc] = hs
        for h in hs:
            out_rows.append(dict(filler_accession=acc, homolog_accession=h.accession,
                                 score=h.score, identity=h.identity, coverage=h.coverage,
                                 length=h.length))
        self_hit = any(h.accession == acc for h in hs)
        print(f"  [{i:>2}/{len(fillers)}] {acc}  len={len(by_acc[acc]):<5}relatives={len(hs):<5}"
              f"self_hit={'yes' if self_hit else 'NO'}    {time.time() - t1:>6.1f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["filler_accession", "homolog_accession", "score",
                                           "identity", "coverage", "length"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {args.out}: {len(out_rows):,} rows over {len(found)} fillers")
    counts = sorted(len(v) for v in found.values())
    print(f"  relatives per filler: min {counts[0]}, median {counts[len(counts) // 2]}, "
          f"max {counts[-1]}")
    no_self = [a for a, v in found.items() if not any(h.accession == a for h in v)]
    print(f"  fillers that did not retrieve themselves at identity 1.0: {len(no_self)}  "
          f"{no_self if no_self else ''}")

    # ---- the inversion residual ------------------------------------------------------------------
    print("\nINVERSION RESIDUAL: catalogue entries that are relatives of BOTH the filler and the "
          "construct's own payload")
    print("  A non-empty intersection means an entry could score as retrieving the payload while being "
          "the filler's family.")
    worst, affected = 0, []
    for r in rows:
        pacc = r["payload_accession"]
        prel = payload_rel.get(pacc, set())
        for a in r["filler_accessions"].split("|"):
            if not a or a not in found:
                continue
            inter = {h.accession for h in found[a]} & prel
            if inter:
                affected.append((r["construct_id"], pacc, a, len(inter), sorted(inter)[:5]))
                worst = max(worst, len(inter))
    if not affected:
        print("  NONE. No filler shares a single catalogue relative with its construct's payload, so "
              "the inversion risk is zero rather than small.")
    else:
        by_pair = {}
        for cid, pacc, facc, n, ex in affected:
            by_pair.setdefault((pacc, facc), (n, ex, 0))
            n0, ex0, c0 = by_pair[(pacc, facc)]
            by_pair[(pacc, facc)] = (n0, ex0, c0 + 1)
        print(f"    {len(affected)} construct-filler pairs affected over {len(by_pair)} distinct "
              f"(payload, filler) pairs; worst intersection {worst} entries")
        print(f"    {'payload':<10}{'filler':<10}{'shared':>8}{'constructs':>12}   examples")
        for (pacc, facc), (n, ex, c) in sorted(by_pair.items(), key=lambda kv: -kv[1][0]):
            print(f"    {pacc:<10}{facc:<10}{n:>8}{c:>12}      {', '.join(ex)}")
        print("\n  THESE CONSTRUCTS MUST BE NAMED AND EXCLUDED FROM THE HEADLINE, per the "
              "pre-registration in results/synthetic_real/prediction.md")
    print(f"\ntotal {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
