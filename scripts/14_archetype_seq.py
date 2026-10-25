#!/usr/bin/env python3
"""Reclassify construct archetypes from sequence evidence, and check the rule against the hand labels.

    # does it reproduce archetype_manual on the 23 audited plasmids?
    python scripts/14_archetype_seq.py --results-dir results/baseline2 --hits results/baseline2/hits_23.tsv --validate

    # classify the corpus
    python scripts/14_archetype_seq.py --results-dir results/corpus --hits results/corpus/hits_corpus.tsv

Same discipline as `12_autolabel.py`: the rule is measured against the only constructs whose architecture
was established by hand before it is trusted on several hundred that were not.

Writes `archetype_seq.csv` into the results directory.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pilot.archetype_seq import BACKBONE_FRACTION, classify
from pilot.autolabel import MIN_COVERAGE, MIN_IDENTITY, read_tabular
from pilot.reference import iter_fasta


def read_faa(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for e in iter_fasta(path):
        oid = e.header.split()[0]
        out[oid.split("|")[0]][oid] = e.seq
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--hits", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--backbone-fraction", type=float, default=BACKBONE_FRACTION)
    ap.add_argument("--min-identity", type=float, default=MIN_IDENTITY)
    ap.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    args = ap.parse_args(argv)

    faa = args.results_dir / "orfs.faa"
    summary = args.results_dir / "pilot_summary.csv"
    for p in (faa, summary, args.hits):
        if not p.is_file():
            print(f"not found: {p}", file=sys.stderr)
            return 2

    orfs = read_faa(faa)
    hits = read_tabular(args.hits)
    with open(summary, newline="", encoding="utf-8-sig") as fh:
        lengths = {r["plasmid_id"]: int(r["length_bp"]) for r in csv.DictReader(fh)}

    calls = {
        pid: classify(orfs.get(pid, {}), hits.get(pid, []), length,
                      backbone_fraction=args.backbone_fraction,
                      min_identity=args.min_identity, min_coverage=args.min_coverage)
        for pid, length in lengths.items()
    }

    rows = [dict(plasmid_id=pid, archetype_seq=c.archetype, flags="|".join(sorted(c.flags)),
                 n_payload_orfs=c.n_payload_orfs, n_cleavage_motifs=c.n_cleavage_motifs,
                 n_tag_motifs=c.n_tag_motifs, payload_fraction=c.payload_fraction,
                 evidence="; ".join(c.evidence))
            for pid, c in sorted(calls.items())]
    out = args.results_dir / "archetype_seq.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(calls)} constructs classified from sequence\n")
    print(f"{'archetype':<24}{'count':>7}")
    for a, n in Counter(c.archetype for c in calls.values()).most_common():
        print(f"    {a:<22}{n:>7}")
    print(f"\n{'flag':<24}{'count':>7}")
    for f, n in Counter(f for c in calls.values() for f in c.flags).most_common():
        print(f"    {f:<22}{n:>7}")

    if args.validate:
        with open(args.manifest, newline="", encoding="utf-8-sig", errors="replace") as fh:
            hand = {(r["filename"] or "").rsplit(".", 1)[0].strip(): (r.get("archetype_manual") or "").strip()
                    for r in csv.DictReader(fh) if (r.get("filename") or "").strip()}
        scored = {p: a for p, a in hand.items() if a and p in calls}
        agree = [p for p in scored if calls[p].archetype == scored[p]]
        # backbone_heavy was an exclusive label in the old scheme and is a flag here, so a hand label of
        # backbone_heavy is honoured by the flag rather than by the archetype
        soft = [p for p in scored if p not in agree and scored[p] == "backbone_heavy"
                and "backbone_heavy" in calls[p].flags]
        print(f"\nValidating against {len(scored)} hand labels")
        print(f"  exact archetype match        {len(agree)}/{len(scored)}")
        print(f"  hand said backbone_heavy, now carried as a flag   {len(soft)}")
        print(f"  combined                     {len(agree) + len(soft)}/{len(scored)}")
        print("\nDisagreements (hand -> sequence [flags]):")
        for p in sorted(scored):
            if p in agree or p in soft:
                continue
            c = calls[p]
            print(f"    {p[:44]:<46}{scored[p]:<20} -> {c.archetype:<20}[{'|'.join(sorted(c.flags))}]")
            if c.evidence:
                print(f"            {'; '.join(c.evidence)}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
