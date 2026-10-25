#!/usr/bin/env python3
"""Assign payloads automatically from DIAMOND/BLAST hits, and check the rule against hand labels first.

    # step 1: does this reproduce the hand-curated labels on the 23 audited plasmids?
    python scripts/12_autolabel.py --hits hits_23.tsv --validate

    # step 2: once a rule is chosen, label a corpus with it
    python scripts/12_autolabel.py --hits hits_corpus.tsv --rule best_bitscore --out data/manifest_auto.csv

`--validate` scores all three selection rules against `payload_accession` in the existing manifest and
prints every disagreement. That is the whole point of doing it in this order: the 23 are the only
constructs whose payload is known from the depositor rather than from alignment, so they are the only
place the rule can be checked rather than merely trusted.

Hits come from `--outfmt 6 qseqid sseqid pident length qlen slen bitscore evalue stitle` (DIAMOND and
BLAST+ emit the same layout).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from pilot.autolabel import MIN_COVERAGE, MIN_IDENTITY, RULES, assign_payload, read_tabular


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hits", type=Path, required=True, help="DIAMOND/BLAST tabular output")
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--validate", action="store_true", help="score every rule against the hand labels")
    ap.add_argument("--rule", default="best_bitscore", choices=RULES)
    ap.add_argument("--out", type=Path, default=Path("data/manifest_auto.csv"))
    ap.add_argument("--min-identity", type=float, default=MIN_IDENTITY)
    ap.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    ap.add_argument("--sweep", action="store_true",
                    help="with --validate: show agreement across a grid of gates, to check that the "
                         "chosen one sits on a plateau rather than a tuned peak")
    ap.add_argument("--exclude-reporters", action="store_true")
    args = ap.parse_args(argv)

    if not args.hits.is_file():
        print(f"not found: {args.hits}", file=sys.stderr)
        return 2
    by_plasmid = read_tabular(args.hits)
    if not by_plasmid:
        print("no usable hits: check the --outfmt 6 column order", file=sys.stderr)
        return 2
    print(f"{len(by_plasmid)} plasmids, {sum(len(v) for v in by_plasmid.values()):,} hits")

    kw = dict(min_identity=args.min_identity, min_coverage=args.min_coverage,
              exclude_reporters=args.exclude_reporters)

    if args.validate:
        with open(args.manifest, newline="", encoding="utf-8-sig", errors="replace") as fh:
            hand = {(r["filename"] or "").rsplit(".", 1)[0].strip(): (r.get("payload_accession") or "").strip()
                    for r in csv.DictReader(fh) if (r.get("filename") or "").strip()}
        scored = {p: a for p, a in hand.items() if a and p in by_plasmid}
        print(f"\nValidating against {len(scored)} hand-labelled plasmids with a protein payload.\n")
        if args.sweep:
            print(f"Gate sweep (rule = {args.rule}). A flat region means the gate is not load-bearing.")
            print(f"{'identity':>9}{'coverage':>10}{'agree':>8}{'abstain':>9}{'WRONG':>7}")
            for ident in (0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50):
                for cov in (0.80, 0.60):
                    a = u = w = 0
                    for pid, want in scored.items():
                        g = assign_payload(by_plasmid[pid], rule=args.rule,
                                           min_identity=ident, min_coverage=cov)
                        acc = g.payload_accession if g and g.passed_gate else ""
                        a, u, w = (a + 1, u, w) if acc == want else (a, u + 1, w) if not acc else (a, u, w + 1)
                    mark = "  <: default" if (ident, cov) == (MIN_IDENTITY, MIN_COVERAGE) else ""
                    print(f"{ident:>9.2f}{cov:>10.2f}{a:>8}{u:>9}{w:>7}{mark}")
            print()
        rows = []
        for rule in RULES:
            agree, miss, wrong = 0, 0, []
            for pid, want in scored.items():
                got = assign_payload(by_plasmid[pid], rule=rule, **kw)
                acc = got.payload_accession if got and got.passed_gate else ""
                if acc == want:
                    agree += 1
                elif not acc:
                    miss += 1
                    wrong.append((pid, want, "(no label)"))
                else:
                    wrong.append((pid, want, acc))
            pct = 100 * agree / len(scored) if scored else 0
            print(f"    {rule:<20}agree {agree}/{len(scored)} ({pct:.0f}%)   "
                  f"ungated {miss}   mislabelled {len(wrong) - miss}")
            rows.append((rule, agree, miss, len(wrong) - miss, wrong))
        best = max(rows, key=lambda r: r[1])
        print(f"\nBest rule: {best[0]} ({best[1]}/{len(scored)})")
        print("\nDisagreements under that rule (hand label -> automatic label):")
        for pid, want, acc in best[4]:
            print(f"    {pid[:46]:<48}{want:<10} -> {acc}")
        print("\nRead each one before accepting the rule. A disagreement may be the rule failing, or the\n"
              "hand label encoding a judgement the rule cannot see: those need different responses.")
        return 0

    out_rows = []
    for pid, hits in sorted(by_plasmid.items()):
        got = assign_payload(hits, rule=args.rule, **kw)
        if got is None:
            continue
        out_rows.append(dict(
            filename=f"{pid}.gbk", source_id=pid, payload_accession=got.payload_accession,
            payload_description=got.payload_description, identity=got.identity, coverage=got.coverage,
            bitscore=got.bitscore, payload_orf_id=got.orf_id, rule=got.rule,
            n_candidates=got.n_candidates, n_excluded_marker=got.n_excluded_marker,
            n_excluded_carrier=got.n_excluded_carrier, passed_gate="yes" if got.passed_gate else "no",
            reason=got.reason, label_source="automatic", annotation_audited="no",
        ))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    passed = sum(1 for r in out_rows if r["passed_gate"] == "yes")
    print(f"\nrule {args.rule}, gate identity>={args.min_identity} coverage>={args.min_coverage}")
    print(f"  labelled   {passed}/{len(out_rows)}")
    print(f"  ungated    {len(out_rows) - passed}  (reported, not discarded: the fraction this cannot label)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
