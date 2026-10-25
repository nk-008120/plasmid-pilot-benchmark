#!/usr/bin/env python3
"""Build the synthetic tier: real payloads in controlled architectures, with exact ground truth.

    python scripts/15_build_synthetic.py --swissprot %USERPROFILE%\\Downloads\\uniprot_sprot.fasta.gz \\
        --out-dir data/synthetic --payloads 24

Why this tier exists: the working record 16.6. The NCBI corpus contains zero 2A peptides and 33 tagged fusions,
so it cannot test the architectures the project is about. Here those architectures are built on purpose,
with dilution as a swept variable rather than whatever the observed constructs happened to have.

Per payload it writes:
  single_cds           dilution 0, ordinary backbone
  backbone_heavy       dilution 0, large backbone
  tagged_fusion        at each requested dilution, 3xFLAG, carrier, TEV site, His6, NLS
  motifless_fusion     at each requested dilution, partner joined directly, nothing for a motif to find
  multi_gene_cassette  at each requested dilution, P2A- and T2A-linked passengers

Payloads are drawn from Swiss-Prot, excluding anything `autolabel.is_vector_part` recognises and anything
listed in `--exclude`, which is how payloads already used in the other tiers are kept out: a synthetic
result that shares payloads with the hand-audited set would not be independent of it.

Writes `.gbk` files plus `synthetic_manifest.csv` carrying exact payload coordinates, so no oracle
alignment is needed to score this tier.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from pilot.fillers import RealProteinFiller, eligible_pool, load_homolog_sets
from pilot.homology import iter_swissprot
from pilot.synthconstruct import build_construct, random_protein, write_genbank

DILUTIONS = (0.2, 0.4, 0.6, 0.8)
MIN_PAYLOAD_AA, MAX_PAYLOAD_AA = 150, 800
# Real resistance markers, by accession. Named rather than discovered because a FASTA carries no parsed
# description to match on, and because the benchmark should make the pipeline exclude an actual marker
# through `autolabel.is_vector_part`: both of these turn up in the corpus DIAMOND output as the highest
# -scoring hit on dozens of constructs, which is exactly the nuisance they are here to reproduce.
MARKERS = (("AmpR", "P62593"), ("KanR", "P00551"), ("CmR", "P62577"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swissprot", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--payloads", type=int, default=24)
    ap.add_argument("--dilutions", default=",".join(str(d) for d in DILUTIONS))
    ap.add_argument("--exclude", type=Path, action="append", default=[],
                    help="CSV with a payload_accession column; repeatable. Keeps the tiers independent.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--filler-source", choices=("random", "swissprot"), default="random",
                    help="where every diluting block comes from. 'random' is `random_protein` and "
                         "reproduces the existing tier byte for byte (condition A). "
                         "'swissprot' draws real proteins, which is the one change that makes the tier's "
                         "216 low-f constructs informative: random sequence has no family in the "
                         "catalogue and so never competes for rank (18.3).")
    ap.add_argument("--filler-homologs", type=Path, default=Path("results/gate_all173/homologs.csv"),
                    help="defines which accessions are a payload's relatives and therefore may never be "
                         "drawn as its filler. Presence at ANY identity disqualifies. A filler that is a "
                         "relative of its own payload would make the filler's family a true homolog of "
                         "the answer, so retrieving it would score as success and the benchmark would "
                         "invert. This is the one correctness requirement in the rebuild.")
    ap.add_argument("--filler-catalogue", type=Path, default=None,
                    help="restrict the filler pool to accessions present in this FASTA: the catalogue "
                         "the tier is scored against. REQUIRED for condition C to mean anything: the "
                         "catalogue is not Swiss-Prot but the gated homologs of this project's 173 "
                         "payloads, so a filler drawn uniformly from Swiss-Prot has almost no chance of "
                         "having a relative in the search database (measured: 2 of 48), and would "
                         "compete for rank exactly as little as random sequence does.")
    ap.add_argument("--filler-min-aa", type=int, default=100,
                    help="shortest Swiss-Prot entry admitted to the filler pool. A very short filler is "
                         "tiled many times over to fill a large budget, which is the most artificial "
                         "case, so the floor trades pool size against that.")
    args = ap.parse_args(argv)

    if not args.swissprot.is_file():
        print(f"not found: {args.swissprot}", file=sys.stderr)
        return 2
    dilutions = [float(x) for x in args.dilutions.split(",") if x.strip()]

    excluded: set[str] = set()
    for p in args.exclude:
        if p.is_file():
            with open(p, newline="", encoding="utf-8-sig", errors="replace") as fh:
                for r in csv.DictReader(fh):
                    acc = (r.get("payload_accession") or "").strip()
                    if acc:
                        excluded.add(acc)
    print(f"excluding {len(excluded)} accessions already used by other tiers")

    print("reading Swiss-Prot...", flush=True)
    entries = list(iter_swissprot(args.swissprot, min_len=MIN_PAYLOAD_AA, max_len=MAX_PAYLOAD_AA))
    print(f"    {len(entries):,} entries in the {MIN_PAYLOAD_AA}-{MAX_PAYLOAD_AA} aa window")

    rng = random.Random(args.seed)
    pool = [(a, s) for a, s in entries if a not in excluded]
    rng.shuffle(pool)
    payloads = pool[: args.payloads]
    if len(payloads) < args.payloads:
        print(f"only {len(payloads)} payloads available", file=sys.stderr)

    by_acc = dict(entries)
    markers = [(label, by_acc[acc]) for label, acc in MARKERS if acc in by_acc]
    if not markers:
        print("none of the named marker accessions were found in this Swiss-Prot release; "
              "the backbone will carry no annotated marker", file=sys.stderr)
    print(f"markers: {[(m[0], len(m[1])) for m in markers]}")

    # ---- the filler source -------------------------------------------------------------------------
    filler = random_protein
    real_filler: RealProteinFiller | None = None
    if args.filler_source == "swissprot":
        tier_payloads = {a for a, _ in payloads}
        rel, cat_payloads = load_homolog_sets(args.filler_homologs)
        print(f"\nfiller pool: reading Swiss-Prot with descriptions "
              f"({args.filler_min_aa}-{MAX_PAYLOAD_AA} aa)...", flush=True)
        described = list(iter_swissprot(args.swissprot, min_len=args.filler_min_aa,
                                        max_len=MAX_PAYLOAD_AA, with_description=True))
        restrict = None
        if args.filler_catalogue:
            from pilot.reference import iter_fasta
            restrict = {e.accession for e in iter_fasta(args.filler_catalogue)}
            print(f"  restricting the pool to the {len(restrict):,} accessions in "
                  f"{args.filler_catalogue}: the database the tier is scored against")
        pool, forbidden = eligible_pool(described, payload_relatives=rel,
                                        tier_payloads=tier_payloads,
                                        catalogue_payloads=cat_payloads,
                                        restrict_to=restrict)
        print(f"    {len(described):,} entries -> {len(pool):,} eligible after is_vector_part and "
              f"removing the {len(cat_payloads)} catalogue payloads and {len(tier_payloads)} tier payloads")
        if restrict is not None:
            print(f"  every filler therefore has at least one catalogue entry at 100% identity "
                  f"(itself), so the filler's family is present by construction")
        real_filler = RealProteinFiller(pool, forbidden, seed=args.seed)
        # Report the binding count rather than the setting: how far the per-payload relative exclusion
        # actually cuts the pool. A payload whose pool is barely larger than `slots` is a silent
        # constraint on which fillers it can have.
        sizes = {a: len([1 for x, _ in pool if x not in forbidden[a]]) for a in sorted(tier_payloads)}
        print(f"  per-payload eligible pool: min {min(sizes.values()):,}, max {max(sizes.values()):,}  "
              f"(global {len(pool):,}); the relative exclusion removes at most "
              f"{len(pool) - min(sizes.values()):,} entries for any one payload")
        filler = real_filler

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    for acc, seq in payloads:
        if real_filler is not None:
            real_filler.begin(acc)
        plan = [("single_cds", 0.0, 0), ("backbone_heavy", 0.0, 12000)]
        plan += [(a, d, 0) for d in dilutions
                 for a in ("tagged_fusion", "motifless_fusion", "multi_gene_cassette")]
        for archetype, dil, pad in plan:
            cid = f"syn_{acc}_{archetype}_{int(dil * 100):02d}"
            if real_filler is not None:
                # per construct, so `last_draws` holds only this construct's blocks; the assignment
                # itself is keyed on the payload and so is identical for all 14 of its constructs
                real_filler.begin(acc)
            c = build_construct(cid, seq, acc, archetype, dil, markers, rng, backbone_pad_bp=pad,
                                filler=filler)
            write_genbank(c, args.out_dir / f"{cid}.gbk")
            n += 1
            draws = list(real_filler.last_draws) if real_filler is not None else []
            rows.append(dict(
                filename=f"{cid}.gbk", construct_id=cid, archetype_manual=archetype,
                payload_accession=acc, payload_aa=c.payload_aa, plasmid_bp=len(c.dna),
                orf_start_nt=c.orf_start_nt, orf_aa=c.orf_aa,
                payload_start_aa=c.payload_start_aa, payload_end_aa=c.payload_end_aa,
                dilution_target=dil, dilution_actual=c.dilution,
                payload_fraction=c.payload_fraction,
                parts="|".join(f"{p.name}:{p.start_aa}-{p.end_aa}" for p in c.parts),
                label_source="exact_by_construction", annotation_audited="n/a",
                filler_source=args.filler_source,
                # Every filler's accession is recorded because condition B cannot mask what the manifest
                # does not carry (this project's rebuild plan, point 2).
                filler_accessions="|".join(d.accession for d in draws),
                filler_modes="|".join(d.mode for d in draws),
                filler_requested_aa="|".join(str(d.requested_aa) for d in draws),
                filler_source_aa="|".join(str(d.source_aa) for d in draws),
            ))
    out = args.out_dir / "synthetic_manifest.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    print(f"\n{n} constructs from {len(payloads)} payloads -> {args.out_dir}/")
    if real_filler is not None:
        modes = Counter(d.mode for d in real_filler.draws)
        print(f"\nfiller blocks placed: {len(real_filler.draws)} over "
              f"{len({d.accession for d in real_filler.draws})} distinct accessions")
        print(f"  by fit: {dict(modes)}:  "
              f"tiled means the source was shorter than the padding budget and was repeated; "
              f"truncated means it was longer and was cut")
        cons_with = Counter()
        for r in rows:
            if r["filler_accessions"]:
                cons_with[len(r["filler_accessions"].split("|"))] += 1
        print(f"  constructs by filler-block count: {dict(sorted(cons_with.items()))}; "
              f"{sum(1 for r in rows if not r['filler_accessions'])} constructs carry no filler")
        req = [d.requested_aa for d in real_filler.draws]
        print(f"  requested block length: min {min(req)}, median "
              f"{sorted(req)[len(req) // 2]}, max {max(req)}")
    print(f"\n{'archetype':<24}{'n':>5}")
    for a, k in Counter(r["archetype_manual"] for r in rows).most_common():
        print(f"    {a:<22}{k:>5}")
    print(f"\n{'dilution':<24}{'n':>5}{'mean actual':>14}")
    for d in sorted({r["dilution_target"] for r in rows}):
        g = [r for r in rows if r["dilution_target"] == d]
        print(f"    {d:<22}{len(g):>5}{sum(r['dilution_actual'] for r in g) / len(g):>14.4f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
