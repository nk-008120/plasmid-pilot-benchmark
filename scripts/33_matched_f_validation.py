#!/usr/bin/env python3
"""Does a real-filler synthetic construct behave like a real construct at the same payload fraction?

    python scripts/33_matched_f_validation.py \
        --reference results/gate_all173/reference_twilight.fasta \
        --homologs results/gate_all173/homologs.csv \
        --synthetic-manifest data/synthetic_real/synthetic_manifest.csv \
        --hand-lengths results/bhs_corpus_probe/hand_orf_lengths.csv \
        --out results/synthetic_real/matched_f_validation

**This is the validation gate, and it is the translation of this project's "score it against the 23".**
The hand tier has no synthetic fillers, so the rule cannot be applied literally. But the 23 audited
constructs *are* real constructs with real non-payload parts, and this project measured their
whole-to-oracle gap stratified by f. So the check is: build condition-C constructs at the hand tier's own
measured f values and architectures, and ask whether the gap comes out like the hand tier's at matched f.

The numbers to reproduce (hand tier, 25% holdout, delta mean rank):

| stratum | n | hand delta mean |
|---|---|---|
| f < 0.50 | 2 | **-37.00** |
| 0.50-0.70 | 1 | -1.00 |
| 0.70-0.95 | 5 | -0.80 |
| f >= 0.95 | 7 | **+0.00** |

If synthetic condition C at matched f does not look like that, **the filler model is wrong and the full
sweep is not worth running.** A null here means the same thing it meant in 18.3: the fillers are not
competing. A gap much larger than the hand tier's would mean the opposite problem: a filler at 100%
identity in the catalogue is a stronger distractor than a real co-encoded gene, which would make the
synthetic tier an upper bound rather than a model.

**Three things this script does not need, because the synthetic tier's boundaries are exact.** It does not
call ORFs (the designed chain *is* the ORF), it does not align to find the payload span (the span is
recorded at construction), and it therefore does not need `01_validate_pilot` or `build_arms`. The queries
are the designed chain and the exact payload substring. That makes the oracle arm here at least as strong
as the hand tier's alignment-derived one, which is the conservative direction for a gap.

**Archetype remapping, reported rather than hidden.** `design` gives `single_cds` and `backbone_heavy` no
diluting parts at all, so a hand construct with one of those archetypes and f < 1 cannot be reproduced
under its own archetype. Those are remapped to `tagged_fusion`, the nearest architecture that carries a
carrier, and every remapping is printed.
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from pilot.align import get_scorer
from pilot.fillers import RealProteinFiller
from pilot.reference import iter_fasta
from pilot.synthconstruct import design

STRATA = (("f < 0.50", 0.0, 0.50), ("0.50-0.70", 0.50, 0.70),
          ("0.70-0.95", 0.70, 0.95), ("f >= 0.95", 0.95, 1.0000001))
#: the working record, hand tier, 25% holdout. The targets this run is checked against.
HAND = {"f < 0.50": (2, -37.00), "0.50-0.70": (1, -1.00),
        "0.70-0.95": (5, -0.80), "f >= 0.95": (7, 0.00)}
#: Archetypes that `design` gives no diluting parts, so they cannot carry a target f < 1.
NO_DILUTION = ("single_cds", "backbone_heavy")


def stratum_of(f: float) -> str:
    for name, lo, hi in STRATA:
        if lo <= f < hi:
            return name
    raise ValueError(f"f={f!r} is outside every stratum")


def rank_of_best_target(row: np.ndarray, tgt: np.ndarray, mask: np.ndarray) -> int | None:
    if tgt.size == 0:
        return None
    r = row.astype(np.float64, copy=True)
    r[mask] = -np.inf
    best = float(np.max(r[tgt]))
    if not np.isfinite(best):
        return None
    return int((r >= best).sum())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--homologs", type=Path, required=True)
    ap.add_argument("--synthetic-manifest", type=Path, required=True,
                    help="the real-filler tier's manifest; supplies each payload's fixed filler "
                         "assignment, so the matched set uses the same fillers as the tier it validates")
    ap.add_argument("--hand-lengths", type=Path, required=True,
                    help="from 27_frame_orf_lengths.py on the hand tier: orf_len and oracle_len per "
                         "construct, which is where the matched f values come from")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--scorer", default="align")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--plan-only", action="store_true",
                    help="print the matched plan and its query-residue cost, then stop without scoring")
    args = ap.parse_args(argv)

    refs = list(iter_fasta(args.reference))
    ref_acc = [e.accession for e in refs]
    idx_of = {a: i for i, a in enumerate(ref_acc)}
    by_acc = {e.accession: e.seq for e in refs}
    db_residues = sum(len(e.seq) for e in refs if 50 <= len(e.seq) <= 5000)
    print(f"CATALOGUE: {args.reference}, {len(refs):,} proteins, {db_residues:,} residues "
          f"in the 50-5000 aa window")

    hom: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with open(args.homologs, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            hom[r["payload_accession"]].append((r["homolog_accession"], float(r["identity"])))

    syn = list(csv.DictReader(open(args.synthetic_manifest, newline="", encoding="utf-8-sig")))
    fillers_of: dict[str, list[str]] = {}
    for r in syn:
        accs = [a for a in r["filler_accessions"].split("|") if a]
        if accs and r["payload_accession"] not in fillers_of:
            fillers_of[r["payload_accession"]] = accs
    payloads = sorted({r["payload_accession"] for r in syn})
    print(f"synthetic payloads available: {len(payloads)}; "
          f"{len(fillers_of)} have a recorded filler assignment")

    hand = []
    with open(args.hand_lengths, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["oracle_len"] == "":
                continue
            hand.append((r["plasmid_id"], int(r["orf_len"]), int(r["oracle_len"]),
                         int(r["oracle_len"]) / int(r["orf_len"])))
    hand.sort(key=lambda x: x[3])
    print(f"hand constructs with an f: {len(hand)}\n")

    # archetypes, from the hand manifest by way of the run's own summary
    man = {r["filename"].strip(): r for r in
           csv.DictReader(open("data/manifest.csv", newline="", encoding="utf-8-sig", errors="replace"))
           if (r.get("filename") or "").strip()}
    sm = {r["plasmid_id"]: r for r in
          csv.DictReader(open("results/gate173_bhs_hand/pilot_summary.csv", newline="",
                              encoding="utf-8-sig"))}

    plan = []
    for i, (pid, orf_len, span_len, f) in enumerate(hand):
        arch = (man.get(sm[pid]["source_file"], {}).get("archetype_manual") or "").strip()
        target_f = f
        used = arch
        note = ""
        if arch in NO_DILUTION and target_f < 0.99:
            used, note = "tagged_fusion", f"remapped from {arch} (carries no diluting part)"
        elif arch not in ("tagged_fusion", "motifless_fusion", "multi_gene_cassette") + NO_DILUTION:
            used, note = "tagged_fusion", f"remapped from {arch!r} (unknown archetype)"
        pacc = payloads[i % len(payloads)]
        plan.append(dict(hand_id=pid, hand_f=round(f, 4), hand_orf=orf_len, hand_span=span_len,
                         hand_archetype=arch, archetype=used, note=note, payload=pacc,
                         dilution=round(1.0 - target_f, 4)))

    print(f"{'hand construct':<46}{'f':>7}    {'archetype -> used':<40}{'syn payload':<10}{'dilution':>9}")
    for p in plan:
        print(f"{p['hand_id'][:44]:<46}{p['hand_f']:>7.3f}    "
              f"{(p['hand_archetype'] + ' -> ' + p['archetype']):<40}{p['payload']:<10}"
              f"{p['dilution']:>9.4f}    {p['note']}")

    # ---- build the matched constructs ------------------------------------------------------------
    built = []
    for p in plan:
        pseq = by_acc.get(p["payload"])
        if not pseq:
            print(f"  payload {p['payload']} not in the catalogue; skipped")
            continue
        faccs = fillers_of.get(p["payload"], [])
        pool = [(a, by_acc[a]) for a in faccs if a in by_acc]
        if not pool:
            print(f"  payload {p['payload']} has no usable filler; skipped")
            continue
        filler = RealProteinFiller(pool, {}, seed=args.seed, slots=min(2, len(pool)))
        filler.begin(p["payload"])
        rng = random.Random(f"{args.seed}:{p['hand_id']}")
        protein, parts = design(pseq, p["payload"], p["archetype"], p["dilution"], rng, filler=filler)
        span = next(x for x in parts if x.kind == "payload")
        f_actual = (span.end_aa - span.start_aa + 1) / len(protein)
        built.append(dict(**p, orf_aa=len(protein), span_start=span.start_aa, span_end=span.end_aa,
                          f_actual=round(f_actual, 4), stratum=stratum_of(f_actual),
                          whole=protein, oracle=protein[span.start_aa - 1: span.end_aa],
                          fillers="|".join(d.accession for d in filler.last_draws),
                          filler_modes="|".join(d.mode for d in filler.last_draws)))

    qres = sum(len(b["whole"]) + len(b["oracle"]) for b in built)
    print(f"\n{len(built)} matched constructs built; {qres:,} query residues over "
          f"{2 * len(built)} catalogue passes")
    print(f"  ORF length: min {min(b['orf_aa'] for b in built)}, "
          f"max {max(b['orf_aa'] for b in built)}")
    print(f"  f achieved vs requested: max |difference| "
          f"{max(abs(b['f_actual'] - b['hand_f']) for b in built):.4f}")
    if args.plan_only:
        print("\n--plan-only: stopping before scoring")
        return 0

    t0 = time.time()
    scorer = get_scorer(args.scorer, [e.seq for e in refs], cache_dir=None)
    print(f"\nscorer {scorer.name} ready in {time.time() - t0:.0f}s; scoring at the "
          f"{args.threshold:.0%} holdout", flush=True)

    thr = args.threshold
    for b in built:
        hs = hom.get(b["payload"], [])
        mask = np.array([idx_of[a] for a, i in hs if i > thr and a in idx_of], dtype=int)
        tgt = np.array([idx_of[a] for a, i in hs if i <= thr and a in idx_of], dtype=int)
        t1 = time.time()
        b["rank_whole"] = rank_of_best_target(scorer.score_all(b["whole"]), tgt, mask)
        b["rank_oracle"] = rank_of_best_target(scorer.score_all(b["oracle"]), tgt, mask)
        b["n_targets"], b["n_masked"] = int(tgt.size), int(mask.size)
        b["seconds"] = round(time.time() - t1, 1)
        print(f"    {b['hand_id'][:40]:<42}{b['payload']:<9}f={b['f_actual']:.3f}  "
              f"orf={b['orf_aa']:<5}whole={b['rank_whole']}  oracle={b['rank_oracle']}    "
              f"targets={b['n_targets']:<5}{b['seconds']:>7.1f}s", flush=True)

    scored = [b for b in built if b["rank_whole"] is not None and b["rank_oracle"] is not None]
    print(f"\nPAIRED on the {len(scored)} of {len(built)} matched constructs scorable under both arms "
          f"({len({b['payload'] for b in scored})} distinct payloads)")
    print(f"\n{'stratum':<12}{'n':>4}{'payld':>7}{'whole t1':>10}{'oracle t1':>11}"
          f"{'whole mean':>12}{'oracle mean':>13}{'d mean':>9}   | hand 25.4: n, d mean")
    for name, _, _ in STRATA:
        g = [b for b in scored if b["stratum"] == name]
        hn, hd = HAND[name]
        if not g:
            print(f"{name:<12}{0:>4}{'':>7}{'--':>10}{'--':>11}{'--':>12}{'--':>13}{'--':>9}"
                  f"   | {hn}, {hd:+.2f}")
            continue
        w = sum(b["rank_whole"] for b in g) / len(g)
        o = sum(b["rank_oracle"] for b in g) / len(g)
        print(f"{name:<12}{len(g):>4}{len({b['payload'] for b in g}):>7}"
              f"{sum(1 for b in g if b['rank_whole'] == 1):>7}/{len(g):<2}"
              f"{sum(1 for b in g if b['rank_oracle'] == 1):>8}/{len(g):<2}"
              f"{w:>12.2f}{o:>13.2f}{o - w:>+9.2f}   | {hn}, {hd:+.2f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cols = [k for k in built[0] if k not in ("whole", "oracle")]
        with open(args.out.with_suffix(".csv"), "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(built)
        print(f"\nwrote {args.out.with_suffix('.csv')}")
    print(f"\ntotal {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
