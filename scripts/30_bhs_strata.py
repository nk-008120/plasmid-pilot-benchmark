#!/usr/bin/env python3
"""The segmenter arm stratified by payload fraction, and whether its own coverage predicts that fraction.

    python scripts/30_bhs_strata.py --detail results/bhs_corpus_full/detail_corpus_25.csv \
        --reference results/gate_all173/reference_twilight.fasta

Reads `26_bhs_protocol.py --detail-csv` and re-scores nothing. The scoring run costs about an hour, so
every table and every threshold here is deliberately downstream of it: a re-table or a different cutoff
must never require the hour back.

**Two different questions, and only the second is new.**

1. *Stratified tables.* The same strata as `29_stratify_by_payload_fraction.py`, imported from it rather
   than restated, so the segmenter's numbers land in the same bands as the alignment arms' and the two
   can be read side by side. `f = oracle_len / orf_len` uses the label; it describes constructs.

2. *Does the segmenter's own best-hit coverage predict f?* `best_hit_cov` is the fraction of the ORF
   claimed by the single highest-scoring catalogue hit, and **a screener can compute it**: it consults
   the catalogue, never the payload. If it tracks `f`, it is a triage signal: it says "this ORF is mostly
   one known thing" or "most of this ORF is unaccounted for, decompose it" without anyone knowing the
   answer. That is a usable result in a way a stratification by `f` can never be, because `f` is only
   available once you already know what you were looking for.

**The threshold is swept, not chosen.** this project's working rules: report the plateau rather than the best value. The
full sweep is printed, the operating point is picked by a stated rule (maximum Youden's J, ties to the
larger threshold), and the confusion table is given at that point. A cutoff tuned on the same 286
constructs it is then evaluated on is in-sample, which is said in the output rather than left implied:
it bounds how good the signal could be, and does not establish it would hold on new constructs.

**Correlation is reported as Spearman as well as Pearson, and that matters here.** `f` is a point mass:
the corpus is dominated by `f = 1.0`. A Pearson r over a point mass plus a short tail is driven almost
entirely by the tail's few constructs, and the macro figure over payloads is given alongside because 286
constructs carry far fewer distinct payloads and one of them appears in 37.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from collections import defaultdict
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

PROTOCOLS = (("rank_whole", "whole", "yes"),
             ("rank_auto_pooled", "auto, pooled", "yes"),
             ("rank_auto_per_segment", "auto, per-segment", "yes"),
             ("rank_pooled", "bhs, pooled", "yes"),
             ("rank_per_segment", "bhs, per-segment", "yes"),
             ("rank_oracle_selected", "bhs, oracle-selected", "NO"),
             ("rank_oracle", "oracle", "NO"))


def _strat_module():
    """`scripts/29_stratify_by_payload_fraction.py`, loaded by path: `scripts/` is not a package.

    The strata, the micro/macro averaging and the paired-transition rule are imported, not restated. A
    second copy would let the segmenter's table and the alignment arms' table drift into different bands
    or different denominators while both still looked right.
    """
    spec = importlib.util.spec_from_file_location(
        "stratify", SCRIPTS / "29_stratify_by_payload_fraction.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def ranks_with_ties(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(ranks_with_ties(x), ranks_with_ties(y))


def confusion(pred: list[bool], truth: list[bool]) -> dict:
    tp = sum(1 for p, t in zip(pred, truth) if p and t)
    fp = sum(1 for p, t in zip(pred, truth) if p and not t)
    fn = sum(1 for p, t in zip(pred, truth) if not p and t)
    tn = sum(1 for p, t in zip(pred, truth) if not p and not t)
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "sens": sens, "spec": spec,
            "ppv": (tp / (tp + fp) if tp + fp else None),
            "npv": (tn / (tn + fn) if tn + fn else None),
            "j": (None if sens is None or spec is None else sens + spec - 1)}


def sweep(cov: list[float], truth: list[bool], grid: list[float]) -> list[tuple[float, dict]]:
    """Predict `truth` by `cov < t`, at every t in the grid."""
    return [(t, confusion([c < t for c in cov], truth)) for t in grid]


def fmt(x, w=6, p=3):
    return f"{'--':>{w}}" if x is None else f"{x:>{w}.{p}f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detail", type=Path, required=True, help="from 26_bhs_protocol.py --detail-csv")
    ap.add_argument("--reference", type=Path, default=None, help="to state the catalogue on every table")
    ap.add_argument("--threshold", type=float, default=0.25, help="the holdout the detail file is from")
    ap.add_argument("--low-f", type=float, default=0.70,
                    help="what counts as a diluted construct for the confusion table; the Step 1 "
                         "stratification put the whole-to-oracle gain below this")
    ap.add_argument("--baseline", default="rank_whole")
    ap.add_argument("--target", default="rank_per_segment")
    args = ap.parse_args(argv)

    m = _strat_module()
    rows = []
    with open(args.detail, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["f"] == "":
                continue
            r["_f"] = float(r["f"])
            r["_cov"] = float(r["best_hit_cov"]) if r["best_hit_cov"] != "" else None
            rows.append(r)

    cat = "catalogue not stated"
    if args.reference:
        from pilot.homology import MAX_LEN, MIN_LEN  # noqa: PLC0415
        from pilot.reference import iter_fasta  # noqa: PLC0415
        refs = [e.seq for e in iter_fasta(args.reference)]
        res = sum(len(s) for s in refs if MIN_LEN <= len(s) <= MAX_LEN)
        cat = (f"{args.reference}: {len(refs):,} proteins, {res:,} residues in the "
               f"{MIN_LEN}-{MAX_LEN} aa window")
    print(f"CATALOGUE: {cat}")
    print(f"detail from {args.detail} (nothing re-scored); holdout {args.threshold:.0%}; "
          f"{len(rows)} constructs, {len({r['payload_accession'] for r in rows})} distinct payloads\n")

    acc = {r["plasmid_id"]: r["payload_accession"] for r in rows}
    ranks = {col: {r["plasmid_id"]: int(r[col]) for r in rows if r.get(col, "") != ""}
             for col, _, _ in PROTOCOLS}
    present = [(col, lbl, free) for col, lbl, free in PROTOCOLS if ranks[col]]

    def table(title: str, ids: list[str]) -> None:
        if not ids:
            return
        print(f"{title}   n={len(ids)}, payloads={len({acc[p] for p in ids})}")
        print(f"    {'protocol':<22}{'free':<6}{'top-1':>9}{'macro':>9}{'mean rank':>11}"
              f"{'macro':>9}{'d t1':>7}{'won':>5}{'lost':>6}{'d mean':>9}")
        base = m.micro_macro(ranks[args.baseline], ids, acc)
        for col, lbl, free in present:
            c = m.micro_macro(ranks[col], ids, acc)
            if not c["n"]:
                continue
            pr = m.paired(ranks[args.baseline], ranks[col], ids)
            print(f"    {lbl:<22}{free:<6}{c['top1']:>5}/{c['n']:<3}{c['top1_macro']:>8.1f}%"
                  f"{c['mean']:>11.2f}{c['mean_macro']:>9.2f}"
                  f"{c['top1'] - base['top1']:>+7}{len(pr['won']):>5}{len(pr['lost']):>6}"
                  f"{pr['d_mean']:>+9.2f}")
        print()

    all_ids = sorted(r["plasmid_id"] for r in rows)
    table("POOLED: every paired construct", all_ids)

    fof = {r["plasmid_id"]: r["_f"] for r in rows}
    print("STRATUM SIZES  " + "  ".join(
        f"{name}: {sum(1 for p in all_ids if m.stratum_of(fof[p]) == name)}" for name, _, _ in m.STRATA)
        + f"  (total {len(all_ids)})\n")
    for name, _, _ in m.STRATA:
        ids = [p for p in all_ids if m.stratum_of(fof[p]) == name]
        warn = "  <: too small to carry a conclusion on its own" if 0 < len(ids) < 12 else ""
        table(f"STRATUM {name}{warn}", ids)

    # ---------------- the new result: a label-free signal for a labelled quantity ----------------
    usable = [r for r in rows if r["_cov"] is not None]
    print("=" * 100)
    print(f"DOES THE SEGMENTER'S OWN BEST-HIT COVERAGE PREDICT f?   "
          f"{len(usable)} of {len(rows)} constructs have a best hit")
    print("  best_hit_cov is label-free: the fraction of the ORF claimed by the top-scoring catalogue")
    print("  hit. f is labelled. If cov tracks f, cov is a triage signal a screener can actually use.\n")
    cov = [r["_cov"] for r in usable]
    f = [r["_f"] for r in usable]
    print(f"  Pearson  r = {fmt(pearson(cov, f))}        Spearman rho = {fmt(spearman(cov, f))}"
          f"   (micro, over {len(usable)} constructs)")
    per: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in usable:
        per[r["payload_accession"]].append((r["_cov"], r["_f"]))
    pc = [sum(a for a, _ in v) / len(v) for v in per.values()]
    pf = [sum(b for _, b in v) / len(v) for v in per.values()]
    print(f"  Pearson  r = {fmt(pearson(pc, pf))}        Spearman rho = {fmt(spearman(pc, pf))}"
          f"   (macro, over {len(per)} payloads, each once)\n")
    print(f"  cov by stratum of f:")
    print(f"        {'stratum':<12}{'n':>4}{'mean cov':>10}{'median cov':>12}{'min':>8}{'max':>8}")
    for name, _, _ in m.STRATA:
        g = [r["_cov"] for r in usable if m.stratum_of(r["_f"]) == name]
        if not g:
            continue
        sg = sorted(g)
        print(f"        {name:<12}{len(g):>4}{sum(g) / len(g):>10.3f}"
              f"{sg[len(sg) // 2]:>12.3f}{sg[0]:>8.3f}{sg[-1]:>8.3f}")

    grid = [round(0.30 + 0.05 * i, 2) for i in range(14)]
    for truth_name, truth in (
        (f"f < {args.low_f:g}  (a diluted construct)",
         [r["_f"] < args.low_f for r in usable]),
        (f"gains from decomposition  ({args.target} better than {args.baseline})",
         [(int(r[args.target]) < int(r[args.baseline])) if r.get(args.target, "") != ""
          and r.get(args.baseline, "") != "" else False for r in usable]),
    ):
        pos = sum(truth)
        print(f"\n  SWEEP: predict \"{truth_name}\" by cov < t     positives {pos} of {len(usable)}")
        if pos == 0:
            print("    no positive cases: nothing to predict, and no threshold can be chosen")
            continue
        sw = sweep(cov, truth, grid)
        print(f"        {'t':>6}{'tp':>5}{'fp':>5}{'fn':>5}{'tn':>5}{'sens':>8}{'spec':>8}"
              f"{'ppv':>8}{'npv':>8}{'J':>8}")
        for t, c in sw:
            print(f"        {t:>6.2f}{c['tp']:>5}{c['fp']:>5}{c['fn']:>5}{c['tn']:>5}"
                  f"{fmt(c['sens'], 8)}{fmt(c['spec'], 8)}{fmt(c['ppv'], 8)}{fmt(c['npv'], 8)}"
                  f"{fmt(c['j'], 8)}")
        best_j = max(c["j"] for _, c in sw if c["j"] is not None)
        t_star = max(t for t, c in sw if c["j"] is not None and c["j"] == best_j)
        plateau = [t for t, c in sw if c["j"] is not None and c["j"] >= best_j - 0.05]
        c = dict(sw)[t_star]
        print(f"    operating point t = {t_star:.2f} (max Youden's J = {best_j:.3f}, ties to the larger "
              f"t); J within 0.05 of the max over t in [{min(plateau):.2f}, {max(plateau):.2f}]")
        print(f"    confusion at t = {t_star:.2f}:  TP {c['tp']}  FP {c['fp']}  FN {c['fn']}    "
              f"TN {c['tn']}   sens {fmt(c['sens'], 5)}  spec {fmt(c['spec'], 5)}    "
              f"ppv {fmt(c['ppv'], 5)}  npv {fmt(c['npv'], 5)}")
        print("    IN-SAMPLE: t was chosen on these same constructs, so this bounds the signal rather "
              "than establishing it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
