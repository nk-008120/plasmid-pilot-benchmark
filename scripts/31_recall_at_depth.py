#!/usr/bin/env python3
"""Recall at review depth, and the f stratification carried to the hand and synthetic tiers. No scoring.

    python scripts/31_recall_at_depth.py --tier corpus
    python scripts/31_recall_at_depth.py --tier hand --hand-per-construct
    python scripts/31_recall_at_depth.py --tier synthetic
    python scripts/31_recall_at_depth.py --cross-tier

**Why depth and not top-1.** `29_stratify_by_payload_fraction.py` reports top-1 only, and top-1 is blind
in both directions at once. `ncbi_AY254174_1` moves 76 -> 2 and `ncbi_AF114752_1` moves 79 -> 7 under the
oracle at the 25% holdout: two of the largest rank movements in the corpus, and top-1 scores both as
zero. `ncbi_AF455027_1` moves 3 -> 1 on a 4-residue trim (f=0.978, so there is nothing to decompose) and
top-1 scores it +1. A metric that records the artefact and not the mechanism is the wrong metric. This
script computes recall@k for k in {1,5,10,25} so the depth at which the two classes separate can be read
off rather than argued.

**`f` is the same quantity in all three tiers, from three sources.** `f = oracle_len / orf_len`: the
fraction of the called ORF that the payload span occupies. For the corpus and hand tiers it is read from
`27_frame_orf_lengths.py` output, which measured it. For the synthetic tier it is a *construction
parameter*: `payload_aa / orf_aa` from the manifest, exact by construction and equal to
`1 - dilution_target`. The definition is shared; the provenance is not, and every table says which.

**Micro and macro, always both**. Micro averages over constructs, macro over
distinct payloads within the stratum. The corpus's 410 constructs carry 138 payloads with one appearing
in 37, so micro alone is a payload-weighted number wearing a construct's label.

Writes nothing. Re-scores nothing: every rank is read from a finished twilight CSV.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

STRATA: tuple[tuple[str, float, float], ...] = (
    ("f < 0.50", 0.0, 0.50),
    ("0.50-0.70", 0.50, 0.70),
    ("0.70-0.95", 0.70, 0.95),
    ("f >= 0.95", 0.95, 1.0000001),
)
DEPTHS = (1, 5, 10, 25)
ARMS = ("whole", "auto", "oracle")

#: One row per tier: the finished twilight CSV, the f source, and how to read it.
TIERS: dict[str, dict] = {
    "hand": {
        "twilight": "results/gate173_control_hand/twilight_sw-blosum62-11-1_hand.csv",
        "f_source": "results/bhs_corpus_probe/hand_orf_lengths.csv",
        "f_kind": "lengths",
        "provenance": "measured (27_frame_orf_lengths.py); payload span is the depositor's stated insert",
    },
    "corpus": {
        "twilight": "results/gate173_control_corpus/twilight_sw-blosum62-11-1_corpus.csv",
        "f_source": "results/bhs_corpus_probe/frame_orf_lengths.csv",
        "f_kind": "lengths",
        "provenance": "measured (27_frame_orf_lengths.py); payload span from gated alignment autolabel",
    },
    "synthetic": {
        "twilight": "results/gate173_control_synthetic/twilight_sw-blosum62-11-1_synthetic.csv",
        "f_source": "data/synthetic/synthetic_manifest.csv",
        "f_kind": "synthetic_manifest",
        "provenance": "exact by construction (payload_aa / orf_aa = 1 - dilution_target); a PARAMETER",
    },
}
REFERENCE = "results/gate_all173/reference_twilight.fasta"


def stratum_of(f: float) -> str:
    for name, lo, hi in STRATA:
        if lo <= f < hi:
            return name
    raise ValueError(f"f={f!r} falls in no stratum; oracle_len > orf_len should be impossible")


def read_f(path: Path, kind: str) -> dict[str, tuple[float, int, int]]:
    """`construct_id -> (f, orf_len, oracle_len)`, from either f source."""
    out: dict[str, tuple[float, int, int]] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if kind == "lengths":
                if r["oracle_len"] == "":
                    continue
                orf, osp, key = int(r["orf_len"]), int(r["oracle_len"]), r["plasmid_id"]
            else:
                orf, osp, key = int(r["orf_aa"]), int(r["payload_aa"]), r["construct_id"]
            out[key] = (osp / orf, orf, osp)
    return out


def read_ranks(path: Path) -> tuple[dict[float, dict[str, dict[str, int]]], dict[str, str]]:
    """`threshold -> arm -> construct -> rank`, plus `construct -> payload`."""
    ranks: dict[float, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    acc: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            acc[r["plasmid_id"]] = r["payload_accession"]
            if r["best_target_rank"] == "":
                continue
            ranks[float(r["threshold"])][r["arm"]][r["plasmid_id"]] = int(r["best_target_rank"])
    return ranks, acc


def catalogue_line(path: Path) -> str:
    from pilot.homology import MAX_LEN, MIN_LEN  # noqa: PLC0415
    from pilot.reference import iter_fasta  # noqa: PLC0415
    refs = [e.seq for e in iter_fasta(path)]
    res = sum(len(s) for s in refs if MIN_LEN <= len(s) <= MAX_LEN)
    return f"{path}: {len(refs):,} proteins, {res:,} residues in the {MIN_LEN}-{MAX_LEN} aa window"


def recall(ranks: dict[str, int], ids: list[str], acc: dict[str, str],
           k: int) -> tuple[float, float, int, int, int]:
    """(micro %, macro %, hits, n constructs, n payloads) for rank <= k."""
    v = [ranks[p] for p in ids if p in ranks]
    if not v:
        return (float("nan"), float("nan"), 0, 0, 0)
    per: dict[str, list[int]] = defaultdict(list)
    for p in ids:
        if p in ranks:
            per[acc[p]].append(ranks[p])
    hits = sum(1 for x in v if x <= k)
    macro = 100 * sum(sum(1 for x in g if x <= k) / len(g) for g in per.values()) / len(per)
    return (100 * hits / len(v), macro, hits, len(v), len(per))


def crossings(a: dict[str, int], b: dict[str, int], ids: list[str], k: int) -> dict:
    """Constructs that cross depth `k` from arm `a` to arm `b`. Paired: only ids scorable under both."""
    both = [p for p in ids if p in a and p in b]
    won = [p for p in both if a[p] > k and b[p] <= k]
    lost = [p for p in both if a[p] <= k and b[p] > k]
    return {"n": len(both), "won": won, "lost": lost, "net": len(won) - len(lost)}


def depth_table(title: str, ids_for: dict[float, list[str]], ranks, acc, thresholds) -> None:
    print(title)
    hdr = f"{'holdout':>8}{'n':>5}{'pyld':>6}"
    for a in ARMS:
        hdr += "".join(f"{a[:3] + '@' + str(k):>9}" for k in DEPTHS)
    hdr += "".join(f"{'d@' + str(k):>6}" for k in DEPTHS)
    print(hdr)
    for thr in thresholds:
        ids = ids_for.get(thr, [])
        if not ids:
            continue
        base = recall(ranks[thr].get("whole", {}), ids, acc, 1)
        if not base[3]:
            continue
        print(f"{thr:>8.0%}{base[3]:>5}{base[4]:>6}"
              + "".join(f"{recall(ranks[thr].get(a, {}), ids, acc, k)[0]:>8.1f}%"
                        for a in ARMS for k in DEPTHS)
              + "".join(f"{crossings(ranks[thr].get('whole', {}), ranks[thr].get('oracle', {}), ids, k)['net']:>+6}"
                        for k in DEPTHS))
        print(f"{'macro':>8}{'':>5}{'':>6}"
              + "".join(f"{recall(ranks[thr].get(a, {}), ids, acc, k)[1]:>8.1f}%"
                        for a in ARMS for k in DEPTHS))
    print()


def f_distribution(frac: dict[str, tuple[float, int, int]], acc: dict[str, str]) -> None:
    print("f DISTRIBUTION over constructs that have an f")
    c = Counter(stratum_of(f) for f, _, _ in frac.values())
    tot = sum(c.values())
    pay = {nm: {acc.get(p, f"(not in twilight) {p}") for p in frac
                if stratum_of(frac[p][0]) == nm} for nm, _, _ in STRATA}
    print(f"{'stratum':>12}{'constructs':>12}{'share':>9}{'payloads':>10}")
    for nm, _, _ in STRATA:
        print(f"{nm:>12}{c.get(nm, 0):>12}{(100 * c.get(nm, 0) / tot):>8.1f}%{len(pay[nm]):>10}")
    print(f"{'total':>12}{tot:>12}{100.0:>8.1f}%{len({acc.get(p, p) for p in frac}):>10}")
    fvals = sorted(f for f, _, _ in frac.values())

    def q(p: float) -> float:
        return fvals[min(len(fvals) - 1, int(p * (len(fvals) - 1)))]

    print(f"  f: min {fvals[0]:.3f}  p25 {q(.25):.3f}  median {q(.5):.3f}  p75 {q(.75):.3f}    "
          f"max {fvals[-1]:.3f}   f<0.70: {sum(1 for x in fvals if x < 0.70)}/{len(fvals)}\n")


def run_tier(tier: str, transitions_at: float) -> None:
    cfg = TIERS[tier]
    tw, fsrc = Path(cfg["twilight"]), Path(cfg["f_source"])
    ranks, acc = read_ranks(tw)
    frac = read_f(fsrc, cfg["f_kind"])
    print("=" * 118)
    print(f"TIER {tier}")
    print(f"CATALOGUE: {catalogue_line(Path(REFERENCE))}")
    print(f"ranks: {tw} (finished run, nothing re-scored)")
    print(f"f:     {fsrc}, {cfg['provenance']}; {len(frac)} constructs have an f")
    print("=" * 118 + "\n")

    thresholds = sorted(ranks, reverse=True)
    all_ids = {thr: sorted(set(ranks[thr].get("whole", {}))) for thr in thresholds}

    f_distribution(frac, acc)
    depth_table("POOLED: every scorable construct, recall@k (d@k = paired whole->oracle net crossings)",
                all_ids, ranks, acc, thresholds)

    missing = {f"{t:.0%}": len([p for p in all_ids[t] if p not in frac]) for t in thresholds}
    if any(missing.values()):
        print(f"NOTE scorable constructs with no f (excluded from every stratum table): "
              f"{ {k: v for k, v in missing.items() if v} }\n")

    print("STRATUM SIZES at each holdout (scorable AND with an f)")
    print(f"{'holdout':>8}" + "".join(f"{nm:>12}" for nm, _, _ in STRATA) + f"{'total':>8}")
    for thr in thresholds:
        ids = [p for p in all_ids[thr] if p in frac]
        cc = {nm: sum(1 for p in ids if stratum_of(frac[p][0]) == nm) for nm, _, _ in STRATA}
        print(f"{thr:>8.0%}" + "".join(f"{cc[nm]:>12}" for nm, _, _ in STRATA) + f"{len(ids):>8}")
    print()

    for nm, _, _ in STRATA:
        ids_for = {thr: [p for p in all_ids[thr] if p in frac and stratum_of(frac[p][0]) == nm]
                   for thr in thresholds}
        n_any = max((len(v) for v in ids_for.values()), default=0)
        if not n_any:
            print(f"STRATUM {nm}: no scorable construct in this tier\n")
            continue
        warn = "  <: too small to carry a conclusion alone" if n_any < 12 else ""
        depth_table(f"STRATUM {nm}  (max n {n_any}){warn}", ids_for, ranks, acc, thresholds)

    thr = transitions_at
    if thr in ranks:
        a, b = ranks[thr].get("whole", {}), ranks[thr].get("oracle", {})
        ids = [p for p in all_ids[thr] if p in frac]
        print(f"EVERY whole->oracle CROSSING AT THE {thr:.0%} HOLDOUT, depth by depth")
        for k in DEPTHS:
            tr = crossings(a, b, ids, k)
            print(f"  depth {k:<3} net {tr['net']:+d} = {len(tr['won'])} won - {len(tr['lost'])} lost "
                  f"(paired n={tr['n']}; {len({acc[p] for p in tr['won']})} distinct payloads among winners)")
            for label, grp in (("won", tr["won"]), ("lost", tr["lost"])):
                for p in sorted(grp, key=lambda x: frac[x][0]):
                    f_, orf, osp = frac[p]
                    print(f"            {label:<5}{stratum_of(f_):<10}{p:<46}{acc[p]:<9}f={f_:.3f}    "
                          f"orf={orf:<5}whole={a[p]:<5}oracle={b[p]}")
        print()


def _spearman_pearson(x: list[float], y: list[float]) -> tuple[float, float]:
    def rank_of(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for t in range(i, j + 1):
                rk[order[t]] = avg
            i = j + 1
        return rk

    def pearson(a: list[float], b: list[float]) -> float:
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        num = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b))
        da = sum((ai - ma) ** 2 for ai in a) ** 0.5
        db = sum((bi - mb) ** 2 for bi in b) ** 0.5
        return num / (da * db) if da and db else float("nan")

    return pearson(x, y), pearson(rank_of(x), rank_of(y))


def hand_per_construct(transitions_at: float) -> None:
    """Part 2: f per construct beside its whole and oracle rank, and the f-vs-gain correlation."""
    cfg = TIERS["hand"]
    ranks, acc = read_ranks(Path(cfg["twilight"]))
    frac = read_f(Path(cfg["f_source"]), "lengths")
    with open("data/manifest.csv", newline="", encoding="utf-8-sig") as fh:
        manifest = list(csv.DictReader(fh))
    thr = transitions_at
    a, b = ranks[thr].get("whole", {}), ranks[thr].get("oracle", {})

    print("=" * 118)
    print(f"HAND TIER, ALL 23 AUDITED CONSTRUCTS, at the {thr:.0%} holdout")
    print(f"CATALOGUE: {catalogue_line(Path(REFERENCE))}")
    print(f"ranks: {cfg['twilight']}   f: {cfg['f_source']} ({cfg['provenance']})")
    print("=" * 118)
    print(f"{'construct':<46}{'payload':<9}{'archetype':<22}{'f':>7}{'orf':>6}{'orcl':>6}"
          f"{'whole':>7}{'oracle':>7}{'gain':>7}")
    rows: list[tuple[float, int, int, int, str, str]] = []
    for m in manifest:
        p = m["filename"][:-4]
        pay = m["payload_accession"] or "(none)"
        f_ = frac.get(p)
        wr, orr = a.get(p), b.get(p)
        gain = (wr - orr) if (wr and orr) else None
        print(f"{p:<46}{pay:<9}{m['archetype_manual']:<22}"
              f"{(f'{f_[0]:.3f}' if f_ else '-'):>7}{(f_[1] if f_ else '-'):>6}{(f_[2] if f_ else '-'):>6}"
              f"{(wr if wr else '-'):>7}{(orr if orr else '-'):>7}"
              f"{(f'{gain:+d}' if gain is not None else '-'):>7}")
        if f_ and wr and orr:
            rows.append((f_[0], wr, orr, wr - orr, p, pay))
    no_payload = sum(1 for m in manifest if not m["payload_accession"])
    print(f"\nscorable at {thr:.0%} with an f: {len(rows)} of 23 "
          f"({no_payload} carry no payload at all, so they have neither f nor rank; "
          f"{23 - len(rows) - no_payload} payload-bearing but unscorable at this holdout)")

    f_v = [r[0] for r in rows]
    print()
    for label, gain_v in (("raw rank gain (whole - oracle)", [float(r[3]) for r in rows]),
                          ("log10 rank ratio log10(whole/oracle)",
                           [math.log10(r[1] / r[2]) for r in rows])):
        pr, sp = _spearman_pearson(f_v, gain_v)
        print(f"  f vs {label:<40} n={len(rows)}  Pearson r={pr:+.3f}  Spearman rho={sp:+.3f}")

    print("\n  per-construct, sorted by f:")
    print(f"        {'f':>7}{'whole':>7}{'oracle':>7}{'gain':>7}  construct")
    for f_, wr, orr, g, p, pay in sorted(rows):
        print(f"        {f_:>7.3f}{wr:>7}{orr:>7}{g:>+7}    {p} ({pay})")

    print()
    for nm, lo, hi in STRATA:
        g = [r for r in rows if lo <= r[0] < hi]
        if not g:
            print(f"    {nm:<10} n=0")
            continue
        print(f"    {nm:<10} n={len(g):<3} payloads={len({r[5] for r in g}):<3}  "
              f"mean whole {sum(r[1] for r in g) / len(g):>7.2f}    "
              f"mean oracle {sum(r[2] for r in g) / len(g):>7.2f}    "
              f"mean gain {sum(r[3] for r in g) / len(g):>+7.2f}    "
              f"median gain {sorted(r[3] for r in g)[len(g) // 2]:>+4d}")
    for k in DEPTHS:
        tr = crossings(a, b, [r[4] for r in rows], k)
        print(f"  recall@{k:<3} whole {sum(1 for r in rows if r[1] <= k)}/{len(rows)}    "
              f"oracle {sum(1 for r in rows if r[2] <= k)}/{len(rows)}    "
              f"net {tr['net']:+d} ({len(tr['won'])} won, {len(tr['lost'])} lost)")
    print()


def cross_tier() -> None:
    """Part 4: the f<0.70 count, one definition, every tier, micro and macro."""
    print("=" * 118)
    print("THE PROJECT'S EFFECTIVE SAMPLE FOR ITS CENTRAL CLAIM: constructs with f < 0.70")
    print(f"CATALOGUE (all three tiers retrieve against the same one): {catalogue_line(Path(REFERENCE))}")
    print("f = oracle_len / orf_len, one definition. Provenance differs by tier and is stated per row.")
    print("=" * 118)
    print(f"{'tier':<11}{'f prov':<11}{'with f':>8}{'f<0.70':>8}{'pyld':>6}"
          f"{'f<0.50':>8}{'pyld':>6}{'scor@25':>9}{'f<.70@25':>10}{'pyld':>6}")
    totals: Counter = Counter()
    detail: dict[str, tuple[dict, list[str], set[str]]] = {}
    for tier, cfg in TIERS.items():
        ranks, acc = read_ranks(Path(cfg["twilight"]))
        frac = read_f(Path(cfg["f_source"]), cfg["f_kind"])
        low = sorted(p for p in frac if frac[p][0] < 0.70)
        vlow = [p for p in frac if frac[p][0] < 0.50]
        s25 = set(ranks.get(0.25, {}).get("whole", {}))
        low25 = [p for p in low if p in s25]
        detail[tier] = (frac, low, set(low25))
        print(f"{tier:<11}{('measured' if cfg['f_kind'] == 'lengths' else 'parameter'):<11}"
              f"{len(frac):>8}{len(low):>8}{len({acc.get(p) for p in low}):>6}"
              f"{len(vlow):>8}{len({acc.get(p) for p in vlow}):>6}"
              f"{len(s25):>9}{len(low25):>10}{len({acc.get(p) for p in low25}):>6}")
        totals["with_f"] += len(frac)
        totals["low"] += len(low)
        totals["low25"] += len(low25)
        totals["pay_low"] += len({acc.get(p) for p in low})
        totals["pay_low25"] += len({acc.get(p) for p in low25})
    print(f"{'ALL THREE':<11}{'':<11}{totals['with_f']:>8}{totals['low']:>8}{totals['pay_low']:>6}"
          f"{'':>8}{'':>6}{'':>9}{totals['low25']:>10}{totals['pay_low25']:>6}")
    print("\n  MICRO is the construct count; MACRO is the payload count beside it: the number of")
    print("  independent cells a macro average over the f<0.70 stratum actually has.")
    print("  Payload counts are summed across tiers and are NOT disjoint between them (the hand and")
    print("  corpus tiers share accessions); within a tier they are distinct.\n")

    for tier, (frac, low, low25) in detail.items():
        print(f"    {tier}: {len(low)} constructs with f<0.70, {len(low25)} of them scorable at 25%")
        if tier == "synthetic":
            byf = Counter(round(frac[p][0], 2) for p in low)
            print(f"    by construction parameter: "
                  + ", ".join(f"f={k:.2f} n={v}" for k, v in sorted(byf.items())))
        else:
            for p in low:
                mark = "" if p in low25 else "   [unscorable at the 25% holdout]"
                print(f"        {p:<46}f={frac[p][0]:.3f}  orf={frac[p][1]:<5}oracle={frac[p][2]}{mark}")
        print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=sorted(TIERS))
    ap.add_argument("--hand-per-construct", action="store_true")
    ap.add_argument("--cross-tier", action="store_true")
    ap.add_argument("--transitions-at", type=float, default=0.25)
    args = ap.parse_args(argv)
    if args.tier:
        run_tier(args.tier, args.transitions_at)
    if args.hand_per_construct:
        hand_per_construct(args.transitions_at)
    if args.cross_tier:
        cross_tier()
    if not (args.tier or args.hand_per_construct or args.cross_tier):
        ap.error("pick --tier, --hand-per-construct or --cross-tier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
