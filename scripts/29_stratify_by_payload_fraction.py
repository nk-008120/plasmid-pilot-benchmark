#!/usr/bin/env python3
"""Does the whole-to-oracle gain live where the ORF is not the payload? A stratification, no re-scoring.

    python scripts/29_stratify_by_payload_fraction.py \
        --twilight results/gate173_control_corpus/twilight_sw-blosum62-11-1_corpus.csv \
        --lengths results/bhs_corpus_probe/frame_orf_lengths.csv \
        --reference results/gate_all173/reference_twilight.fasta

**The quantity.** `f = oracle_len / orf_len`: the fraction of the called ORF that the payload span
actually occupies. `f = 1.0` means the ORF *is* the payload and there is nothing to decompose; a low `f`
means the payload is a minority of a longer ORF, which is the regime the project's central claim is
about. `f` uses the label (the oracle span is built from the payload), so it is a way of *describing*
constructs, never a method.

**Why a stratification and not a correlation.** The corpus is overwhelmingly `f = 1.0`: 247 of 286
constructs at `f >= 0.95` and 180 at exactly 1.000. A correlation over that distribution is dominated by
a single point mass and would report "no relationship" whatever the low-`f` constructs did. The strata
are declared before the numbers are looked at, and the two low ones are small (5 and 10), which is stated
in the output rather than left for a reader to discover: a stratum of 5 cannot distinguish a real
effect from two lucky constructs, and the per-construct transition table is printed for exactly that
reason.

**Micro and macro, always both.** 286 corpus constructs carry 84 distinct payloads and one appears in 37
of them, so a micro average over constructs counts that payload 37 times. The macro
average weights each payload once within each stratum. Neither is the right answer alone and reporting
one without the other has already misled this project once.

**The paired delta is printed as transitions, not as a difference of two counts.** `whole_top1 214` and
`oracle_top1 226` is a net `+12`, which is consistent with 12 constructs gaining and none losing, or with
20 gaining and 8 losing. Those are different findings (the working record is the precedent: a drop that looked
like one changed rank was five removed constructs). So every table carries `won` and `lost` as well as
the net.

Writes nothing unless `--out` is given. Re-scores nothing: every rank is read from a finished run.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

#: Declared in advance. Lower bound inclusive, upper bound exclusive, so every f lands in exactly one.
STRATA: tuple[tuple[str, float, float], ...] = (
    ("f < 0.50", 0.0, 0.50),
    ("0.50-0.70", 0.50, 0.70),
    ("0.70-0.95", 0.70, 0.95),
    ("f >= 0.95", 0.95, 1.0000001),
)


def stratum_of(f: float) -> str:
    """The one stratum `f` belongs to. An f outside every band is a bug, not a category."""
    for name, lo, hi in STRATA:
        if lo <= f < hi:
            return name
    raise ValueError(f"f={f!r} falls in no stratum; oracle_len > orf_len should be impossible")


def payload_fractions(lengths_csv: Path) -> dict[str, tuple[float, int, int]]:
    """`plasmid_id -> (f, orf_len, oracle_len)` from `27_frame_orf_lengths.py`'s output."""
    out = {}
    with open(lengths_csv, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["oracle_len"] == "":
                continue
            orf, osp = int(r["orf_len"]), int(r["oracle_len"])
            out[r["plasmid_id"]] = (osp / orf, orf, osp)
    return out


def read_ranks(twilight_csv: Path) -> dict[float, dict[str, dict[str, int]]]:
    """`threshold -> arm -> plasmid_id -> rank`, skipping unscorable cells."""
    out: dict[float, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    with open(twilight_csv, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["best_target_rank"] == "":
                continue
            out[float(r["threshold"])][r["arm"]][r["plasmid_id"]] = int(r["best_target_rank"])
    return out


def payload_of(twilight_csv: Path) -> dict[str, str]:
    with open(twilight_csv, newline="", encoding="utf-8-sig") as fh:
        return {r["plasmid_id"]: r["payload_accession"] for r in csv.DictReader(fh)}


def micro_macro(ranks: dict[str, int], ids: list[str], acc: dict[str, str]) -> dict:
    """top-1 and mean rank, counted once per construct (micro) and once per payload (macro)."""
    v = [ranks[p] for p in ids if p in ranks]
    if not v:
        return {"n": 0, "n_payloads": 0, "top1": 0, "top1_micro": None, "mean": None,
                "top1_macro": None, "mean_macro": None}
    per: dict[str, list[int]] = defaultdict(list)
    for p in ids:
        if p in ranks:
            per[acc[p]].append(ranks[p])
    return {
        "n": len(v), "n_payloads": len(per), "top1": sum(1 for x in v if x == 1),
        "top1_micro": 100 * sum(1 for x in v if x == 1) / len(v),
        "mean": sum(v) / len(v),
        "top1_macro": 100 * sum(sum(1 for x in g if x == 1) / len(g) for g in per.values()) / len(per),
        "mean_macro": sum(sum(g) / len(g) for g in per.values()) / len(per),
    }


def paired(a: dict[str, int], b: dict[str, int], ids: list[str]) -> dict:
    """Top-1 transitions from arm `a` to arm `b`, on the constructs scorable under both."""
    both = [p for p in ids if p in a and p in b]
    won = [p for p in both if a[p] != 1 and b[p] == 1]
    lost = [p for p in both if a[p] == 1 and b[p] != 1]
    moved = [p for p in both if a[p] != b[p]]
    return {"n_paired": len(both), "won": won, "lost": lost, "net": len(won) - len(lost),
            "n_rank_changed": len(moved),
            "d_mean": ((sum(b[p] for p in both) - sum(a[p] for p in both)) / len(both)) if both else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twilight", type=Path, required=True, help="a FINISHED twilight CSV; not re-scored")
    ap.add_argument("--lengths", type=Path, required=True, help="from 27_frame_orf_lengths.py")
    ap.add_argument("--reference", type=Path, default=None,
                    help="only to state the catalogue and its residue total on every table")
    ap.add_argument("--arms", default="whole,auto,oracle")
    ap.add_argument("--baseline", default="whole")
    ap.add_argument("--target", default="oracle")
    ap.add_argument("--pooled-only", action="store_true",
                    help="reproduce the run's own sweep table and stop; this is the validation path")
    ap.add_argument("--transitions-at", type=float, default=0.25,
                    help="holdout whose per-construct transitions are listed by stratum")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    arms = [a for a in args.arms.split(",") if a]
    frac = payload_fractions(args.lengths)
    ranks = read_ranks(args.twilight)
    acc = payload_of(args.twilight)

    cat = "catalogue not stated"
    if args.reference:
        from pilot.homology import MAX_LEN, MIN_LEN  # noqa: PLC0415
        from pilot.reference import iter_fasta  # noqa: PLC0415
        refs = [e.seq for e in iter_fasta(args.reference)]
        res = sum(len(s) for s in refs if MIN_LEN <= len(s) <= MAX_LEN)
        cat = (f"{args.reference}: {len(refs):,} proteins, {res:,} residues in the "
               f"{MIN_LEN}-{MAX_LEN} aa window")
    print(f"CATALOGUE: {cat}")
    print(f"ranks read from {args.twilight} (nothing re-scored); "
          f"f from {args.lengths} for {len(frac)} constructs\n")

    thresholds = sorted(ranks, reverse=True)
    rows_out = []

    def table(title: str, ids_for: dict[float, list[str]]) -> None:
        print(title)
        print(f"{'holdout':>8}{'n':>5}{'payld':>7}"
              + "".join(f"{a + ' t1':>11}{a + ' mean':>12}" for a in arms)
              + f"{'d t1':>7}{'won':>5}{'lost':>6}{'d mean':>9}")
        for thr in thresholds:
            ids = ids_for.get(thr, [])
            if not ids:
                continue
            cells = {a: micro_macro(ranks[thr].get(a, {}), ids, acc) for a in arms}
            base = cells[args.baseline]
            if not base["n"]:
                continue
            pr = paired(ranks[thr].get(args.baseline, {}), ranks[thr].get(args.target, {}), ids)
            print(f"{thr:>8.0%}{base['n']:>5}{base['n_payloads']:>7}"
                  + "".join(f"{cells[a]['top1']:>7}/{cells[a]['n']:<3}{cells[a]['mean']:>12.2f}"
                            for a in arms)
                  + f"{pr['net']:>+7}{len(pr['won']):>5}{len(pr['lost']):>6}{pr['d_mean']:>+9.2f}")
            print(f"{'macro':>8}{'':>5}{'':>7}"
                  + "".join(f"{cells[a]['top1_macro']:>10.1f}%{cells[a]['mean_macro']:>12.2f}"
                            for a in arms))
            rows_out.append(dict(stratum=title, threshold=thr, n=base["n"],
                                 n_payloads=base["n_payloads"],
                                 **{f"{a}_{k}": cells[a][k] for a in arms
                                    for k in ("top1", "top1_micro", "top1_macro", "mean", "mean_macro")},
                                 net=pr["net"], won=len(pr["won"]), lost=len(pr["lost"]),
                                 n_rank_changed=pr["n_rank_changed"], d_mean=pr["d_mean"]))
        print()

    all_ids = {thr: sorted(set(ranks[thr].get(args.baseline, {}))) for thr in thresholds}
    table(f"POOLED: every scorable construct ({args.baseline} -> {args.target})", all_ids)
    if args.pooled_only:
        return 0

    missing = {thr: [p for p in all_ids[thr] if p not in frac] for thr in thresholds}
    if any(missing.values()):
        print(f"WARNING: constructs with no f at some holdout (excluded from the strata): "
              f"{ {f'{t:.0%}': len(v) for t, v in missing.items() if v} }\n")

    print("STRATUM SIZES at each holdout (constructs with an f)")
    print(f"{'holdout':>8}" + "".join(f"{name:>12}" for name, _, _ in STRATA) + f"{'total':>8}")
    for thr in thresholds:
        ids = [p for p in all_ids[thr] if p in frac]
        c = {name: sum(1 for p in ids if stratum_of(frac[p][0]) == name) for name, _, _ in STRATA}
        print(f"{thr:>8.0%}" + "".join(f"{c[name]:>12}" for name, _, _ in STRATA) + f"{len(ids):>8}")
    print()

    for name, _, _ in STRATA:
        ids_for = {thr: [p for p in all_ids[thr] if p in frac and stratum_of(frac[p][0]) == name]
                   for thr in thresholds}
        n_any = max((len(v) for v in ids_for.values()), default=0)
        warn = "  <: too small to carry a conclusion on its own" if n_any < 12 else ""
        table(f"STRATUM {name}  (max n {n_any}){warn}", ids_for)

    thr = args.transitions_at
    if thr in ranks:
        a, b = ranks[thr].get(args.baseline, {}), ranks[thr].get(args.target, {})
        ids = [p for p in all_ids[thr] if p in frac]
        pr = paired(a, b, ids)
        print(f"EVERY TOP-1 TRANSITION AT THE {thr:.0%} HOLDOUT, {args.baseline} -> {args.target}  "
              f"(paired n={pr['n_paired']})")
        print(f"  net {pr['net']:+d} = {len(pr['won'])} won - {len(pr['lost'])} lost; "
              f"{pr['n_rank_changed']} constructs changed rank at all")
        for label, group in (("WON", pr["won"]), ("LOST", pr["lost"])):
            by: dict[str, list[str]] = defaultdict(list)
            for p in group:
                by[stratum_of(frac[p][0])].append(p)
            print(f"    {label} by stratum: "
                  + ", ".join(f"{nm}  {len(by.get(nm, []))}" for nm, _, _ in STRATA))
            for nm, _, _ in STRATA:
                for p in sorted(by.get(nm, [])):
                    f_, orf, osp = frac[p]
                    print(f"        {nm:<10}{p:<22}{acc[p]:<9}f={f_:.3f}  orf={orf:<5}oracle={osp:<5}"
                          f"{args.baseline}={a[p]:<5}{args.target}={b[p]}")
        print(f"\n  payloads among the winners: "
              f"{len({acc[p] for p in pr['won']})} distinct over {len(pr['won'])} constructs")
    if args.out and rows_out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
