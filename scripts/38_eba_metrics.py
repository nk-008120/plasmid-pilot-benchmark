"""Read an `18_eba_sweep.py` run under the working record instead of off its printed table.

    python scripts/38_eba_metrics.py --sweep-csv results/eba_regime_probe/eba_sweep_blosum.csv

`18_eba_sweep.py` prints **top-1 only**. this project's working rules has said since 17 Oct that top-1 is a column and never
a headline, and the probe run of 19 Oct is the case that shows why: at the 25% holdout EBA at shift 0.42
reads 9/15 against exact alignment's 7/15 on top-1, an apparent win, while `recall@10` goes the other
way (60.0% against 65.0%) and mean rank is **42.60 against 5.67**. Reading the printed table would have
recorded a positive result for a scorer whose rank distribution had just come apart. See the working record.

What this emits, per holdout and per arm, paired across scorers on the constructs both scored:

* **`recall@10` and MRR**, the primary pair for a non-alignment scorer.
* **mean rank beside its miss count and its median**, never alone. A mean far above its median is the
  bimodality signature of 32.2 and is flagged in the output rather than left for the reader to notice.
* **top-1 as a column.**
* **the decomposition gap** (`oracle - whole`) computed *within* each scorer, which is the quantity the
  sweep exists to measure, with the per-construct improved/worsened split beside it.
* **f strata** where a payload-fraction source is given, with the count of constructs that have no `f`
  printed rather than silently dropped (report binding not settings).

It reads finished CSVs and runs no scorer, so it costs seconds and reproduces from the repository.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DEPTH = 10
STRATA = [(0.0, 0.50, "f < 0.50"), (0.50, 0.70, "0.50-0.70"),
          (0.70, 0.95, "0.70-0.95"), (0.95, 1.01, "f >= 0.95")]


def read_rank(value: str | None) -> int | None:
    """A rank, or `None` for a construct this arm did not retrieve.

    Blank is NOT zero and NOT rank 1: the same trap `37_gap_closure.py` records.
    """
    v = (value or "").strip()
    if v == "" or v.lower() in {"na", "none", "nan"}:
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on the discordant pairs only. `b + c == 0` is no evidence, p = 1."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n), 1.0)


def summarise(ranks: list[int | None]) -> dict:
    found = [x for x in ranks if x is not None]
    n = len(ranks)
    return {
        "n": n,
        "recall_at_10": round(100 * sum(1 for x in found if x <= DEPTH) / n, 1) if n else None,
        "top_1": round(100 * sum(1 for x in found if x == 1) / n, 1) if n else None,
        # A miss contributes 0 to MRR; dropping it instead would score an arm on what it did retrieve.
        "mrr": round(statistics.fmean([1 / x for x in found] + [0.0] * (n - len(found))), 3) if n else None,
        "mean_rank": round(statistics.fmean(found), 2) if found else None,
        "median_rank": statistics.median(found) if found else None,
        "not_retrieved": n - len(found),
    }


def hit(r: int | None) -> bool:
    return r is not None and r <= DEPTH


def paired(a: dict, b: dict, pids: list[str]) -> dict:
    """`b` against `a` at depth 10, within construct."""
    wins = sum(1 for p in pids if not hit(a[p]) and hit(b[p]))
    losses = sum(1 for p in pids if hit(a[p]) and not hit(b[p]))
    better = sum(1 for p in pids if a[p] and b[p] and b[p] < a[p])
    worse = sum(1 for p in pids if a[p] and b[p] and b[p] > a[p])
    return {"n": len(pids), "wins": wins, "losses": losses, "p_mcnemar": mcnemar_exact(wins, losses),
            "rank_better": better, "rank_worse": worse}


def load_f(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            pid = r.get("plasmid_id") or r.get("id")
            if pid and (r.get("f") or "").strip():
                out[pid] = float(r["f"])
    return out


def index(rows: list[dict]) -> dict[tuple[str, str, str], dict[str, int | None]]:
    idx: dict[tuple[str, str, str], dict[str, int | None]] = defaultdict(dict)
    for r in rows:
        idx[(r["threshold"], r["arm"], r["scorer"])][r["plasmid_id"]] = read_rank(r["best_target_rank"])
    return idx


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-csv", type=Path, required=True)
    ap.add_argument("--f-source", type=Path, default=Path("results/bhs_corpus_full/detail_hand_25.csv"),
                    help="a detail CSV carrying plasmid_id and f; the hand tier's by default")
    ap.add_argument("--control", default="align", help="the scorer every other one is paired against")
    args = ap.parse_args(argv)

    if not args.sweep_csv.exists():
        print(f"missing: {args.sweep_csv}", file=sys.stderr)
        return 2
    with open(args.sweep_csv, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    idx = index(rows)
    f_of = load_f(args.f_source)
    scorers = sorted({r["scorer"] for r in rows})
    others = [s for s in scorers if s != args.control]
    arms = sorted({r["arm"] for r in rows})
    all_pids = sorted({r["plasmid_id"] for r in rows})

    print(f"{args.sweep_csv}")
    print(f"  scorers: {scorers}")
    print(f"  control: {args.control}; arms: {arms}")
    # Binding count, printed rather than assumed: a construct with no f cannot enter a stratum row.
    no_f = [p for p in all_pids if p not in f_of]
    print(f"  payload fraction known for {len(all_pids) - len(no_f)} of {len(all_pids)} constructs"
          f"{'' if not no_f else f'; {len(no_f)} have no f and are absent from every stratum row'}")
    if not others:
        print("\nonly the control is present; nothing to pair against", file=sys.stderr)

    for thr in sorted({r["threshold"] for r in rows}, key=float):
        print(f"\n================ holdout {float(thr):.0%} ================")
        print(f"    {'arm':<8}{'scorer':<18}{'n':>4}{'r@10':>8}{'MRR':>7}{'top1':>7}"
              f"{'mean':>9}{'median':>8}{'miss':>6}")
        for arm in arms:
            for sc in scorers:
                d = idx.get((thr, arm, sc))
                if not d:
                    continue
                s = summarise([d[p] for p in sorted(d)])
                flag = ""
                if s["mean_rank"] and s["median_rank"] and s["mean_rank"] > 10 * max(s["median_rank"], 1):
                    flag = "  <: mean >> median: bimodal, do not quote the mean alone (32.2)"
                print(f"    {arm:<8}{sc:<18}{s['n']:>4}{s['recall_at_10']:>7.1f}%{s['mrr']:>7.3f}"
                      f"{s['top_1']:>6.1f}%{s['mean_rank']:>9.2f}{s['median_rank']:>8.1f}"
                      f"{s['not_retrieved']:>6}{flag}")
            for sc in others:
                a, b = idx.get((thr, arm, args.control)), idx.get((thr, arm, sc))
                if not a or not b:
                    continue
                pids = sorted(set(a) & set(b))
                pr = paired(a, b, pids)
                print(f"    paired {sc} vs {args.control}: r@10 {sc}-only {pr['wins']}, "
                      f"{args.control}-only {pr['losses']}, p = {pr['p_mcnemar']:.3g}; "
                      f"rank better {pr['rank_better']} / worse {pr['rank_worse']} of {pr['n']}")

        if "whole" in arms and "oracle" in arms:
            print(", the decomposition gap (oracle - whole), within each scorer,")
            for sc in scorers:
                w, o = idx.get((thr, "whole", sc)), idx.get((thr, "oracle", sc))
                if not w or not o:
                    continue
                pids = sorted(set(w) & set(o))
                both = [(w[p], o[p]) for p in pids if w[p] and o[p]]
                print(f"        {sc:<18} n={len(pids):<4} r@10 {sum(1 for p in pids if hit(w[p]))}"
                      f" -> {sum(1 for p in pids if hit(o[p]))}"
                      f"   mean rank {statistics.fmean([x for x, _ in both]):.2f}"
                      f" -> {statistics.fmean([y for _, y in both]):.2f}"
                      f"   improved {sum(1 for x, y in both if y < x)}"
                      f" / worsened {sum(1 for x, y in both if y > x)}")

        if f_of:
            print(", recall@10 by payload fraction,")
            for lo, hi, lab in STRATA:
                sub = [p for p in all_pids if lo <= f_of.get(p, -1) < hi]
                if not sub:
                    continue
                cells = []
                for arm in arms:
                    for sc in scorers:
                        d = idx.get((thr, arm, sc))
                        if not d:
                            continue
                        ps = [p for p in sub if p in d]
                        if ps:
                            cells.append(f"{arm}/{sc.split()[0]}  "
                                         f"{100 * sum(1 for p in ps if hit(d[p])) / len(ps):.0f}%")
                print(f"        {lab:<11} n={len(sub):<4}  " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
