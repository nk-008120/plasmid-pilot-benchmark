"""How much of the oracle-minus-whole gap does the LABEL-FREE arm actually close?

Every slide and section in this project so far reports `oracle - whole` as the available headroom and
`auto - whole` as what the hand-written motif rules capture. Neither number answers the question the
project exists to answer, which is what the **deployable** arm captures: `bhs_per_segment`, the
label-free best-hit segmenter. This script computes it from the four committed detail CSVs,
per tier and stratified by payload fraction, with the paired test beside it.

**It reads committed files and runs no scorer**, so it is reproducible from the repository alone and
costs seconds. That matters: the number it produces is the project's headline and it should not depend
on a machine being available.

Three things it is careful about, all of them this project's working rules rules:

* **Paired, restricted to constructs present in both arms.** A tier's rows are the same constructs for
  every arm, so the comparison is within-construct and McNemar applies. Reporting four marginal rates and
  subtracting them would sum a sample effect and an effect effect, which is this project's error.
* **Stratified by payload fraction, never pooled.** Three of the four tiers are dominated by constructs
  the effect cannot act on (corpus median f = 1.000), so the pooled figure is a measurement of the
  sampling frame. The stratified table is the result; the pooled row is there to show the difference.
* **The closure fraction is undefined, not zero, where there is no gap.** On `synth_random` oracle equals
  whole, so there is nothing to close and the ratio has no denominator. Printing 0% there would read as a
  failure of the arm when it is a property of the tier, and that tier is the control.

Writes `gap_closure.json`, `gap_closure.csv` and `gap_closure.log` to `--out-dir`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# (label, path relative to the repo root, whose ground truth)
TIERS = [
    ("hand", "results/bhs_corpus_full/detail_hand_25.csv", "depositor"),
    ("corpus", "results/bhs_corpus_full/detail_corpus_25.csv", "alignment (partly circular for this arm)"),
    ("synth_random", "results/synthetic_real/A_detail_238.csv", "exact by construction"),
    ("synth_competing", "results/synthetic_real/C_detail_238.csv", "exact by construction"),
]
ARMS = [("whole", "rank_whole"), ("auto_per_segment", "rank_auto_per_segment"),
        ("bhs_per_segment", "rank_per_segment"), ("oracle", "rank_oracle")]
STRATA = [(0.0, 0.50, "f < 0.50"), (0.50, 0.70, "0.50-0.70"),
          (0.70, 0.95, "0.70-0.95"), (0.95, 1.01, "f >= 0.95")]
DEPTH = 10


def read_rank(value: str | None) -> int | None:
    """A rank, or `None` for a construct the arm did not retrieve at any depth.

    Blank is NOT zero and NOT rank 1. An earlier version of this read coerced blanks with `int(x or 0)`,
    which silently promoted every miss to the best possible rank.
    """
    v = (value or "").strip()
    if v == "" or v.lower() in {"na", "none", "nan"}:
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def hits(rows: list[dict], col: str, depth: int = DEPTH) -> int:
    return sum(1 for r in rows if (x := read_rank(r.get(col))) is not None and x <= depth)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on the discordant pairs only. `b + c == 0` is no evidence, p = 1."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n), 1.0)


def arm_summary(rows: list[dict], col: str) -> dict:
    ranks = [read_rank(r.get(col)) for r in rows]
    found = [x for x in ranks if x is not None]
    return {
        "n": len(ranks),
        "recall_at_10": round(100 * sum(1 for x in found if x <= DEPTH) / len(ranks), 1),
        "top_1": round(100 * sum(1 for x in found if x == 1) / len(ranks), 1),
        # Mean rank is reported beside its miss count, never alone.
        "mean_rank": round(statistics.fmean(found), 2) if found else None,
        "not_retrieved": len(ranks) - len(found),
    }


def closure(rows: list[dict]) -> dict:
    w, b, o = (hits(rows, "rank_whole"), hits(rows, "rank_per_segment"), hits(rows, "rank_oracle"))
    gap = o - w
    return {
        "n": len(rows), "whole": w, "bhs_per_segment": b, "oracle": o,
        "gap_oracle_minus_whole": gap,
        # Undefined where the tier offers no gap: see the module docstring.
        "closed_pct": round(100 * (b - w) / gap, 1) if gap else None,
        "residual_to_oracle": o - b,
    }


def paired(rows: list[dict], a: str, b: str) -> dict:
    """Discordant counts for `b` against `a` at depth 10, within construct."""
    wins = sum(1 for r in rows
               if not ((x := read_rank(r.get(a))) is not None and x <= DEPTH)
               and ((y := read_rank(r.get(b))) is not None and y <= DEPTH))
    losses = sum(1 for r in rows
                 if ((x := read_rank(r.get(a))) is not None and x <= DEPTH)
                 and not ((y := read_rank(r.get(b))) is not None and y <= DEPTH))
    return {"wins": wins, "losses": losses, "p_mcnemar": mcnemar_exact(wins, losses)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out-dir", type=Path, default=Path("results/gap_closure"))
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = open(args.out_dir / "gap_closure.log", "w", encoding="utf-8")

    def say(line: str = "") -> None:
        print(line, flush=True)
        log.write(line + "\n")

    out: dict = {"depth": DEPTH, "tiers": {}}
    csv_rows: list[dict] = []

    for tier, rel, truth in TIERS:
        path = args.repo_root / rel
        if not path.exists():
            say(f"missing: {rel}, skipped")
            continue
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        say(f"\n=== {tier}  (n = {len(rows)}, ground truth: {truth}) ===")
        rec: dict = {"n": len(rows), "ground_truth": truth, "arms": {}, "strata": {}}
        say(f"    {'arm':<18}{'recall@10':>11}{'top-1':>9}{'mean rank':>11}{'not retrieved':>15}")
        for arm, col in ARMS:
            s = arm_summary(rows, col)
            rec["arms"][arm] = s
            say(f"    {arm:<18}{s['recall_at_10']:>10.1f}%{s['top_1']:>8.1f}%"
                f"{s['mean_rank'] if s['mean_rank'] is not None else float('nan'):>11.2f}"
                f"{s['not_retrieved']:>15}")

        rec["closure"] = closure(rows)
        c = rec["closure"]
        frac = "undefined (no gap on this tier)" if c["closed_pct"] is None else f"{c['closed_pct']}%"
        say(f"  gap oracle-whole = {c['gap_oracle_minus_whole']} constructs;"
            f" label-free arm closes {frac}; residual to oracle {c['residual_to_oracle']}")

        rec["paired_bhs_vs_whole"] = paired(rows, "rank_whole", "rank_per_segment")
        rec["paired_bhs_vs_auto"] = paired(rows, "rank_auto_per_segment", "rank_per_segment")
        p = rec["paired_bhs_vs_whole"]
        say(f"  paired vs whole: bhs wins {p['wins']}, whole wins {p['losses']}, "
            f"p = {p['p_mcnemar']:.3g}")

        for lo, hi, lab in STRATA:
            sub = [r for r in rows if (r.get("f") or "").strip()
                   and lo <= float(r["f"]) < hi]
            if not sub:
                continue
            st = {"n": len(sub), **closure(sub),
                  "paired_bhs_vs_whole": paired(sub, "rank_whole", "rank_per_segment")}
            rec["strata"][lab] = st
            fr = "--" if st["closed_pct"] is None else f"{st['closed_pct']:.0f}%"
            say(f"        {lab:<11} n={st['n']:<4} whole {100*st['whole']/st['n']:>5.1f}%  "
                f"bhs {100*st['bhs_per_segment']/st['n']:>5.1f}%  "
                f"oracle {100*st['oracle']/st['n']:>5.1f}%  closed {fr}")
            csv_rows.append({"tier": tier, "stratum": lab, "n": st["n"], "whole": st["whole"],
                             "bhs_per_segment": st["bhs_per_segment"], "oracle": st["oracle"],
                             "closed_pct": st["closed_pct"],
                             "wins": st["paired_bhs_vs_whole"]["wins"],
                             "losses": st["paired_bhs_vs_whole"]["losses"]})
        out["tiers"][tier] = rec

    say("\nREAD THIS AS THE RESULT. The deployable arm is `bhs_per_segment`; `oracle` uses the label and "
        "is a\nceiling. Where a tier shows no gap, the arm is not failing: the tier cannot host the "
        "effect, and\n`synth_random` is exactly that control. The corpus tier's payload labels come from "
        "alignment, so a\nsegmenter that cuts on alignment hits is partly circular there; the hand tier "
        "and both synthetic\ntiers are not.")

    (args.out_dir / "gap_closure.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with open(args.out_dir / "gap_closure.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    say(f"\nwrote gap_closure.json, gap_closure.csv, gap_closure.log in {args.out_dir}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
