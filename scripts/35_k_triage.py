#!/usr/bin/env python3
"""Does the segment count `k` predict when decomposition changes the answer? (open item 34)

    python scripts/35_k_triage.py
    python scripts/35_k_triage.py --out-dir results/k_triage --depth 10

Runs no retrieval. Everything it needs is in the four committed per-construct detail files, so this is
an analysis of measurements already made, on the 40,750-protein catalogue
(`results/gate_all173/reference_twilight.fasta`, 20,806,731 residues) at the 25% holdout.

**The question.** A screening operator cannot compute payload fraction `f`: it needs the answer. So the
f-stratified result describes the benchmark but cannot be a screening rule. `k_bhs`,
how many segments the label-free best-hit segmenter emits, is computed from the construct alone, and
this project found the segmenter's whole effect lives at `k >= 2`. This scores that as a triage rule:
*should this construct be decomposed before it is searched?*

**The comparator is 30.5's `best_hit_cov < 0.65`,** the project's existing triage signal, which is
in-sample, safe only in the negative direction, and whose specificity falls to 0.583 on real
constructs. The two rules are scored on the **same** constructs so the comparison is paired, comparing them across different samples would sum a sample change and an effect change
into one number, which is the error.

**Validated against the 23 first**, as this project's working rules requires, and reported as *consistent* rather than as
passing: the hand tier has 15 scorable constructs and no power to settle anything here.

Writes `k_triage.csv` (one row per tier per rule), `k_triage_summary.json` and `k_triage.log`.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from math import comb
from pathlib import Path

CATALOGUE = ("results/gate_all173/reference_twilight.fasta: 40,750 proteins, 20,806,731 residues, "
             "holdout 25%, gate E <= 1e-3")
# The figures this script re-derives as a cross-check. A mismatch means either the detail files or
# CONTEXT is wrong, and the detail files are the authority: this guard is how the 18 Oct conflation of
# `bhs_iterations` with `k_bhs` was found, after it had already reached open item 25 as an acceptance
# target. They are different quantities: an iteration whose claimed span is shorter than `min_len`
# emits no segment, so k <= iterations, and the floor decides *iterations* while k is what carries the
# value. A stopping rule must be checked against both.
EXPECTED_MEAN_ITERATIONS = {"synth_random": 1.01, "corpus": 1.10, "hand": 1.67,
                            "synth_competing": 3.46}
EXPECTED_MEAN_K = {"synth_random": 1.01, "corpus": 1.10, "hand": 1.60, "synth_competing": 3.34}
DEFAULT_INPUTS = {
    "corpus": "results/bhs_corpus_full/detail_corpus_25.csv",
    "hand": "results/bhs_corpus_full/detail_hand_25.csv",
    "synth_random": "results/synthetic_real/A_detail_238.csv",
    "synth_competing": "results/synthetic_real/C_detail_238.csv",
}
# The hand tier is scored first and reported first. Not cosmetic: this project's rule is that a rule is
# validated against the 23 before it is applied anywhere else, and print order is how that is honoured.
TIER_ORDER = ("hand", "corpus", "synth_random", "synth_competing")


def rank(value: str) -> int | None:
    """A rank, or None for 'not retrieved'. Blank and non-positive both mean unscorable."""
    try:
        r = int(float(value))
    except (TypeError, ValueError):
        return None
    return r if r > 0 else None


def read_detail(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p for a `b` against `c` discordant split."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(comb(n, i) for i in range(k + 1)) / 2 ** (n - 1))


def table_2x2(flags: list[bool], labels: list[bool]) -> dict:
    """Sensitivity, specificity, PPV, NPV of a boolean rule against a boolean outcome."""
    tp = sum(1 for f, l in zip(flags, labels) if f and l)
    fp = sum(1 for f, l in zip(flags, labels) if f and not l)
    fn = sum(1 for f, l in zip(flags, labels) if not f and l)
    tn = sum(1 for f, l in zip(flags, labels) if not f and not l)

    def ratio(num: int, den: int) -> float | None:
        return None if den == 0 else round(num / den, 4)

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": tp + fp + fn + tn,
            "n_positive_label": tp + fn, "n_flagged": tp + fp,
            "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
            "ppv": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn)}


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return None if dx == 0 or dy == 0 else round(num / (dx * dy), 4)


def analyse(tier: str, rows: list[dict], depth: int, cov_cut: float, out) -> dict:
    def say(line: str = "") -> None:
        print(line)
        out.write(line + "\n")

    # The outcome: per-segment ranking puts this construct inside the review depth where the
    # undecomposed query does not. That is the thing a triage rule would be used to predict.
    whole = [rank(r["rank_whole"]) for r in rows]
    per_seg = [rank(r["rank_per_segment"]) for r in rows]
    labels = [(p is not None and p <= depth) and not (w is not None and w <= depth)
              for w, p in zip(whole, per_seg)]
    k = [int(r["k_bhs"]) for r in rows]
    f = [float(r["f"]) for r in rows]
    cov = [float(r["best_hit_cov"]) if r["best_hit_cov"] not in ("", None) else None
           for r in rows]

    say(f"=== {tier}: n = {len(rows)}, outcome = per-segment enters top {depth} where whole does not "
        f"({sum(labels)} constructs)")

    res: dict = {"tier": tier, "n": len(rows), "n_label_positive": sum(labels), "depth": depth,
                 "catalogue": CATALOGUE}

    # --- rule 1: k >= 2 -------------------------------------------------------------------------
    k_flags = [x >= 2 for x in k]
    res["rule_k_ge_2"] = table_2x2(k_flags, labels)
    # --- rule 2: best_hit_cov < cut, on the SAME constructs ---------------------------------------
    have_cov = [i for i, c in enumerate(cov) if c is not None]
    cov_flags = [cov[i] < cov_cut for i in have_cov]
    cov_labels = [labels[i] for i in have_cov]
    res["rule_best_hit_cov"] = table_2x2(cov_flags, cov_labels) | {"cut": cov_cut}
    res["rule_best_hit_cov"]["n_missing_cov"] = len(rows) - len(have_cov)

    for name, key in (("k >= 2", "rule_k_ge_2"), (f"best_hit_cov < {cov_cut}", "rule_best_hit_cov")):
        d = res[key]
        say(f"    {name:<22} tp={d['tp']:>3} fp={d['fp']:>3} fn={d['fn']:>3} tn={d['tn']:>3}      "
            f"sens={d['sensitivity']} spec={d['specificity']} ppv={d['ppv']} npv={d['npv']}")

    # paired: do the two rules differ in how often they are RIGHT, on the constructs both can score?
    if have_cov:
        k_right = [k_flags[i] == labels[i] for i in have_cov]
        c_right = [cov_flags[j] == cov_labels[j] for j in range(len(have_cov))]
        b = sum(1 for a, c in zip(k_right, c_right) if a and not c)
        cc = sum(1 for a, c in zip(k_right, c_right) if c and not a)
        res["paired_vs_best_hit_cov"] = {"n": len(have_cov), "k_right_cov_wrong": b,
                                         "cov_right_k_wrong": cc, "p": round(mcnemar_exact(b, cc), 6)}
        say(f"  paired on {len(have_cov)}: k right / cov wrong {b}, cov right / k wrong {cc}, "
            f"McNemar exact p = {mcnemar_exact(b, cc):.4g}")

    # --- the k distribution, and the cross-check against the working record -----------------------------
    iters = [int(r["bhs_iterations"]) for r in rows]
    res["k"] = {}
    for name, vals, expected in (("k", k, EXPECTED_MEAN_K),
                                 ("iterations", iters, EXPECTED_MEAN_ITERATIONS)):
        hist = {str(v): vals.count(v) for v in sorted(set(vals))}
        mean_v = round(statistics.fmean(vals), 2)
        exp = expected.get(tier)
        ok = None if exp is None else abs(mean_v - exp) <= 0.01
        res["k"][name] = {"mean": mean_v, "median": statistics.median(vals), "max": max(vals),
                          "histogram": hist, "expected_mean_from_context": exp, "matches_context": ok}
        flag = "" if exp is None else ("  MATCHES CONTEXT" if ok
                                       else f"  *** MISMATCH: CONTEXT says {exp} ***")
        say(f"    {name}: mean {mean_v}, median {statistics.median(vals)}, max {max(vals)}, "
            f"{hist}{flag}")
    # kept flat for the CSV and for readers of earlier versions
    res["k"]["mean"] = res["k"]["k"]["mean"]
    res["k"]["max"] = res["k"]["k"]["max"]
    res["k"]["matches_context"] = (res["k"]["k"]["matches_context"] is not False
                                   and res["k"]["iterations"]["matches_context"] is not False)
    # A k = 1 construct is NOT the same query as `whole`: the segmenter emits one *sub-span* of the ORF,
    # so a single-segment construct can still be rescued or lost. 33.5 asserted it could not.
    res["k"]["n_k1_query_differs_from_whole"] = sum(
        1 for i in range(len(rows)) if k[i] == 1 and whole[i] != per_seg[i])

    # --- the falsifying cases, named before they are looked for -----------------------------------
    # 33.5 claims per-segment ranking never costs a construct its place in the top `depth`, and that
    # k = 1 constructs are never rescued. Either would falsify "k >= 2 is the triage rule".
    lost = [rows[i]["plasmid_id"] for i in range(len(rows))
            if (whole[i] is not None and whole[i] <= depth)
            and not (per_seg[i] is not None and per_seg[i] <= depth)]
    rescued_at_k1 = [rows[i]["plasmid_id"] for i in range(len(rows)) if labels[i] and k[i] == 1]
    res["falsifying"] = {"lost_top_n_under_per_segment": lost,
                         "rescued_at_k_1": rescued_at_k1}
    say(f"  falsifying check: lost at depth {depth} under per-segment = {len(lost)}  "
        f"{lost or ''}; rescued at k = 1 = {len(rescued_at_k1)}  {rescued_at_k1 or ''}")

    # --- k against f: different mechanisms, or one a proxy for the other? -------------------------
    fb = [("f<0.50", 0.0, 0.50), ("0.50-0.70", 0.50, 0.70), ("0.70-0.95", 0.70, 0.95),
          ("f>=0.95", 0.95, 1.0001)]
    kb = [("k=1", 1, 2), ("k=2", 2, 3), ("k>=3", 3, 10 ** 9)]
    cont = {fl: {kl: sum(1 for i in range(len(rows))
                         if flo <= f[i] < fhi and klo <= k[i] < khi)
                 for kl, klo, khi in kb} for fl, flo, fhi in fb}
    res["k_vs_f"] = {"contingency": cont,
                     "pearson_k_vs_f": pearson([float(x) for x in k], f),
                     "pearson_k_vs_f_spearmanlike": pearson(
                         [float(sorted(k).index(x)) for x in k], [float(sorted(f).index(x)) for x in f])}
    say(f"  k vs f: Pearson r = {res['k_vs_f']['pearson_k_vs_f']}")
    say(f"        {'':<11}" + "".join(f"{kl:>8}" for kl, _, _ in kb))
    for fl, _, _ in fb:
        say(f"        {fl:<11}" + "".join(f"{cont[fl][kl]:>8}" for kl, _, _ in kb))

    # `whole` alone, by k. The reason k = 1 means "do not bother": the undecomposed arm is already there.
    by_k = {}
    for lab, lo, hi in kb:
        sel = [i for i in range(len(rows)) if lo <= k[i] < hi]
        if not sel:
            continue
        got = [whole[i] for i in sel if whole[i] is not None]
        by_k[lab] = {"n": len(sel),
                     "whole_recall_at_depth": round(100 * sum(1 for x in got if x <= depth) / len(sel), 1),
                     "per_segment_recall_at_depth": round(
                         100 * sum(1 for i in sel if per_seg[i] is not None and per_seg[i] <= depth)
                         / len(sel), 1)}
    res["by_k"] = by_k
    say("  recall@%d by k: %s" % (depth, ", ".join(
        f"{lab} n={d['n']} whole {d['whole_recall_at_depth']}% -> per-segment "
        f"{d['per_segment_recall_at_depth']}%" for lab, d in by_k.items())))
    say()
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("results/k_triage"))
    ap.add_argument("--depth", type=int, default=10,
                    help="review depth the outcome is defined at. 10 per the working record; open item 23 "
                         "is the unanswered question of what an operator actually works to.")
    ap.add_argument("--cov-cut", type=float, default=0.65,
                    help="the comparator rule's threshold, from the working record")
    for tier, default in DEFAULT_INPUTS.items():
        ap.add_argument(f"--{tier.replace('_', '-')}", type=Path, default=Path(default))
    args = ap.parse_args(argv)

    inputs = {t: getattr(args, t) for t in DEFAULT_INPUTS}
    missing = [str(p) for p in inputs.values() if not p.is_file()]
    if missing:
        print("not found: " + ", ".join(missing), file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = open(args.out_dir / "k_triage.log", "w", encoding="utf-8")
    out.write(f"catalogue: {CATALOGUE}\ndepth: {args.depth}\ncomparator cut: {args.cov_cut}\n\n")
    print(f"catalogue: {CATALOGUE}")
    print(f"the hand tier is scored first; n = 15 and it settles nothing\n")

    results = {}
    for tier in TIER_ORDER:
        results[tier] = analyse(tier, read_detail(inputs[tier]), args.depth, args.cov_cut, out)

    # One CSV row per tier per rule, so the tables above are re-derivable without rerunning.
    csv_path = args.out_dir / "k_triage.csv"
    fields = ["tier", "rule", "n", "n_label_positive", "tp", "fp", "fn", "tn",
              "sensitivity", "specificity", "ppv", "npv", "mean_k", "max_k"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for tier, r in results.items():
            for rule_key, rule_name in (("rule_k_ge_2", "k>=2"),
                                        ("rule_best_hit_cov", f"best_hit_cov<{args.cov_cut}")):
                d = r[rule_key]
                w.writerow({"tier": tier, "rule": rule_name, "n": d["n"],
                            "n_label_positive": r["n_label_positive"],
                            **{x: d[x] for x in ("tp", "fp", "fn", "tn", "sensitivity",
                                                 "specificity", "ppv", "npv")},
                            "mean_k": r["k"]["mean"], "max_k": r["k"]["max"]})

    bad = [t for t, r in results.items() if r["k"]["matches_context"] is False]
    summary = {"catalogue": CATALOGUE, "depth": args.depth, "cov_cut": args.cov_cut,
               "context_33_5_mean_k_reproduced": not bad, "tiers": results}
    (args.out_dir / "k_triage_summary.json").write_text(json.dumps(summary, indent=2),
                                                        encoding="utf-8")
    line = ("every tier's mean k reproduces the working record" if not bad
            else f"*** mean k does NOT reproduce 33.5 for: {', '.join(bad)}, the detail files are "
                 f"the authority, so section 33 is wrong ***")
    print(line)
    out.write(line + "\n")
    out.close()
    print(f"wrote {csv_path.name}, k_triage_summary.json, k_triage.log in {args.out_dir}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
