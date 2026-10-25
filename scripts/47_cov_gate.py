"""Does `best_hit_cov` decide *whether to decompose*? Test the gate the working record pre-registered.

51.3 found decomposition's first measured cost: at f 0.70-0.95 the segmenter loses 9,446 rank points
on three GAL4 constructs where the whole-ORF query is at rank 2, because those constructs are high-f
(0.896) and low-coverage (payload coverage 0.196): a DNA-binding domain standing in for an
881-residue protein. It then proposed a gate: decline to segment when 30.5's triage signal says the
ORF is not diluted, which is the one direction that signal is sound in (NPV 0.996-1.000), and named
the falsifier: **the gate works if it leaves the GAL4 constructs on `whole` and costs nothing in the
f < 0.50 stratum.**

The gate is label-free by construction. `best_hit_cov` is the fraction of the called ORF claimed by
the single highest-scoring catalogue hit (`26_bhs_protocol.py`); it consults the catalogue and never
the payload, so it is a screening rule and not a description. `f` is labelled and appears here only to
stratify, per 26.3.

Three things are measured, not one, because a gate that fails can fail for two different reasons:

1. **The stated gate at 0.65.** Where does it route each construct, and what does that cost?
2. **A sweep of the threshold**, because this project's working rules requires a tuned threshold to be swept with its
   plateau shown rather than reported at its best value. If no threshold works, that is a closed
   measurement rather than an invitation to keep looking.
3. **An oracle gate**: route each construct to whichever arm actually wins. This uses the label and
   is a diagnostic, never an arm, exactly as 26's oracle-selected protocol is. It separates "the
   signal is the wrong signal" from "no routing of these two arms could have helped", which are
   different findings and need opposite responses.
"""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE_COV = 0.65          # in-sample and stated to be so
F_STRATA = (0.50, 0.70, 0.95)
STRATUM_NAMES = ("<0.50", "0.50-0.70", "0.70-0.95", ">=0.95")
SWEEP = (0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 0.90)


def stratum(f: float) -> str:
    if f < F_STRATA[0]:
        return STRATUM_NAMES[0]
    if f < F_STRATA[1]:
        return STRATUM_NAMES[1]
    if f < F_STRATA[2]:
        return STRATUM_NAMES[2]
    return STRATUM_NAMES[3]


def k_gated_rank(row: dict, min_segments: int = 2) -> int:
    """The alternative gate: decompose only when the segmenter claimed at least `min_segments`.

    `k_bhs` is the number of segments the best-hit segmenter emitted. Like `best_hit_cov` it is
    label-free: it consults the catalogue and never the payload.

    **`min_segments = 2` is a swept parameter. It was argued here to be a definition and it is not**
    (19 Oct). The dead argument ran: k = 1 means the segmenter found one thing, so
    "decompose" and "query the whole ORF" are the same claim about the sequence, and the per-segment arm
    differs from `whole` only by having thrown the unclaimed residues away. The premise holds; the
    inference does not, because the residues thrown away are the ones competing for the rank: one
    claimed window can beat the ORF it was cut from, 6 times in 476 synthetic constructs, largest
    66 -> 1. What justifies 2 is the sweep: of the three candidate settings on route B, 1 (which is
    decomposing unconditionally) loses f 0.70-0.95 to 174.18 and 3 breaks >= 0.95 and 0.50-0.70, so 2 is
    the interior winner with both neighbours failing differently. In-sample, and quoted as such.

    It is still in-sample on route B. this project's working rules is explicit that a threshold chosen on the data it is
    then evaluated on is the same error as a tuned one wearing the other hat, so `--validate` scores
    it against the 23 hand-audited constructs before it is believed.
    """
    return row["rank_per_segment"] if row["k_bhs"] >= min_segments else row["rank_whole"]


def gated_rank(row: dict, cov_threshold: float) -> int:
    """The arm the gate picks. High coverage means 'mostly one known thing': do not decompose.

    Below the threshold the gate decomposes, which is the *positive* use of a signal 30.5 says is
    only sound negatively. That is the thing under test, not an assumption being made here.
    """
    cov = row["cov"]
    if cov is not None and cov >= cov_threshold:
        return row["rank_whole"]
    return row["rank_per_segment"]


def read_detail(path: Path) -> list:
    """One row per construct, with the three ranks and the gate signal."""
    out = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if not r.get("f") or not r.get("rank_whole"):
                continue
            out.append({
                "plasmid_id": r["plasmid_id"],
                "payload": r["payload_accession"],
                "f": float(r["f"]),
                "cov": float(r["best_hit_cov"]) if r["best_hit_cov"] else None,
                "best_hit": r["best_hit_accession"],
                "rank_whole": int(r["rank_whole"]),
                "rank_per_segment": int(r["rank_per_segment"]),
                "rank_oracle": int(r["rank_oracle"]),
                "k_bhs": int(r["k_bhs"]) if r.get("k_bhs") else 1,
            })
    return out


def metrics(ranks: list) -> dict:
    return {"n": len(ranks),
            "mean": round(statistics.fmean(ranks), 2),
            "median": statistics.median(ranks),
            "mrr": round(statistics.fmean(1.0 / r for r in ranks), 3),
            "r10": round(sum(1 for r in ranks if r <= 10) / len(ranks), 3)}


def macro_mean(rows: list, pick) -> float:
    by_payload = defaultdict(list)
    for r in rows:
        by_payload[r["payload"]].append(pick(r))
    return round(statistics.fmean(statistics.fmean(v) for v in by_payload.values()), 2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detail", type=Path,
                    default=ROOT / "results/route_b_gate/detail_h25.csv",
                    help="a 26_bhs_protocol.py --detail-csv for ONE holdout")
    ap.add_argument("--cov-threshold", type=float, default=GATE_COV)
    args = ap.parse_args(argv)

    rows = read_detail(args.detail)
    arms = {
        "whole (always)": lambda r: r["rank_whole"],
        "per_segment (always)": lambda r: r["rank_per_segment"],
        f"GATED at cov>={args.cov_threshold}": lambda r: gated_rank(r, args.cov_threshold),
        "GATED at k_bhs>=2": lambda r: k_gated_rank(r, 2),
        "GATED at k_bhs>=3": lambda r: k_gated_rank(r, 3),
        "oracle gate (uses the label)": lambda r: min(r["rank_whole"], r["rank_per_segment"]),
        "oracle span (uses the label)": lambda r: r["rank_oracle"],
    }

    n_hi = sum(1 for r in rows if r["cov"] is not None and r["cov"] >= args.cov_threshold)
    print(f"the best_hit_cov gate, from {args.detail.name}")
    print(f"  constructs: {len(rows)}")
    print(f"  BINDING, routed to `whole` by cov >= {args.cov_threshold}: {n_hi} of {len(rows)}; "
          f"the other {len(rows) - n_hi} are decomposed")
    print(f"  BINDING, no best hit at all (gate has no signal): "
          f"{sum(1 for r in rows if r['cov'] is None)}")
    covs = sorted(r["cov"] for r in rows if r["cov"] is not None)
    print(f"  best_hit_cov: min {covs[0]:.3f}  median {covs[len(covs) // 2]:.3f}  max {covs[-1]:.3f}")

    print("\n--- the case the gate was proposed for " + "-" * 36)
    worst = sorted(rows, key=lambda r: r["rank_per_segment"] - r["rank_whole"], reverse=True)[:5]
    print(f"{'construct':<40}{'f':>7}{'cov':>7}{'k':>4}{'whole':>8}{'per_seg':>9}"
          f"{'cov gate':>8}  cov sends   k sends")
    for r in worst:
        g = gated_rank(r, args.cov_threshold)
        to = "whole" if (r["cov"] is not None and r["cov"] >= args.cov_threshold) else "DECOMPOSE"
        kto = "whole" if r["k_bhs"] < 2 else "DECOMPOSE"
        print(f"{r['plasmid_id'][:40]:<40}{r['f']:>7.3f}"
              f"{(r['cov'] if r['cov'] is not None else float('nan')):>7.3f}{r['k_bhs']:>4}"
              f"{r['rank_whole']:>8}{r['rank_per_segment']:>9}{g:>8}    {to:<11}{kto}")

    print("\n--- by f stratum, mean rank / MRR / recall@10 (never pooled, 26.3) " + "-" * 22)
    print(f"{'f stratum':<12}{'arm':<32}{'n':>4}{'mean':>9}{'med':>6}{'MRR':>7}{'r@10':>8}"
          f"{'macro mean':>12}")
    for s in STRATUM_NAMES:
        rows_s = [r for r in rows if stratum(r["f"]) == s]
        if not rows_s:
            continue
        for name, pick in arms.items():
            m = metrics([pick(r) for r in rows_s])
            print(f"{s:<12}{name:<32}{m['n']:>4}{m['mean']:>9}{m['median']:>6}{m['mrr']:>7}"
                  f"{m['r10']:>8}{macro_mean(rows_s, pick):>12}")
        print()

    print("--- threshold sweep: is there ANY cov threshold that beats both arms? " + "-" * 18)
    print("  (this project's working rules: sweep a tuned threshold and show the plateau, never report the best value)")
    print(f"{'threshold':<11}{'n to whole':>11}", end="")
    for s in STRATUM_NAMES:
        print(f"{s + ' mean':>18}", end="")
    print()
    for thr in SWEEP:
        n_w = sum(1 for r in rows if r["cov"] is not None and r["cov"] >= thr)
        print(f"{thr:<11}{n_w:>11}", end="")
        for s in STRATUM_NAMES:
            rows_s = [r for r in rows if stratum(r["f"]) == s]
            if not rows_s:
                print(f"{'-':>18}", end="")
                continue
            print(f"{metrics([gated_rank(r, thr) for r in rows_s])['mean']:>18}", end="")
        print()
    print("\n  for reference, the same four strata under each fixed arm:")
    for name in ("whole (always)", "per_segment (always)", "oracle gate (uses the label)"):
        print(f"{name:<22}", end="")
        for s in STRATUM_NAMES:
            rows_s = [r for r in rows if stratum(r["f"]) == s]
            print(f"{(metrics([arms[name](r) for r in rows_s])['mean'] if rows_s else '-'):>18}",
                  end="")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
