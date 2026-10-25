"""Hunt this project's named falsifier on the synthetic tier: k_bhs = 1 where per_segment beats whole.

52.2 replaced the refuted `best_hit_cov` gate with a segment-count gate, decompose when the best-hit
segmenter claimed `k_bhs >= 2`, query the whole ORF when it claimed 1, and argued the k = 1 branch
from a premise rather than a threshold: **at k = 1 the segmenter found one catalogue-supported thing,
so decomposing can only discard residues.** 52.3 then named the case that refutes it, a construct with
k = 1 where `per_segment` nonetheless beats `whole`, counted 0 of 22 across route B and the 23, and
said plainly why that count does not settle it: **the hand tier cannot produce the case at all** (all
9 of its k = 1 constructs have `whole == per_segment` exactly), so the branch rests on route B,
in-sample.

The synthetic tier is where the case can be built rather than waited for, because dilution is an
independent variable there with exact boundaries by construction (the three tiers).

**Two conditions are read, not one, and that is the point.** this project's working rules forbids tuning a constructed
condition until it matches reality and requires an overshoot to be reported as a bound with the real
case bracketed between conditions (30.6, the user's decision at open item 12):

* **A `random`**: `random_protein()` fillers. No non-payload material is catalogue-supported, so the
  segmenter has nothing but the payload to find. The **floor** for segmentation, and 30.2's measured
  null: whole-to-oracle is -0.00 in every stratum.
* **C `real_competing`**: real Swiss-Prot fillers with their families present. Maximal rank
  competition and an explicit **upper bound**, 2.5x the hand tier's gap at f < 0.50 (30.6).

Condition B (`real_masked`, the middle of the bracket) was run on 17 Oct but its detail CSV was never
committed, and the 40,750-protein catalogue it needs is not in this checkout, so it is absent here and
counted as such rather than quietly dropped.

Nothing is re-scored. These are `26_bhs_protocol.py --detail-csv` artefacts from this project's run,
holdout 25%, exact Smith-Waterman, gate E <= 1e-3, catalogue
`results/gate_all173/reference_twilight.fasta` (**40,750 proteins, 20,806,731 residues** in the
50-5,000 aa window; this project's existing row, **not a new catalogue**), so the gate's definition is
imported from `47_cov_gate.py` and only the census is new.
"""
import argparse
import csv
import importlib.util
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("cov_gate", ROOT / "scripts/47_cov_gate.py")
cov_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cov_gate)

# The gate under test lives in 47 and is imported rather than restated, so the two scripts cannot
# drift into testing different rules.
k_gated_rank = cov_gate.k_gated_rank
stratum = cov_gate.stratum
metrics = cov_gate.metrics
macro_mean = cov_gate.macro_mean
STRATUM_NAMES = cov_gate.STRATUM_NAMES

TIER_TOTAL = 336          # GenBank files in data/synthetic/, the whole tier before any holdout
CONDITIONS = {"A": ("results/synthetic_real/A_detail_238.csv", "random (control / floor)"),
              "B": ("results/synthetic_real/B_detail_238.csv", "real_masked (the bracket's middle)"),
              "C": ("results/synthetic_real/C_detail_238.csv", "real_competing (upper bound)")}


def read_detail(path: Path) -> list[dict]:
    """47's rows plus the three segmenter diagnostics the census needs.

    `best_hit_emitted` matters here and nowhere else: at k = 1 it says whether the single emitted
    segment IS the highest-scoring hit or something the segmenter preferred over it, which is the
    difference between "one thing found" and "one thing chosen over a better-scoring one".
    """
    rows = cov_gate.read_detail(path)
    extra = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            extra[r["plasmid_id"]] = {
                "orf_len": int(r["orf_len"]) if r.get("orf_len") else None,
                "bhs_iterations": int(r["bhs_iterations"]) if r.get("bhs_iterations") else None,
                "bhs_stop": r.get("bhs_stop", ""),
                "bhs_fellback": int(r["bhs_fellback"]) if r.get("bhs_fellback") else 0,
                "best_hit_emitted": (int(r["best_hit_emitted"])
                                     if r.get("best_hit_emitted") not in (None, "") else None),
                "n_targets": int(r["n_targets"]) if r.get("n_targets") else None,
                "db_residues_searched": (int(r["db_residues_searched"])
                                         if r.get("db_residues_searched") else None),
                "condition": r.get("condition", ""),
            }
    for row in rows:
        row.update(extra[row["plasmid_id"]])
    return rows


def verdict(row: dict) -> str:
    """Which arm actually won on this construct. Uses the ranks, so it is a diagnostic."""
    if row["rank_per_segment"] < row["rank_whole"]:
        return "per_segment"
    if row["rank_whole"] < row["rank_per_segment"]:
        return "whole"
    return "tie"


def census(rows: list[dict], min_segments: int = 2) -> dict:
    """Count, per gate branch, which arm wins. The k = 1 / per_segment cell is the falsifier."""
    out = {}
    for branch, keep in (("k=1", lambda r: r["k_bhs"] < min_segments),
                         (f"k>={min_segments}", lambda r: r["k_bhs"] >= min_segments)):
        g = [r for r in rows if keep(r)]
        c = Counter(verdict(r) for r in g)
        out[branch] = {"n": len(g), "tie": c["tie"], "whole": c["whole"],
                       "per_segment": c["per_segment"],
                       "payloads_whole": sorted({r["payload"] for r in g if verdict(r) == "whole"}),
                       "payloads_per_segment": sorted({r["payload"] for r in g
                                                       if verdict(r) == "per_segment"})}
    return out


def falsifiers(rows: list[dict], min_segments: int = 2) -> list[dict]:
    """this project's named case: k below the gate's boundary, and per_segment strictly better."""
    return [r for r in rows
            if r["k_bhs"] < min_segments and r["rank_per_segment"] < r["rank_whole"]]


def branch_rank_points(rows: list[dict], min_segments: int = 2) -> dict:
    """What the k = 1 branch is worth in rank points against decomposing unconditionally.

    Positive `net` means the branch earns its place on this condition. The sign is the quantity of
    interest, because a branch that helps on the control and hurts on the bound is bracketed rather
    than settled, and this project's working rules forbids reading either end as the answer.
    """
    g = [r for r in rows if r["k_bhs"] < min_segments]
    saved = sum(r["rank_per_segment"] - r["rank_whole"] for r in g if verdict(r) == "whole")
    lost = sum(r["rank_whole"] - r["rank_per_segment"] for r in g if verdict(r) == "per_segment")
    return {"n": len(g), "saved": saved, "lost": lost, "net": saved - lost}


def binding(rows: list[dict], tier_total: int = TIER_TOTAL) -> dict:
    """Every limit that could have truncated the subset this gate is measured on.

    A gate measured on a silently truncated subset is exactly the failure 52.3 is about, so the
    counts are printed whether or not any of them bind.
    """
    return {"tier_total": tier_total,
            "scored": len(rows),
            "absent_at_holdout": tier_total - len(rows),
            "payloads": len({r["payload"] for r in rows}),
            "no_best_hit": sum(1 for r in rows if r["cov"] is None),
            "segmenter_fell_back": sum(1 for r in rows if r["bhs_fellback"]),
            "stop_reasons": dict(Counter(r["bhs_stop"] for r in rows)),
            "max_iterations": max((r["bhs_iterations"] or 0) for r in rows) if rows else 0,
            "min_targets": min((r["n_targets"] or 0) for r in rows) if rows else 0,
            "distinct_databases": len({r["db_residues_searched"] for r in rows}),
            "k_distribution": dict(sorted(Counter(r["k_bhs"] for r in rows).items())),
            "f_strata": {s: sum(1 for r in rows if stratum(r["f"]) == s) for s in STRATUM_NAMES}}


def emitted_probe(rows: list[dict], min_segments: int = 2) -> dict:
    """Does `best_hit_emitted == 0` separate the falsifiers? IN-SAMPLE, found by looking at them.

    Reported because it is cheap and label-free, and reported as a hypothesis because this project's working rules is
    explicit that a rule chosen on the data it is then scored on is not validated. It is not proposed
    as a replacement gate here.
    """
    k1 = [r for r in rows if r["k_bhs"] < min_segments]
    fired = [r for r in k1 if r["best_hit_emitted"] == 0]
    bad = [r for r in k1 if verdict(r) == "per_segment"]
    return {"k1": len(k1), "fired": len(fired),
            "fired_and_falsifying": sum(1 for r in fired if verdict(r) == "per_segment"),
            "falsifying": len(bad),
            "missed": [r["plasmid_id"] for r in bad if r["best_hit_emitted"] != 0]}


def write_csvs(out_dir: Path, loaded: list, min_segments: int = 2) -> None:
    """One census row per condition x f stratum, and one row per falsifying construct."""
    out_dir.mkdir(parents=True, exist_ok=True)
    census_rows, fal_rows = [], []
    for key, label, rows in loaded:
        for s in ("ALL",) + STRATUM_NAMES:
            rs = rows if s == "ALL" else [r for r in rows if stratum(r["f"]) == s]
            if not rs:
                continue
            c = census(rs, min_segments)
            rp = branch_rank_points(rs, min_segments)
            census_rows.append({
                "condition": key, "condition_label": label, "f_stratum": s, "n": len(rs),
                "k1_n": c["k=1"]["n"], "k1_tie": c["k=1"]["tie"],
                "k1_whole_wins": c["k=1"]["whole"], "k1_per_segment_wins": c["k=1"]["per_segment"],
                "k1_payloads_helped": " ".join(c["k=1"]["payloads_whole"]),
                "k1_payloads_hurt": " ".join(c["k=1"]["payloads_per_segment"]),
                "kge_n": c[f"k>={min_segments}"]["n"],
                "kge_whole_wins": c[f"k>={min_segments}"]["whole"],
                "kge_per_segment_wins": c[f"k>={min_segments}"]["per_segment"],
                "branch_rank_points_saved": rp["saved"], "branch_rank_points_lost": rp["lost"],
                "branch_rank_points_net": rp["net"],
                "mean_whole": metrics([r["rank_whole"] for r in rs])["mean"],
                "mean_per_segment": metrics([r["rank_per_segment"] for r in rs])["mean"],
                "mean_gated": metrics([k_gated_rank(r, min_segments) for r in rs])["mean"],
                "mean_oracle_router": metrics(
                    [min(r["rank_whole"], r["rank_per_segment"]) for r in rs])["mean"],
                "mean_oracle_span": metrics([r["rank_oracle"] for r in rs])["mean"]})
        for r in falsifiers(rows, min_segments):
            fal_rows.append({"condition": key, "f_stratum": stratum(r["f"]), **r})
    for name, rows_out in (("census.csv", census_rows), ("falsifiers.csv", fal_rows)):
        if not rows_out:
            continue
        with open(out_dir / name, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)


def load(args) -> list[tuple[str, str, list[dict]]]:
    out = []
    for key in args.conditions.split(","):
        key = key.strip()
        rel, label = CONDITIONS[key]
        path = args.detail_dir / Path(rel).name if args.detail_dir else ROOT / rel
        if not path.is_file():
            print(f"condition {key} ({label}): ABSENT, {path} is not in this checkout, "
                  f"counted rather than dropped\n")
            continue
        out.append((key, label, read_detail(path)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", default="A,B,C",
                    help="which of this project's paired conditions to read, comma-separated")
    ap.add_argument("--detail-dir", type=Path, default=None,
                    help="look for the detail CSVs here instead of results/synthetic_real/")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write census.csv and falsifiers.csv here, so the census can be re-tabled "
                         "without re-reading the detail CSVs")
    ap.add_argument("--min-segments", type=int, default=2,
                    help="the gate's boundary. 2 is 52.2's definition, not a tuned value; it is an "
                         "argument so the census can be re-run at another k, never so it can be fitted")
    args = ap.parse_args(argv)

    loaded = load(args)
    if not loaded:
        print("no condition is present; nothing to census")
        return 1

    print("this project's named falsifier, hunted on the synthetic tier")
    print(f"  the case: k_bhs < {args.min_segments} and rank_per_segment < rank_whole")
    print("  catalogue: results/gate_all173/reference_twilight.fasta, 40,750 proteins, "
          "20,806,731 residues (this project's existing row)")
    print("  holdout 25%, exact Smith-Waterman, gate E <= 1e-3; nothing re-scored\n")

    for key, label, rows in loaded:
        b = binding(rows)
        print(f"=== condition {key}: {label}  " + "=" * max(0, 46 - len(label)))
        print(f"  BINDING, tier: {b['scored']} of {b['tier_total']} constructs scorable at the 25% "
              f"holdout; {b['absent_at_holdout']} absent, over {b['payloads']} payloads")
        print(f"  BINDING, no best hit at all (the gate has no signal): {b['no_best_hit']}")
        print(f"  BINDING, segmenter fell back to the whole ORF: {b['segmenter_fell_back']}")
        print(f"  BINDING, segmenter stop reasons: {b['stop_reasons']}  "
              f"(max {b['max_iterations']} iterations; an iteration cap would show here)")
        print(f"  BINDING, fewest retrievable targets on any construct: {b['min_targets']}; "
              f"distinct databases searched: {b['distinct_databases']}")
        print(f"  k_bhs distribution: {b['k_distribution']}")
        print(f"  f strata: {b['f_strata']}")

        cs = census(rows, args.min_segments)
        print(f"\n  {'branch':<10}{'n':>5}{'tie':>6}{'whole wins':>12}{'per_seg wins':>14}"
              f"   payloads (whole / per_seg)")
        for branch, c in cs.items():
            print(f"    {branch:<10}{c['n']:>5}{c['tie']:>6}{c['whole']:>12}{c['per_segment']:>14}"
                  f"      {len(c['payloads_whole'])} / {len(c['payloads_per_segment'])}")
        f1 = cs["k=1"]["per_segment"]
        print(f"\n  THE FALSIFIER: {f1} construct(s) with k = 1 where per_segment is strictly better"
              + (", the k = 1 branch is REFUTED on this condition" if f1
                 else ", not produced on this condition"))
        print(f"  the k >= {args.min_segments} branch: "
              f"{cs[f'k>={args.min_segments}']['whole']} construct(s) where whole beats per_segment "
              f"(0 is the claim)")

        fal = falsifiers(rows, args.min_segments)
        if fal:
            print(f"\n  {'construct':<42}{'payload':<9}{'f':>7}{'cov':>7}{'whole':>7}{'perSeg':>8}"
                  f"{'oracle':>8}{'iters':>7}{'bestHitEmitted':>16}")
            for r in sorted(fal, key=lambda r: r["plasmid_id"]):
                print(f"    {r['plasmid_id'][:40]:<42}{r['payload']:<9}{r['f']:>7.3f}"
                      f"{(r['cov'] if r['cov'] is not None else float('nan')):>7.3f}"
                      f"{r['rank_whole']:>7}{r['rank_per_segment']:>8}{r['rank_oracle']:>8}"
                      f"{r['bhs_iterations']:>7}{r['best_hit_emitted']:>16}")

        print(f"\n  by f stratum: the k = 1 cell, never pooled (26.3):")
        print(f"    {'f stratum':<12}{'n':>5}{'k=1':>6}{'k=1 whole wins':>16}"
              f"{'k=1 per_seg wins':>18}{'k>=2 whole wins':>17}")
        for s in STRATUM_NAMES:
            rs = [r for r in rows if stratum(r["f"]) == s]
            if not rs:
                continue
            c = census(rs, args.min_segments)
            print(f"    {s:<12}{len(rs):>5}{c['k=1']['n']:>6}{c['k=1']['whole']:>16}"
                  f"{c['k=1']['per_segment']:>18}{c[f'k>={args.min_segments}']['whole']:>17}")

        rp = branch_rank_points(rows, args.min_segments)
        print(f"\n  the k = 1 branch in rank points, against decomposing unconditionally:"
              f" saved {rp['saved']:,}, lost {rp['lost']:,}, net {rp['net']:+,}")
        print(f"  payloads the branch helps: {cs['k=1']['payloads_whole']}")
        print(f"  payloads the branch hurts: {cs['k=1']['payloads_per_segment']}")

        print(f"\n  the arms and the gates by stratum, mean / MRR / recall@10 (26.3):")
        arms = {"whole (always)": lambda r: r["rank_whole"],
                "per_segment (always)": lambda r: r["rank_per_segment"],
                f"GATED at k>={args.min_segments}": lambda r: k_gated_rank(r, args.min_segments),
                "oracle router (label)": lambda r: min(r["rank_whole"], r["rank_per_segment"]),
                "oracle span (label)": lambda r: r["rank_oracle"]}
        print(f"    {'f stratum':<12}{'arm':<26}{'n':>5}{'mean':>10}{'med':>7}{'MRR':>7}{'r@10':>8}"
              f"{'macro mean':>12}")
        for s in STRATUM_NAMES:
            rs = [r for r in rows if stratum(r["f"]) == s]
            if not rs:
                continue
            for name, pick in arms.items():
                m = metrics([pick(r) for r in rs])
                print(f"    {s:<12}{name:<26}{m['n']:>5}{m['mean']:>10}{m['median']:>7}{m['mrr']:>7}"
                      f"{m['r10']:>8}{macro_mean(rs, pick):>12}")
            print()

        ep = emitted_probe(rows, args.min_segments)
        print(f"  IN-SAMPLE probe, not a rule: `best_hit_emitted == 0` fires on {ep['fired']} of "
              f"{ep['k1']} k = 1 constructs, {ep['fired_and_falsifying']} of them falsifying; "
              f"it misses {len(ep['missed'])} of {ep['falsifying']} falsifiers {ep['missed']}\n")

    if len(loaded) > 1:
        print("=== the bracket " + "=" * 58)
        print("  this project's working rules: a constructed condition that overshoots is a BOUND, and the real case is"
              "\n  bracketed between conditions rather than fitted to one of them.")
        print(f"    {'condition':<18}{'k=1 n':>7}{'falsifiers':>12}{'k>=2 whole wins':>17}"
              f"{'branch net rank points':>24}")
        for key, label, rows in loaded:
            cs = census(rows, args.min_segments)
            rp = branch_rank_points(rows, args.min_segments)
            print(f"    {key + ' ' + label.split()[0]:<18}{cs['k=1']['n']:>7}"
                  f"{cs['k=1']['per_segment']:>12}"
                  f"{cs[f'k>={args.min_segments}']['whole']:>17}{rp['net']:>+24,}")
        nets = [branch_rank_points(r, args.min_segments)["net"] for _, _, r in loaded]
        fals = [census(r, args.min_segments)["k=1"]["per_segment"] for _, _, r in loaded]
        print(f"\n  the falsifier fires on {sum(1 for f in fals if f)} of {len(fals)} conditions, so "
              f"it is {'NOT ' if sum(1 for f in fals if f) > 1 else ''}an artefact of one condition's "
              f"extremity.")
        print(f"  the branch's net rank points run {min(nets):+,} to {max(nets):+,} across the "
              f"bracket, so its SIGN"
              + (" CHANGES and the branch's value is bracketed, not settled."
                 if min(nets) < 0 < max(nets) else " is the same at both ends of the bracket."))

    if args.out_dir:
        write_csvs(args.out_dir, loaded, args.min_segments)
        print(f"\nwrote {args.out_dir}/census.csv and falsifiers.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
