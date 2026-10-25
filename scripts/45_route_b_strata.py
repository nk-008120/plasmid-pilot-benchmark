"""Report route B's retrieval stratified by payload fraction f, the only form this project's working rules allows.

the working record, decided: mean rank and MRR are primary and must be **stratified by f**, with
recall@10 as the operational secondary, top-1 as a column and **no pooled scalar across strata**. The
runner in `10_twilight.py` does not know f, it is computed by `43_route_b_f.py` from `build_arms`
before any retrieval happens, so the join lives here. f uses the label and can therefore never be
part of a screening rule; it is a description of the benchmark item.

Route B is the first tier where this stratification has much to work with. The corpus puts 86% of its
mass at f >= 0.95, so a corpus stratum below 0.70 holds 15 constructs; route B puts 29 of 46 below
0.70. Both route B passes are read, because one manifest row per file cannot carry a
construct's second payload and three pairs went to a second pass (`44_route_b_bench.py`).

The micro/macro split is not optional either (27.4): the 46 pairs carry 19 distinct payloads, several
of them the same cassette in a different backbone, so constructs are not independent trials and the
macro figure governs.
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F_STRATA = (0.50, 0.70, 0.95)
STRATUM_NAMES = ("<0.50", "0.50-0.70", "0.70-0.95", ">=0.95")
REPORT_ARMS = ("whole", "auto", "bhs", "oracle", "auto_per_segment", "bhs_per_segment")


def stratum(f: float) -> str:
    if f < F_STRATA[0]:
        return STRATUM_NAMES[0]
    if f < F_STRATA[1]:
        return STRATUM_NAMES[1]
    if f < F_STRATA[2]:
        return STRATUM_NAMES[2]
    return STRATUM_NAMES[3]


def read_f_by_pair(paths) -> dict:
    """(source_id, payload_accession) -> f, from the benchmark manifests."""
    out = {}
    for p in paths:
        with open(p, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r.get("f"):
                    out[(r["source_id"], r["payload_accession"])] = float(r["f"])
    return out


def read_ranks(paths) -> list:
    rows = []
    for p in paths:
        with open(p, newline="", encoding="utf-8-sig") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def source_id_of(plasmid_id: str) -> str:
    """`addgene-plasmid-176139-sequence-353012` -> `addgene_176139`, the manifest's source_id."""
    parts = plasmid_id.split("-")
    return "addgene_" + parts[2] if len(parts) > 2 else plasmid_id


def summarise(ranks) -> dict:
    """Mean rank, MRR and recall@10, plus the >=10,000 count 32.2 requires beside any mean."""
    return {"n": len(ranks),
            "mean_rank": round(statistics.fmean(ranks), 2),
            "median_rank": statistics.median(ranks),
            "mrr": round(statistics.fmean(1.0 / r for r in ranks), 3),
            "recall_at_10": round(sum(1 for r in ranks if r <= 10) / len(ranks), 3),
            "top1": sum(1 for r in ranks if r == 1),
            "n_ge_10000": sum(1 for r in ranks if r >= 10_000)}


def macro(by_payload: dict) -> dict:
    """Average the per-payload means, so one cassette in four backbones counts once."""
    per = {a: statistics.fmean(rs) for a, rs in by_payload.items()}
    mrr = {a: statistics.fmean(1.0 / r for r in rs) for a, rs in by_payload.items()}
    rec = {a: sum(1 for r in rs if r <= 10) / len(rs) for a, rs in by_payload.items()}
    return {"n_payloads": len(per),
            "mean_rank": round(statistics.fmean(per.values()), 2),
            "mrr": round(statistics.fmean(mrr.values()), 3),
            "recall_at_10": round(statistics.fmean(rec.values()), 3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank-csv", type=Path, nargs="+", default=[
        ROOT / "results/route_b_pilot/twilight_sw-blosum62-11-1_route_b.csv",
        ROOT / "results/route_b_pilot/twilight_sw-blosum62-11-1_route_b_second.csv"])
    ap.add_argument("--manifest", type=Path, nargs="+", default=[
        ROOT / "data/manifest_route_b.csv", ROOT / "data/manifest_route_b_second.csv"])
    ap.add_argument("--thresholds", nargs="+", default=["0.25", "0.5", "1.0"])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/route_b_strata")
    args = ap.parse_args(argv)

    f_by_pair = read_f_by_pair(args.manifest)
    rows = read_ranks(args.rank_csv)
    scored = [r for r in rows if r["best_target_rank"] != ""]
    missing = sorted({(source_id_of(r["plasmid_id"]), r["payload_accession"]) for r in scored
                      if (source_id_of(r["plasmid_id"]), r["payload_accession"]) not in f_by_pair})

    print("route B retrieval, stratified by payload fraction f ")
    print("  rank rows: {} ({} scorable) over {} pass(es)".format(
        len(rows), len(scored), len(args.rank_csv)))
    print("  f known for {} (deposit, payload) pairs".format(len(f_by_pair)))
    print("  BINDING, pairs with a rank but no f: {} {}".format(len(missing), missing))
    counts = defaultdict(set)
    for r in scored:
        key = (source_id_of(r["plasmid_id"]), r["payload_accession"])
        if key in f_by_pair:
            counts[stratum(f_by_pair[key])].add(key)
    print("  stratum sizes (pairs): "
          + "  ".join("{} {}".format(s, len(counts[s])) for s in STRATUM_NAMES))
    print("  NEVER pool these rows into one scalar; 26.3 forbids it, and on this tier the strata "
          "do not even order the arms the same way.\n")

    out = {"thresholds": {}}
    for thr in args.thresholds:
        at = [r for r in scored if r["threshold"] == thr]
        print("=== holdout {} ".format(thr) + "=" * 74)
        print("{:<12}{:<20}{:>4}{:>9}{:>6}{:>7}{:>8}{:>6}{:>7} | {:>8}{:>9}{:>7}{:>8}".format(
            "f stratum", "arm", "n", "mean", "med", "MRR", "r@10", "top1", ">=1e4",
            "macro n", "mean", "MRR", "r@10"))
        out["thresholds"][thr] = {}
        for s in STRATUM_NAMES:
            rows_s = [r for r in at
                      if (source_id_of(r["plasmid_id"]), r["payload_accession"]) in f_by_pair
                      and stratum(f_by_pair[(source_id_of(r["plasmid_id"]),
                                             r["payload_accession"])]) == s]
            if not rows_s:
                continue
            out["thresholds"][thr][s] = {}
            for arm in REPORT_ARMS:
                got = [r for r in rows_s if r["arm"] == arm]
                if not got:
                    continue
                ranks = [int(r["best_target_rank"]) for r in got]
                by_payload = defaultdict(list)
                for r in got:
                    by_payload[r["payload_accession"]].append(int(r["best_target_rank"]))
                mi, ma = summarise(ranks), macro(by_payload)
                out["thresholds"][thr][s][arm] = {"micro": mi, "macro": ma}
                print("{:<12}{:<20}{:>4}{:>9}{:>6}{:>7}{:>8}{:>6}{:>7} | "
                      "{:>8}{:>9}{:>7}{:>8}".format(
                          s, arm, mi["n"], mi["mean_rank"], mi["median_rank"], mi["mrr"],
                          mi["recall_at_10"], mi["top1"], mi["n_ge_10000"],
                          ma["n_payloads"], ma["mean_rank"], ma["mrr"], ma["recall_at_10"]))
            print()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "route_b_strata.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote {}".format(args.out_dir / "route_b_strata.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
