#!/usr/bin/env python3
"""The decomposition gap against catalogue size, on top-1 and on rank-based measures.

    python scripts/23_size_curve.py

**Why not top-1 alone.** Top-1 over 19 constructs moves in steps of 5.3 percentage points, and across the
four clustered catalogues measured so far it sat flat at 9/19 and 15/19 for three of them while the
underlying ranks were plainly moving. A measure that cannot resolve anything smaller than one construct is
the wrong instrument for a decay curve. So this reports, per arm and holdout:

* **top-1** (what a screening service that looks at one answer sees)
* **MRR**, the mean of 1/rank, which rewards getting closer without reaching rank 1
* **median rank**, which is robust to the few constructs that rank in the thousands
* **mean rank improvement**, the mean of (whole rank - oracle rank), which is the decomposition gap
  measured in places rather than in constructs

**Two series, deliberately not pooled.** The `size_*` catalogues share one 10,286-protein core and differ
only by uniformly sampled Swiss-Prot decoys, so they isolate size. The others grew by absorbing other
payloads' homolog families, so they vary size *and* the density of competing near-relatives. They are
plotted as separate markers because the difference between them is the finding, not noise.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

RANDOM = "random decoys"
CLUSTERED = "clustered families"
HOLDOUTS = (1.00, 0.90, 0.70, 0.50, 0.35, 0.25)  # compared as floats: the CSV writes 0.5, not 0.50


def load(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def arm_stats(rows: list[dict], thr: str, arm: str) -> dict:
    got = [r for r in rows if float(r["threshold"]) == thr and r["arm"] == arm and r["best_target_rank"] != ""]
    if not got:
        return {}
    ranks = np.array([int(r["best_target_rank"]) for r in got], dtype=float)
    per: dict[str, list[float]] = {}
    for r, rank in zip(got, ranks):
        per.setdefault(r["payload_accession"], []).append(1.0 if rank == 1 else 0.0)
    return {"n": len(got), "top1": int((ranks == 1).sum()),
            "top1_macro": round(100 * float(np.mean([np.mean(v) for v in per.values()])), 1),
            "mrr": round(float(np.mean(1.0 / ranks)), 4),
            "median_rank": float(np.median(ranks)), "mean_rank": round(float(ranks.mean()), 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results/baseline2"))
    ap.add_argument("--scorer", default="sw-blosum62-11-1")
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    args = ap.parse_args(argv)

    # catalogue sizes that are not encoded in the tier name
    known = {"hand_sw": 8337, "hand_sw_merged": 34159, "aud15k": 15205, "aud100k": 100685}
    points: list[dict] = []
    for p in sorted(args.results_dir.glob(f"twilight_{args.scorer}_*.csv")):
        tier = p.stem.replace(f"twilight_{args.scorer}_", "")
        m = re.fullmatch(r"size(\d+)", tier)
        if m:
            size, series = int(m.group(1)), RANDOM
        elif tier in known:
            size, series = known[tier], CLUSTERED
        else:
            continue
        rows = load(p)
        points.append({"tier": tier, "size": size, "series": series,
                       "by_threshold": {f"{t:.2f}": {a: arm_stats(rows, t, a) for a in ("whole", "auto", "oracle")}
                                        for t in HOLDOUTS},
                       "rows": rows})
    if not points:
        print("no twilight CSVs matched")
        return 2
    points.sort(key=lambda d: (d["series"], d["size"]))

    for series in (RANDOM, CLUSTERED):
        sel = [d for d in points if d["series"] == series]
        if not sel:
            continue
        print(f"\n{'=' * 118}\n{series.upper()}\n{'=' * 118}")
        for thr in HOLDOUTS:
            print(f"\n  holdout {float(thr):.0%}")
            print(f"        {'catalogue':>10}{'n':>5}" + "".join(
                f"{a + ' top1':>13}{a + ' MRR':>12}{a + ' medR':>11}" for a in ("whole", "oracle"))
                + f"{'gap top1':>10}{'gap MRR':>10}{'mean rank gain':>16}")
            for d in sel:
                w, o = d["by_threshold"][f"{thr:.2f}"]["whole"], d["by_threshold"][f"{thr:.2f}"]["oracle"]
                if not w or not o:
                    continue
                pair = {(r["plasmid_id"], r["arm"]): r["best_target_rank"]
                        for r in d["rows"] if float(r["threshold"]) == thr}
                gains = [int(pair[(pid, "whole")]) - int(pair[(pid, "oracle")])
                         for pid in {k[0] for k in pair}
                         if pair.get((pid, "whole")) not in ("", None)
                         and pair.get((pid, "oracle")) not in ("", None)]
                print(f"        {d['size']:>10,}{w['n']:>5}"
                      f"{w['top1']:>8}/{w['n']:<4}{w['mrr']:>12.4f}{w['median_rank']:>11.0f}"
                      f"{o['top1']:>8}/{o['n']:<4}{o['mrr']:>12.4f}{o['median_rank']:>11.0f}"
                      f"{o['top1'] - w['top1']:>+10}{o['mrr'] - w['mrr']:>+10.4f}"
                      f"{(float(np.mean(gains)) if gains else float('nan')):>16.1f}")

    out = {"series": [{k: v for k, v in d.items() if k != "rows"} for d in points]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "size_curve.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("\nmatplotlib not installed; skipping the figure")
        return 0

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for ax, (key, label) in zip(axes, (("top1", "oracle - whole, constructs at rank 1"),
                                       ("mrr", "oracle - whole, MRR"))):
        for series, colour, marker in ((RANDOM, "#2f6fb3", "-o"), (CLUSTERED, "#b3402f", "--s")):
            sel = [d for d in points if d["series"] == series]
            xs, ys = [], []
            for d in sel:
                w, o = d["by_threshold"]["0.25"]["whole"], d["by_threshold"]["0.25"]["oracle"]
                if not w or not o:
                    continue
                xs.append(d["size"])
                ys.append(o[key] - w[key] if key == "mrr" else o["top1"] - w["top1"])
            if xs:
                ax.plot(xs, ys, marker, color=colour, ms=6, lw=1.8, label=series)
        ax.set_xscale("log")
        ax.set_xlabel("catalogue size (proteins, log scale)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, lw=0.6)
        ax.axhline(0, color="#999999", lw=0.8)
    axes[0].set_title("The decomposition gap at the 25% holdout", fontsize=11, loc="left", pad=18)
    axes[0].text(0.0, 1.02, "solid = size varied with random decoys; dashed = catalogues that also "
                 "added competing homolog families", transform=axes[0].transAxes, fontsize=7.8,
                 color="#777777", va="bottom")
    axes[1].legend(fontsize=8, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"fig_d8_gap_vs_size.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {args.out}/fig_d8_gap_vs_size.png / .pdf and size_curve.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
