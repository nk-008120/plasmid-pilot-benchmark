#!/usr/bin/env python3
"""Put the three tiers' twilight results side by side, and draw the synthetic dose-response.

    python scripts/17_compare_tiers.py --results results/baseline2 --results results/corpus \\
        --results results/synthetic --scorer sw-blosum62-11-1

Reads every `twilight_<scorer>_<tier>_summary.json` it is pointed at. The point of the comparison is not
that the tiers agree, they measure different things on purpose, but that each one's weakness is
covered by another's strength:

* **hand-audited (n=23)**: real constructs, ground truth from the depositor, too small to resolve anything.
* **corpus (n~410)**: statistical power and breadth, labels derived by alignment and 80% validated, but
  with no 2A cassettes at all and only 33 tagged fusions (the working record 16.6).
* **synthetic (n=336)**: the architectures the corpus lacks, exact boundaries, dilution swept rather than
  observed, and artificial, so it proves nothing on its own about real deposits.

A claim that holds on all three is worth more than a better number on any one of them.
"""
from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pilot.figstyle import (METRICS, PRIMARY_METRICS, arm_display, arms_present, legend_note,
                            missing_metrics, series, thresholds_of)

TIER_STYLE = {"hand": ("#333333", "-o"), "corpus": ("#2f6fb3", "-s"), "synthetic": ("#b3402f", "-^"),
              # the real-filler rebuild: a fourth tier, and the only one whose
              # fillers compete in the catalogue, so it must not fall back to the grey default
              # and read as "some other run"
              "synth_real": ("#7a4fb3", "-D")}

#: `fig_d5_tiers` is coloured by TIER, so it cannot also colour by arm. It shows the three arms that
#: carry the argument, the status quo, the best label-free arm, and the ceiling, as one faint, one
#: dashed and one solid line per tier. Every arm is in `fig_d10_tiers_metrics` instead.
D5_ARMS = (("whole", 0.35, 2.6, "-"), ("bhs_per_segment", 0.95, 1.6, "--"), ("oracle", 1.0, 1.8, "-"))
D5_LABEL = {"whole": "ORF as called", "bhs_per_segment": "best-hit segments, per-segment",
            "oracle": "oracle span"}


# Which palette the figures use. A one-element list rather than a bare module global because `--mono`
# is parsed in `main` and read inside helpers: it is the smallest thing that lets one choice reach all
# of them without threading an argument through every drawing function. `figstyle.arm_display` is the
# only place the two palettes are defined, so the encoding cannot differ between them.
MONO = [False]

def discover_summaries(dirs: list[Path], scorer: str) -> tuple[dict[str, tuple[Path, dict]], list[Path]]:
    """Tiered summaries keyed by tier, plus the untiered files that were deliberately left out.

    `results/baseline2` still holds `twilight_<scorer>_summary.json` from the section 15.13 run, which
    predates `--tier` and was scored against a different (8,328-protein) reference. A glob on the scorer
    name matches it, and it then appears alongside the three tiers as a fourth line and a fourth curve,
    comparing numbers that were never comparable. Requiring a tier label is what keeps that out.
    """
    found: dict[str, tuple[Path, dict]] = {}
    skipped: list[Path] = []
    for d in dirs:
        for p in sorted(d.glob(f"twilight_{scorer}*_summary.json")):
            agg = json.loads(p.read_text(encoding="utf-8"))
            tier = (agg.get("tier") or "").strip()
            if not tier:
                skipped.append(p)
                continue
            found[tier] = (p, agg)
    return found, skipped


def fig_tier_metrics(found: dict[str, dict], out: Path) -> None:
    """Every arm, every tier, in the metrics this project made primary.

    Rows are mean rank, MRR and recall@10; columns are tier x (micro, macro). Here colour is the ARM
    rather than the tier, the tier is the column, so a reader compares aggregations down a column
    and tiers across a row. The pair of lines in one colour is the same segments under the two
    aggregations, which is the comparison sections 24.3, 25.5 and 28.1 rest on.

    Top-1 is not a row: it is `fig_d5_tiers`. A tier whose summary predates a metric gets an empty
    panel and a printed reason, never a line at zero.
    """
    tiers = sorted(found)
    cols = [(t, macro) for t in tiers for macro in (False, True)]
    fig, axes = plt.subplots(len(PRIMARY_METRICS), len(cols), squeeze=False,
                             figsize=(4.6 * len(cols), 3.2 * len(PRIMARY_METRICS)))
    handles: list = []
    labels: list[str] = []
    for j, (tier, macro) in enumerate(cols):
        agg = found[tier]
        thr = thresholds_of(agg)
        xs = [t * 100 for t in thr]
        arms = arms_present(agg)
        gone = missing_metrics(agg, macro=macro)
        if gone:
            print(f"    {tier} [{'macro' if macro else 'micro'}]: this summary carries no "
                  f"{', '.join(gone)}: panel left empty, not drawn flat "
                  f"(re-run 10_twilight.py on this tier to fill it)")
        for i, mkey in enumerate(PRIMARY_METRICS):
            ax, m = axes[i][j], METRICS[mkey]
            for arm in arms:
                st = arm_display(MONO[0])[arm]
                kw = dict(color=st.colour, lw=st.lw, ms=st.ms, label=st.label)
                if st.hollow:
                    kw.update(markerfacecolor="white", markeredgecolor=st.colour, markeredgewidth=1.4)
                line, = ax.plot(xs, series(agg, arm, mkey, thr, macro=macro), st.fmt, **kw)
                if st.label not in labels:
                    handles.append(line)
                    labels.append(st.label)
            ax.set_xlim(103, 22)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{int(x)}%" for x in xs])
            ax.grid(alpha=0.25, lw=0.6)
            # Only scale an axis that has something on it: inverting an empty log axis asks
            # matplotlib to set a non-positive ylim, which it warns about and ignores.
            if mkey not in gone:
                if m.log:
                    ax.set_yscale("log")
                    ax.invert_yaxis()
                elif mkey == "recall10":
                    ax.set_ylim(-4, 104)
                else:
                    ax.set_ylim(-0.04, 1.04)
            if j == 0:
                ax.set_ylabel(m.label, fontsize=8.5)
            if i == 0:
                ax.set_title(f"{tier}: {'macro, per payload' if macro else 'micro, per construct'}",
                             fontsize=10, loc="left")
            if i == len(PRIMARY_METRICS) - 1:
                ax.set_xlabel("holdout", fontsize=8.5)
            if mkey in gone:
                # nothing was plotted, so matplotlib's autoscaled decade ticks are noise that reads
                # like data; blank them and leave the panel's label and reason as the whole content
                ax.text(0.5, 0.5, f"no {mkey} in this summary", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="#999999")
                ax.set_yticks([])
    fig.suptitle("Every arm, every tier, in the primary metrics ", fontsize=12, x=0.0,
                 ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if handles:
        fig.legend(handles, labels, fontsize=7.6, frameon=False, ncol=min(3, len(handles)),
                   loc="lower center", bbox_to_anchor=(0.5, 0.01))
    fig.text(0.0, -0.012, textwrap.fill(legend_note(MONO[0]), 104), fontsize=7.4, color="#777777",
             ha="left", va="top")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_d10_tiers_metrics.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/fig_d10_tiers_metrics.png / .pdf")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mono", action="store_true",
                    help="draw in greyscale. The encoding is unchanged: grey value "
                         "replaces hue as the channel saying which segments a line is drawn "
                         "from, and line style still says how they were aggregated. For print, "
                         "for photocopies, and for readers who cannot separate the oracle red "
                         "from the bhs green.")
    ap.add_argument("--results", type=Path, action="append", required=True)
    ap.add_argument("--scorer", default="sw-blosum62-11-1")
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    args = ap.parse_args(argv)
    MONO[0] = bool(args.mono)

    found, skipped = discover_summaries(args.results, args.scorer)
    for tier, (p, _) in sorted(found.items()):
        print(f"    {tier:<14}{p}")
    for p in skipped:
        print(f"    {'(no tier)':<14}{p}  skipped")
    found = {t: a for t, (_, a) in found.items()}
    if not found:
        print(f"no tiered twilight summaries for scorer {args.scorer!r} in the given directories")
        return 2

    thresholds = sorted({float(k) for a in found.values() for k in a["by_threshold"]}, reverse=True)
    # One block per tier, every arm that tier carries. This used to print `whole` and `oracle` as two
    # literals, which is why a tier gaining an arm changed nothing visible here. Mean rank leads, per
    # the working record; top-1 is kept as the last column and is never the only number shown.
    for tier in sorted(found):
        agg = found[tier]
        arms = arms_present(agg)
        print(f"\n{tier}  ({', '.join(arms)})")
        print(f"{'holdout':>9}    {'arm':<31}{'mean rank':>11}{'MRR':>8}{'recall@10':>11}{'top-1':>12}")
        for thr in thresholds:
            d = agg["by_threshold"].get(f"{thr:.2f}")
            if not d:
                continue
            for arm in arms:
                mr = d.get(f"{arm}_mean_rank")
                mrr = d.get(f"{arm}_mrr")
                n = d.get(f"{arm}_n") or 0
                print(f"{thr:>8.0%}    {arm_display(MONO[0])[arm].label:<31}"
                      f"{'-' if mr is None else format(mr, '.2f'):>11}"
                      f"{'-' if mrr is None else format(mrr, '.3f'):>8}"
                      f"{(100 * d[f'{arm}_top10'] / n if n else 0):>10.1f}%"
                      f"{d[f'{arm}_top1']:>7}/{n:<5}")

    args.out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for tier in sorted(found):
        colour, style = TIER_STYLE.get(tier, ("#777777", "-o"))
        agg = found[tier]
        thr = thresholds_of(agg)
        xs = [t * 100 for t in thr]
        have = arms_present(agg)
        for arm, alpha, lw, ls in D5_ARMS:
            if arm not in have:
                continue
            ys = series(agg, arm, "top1", thr)
            ax.plot(xs, ys, style[1:], ls=ls, color=colour, lw=lw, ms=5, alpha=alpha,
                    markerfacecolor="white" if arm == "bhs_per_segment" else colour,
                    label=f"{tier}: {D5_LABEL[arm]}")
    ax.set_xlim(103, 22)
    ax.set_ylim(-4, 104)
    ax.set_xticks([t * 100 for t in thresholds])
    ax.set_xticklabels([f"{int(t * 100)}%" for t in thresholds])
    ax.set_xlabel("holdout: nothing above this identity is in the catalogue")
    ax.set_ylabel("payload's family retrieved at rank 1 (%)")
    # "all three tiers" was a literal until the real-filler rebuild made it four. The count comes from
    # the data for the same reason the arms do: a hardcoded number in a title is a caption that lies.
    ax.set_title(f"The decomposition gap across {len(found)} tiers: {', '.join(sorted(found))}",
                 fontsize=11, loc="left", pad=26)
    ax.text(0.0, 1.055, "faint = ORF as called, dashed = best-hit segments ranked one at a time "
            "(label-free), solid = oracle payload span", transform=ax.transAxes, fontsize=7.8,
            color="#777777", va="bottom")
    ax.text(0.0, 1.015, "top-1 is a column, not a headline: the primary metrics are in "
            "fig_d10_tiers_metrics", transform=ax.transAxes, fontsize=7.8, color="#777777", va="bottom")
    ax.grid(alpha=0.25, lw=0.6)
    # one column per arm, not per tier: with four tiers the per-tier layout ran off the axes to the
    # right and `bbox_inches="tight"` widened the whole canvas to follow it
    ax.legend(fontsize=7.2, ncol=len(D5_ARMS), frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"fig_d5_tiers.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {args.out}/fig_d5_tiers.png / .pdf")

    fig_tier_metrics(found, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
