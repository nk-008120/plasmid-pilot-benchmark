#!/usr/bin/env python3
"""Figures for the decomposition result.

    python scripts/07_figures.py --results-dir results/baseline2 --embedder kmer3-1024

Reads `decompose_<embedder>.csv` and `_summary.json` from a `06_decompose.py` run and writes PNG + PDF to
`<results-dir>/figures/`. PDF as well as PNG because the proposal is a print document and a rasterised
scatter plot at 300 dpi still looks like a rasterised scatter plot.
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
import numpy as np

from pilot.figstyle import (METRICS, PRIMARY_METRICS, arm_display, arms_present, legend_note,
                            missing_metrics, series, thresholds_of)

WHOLE, AUTO, ORACLE = "#b4b4b4", "#2f6fb3", "#b3402f"
# References that are a long polyprotein rather than the mature protein; cosine against these tracks query
# length, so they are shown but excluded from the dose-response fit. Flagged in the manifest before the run.
POLYPROTEIN_REF = {"P04517"}


# Which palette the figures use. A one-element list rather than a bare module global because `--mono`
# is parsed in `main` and read inside helpers: it is the smallest thing that lets one choice reach all
# of them without threading an argument through every drawing function. `figstyle.arm_display` is the
# only place the two palettes are defined, so the encoding cannot differ between them.
MONO = [False]

def _save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png / .pdf")


def short(pid: str) -> str:
    """'addgene_73037_pCDH_Nluc_P2A_copGFP_T2A_Puro' -> '#73037'."""
    parts = pid.split("_")
    return f"#{parts[1]}" if len(parts) > 1 and parts[1].isdigit() else pid[:12]


def fig_dilution(rows: list[dict], out: Path) -> None:
    """Dose-response: how much of the payload's signal the fusion costs, against how diluted it is.

    One column per plasmid: grey = the ORF as called, red = the oracle payload span. The vertical span
    between them is the recoverable loss, and it grows with dilution. This is the plot that turns the
    project's premise from an assertion into a measurement.
    """
    rows = sorted(rows, key=lambda r: float(r["dilution"]))
    d = np.array([float(r["dilution"]) for r in rows])
    w = np.array([float(r["whole_cos"]) for r in rows])
    o = np.array([float(r["oracle_cos"]) for r in rows])
    a = np.array([float(r["auto_cos"]) for r in rows])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    for x, y0, y1 in zip(d, w, o):
        ax.plot([x, x], [y0, y1], color="#cccccc", lw=1.1, zorder=1)
    ax.scatter(d, w, s=46, color=WHOLE, edgecolor="#7a7a7a", lw=0.7, zorder=3, label="ORF as called")
    ax.scatter(d, a, s=26, marker="D", color=AUTO, zorder=4, label="motif decomposition")
    ax.scatter(d, o, s=46, color=ORACLE, zorder=5, label="oracle payload span")
    # label only the fusions; alternate the offset so the cluster around 0.28 stays readable
    lab = [(x, y, r) for x, y, r in zip(d, o, rows) if float(r["dilution"]) > 0.25]
    for i, (x, y, r) in enumerate(lab):
        ax.annotate(short(r["plasmid_id"]), (x, y), textcoords="offset points",
                    xytext=(6, 5 if i % 2 == 0 else -11), fontsize=7.5, color="#444444")
    ax.set_xlabel("dilution: fraction of the called ORF that is not the payload")
    ax.set_ylabel("cosine similarity to the correct protein")
    ax.set_title("Fusion dilutes the payload's signal", fontsize=11, loc="left")
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    ax.grid(alpha=0.25, lw=0.6)

    loss = o - w
    # #171782's reference is the whole 3,054 aa TEV genome polyprotein, so cosine against it tracks query
    # length rather than identity and the "recovered" figure goes negative. The manifest flagged that
    # before the run, so it is excluded from the fit and drawn hollow rather than quietly dropped.
    flagged = np.array([len(POLYPROTEIN_REF & {r["payload_accession"]}) > 0 for r in rows])
    keep = ~flagged
    ax2.scatter(d[keep], loss[keep], s=44, color=ORACLE, zorder=3)
    ax2.scatter(d[flagged], loss[flagged], s=44, facecolor="none", edgecolor=ORACLE, lw=1.3, zorder=3)
    xs = np.linspace(0, max(d) * 1.05, 50)
    for mask, style, lbl in ((np.ones_like(keep), ":", "all"), (keep, "--", "excl. polyprotein ref")):
        sl, ic = np.polyfit(d[mask], loss[mask], 1)
        rr = np.corrcoef(d[mask], loss[mask])[0, 1]
        ax2.plot(xs, sl * xs + ic, color="#333333", lw=1.2, ls=style,
                 label=f"{lbl}: slope {sl:.2f}, r = {rr:.2f}")
    ax2.axhline(0, color="#888888", lw=0.8)
    ax2.set_xlabel("dilution")
    ax2.set_ylabel("cosine recovered by decomposition")
    ax2.set_title("The loss is proportional to the dilution", fontsize=11, loc="left")
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(alpha=0.25, lw=0.6)
    _save(fig, out, "fig_d1_dilution_doseresponse")


def fig_ranks(rows: list[dict], out: Path) -> None:
    """Slope chart of the payload's rank through the three arms, on a log axis.

    Only plasmids whose rank changes are labelled; the flat lines at rank 1 are the majority and are the
    point that the rank metric saturates at this embedder.

    These three arms are `decompose_<embedder>.csv`'s columns, from `06_decompose.py`, **not** the
    twilight benchmark's. That run emits no per-segment arm, so there is nothing to add here and the
    literals below are the file's actual schema rather than a hardcoded subset: the per-segment arms
    appear in `fig_d4_twilight_curve` and `fig_d9_twilight_metrics`, which read the twilight summaries.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    arms = ["whole_rank", "auto_rank", "oracle_rank"]
    xs = [0, 1, 2]
    n_moved = 0
    for r in rows:
        y = [int(r[k]) for k in arms]
        moved = len(set(y)) > 1
        ax.plot(xs, y, "-o", ms=5, lw=1.6 if moved else 0.8,
                color="#2f6fb3" if moved else "#d2d2d2",
                zorder=3 if moved else 1, alpha=1 if moved else 0.8)
        if moved:
            ax.annotate(f"{short(r['plasmid_id'])}    {y[0]}->{y[-1]}", (2, y[-1]),
                        textcoords="offset points", xytext=(8, -3), fontsize=8.5, color="#2f6fb3")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    ax.set_xticks(xs)
    ax.set_xticklabels(["ORF as called", "motif\ndecomposition", "oracle\npayload span"])
    ax.set_xlim(-0.25, 3.05)
    ax.set_ylabel("rank of the correct protein (lower is better)")
    ax.set_title(f"Rank moves for {n_moved} of {len(rows)} plasmids at this embedder", fontsize=11, loc="left")
    ax.text(0.0, 1.015, "the rank metric saturates: a k-mer query containing the reference verbatim "
            "already ranks it first", transform=ax.transAxes, fontsize=7.8, color="#777777", va="bottom")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    _save(fig, out, "fig_d2_rank_slopes")


def fig_masking(agg: dict, out: Path) -> None:
    """A second gene inside a fusion ORF is invisible to a whole-construct query.

    Each pair is one segment whose own top hit is a protein that is *not* the construct's payload: a
    gene the whole-ORF embedding averaged away. The screening-relevant direction of the same effect.
    """
    s = agg.get("nonpayload_genes_surfaced_by_decomposition") or []
    if not s:
        print("  (no surfaced genes to plot)")
        return
    fig, ax = plt.subplots(figsize=(7.2, 1.15 * len(s) + 1.9))
    y = np.arange(len(s))
    wr = [d["whole_orf_rank"] for d in s]
    ax.hlines(y, 1, wr, color="#cccccc", lw=2.2, zorder=1)
    ax.scatter(wr, y, s=60, color=WHOLE, edgecolor="#7a7a7a", lw=0.7, zorder=3, label="whole-ORF query")
    ax.scatter([1] * len(s), y, s=60, color=AUTO, zorder=4, label="decomposed segment")
    for i, d in enumerate(s):
        ax.annotate(f"rank {d['whole_orf_rank']}\ncos {d['whole_orf_cos']}", (d["whole_orf_rank"], i),
                    textcoords="offset points", xytext=(0, 13), fontsize=8, color="#555555",
                    ha="center")
        ax.annotate(f"rank 1\ncos {d['segment_cos']}", (1, i), textcoords="offset points",
                    xytext=(0, 13), fontsize=8, color=AUTO, ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{short(d['plasmid_id'])}    {d['accession']}\n{d['segment']} ({d['length_aa']} aa)"
                        for d in s], fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("rank of the second gene among 5,016 reference proteins")
    ax.set_xlim(0.75, max(wr) * 2.2)
    ax.set_ylim(-0.75, len(s) - 0.25)
    ax.set_title("Whole-ORF querying hides co-encoded genes", fontsize=11, loc="left")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    _save(fig, out, "fig_d3_second_gene_masking")


def wrapped_note(width: int = 104) -> str:
    """`legend_note()` broken over lines.

    `_save` uses `bbox_inches="tight"`, which grows the canvas to fit whatever is drawn, so a
    single-line caption under a 5-inch panel silently produced a 14-inch figure.
    """
    return textwrap.fill(legend_note(MONO[0]), width)


def _plot_arm(ax, xs, ys, arm: str) -> None:
    """One arm's line, styled from the single table in `pilot.figstyle`."""
    st = arm_display(MONO[0])[arm]
    kw = dict(color=st.colour, lw=st.lw, ms=st.ms, label=st.label)
    if st.hollow:
        kw.update(markerfacecolor="white", markeredgecolor=st.colour, markeredgewidth=1.4)
    ax.plot(xs, ys, st.fmt, **kw)


def _holdout_axis(ax, xs: list[float]) -> None:
    ax.set_xlim(103, 22)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(x)}%" for x in xs])
    ax.grid(alpha=0.25, lw=0.6)


def fig_twilight(results_dir: Path, out: Path) -> None:
    """Top-1 homolog retrieval as the catalogue is stripped of the payload's close relatives.

    One panel per scorer. The x axis walks left to right into the twilight zone; the vertical gap between
    the grey line (ORF as called) and the red line (oracle payload span) is what decomposition is worth,
    and it is near zero at 100% identity and widest at 25%. That shape is the project's thesis: dilution
    costs little when the match is exact and a great deal when it is remote.

    **The quantity here is top-1, which the working record retired as a headline.** The figure is kept, and
    kept in top-1, for continuity with sections 15 to 21, which are all stated in it; the primary
    metrics are in `fig_d9_twilight_metrics`. What changed is that the arms are no longer three
    literals: every arm the summary carries is drawn, so a new arm cannot go missing here again.
    """
    files = sorted(results_dir.glob("twilight_*_summary.json"))
    if not files:
        print("  (no twilight runs to plot)")
        return
    fig, axes = plt.subplots(1, len(files), figsize=(5.4 * len(files), 4.4), squeeze=False)
    for ax, f in zip(axes[0], files):
        agg = json.loads(f.read_text(encoding="utf-8"))
        thr = thresholds_of(agg)
        xs = [t * 100 for t in thr]
        arms = arms_present(agg)
        for arm in arms:
            _plot_arm(ax, xs, series(agg, arm, "top1", thr), arm)
        _holdout_axis(ax, xs)
        ax.set_ylim(-4, 104)
        ax.set_xlabel("holdout: nothing above this identity is in the catalogue")
        ax.set_ylabel("payload's family retrieved at rank 1 (%)")
        ax.set_title(f"{agg['scorer']}: {len(arms)} arms", fontsize=11, loc="left")
        ax.legend(fontsize=7.5, loc="lower left", frameon=False)
        print(f"    {f.name}: arms drawn -> {', '.join(arms)}")
    fig.suptitle("Walking the query into the twilight zone", fontsize=12, x=0.0, ha="left")
    fig.text(0.0, -0.02, "top-1 is a column, not a headline: the primary metrics are "
                         "in fig_d9_twilight_metrics." + chr(10) + wrapped_note(),
             fontsize=7.4, color="#777777", ha="left", va="top")
    fig.tight_layout()
    _save(fig, out, "fig_d4_twilight_curve")


def fig_twilight_metrics(results_dir: Path, out: Path) -> None:
    """The primary metrics of mean rank and MRR, with recall@10 beside them.

    Rows are metrics, columns are (summary x micro, summary x macro). Micro averages over constructs and
    macro over payloads, and both are always shown, because 410 corpus constructs carry 138 distinct
    payloads and one appears in 37 of them: a micro average is then 37 repetitions of one answer
.

    Mean rank is drawn on an inverted log axis: 1 is the best attainable value, the tail runs to four
    figures, and it is the only quantity in the project that moves monotonically with dilution without
    saturating. Top-1 is deliberately **not** a row here; it is `fig_d4_twilight_curve`.

    A summary written before 17 Oct carries neither `*_mean_rank` nor `*_top10_macro`. Those panels are
    left empty with the reason printed, rather than drawn flat at zero.
    """
    files = sorted(results_dir.glob("twilight_*_summary.json"))
    if not files:
        print("  (no twilight runs to plot)")
        return
    aggs = [(f, json.loads(f.read_text(encoding="utf-8"))) for f in files]
    cols = [(f, agg, macro) for f, agg in aggs for macro in (False, True)]
    fig, axes = plt.subplots(len(PRIMARY_METRICS), len(cols), squeeze=False,
                             figsize=(4.9 * len(cols), 3.3 * len(PRIMARY_METRICS)))
    for j, (f, agg, macro) in enumerate(cols):
        thr = thresholds_of(agg)
        xs = [t * 100 for t in thr]
        arms = arms_present(agg)
        gone = missing_metrics(agg, macro=macro)
        if gone:
            print(f"    {f.name} [{'macro' if macro else 'micro'}]: this summary carries no "
                  f"{', '.join(gone)}: panel left empty, not drawn flat "
                  f"(re-run 10_twilight.py to fill it)")
        for i, mkey in enumerate(PRIMARY_METRICS):
            ax, m = axes[i][j], METRICS[mkey]
            for arm in arms:
                _plot_arm(ax, xs, series(agg, arm, mkey, thr, macro=macro), arm)
            _holdout_axis(ax, xs)
            # Only scale an axis that has something on it: inverting an empty log axis asks
            # matplotlib to set a non-positive ylim, which it warns about and ignores.
            if mkey not in gone:
                if m.log:
                    ax.set_yscale("log")
                    ax.invert_yaxis()  # best (1) at the top, as in fig_d2
                elif mkey == "recall10":
                    ax.set_ylim(-4, 104)
                else:
                    ax.set_ylim(-0.04, 1.04)
            if j == 0:
                ax.set_ylabel(m.label, fontsize=8.5)
            if i == 0:
                ax.set_title(f"{agg.get('tier') or agg['scorer']}: "
                             f"{'macro, per payload' if macro else 'micro, per construct'}",
                             fontsize=10, loc="left")
            if i == len(PRIMARY_METRICS) - 1:
                ax.set_xlabel("holdout: nothing above this identity is in the catalogue", fontsize=8.5)
            if mkey in gone:
                # nothing was plotted, so matplotlib's autoscaled decade ticks are noise that reads
                # like data; blank them and leave the panel's label and reason as the whole content
                ax.text(0.5, 0.5, f"no {mkey} in this summary", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="#999999")
                ax.set_yticks([])
    fig.suptitle("Mean rank and MRR are the primary metrics ", fontsize=12, x=0.0,
                 ha="left")
    # One figure-level legend under the panels rather than one inside the first: with six arms it is
    # large enough to sit on top of the data, and the mean-rank panel is the densest in the figure.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    if handles:
        fig.legend(handles, labels, fontsize=7.6, frameon=False, ncol=min(3, len(handles)),
                   loc="lower center", bbox_to_anchor=(0.5, 0.012))
    fig.text(0.0, -0.012, wrapped_note(), fontsize=7.4, color="#777777", ha="left", va="top")
    # The 100% column is not a holdout: the only target is the payload's own catalogue entry, so a
    # segment arm is querying with part of an exact match and is expected to rank it worse than the
    # whole ORF does. It is a real measurement and it sets the top of the log axis, so it is explained
    # here rather than clipped: this project's "level at 100% identity" is the same fact.
    fig.text(0.0, -0.045, "the 100% column is not a holdout (the target is the payload's own entry), "
                          "so a segment arm" + chr(10) + "queries with part of an exact match; that "
                          "point is real and sets the top of the rank axis",
             fontsize=7.4, color="#777777", ha="left", va="top")
    _save(fig, out, "fig_d9_twilight_metrics")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mono", action="store_true",
                    help="draw in greyscale. The encoding is unchanged: grey value "
                         "replaces hue as the channel saying which segments a line is drawn "
                         "from, and line style still says how they were aggregated. For print, "
                         "for photocopies, and for readers who cannot separate the oracle red "
                         "from the bhs green.")
    ap.add_argument("--results-dir", type=Path, default=Path("results/baseline2"))
    ap.add_argument("--embedder", default="kmer3-1024", help="the embedder *name*, as it appears in the filename")
    ap.add_argument("--out", type=Path, default=None,
                    help="where the figures go; default <results-dir>/figures. Point it at a new "
                         "directory to redraw a finished run without writing into it.")
    args = ap.parse_args(argv)
    MONO[0] = bool(args.mono)

    stem = args.embedder.replace("/", "_")
    csv_path = args.results_dir / f"decompose_{stem}.csv"
    out = args.out or (args.results_dir / "figures")
    # The twilight figures read `twilight_*_summary.json` and nothing else, so a directory holding only
    # a `10_twilight.py` run is a legitimate input. It used to be rejected, which is why the benchmark's
    # own results folders could not be plotted without first copying a `06_decompose.py` output in.
    twilight_only = not csv_path.is_file()
    if twilight_only:
        if not sorted(args.results_dir.glob("twilight_*_summary.json")):
            print(f"not found: {csv_path} and no twilight_*_summary.json either: "
                  f"run scripts/06_decompose.py or scripts/10_twilight.py first")
            return 2
        print(f"no {csv_path.name}: drawing the twilight figures only -> {out}/")
    else:
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            rows = [r for r in csv.DictReader(fh) if r["whole_cos"]]
        agg = json.loads(
            (args.results_dir / f"decompose_{stem}_summary.json").read_text(encoding="utf-8"))
        print(f"{len(rows)} scored plasmids -> {out}/")
        fig_dilution(rows, out)
        fig_ranks(rows, out)
        fig_masking(agg, out)
    fig_twilight(args.results_dir, out)
    fig_twilight_metrics(args.results_dir, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
