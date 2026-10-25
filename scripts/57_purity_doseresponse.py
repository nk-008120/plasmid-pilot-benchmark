#!/usr/bin/env python3
"""Draw the filler effect is monotone in the purity of the **queried segment**, not in f.

    python scripts/57_purity_doseresponse.py
    python scripts/57_purity_doseresponse.py --mono
    python scripts/57_purity_doseresponse.py --geom results/auto_geometry_postr10/auto_geom.csv \\
        --out results/figures_purity

Reads the per-construct geometry the working record computed, one row per `synth_real` construct, carrying
the motif decomposition's queried-segment purity on both filler tiers beside the ranks already on disk
in `results/maskfix_{synthreal,esm_synthreal,synthrandom,esm_synthrandom}`, and writes PNG + PDF plus
the drawn numbers as CSV, so a caption can be checked against the figure without re-running anything.

**What the figure is.** 60.2 measured a 1,831,284-point sign reversal between random and SwissProt
filler on the whole-ORF query, and 66.2 showed it goes to exactly zero under `bhs_per_segment`. 66.4
then found it *surviving* `auto_per_segment`, the motif split, and named the mechanism as "a split that
leaves filler residues inside a segment leaves the effect inside with them". That sentence is wrong and
this figure is what replaces it: splits leaving up to 9.6% filler inside a segment carry **zero** effect,
and what carries the whole residual is the **absence** of a split. The effect is monotone in how much of
the sequence the arm actually hands the scorer, `overlap / segment length`, the queried segment's
**purity**, and flat in the dilution of the construct.

**Why both panels are needed, and what each one controls for.** On the undecomposed subset purity
*equals* f, so the left panel alone cannot separate a purity effect from 59.2's dilution effect all over
again. The right panel is the control: `multi_gene_cassette` spans the same four f values at a purity
that barely moves, so the comparison is f-matched 24 against 24 at every level. A fourfold change in
payload fraction moves it not at all while the same change moves the undecomposed arm from `recall@10`
0.000 to 0.708. **So what a pooled embedding is sensitive to is the dilution of the sequence it is
handed, not the dilution of the construct**: this project's thesis as a measurement on the query side.

**How the architectures are drawn, and why that is `figstyle`'s own encoding rather than a new one.**
All three series are one arm, `auto_per_segment`, so colouring them apart by architecture would break
the rule that colour says which segments the query is. It does not need to: on `motifless_fusion` the
arm emits one segment spanning the whole ORF and is **bit-identical to `whole`** on every undecomposed
cell (812 of 812 and 822 of 822 construct x threshold cells under both scorers, 68.1), so that series
*is* the whole-ORF query under another name and is drawn in `whole`'s style. The two decomposed
architectures are drawn in `auto_per_segment`'s, told apart on the x axis, which separates them
anyway, and by name on the plot.

**Holdout 1.00, which holds nothing out.** The payload's own catalogue entry is present at 100% identity,
so neither the holdout nor the segmenter is in the causal path and the filler is the only variable.
Quote it as `synth_real` against `synth_random`, f recomputed from `payload_aa / orf_aa`, 40,750-protein
catalogue, mean-pooled ESM-2: per this project's working rules the tier is established by `orfs.faa`'s md5 and never by
the free-text label.

**Mean rank is not drawn.** This is an embedding arm, so the working record permits a mean only beside its
count of rank >= 10,000 failures; the drawn quantities are net paired rank points and `recall@10`, and
the failure counts go in the caption and the CSV. Top-1 is not drawn either: it is a column and never a
headline (26.3).

**What this does not measure.** The tier's purity distribution is empty between 0.842 and 0.904, so
where in that gap the effect ends is not located: four measured points and a floor, not a threshold
(64.3's rule: two readings at the ends of a range do not establish what happens between them). One
synthetic tier, five architectures, 24 payloads, contiguous exact-boundary payloads. It transfers to the
corpus only as a hypothesis: 86% of the corpus sits at f >= 0.95, where purity and f are both near 1 and
nothing here would be visible.
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

from pilot.figstyle import METRICS, arm_display, legend_note

ROOT = Path(__file__).resolve().parents[1]

#: The purity bands the working record reports, as (nominal centre, lo, hi) with the floor last.
#: Half-open on the right except the floor. The nominal centres coincide with `motifless_fusion`'s four
#: dilution levels because on that architecture purity *is* f, which is exactly why the right panel
#: exists. `BAND_FLOOR` is 0.90 and not 0.904: the four `syn_Q6F1K1_multi_gene_cassette_*` segments sit
#: at 0.9037, and a floor written at the rounded value would drop them and report 236 where 68.1's
#: table reports 240.
BAND_FLOOR = 0.90
BANDS: tuple[tuple[float, float, float], ...] = (
    (0.20, 0.15, 0.25),
    (0.40, 0.35, 0.45),
    (0.60, 0.55, 0.65),
    (0.80, 0.75, 0.86),
    (BAND_FLOOR, BAND_FLOOR, 1.01),
)

#: The three architectures the f-matched contrast is drawn over, each with the arm whose style it is
#: drawn in. `motifless_fusion` is undecomposed and the arm is bit-identical to `whole` there.
#: The last two fields are the label offsets in points on each panel: all three series carry only two
#: distinct styles between them, so the architecture is named on the plot and the offsets are what keep
#: those names off each other. multi_gene_cassette and tagged_fusion both sit at net 0 on the left.
SERIES: tuple[tuple[str, str, tuple[float, float], tuple[float, float]], ...] = (
    ("motifless_fusion", "whole", (12, 16), (6, -30)),
    ("multi_gene_cassette", "auto_per_segment", (-8, 26), (6, -30)),
    ("tagged_fusion", "auto_per_segment", (-8, -32), (6, 10)),
)
#: The two undecomposed architectures with essentially no filler to be sensitive to. Counted and
#: reported, never drawn: a series of coincident points near (1.00, 0) reads as a measurement.
#: 68.1's table calls them f = 1.00; measured, every one of them is f = 0.9967 or 1.0000, so the
#: label is a rounding and `F1_MIN` is what the data supports.
F1_ARCHITECTURES = ("backbone_heavy", "single_cds")
F1_MIN = 0.9967

#: The f values `motifless_fusion` and the two decomposed architectures share, which is what makes the
#: right panel a 24-against-24 contrast rather than two different samples.
F_LEVELS = (0.20, 0.40, 0.60, 0.80)

#: real-filler column, random-filler column. The paired contrast is random against SwissProt, so a
#: negative net means random filler costs the arm rank, the sign 60.2 and 66.2 report.
RANK_COLS = {
    "esm_auto_per_segment": ("esm_auto_ps_real", "esm_auto_ps_random"),
    "align_auto_per_segment": ("align_auto_ps_real", "align_auto_ps_random"),
    "esm_whole": ("esm_whole_real", "esm_whole_random"),
    "esm_bhs_per_segment": ("esm_bhs_ps_real", "esm_bhs_ps_random"),
    "esm_oracle": ("esm_oracle_real", "esm_oracle_random"),
}

RECALL_DEPTH = 10
#: this project's threshold: an embedding arm's mean rank may be reported only beside this count.
FAIL_RANK = 10_000


def architecture(plasmid_id: str) -> str:
    """'syn_A1ABP5_motifless_fusion_20' -> 'motifless_fusion'. The trailing field is the dilution."""
    return plasmid_id.split("_", 2)[2].rsplit("_", 1)[0]


def band_of(purity: float) -> float | None:
    """Which reported band a purity falls in, or `None`, which is counted and printed, not dropped."""
    for centre, lo, hi in BANDS:
        if lo <= purity < hi:
            return centre
    return None


def read_geometry(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["f"] = float(r["f"])
        r["purity"] = float(r["purity_real"])
        r["purity_random"] = float(r["purity_random"])
        r["architecture"] = architecture(r["plasmid_id"])
        r["undecomposed"] = int(r["payload_seg_len_real"]) == int(r["orf_len"])
    return rows


def paired(rows: list[dict], arm: str) -> dict:
    """The paired random-against-SwissProt contrast for one arm over one subset of constructs.

    Returned rather than printed so the figure, the CSV and the caption all read the same numbers.
    """
    real_col, rnd_col = RANK_COLS[arm]
    sp = [int(r[real_col]) for r in rows]
    rnd = [int(r[rnd_col]) for r in rows]
    n = len(rows)
    return {
        "n": n,
        "net": sum(a - b for a, b in zip(sp, rnd)),
        "better": sum(1 for a, b in zip(sp, rnd) if b < a),
        "equal": sum(1 for a, b in zip(sp, rnd) if b == a),
        "worse": sum(1 for a, b in zip(sp, rnd) if b > a),
        "fail_random": sum(1 for x in rnd if x >= FAIL_RANK),
        "fail_swissprot": sum(1 for x in sp if x >= FAIL_RANK),
        "recall_random": (sum(1 for x in rnd if x <= RECALL_DEPTH) / n) if n else float("nan"),
        "recall_swissprot": (sum(1 for x in sp if x <= RECALL_DEPTH) / n) if n else float("nan"),
    }


def by_band(rows: list[dict], arm: str) -> list[dict]:
    """One record per reported purity band, with the band's membership and f spread carried along."""
    out = []
    for centre, lo, hi in BANDS:
        sub = [r for r in rows if lo <= r["purity"] < hi]
        if not sub:
            continue
        rec = {"band": centre, "lo": lo, "hi": hi,
               "purity_min": min(r["purity"] for r in sub),
               "purity_max": max(r["purity"] for r in sub),
               "f_min": min(r["f"] for r in sub), "f_max": max(r["f"] for r in sub)}
        rec.update(paired(sub, arm))
        out.append(rec)
    return out


def by_series(rows: list[dict], arm: str) -> dict[str, list[dict]]:
    """Per architecture, one record per shared f level: the f-matched contrast the right panel draws."""
    out: dict[str, list[dict]] = {}
    for arch, _style_arm, _off_l, _off_r in SERIES:
        recs = []
        for fv in F_LEVELS:
            sub = [r for r in rows if r["architecture"] == arch and abs(r["f"] - fv) < 0.01]
            if not sub:
                continue
            rec = {"architecture": arch, "f": fv,
                   "purity_min": min(r["purity"] for r in sub),
                   "purity_max": max(r["purity"] for r in sub),
                   "purity_mean": sum(r["purity"] for r in sub) / len(sub)}
            rec.update(paired(sub, arm))
            recs.append(rec)
        out[arch] = recs
    return out


def figure(bands: list[dict], series: dict[str, list[dict]], notes: list[str], *,
           title: str, mono: bool, caption: bool = True) -> plt.Figure:
    display = arm_display(mono)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.4, 5.5))

    # ---- left: the dose-response in the purity of the queried segment
    for arch, style_arm, off_l, _off_r in SERIES:
        recs = series[arch]
        if not recs:
            continue
        st = display[style_arm]
        xs = [r["purity_mean"] for r in recs]
        ys = [r["net"] for r in recs]
        kw = dict(color=st.colour, lw=st.lw, ms=st.ms, zorder=3, label=st.label)
        if st.hollow:
            kw.update(markerfacecolor="white", markeredgecolor=st.colour, markeredgewidth=1.6)
        axl.plot(xs, ys, st.fmt, **kw)
        # The architecture, on the plot: the legend names the arm, and two of the three series share
        # one arm, so the subset has to be said somewhere.
        i = -1 if arch == "tagged_fusion" else 0
        axl.annotate(arch.replace("_", " "), (xs[i], ys[i]), textcoords="offset points",
                     xytext=off_l, fontsize=8.5, color=st.colour,
                     ha="left" if arch == "motifless_fusion" else "right")
    axl.axhline(0.0, color="#999999", lw=0.9, ls=":", zorder=1)
    axl.set_yscale("symlog", linthresh=100)
    axl.set_xlim(0.14, 1.06)
    axl.set_xlabel("purity of the queried segment  (payload residues / residues handed to the scorer)")
    axl.set_ylabel("net paired rank points, random filler against SwissProt filler\n"
                   "(below zero = random filler costs the arm rank)")
    axl.set_title("the effect is monotone in what the arm queries", fontsize=10.5, loc="left")
    axl.grid(axis="y", color="#e6e6e6", lw=0.75)
    axl.set_axisbelow(True)
    for rec in bands:
        if rec["band"] == BAND_FLOOR:
            axl.axvspan(rec["lo"], 1.06, color="#f4f4f4", zorder=0)
            axl.annotate(f"purity >= {BAND_FLOOR:.2f}:  n = {rec['n']},  net {rec['net']:+d}",
                         (1.045, 0.03), xycoords=("data", "axes fraction"),
                         fontsize=8, color="#595959", rotation=90, va="bottom", ha="center")

    # ---- right: the f-matched control
    recall = METRICS["recall10"]
    for arch, style_arm, _off_l, off_r in SERIES:
        recs = series[arch]
        if not recs:
            continue
        st = display[style_arm]
        kw = dict(color=st.colour, lw=st.lw, ms=st.ms, zorder=3, label=st.label)
        if st.hollow:
            kw.update(markerfacecolor="white", markeredgecolor=st.colour, markeredgewidth=1.6)
        axr.plot([r["f"] for r in recs], [100.0 * r["recall_random"] for r in recs], st.fmt, **kw)
        lo = min(r["purity_min"] for r in recs)
        hi = max(r["purity_max"] for r in recs)
        first = recs[0]
        axr.annotate(f"{arch.replace('_', ' ')}\npurity {lo:.3f} to {hi:.3f}",
                     (first["f"], 100.0 * first["recall_random"]), textcoords="offset points",
                     xytext=off_r, fontsize=8.5, color=st.colour, ha="left")
    axr.set_xlabel("f = payload fraction of the construct  (payload_aa / orf_aa)")
    axr.set_ylabel(recall.label + ", random filler")
    axr.set_title("and flat in the dilution of the construct", fontsize=10.5, loc="left")
    axr.set_ylim(-22, 112)
    axr.set_xlim(0.15, 0.87)
    axr.set_xticks(list(F_LEVELS))
    axr.grid(axis="y", color="#e6e6e6", lw=0.75)
    axr.set_axisbelow(True)

    for ax in (axl, axr):
        # Two of the three series are one arm, so the handles repeat; a legend naming it twice reads
        # as two arms. Deduplicated on the label rather than drawn with invented per-series labels.
        handles, labels = ax.get_legend_handles_labels()
        seen: dict[str, object] = {}
        for h, lab in zip(handles, labels):
            seen.setdefault(lab, h)
        ax.legend(list(seen.values()), list(seen), fontsize=8, loc="center left", framealpha=0.95)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle(title, fontsize=12.5, x=0.012, ha="left", y=0.995)
    if caption:
        cap = "\n".join(textwrap.fill(n, 185) for n in notes)
        fig.text(0.012, -0.03, cap, fontsize=7.8, color="#595959", va="top", ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return fig


def write_table(bands: list[dict], series: dict[str, list[dict]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["subset", "key", "n", "purity_min", "purity_max", "f_min", "f_max",
            "net", "better", "equal", "worse",
            "recall_random", "recall_swissprot", "fail_random", "fail_swissprot"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rec in bands:
            w.writerow({**rec, "subset": "purity_band", "key": f"{rec['band']:.2f}"})
        for arch, recs in series.items():
            for rec in recs:
                w.writerow({**rec, "subset": "architecture", "key": f"{arch}@f={rec['f']:.2f}",
                            "f_min": rec["f"], "f_max": rec["f"]})
    print(f"  wrote {path.name}")


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png / .pdf")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mono", action="store_true",
                    help="draw in greyscale. The encoding is unchanged: grey value replaces hue as "
                         "the channel saying which segments a line is drawn from, and line style "
                         "still says how they were aggregated.")
    ap.add_argument("--geom", type=Path, default=ROOT / "results/auto_geometry_postr10/auto_geom.csv",
                    help="the per-construct geometry the working record computed")
    ap.add_argument("--arm", default="esm_auto_per_segment", choices=sorted(RANK_COLS),
                    help="which arm's paired contrast to draw; the default is the one 66.4 named")
    ap.add_argument("--out", type=Path, default=ROOT / "results/figures_purity",
                    help="where the figure goes. A new directory per run, never a finished one.")
    args = ap.parse_args(argv)

    if not args.geom.is_file():
        print(f"not found: {args.geom}")
        return 2
    rows = read_geometry(args.geom)
    print(f"{len(rows)} constructs read from {args.geom.name} -> {args.out}/")

    # Report binding, not settings: every band prints its membership, and so does what no band holds.
    bands = by_band(rows, args.arm)
    unbanded = [r for r in rows if band_of(r["purity"]) is None]
    print(f"  purity bands: " + ", ".join(f"{b['band']:.2f} n={b['n']}" for b in bands))
    print(f"  constructs in no reported band: {len(unbanded)}"
          + (" (" + ", ".join(f"{r['plasmid_id']}  {r['purity']:.4f}" for r in unbanded) + ")"
             if unbanded else ""))
    # Counted on both tiers and printed as two numbers on purpose. 68.1's table reports 144
    # undecomposed against 192 decomposed; that is the random-filler tier, and the SwissProt tier is
    # two lower because the motif catalogue finds a 9-residue trim in the real filler on two
    # constructs. Quoting one of them as "the" count is what the row below exists to prevent.
    undec = [r for r in rows if r["undecomposed"]]
    undec_rnd = [r for r in rows
                 if int(r["payload_seg_len_random"]) == int(r["orf_len"])]
    print(f"  undecomposed (queried segment is the whole ORF): "
          f"{len(undec)} of {len(rows)} on SwissProt filler, {len(undec_rnd)} on random filler; "
          f"decomposed {len(rows) - len(undec)} and {len(rows) - len(undec_rnd)}")
    # The two tiers are identical but for the filler, for the construct, and not for every quantity
    # derived from it. The motif catalogue scans the filler too, so this count is printed per 68.1.
    moved = [r for r in rows if abs(r["purity"] - r["purity_random"]) > 1e-4]
    print(f"  constructs whose motif decomposition differs between the two fillers: {len(moved)}"
          + (" (" + ", ".join(f"{r['plasmid_id']}  {r['purity']:.4f}/{r['purity_random']:.4f}"
                              for r in moved) + ")" if moved else ""))
    for arch in F1_ARCHITECTURES:
        sub = [r for r in rows if r["architecture"] == arch]
        lo = min((r["f"] for r in sub), default=float("nan"))
        print(f"  not drawn, f >= {lo:.4f} and so nothing to dilute: {arch} n={len(sub)}")

    series = by_series(rows, args.arm)
    whole = paired(rows, "esm_whole")
    bhs = paired(rows, "esm_bhs_per_segment")

    notes = [
        "synth_real against synth_random: the same 336 constructs, the same 24 payloads, the same "
        "payload_aa and orf_aa residue for residue, differing only in what the non-payload residues "
        "are made of. Holdout 1.00, which holds nothing out: the payload's own entry sits in the "
        "40,750-protein catalogue at 100% identity, so neither the holdout nor the segmenter is in "
        "the causal path and the filler is the only variable. Mean-pooled ESM-2. f recomputed from "
        "payload_aa / orf_aa, never read from the column of that name.",
        "Read the left panel as a dose-response and the right one as its control. On the undecomposed "
        "architecture purity equals f, so the left panel alone cannot separate a purity effect from a "
        "dilution effect; multi_gene_cassette spans the same four f values at a purity that barely "
        "moves, which makes the right panel an f-matched 24-against-24 contrast at every level.",
        "The undecomposed series is auto_per_segment and is bit-identical to the whole-ORF query on "
        "every undecomposed cell, 812 of 812 and 822 of 822 construct x threshold cells under both "
        "scorers, which is the arm's definition on that subset rather than a coincidence of ties "
        ". It is drawn in the whole arm's style for that reason.",
        f"Mean rank is not drawn: this is an embedding arm, so a mean is admissible only beside its "
        f"count of rank >= {FAIL_RANK:,} failures. Over all 336 constructs the "
        f"whole-ORF query fails that way on {whole['fail_random']} constructs with random filler "
        f"against {whole['fail_swissprot']} with SwissProt, and the best-hit segmenter on "
        f"{bhs['fail_random']} against {bhs['fail_swissprot']}, paired "
        f"{bhs['better']} better / {bhs['equal']} equal / {bhs['worse']} worse. Top-1 is not drawn "
        "either: it is a column and never a headline.",
        legend_note(mono=args.mono),
        "Four measured points and a floor, not a located threshold: the tier's purity distribution is "
        "empty between 0.842 and 0.904, so where in that gap the effect ends is unmeasured (CONTEXT "
        "64.3). One synthetic tier, five architectures, 24 payloads, contiguous exact-boundary "
        "payloads; 86% of the real corpus sits at f >= 0.95, where purity and f are both near 1 and "
        "nothing here would be visible.",
    ]
    title = ("What a pooled embedding is sensitive to is the dilution of the sequence it is "
             "handed, not of the construct ")
    stem = "fig_d13_purity_doseresponse" + ("_mono" if args.mono else "")
    save(figure(bands, series, notes, mono=args.mono, title=title), args.out, stem)
    # A second copy without the caption block, for embedding. At a document's 6.5 inches the caption's
    # 7.8pt sets at about 2pt and is unreadable, so carrying it there costs most of the file size and
    # buys nothing; the deliverables set the same qualifiers in legible type beside the picture. The
    # captioned copy above is the archival figure and nothing else differs between them.
    save(figure(bands, series, notes, mono=args.mono, title=title, caption=False),
         args.out, stem + "_nocap")
    write_table(bands, series, args.out / "purity_doseresponse.csv")
    (args.out / "purity_doseresponse.json").write_text(
        json.dumps({"arm": args.arm, "geom": str(args.geom), "bands": bands,
                    "series": series, "undecomposed": len(undec),
                    "esm_whole": whole, "esm_bhs_per_segment": bhs}, indent=2),
        encoding="utf-8")
    print(" NEVER pool these bands into one scalar, and read the right panel before "
          "calling the left one a purity effect: on the undecomposed subset purity is f.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
