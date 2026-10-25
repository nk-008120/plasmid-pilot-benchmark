#!/usr/bin/env python3
"""Draw route B's payload-fraction dose-response: the paired whole-to-oracle gap, one stratum at a time.

    python scripts/46_route_b_dose_response.py
    python scripts/46_route_b_dose_response.py --strata results/route_b_strata/route_b_strata.json \\
        --out results/route_b_figures

Reads `route_b_strata.json` from `scripts/45_route_b_strata.py`, the only reporting form the working record
permits, and writes PNG + PDF plus the drawn numbers as CSV, so a caption can be checked against the
figure without re-running anything.

**What the figure is.** Route B is the project's first tier of real deposited constructs sitting inside
the diluted regime (29 of 46 pairs below f = 0.70, against the corpus's 15 of 286), and the working record is
the dose-response that tier was built to measure: at the 25% holdout the whole-to-oracle mean rank goes
81.19 -> 7.81 at f < 0.50, 28.62 -> 5.77 at 0.50-0.70, 3.00 -> 1.82 at 0.70-0.95, and 19.33 -> 19.33 at
f >= 0.95. Section 15.13 measured the same effect as "6 plasmids" on a holdout series; here it is a
monotone response in the moderating variable itself.

**Why it is drawn as four isolated pairs and not as two curves.** A line through one arm across the
strata is the figure this result most invites and it would be wrong: `whole` has mean rank 3.00 at
0.70-0.95 and 19.33 at f >= 0.95, which is not evidence that dilution helps: the strata hold different
payloads with different family densities, so a cross-stratum step mixes an effect with a sample change
(and 21.3 before it). So nothing here is ever connected across strata: every drawn line is
a **vertical** connector between two arms measured on the same constructs, the slots are separated by a
rule, and `test_no_line_in_the_figure_spans_two_strata` keeps it that way.

**Aggregation.** 46 pairs carry 19 distinct payloads, 7 payloads hold the f < 0.50 stratum's 16 pairs,
three POLB deposits are point mutants of one construct, `PB-TO-hNIL` and `PB-tetO-hNIL` are one cassette
in two backbones, so micro and macro are both drawn and the macro figure governs.
Following `pilot.figstyle`, colour says which residues the query is and line style says how the pairs
were aggregated: solid micro, dashed macro, same colours either way.

**n is on the axis and in the marker.** A 6-pair stratum must not read like a 16-pair one, so every slot
carries its pair and payload counts in the tick label and the marker area scales with the count behind
that marker.

**The f >= 0.95 slot is the control and is shaded.** With no dilution the oracle span is the called ORF,
so both arms issue the same query and the gap is zero by construction rather than by measurement. The
script does not take that on trust: `zero_gap_strata` recovers it from the numbers, and the f census
prints how many of the slot's pairs sit at f exactly 1.000, 3 of 6 on this tier, the rest within 2.4%
of it, because "the oracle span *is* the ORF" is exactly true only for the first three.

Top-1 is not drawn. It is a column and never a headline, and on this tier it is blind to
the f < 0.50 stratum's rank moves; it stays in `45_route_b_strata.py`'s table.
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
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from pilot.figstyle import METRICS, PRIMARY_METRICS, arm_display

ROOT = Path(__file__).resolve().parents[1]

#: Must stay identical to `45_route_b_strata.py`'s `STRATUM_NAMES`; a test asserts it.
STRATUM_ORDER = ("<0.50", "0.50-0.70", "0.70-0.95", ">=0.95")
F_STRATA = (0.50, 0.70, 0.95)
#: ASCII, because this one is printed to a Windows console that is cp1252 and not UTF-8.
STRATUM_LABEL = {"<0.50": "f < 0.50", "0.50-0.70": "0.50 <= f < 0.70",
                 "0.70-0.95": "0.70 <= f < 0.95", ">=0.95": "f >= 0.95"}
#: the same strata as a reader should see them on the axis, where the font is not the terminal's
FIG_LABEL = {"<0.50": "f < 0.50", "0.50-0.70": "0.50 <= f < 0.70",
             "0.70-0.95": "0.70 <= f < 0.95", ">=0.95": "f >= 0.95"}

#: The dose-response is the paired contrast between these two arms, in this order.
PAIR = ("whole", "oracle")
#: The companion figure: the deployable, label-free arm against the labelled ceiling.
CEILING_PAIR = ("oracle", "bhs_per_segment")

#: metric key -> (field in the strata JSON, scale into `figstyle.METRICS`' units)
JSON_FIELD = {"mean_rank": ("mean_rank", 1.0), "mrr": ("mrr", 1.0),
              "recall10": ("recall_at_10", 100.0)}

#: (name, matplotlib line style, filled marker, x offset within the stratum's slot)
AGGREGATIONS = (("micro", "-", True, -0.19), ("macro", "--", False, 0.19))

CONNECTOR = "#bbbbbb"
#: marker-area multiplier for a pair's first arm, so a zero gap reads as a ring (see `_panel`)
HALO = 1.9
CONTROL_SHADE = "#f1f1f1"
RANK_TICKS = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 3000)


# --------------------------------------------------------------------------------------- reading


# Which palette the figures use. A one-element list rather than a bare module global because `--mono`
# is parsed in `main` and read inside helpers: it is the smallest thing that lets one choice reach all
# of them without threading an argument through every drawing function. `figstyle.arm_display` is the
# only place the two palettes are defined, so the encoding cannot differ between them.
MONO = [False]

def stratum(f: float) -> str:
    """The project's f strata, duplicated from script 45 rather than imported across a numbered name."""
    if f < F_STRATA[0]:
        return STRATUM_ORDER[0]
    if f < F_STRATA[1]:
        return STRATUM_ORDER[1]
    if f < F_STRATA[2]:
        return STRATUM_ORDER[2]
    return STRATUM_ORDER[3]


def thresholds_of(doc: dict) -> list[str]:
    """The holdout keys, loosest first, which is left-to-right into the twilight zone (as in 17_*)."""
    return sorted(doc.get("thresholds") or {}, key=float, reverse=True)


def strata_present(doc: dict) -> tuple[str, ...]:
    """Strata the run actually carries, in canonical order: an empty one is skipped by script 45."""
    seen = {s for cells in (doc.get("thresholds") or {}).values() for s in cells}
    return tuple(s for s in STRATUM_ORDER if s in seen)


def value(doc: dict, thr: str, strat: str, arm: str, metric: str, *,
          macro: bool = False) -> float | None:
    """One drawn y value, or `None` where the run does not carry it, so a gap stays a gap."""
    cell = ((doc.get("thresholds") or {}).get(thr) or {}).get(strat) or {}
    got = cell.get(arm)
    if not got:
        return None
    field, scale = JSON_FIELD[metric]
    v = (got.get("macro" if macro else "micro") or {}).get(field)
    return None if v is None else float(v) * scale


def counts(doc: dict, thr: str, strat: str) -> tuple[int | None, int | None]:
    """(pairs, distinct payloads) in this cell, read from the arms rather than assumed."""
    cell = ((doc.get("thresholds") or {}).get(thr) or {}).get(strat) or {}
    pairs = {c["micro"]["n"] for c in cell.values() if c.get("micro")}
    payloads = {c["macro"]["n_payloads"] for c in cell.values() if c.get("macro")}
    return (pairs.pop() if len(pairs) == 1 else None,
            payloads.pop() if len(payloads) == 1 else None)


def disagreeing_counts(doc: dict) -> list[tuple[str, str, list[int]]]:
    """Cells whose arms were scored on different numbers of constructs.

    The arms are supposed to be the same constructs queried differently, so a disagreement here means
    the paired contrast is not paired and the figure would be comparing two samples. Reported rather
    than crashed on, in the shape this project's working rules asks of any limit: print how many items hit it.
    """
    out = []
    for thr in thresholds_of(doc):
        for strat in strata_present(doc):
            cell = (doc["thresholds"][thr] or {}).get(strat) or {}
            ns = sorted({c["micro"]["n"] for c in cell.values() if c.get("micro")})
            if len(ns) > 1:
                out.append((thr, strat, ns))
    return out


def rank_failures(doc: dict, pair: tuple[str, str]) -> int:
    """Rows at rank >= 10,000 across the drawn cells, which is what licenses quoting a mean at all.

    under mean-pooling the rank distribution is bimodal, so a mean rank is reportable
    only beside this count. It is summed from the run rather than written into the caption, because a
    caption that states a number a future run could falsify is the defect this project keeps finding.
    """
    total = 0
    for thr in thresholds_of(doc):
        for strat in strata_present(doc):
            cell = ((doc.get("thresholds") or {}).get(thr) or {}).get(strat) or {}
            for arm in pair:
                total += ((cell.get(arm) or {}).get("micro") or {}).get("n_ge_10000") or 0
    return total


def fold(a: float | None, b: float | None) -> float | None:
    """How many times better `b` is than `a` on a lower-is-better metric."""
    if a is None or b is None or not b:
        return None
    return a / b


def zero_gap_strata(doc: dict, pair: tuple[str, str] = PAIR) -> tuple[str, ...]:
    """Strata where the two arms are identical on every drawn metric, micro and macro, at every holdout.

    This is the control recovered from the data instead of asserted from the stratum's name. At
    f >= 0.95 the oracle span is the called ORF, so the two arms are the same query and the gap has to
    be exactly zero; if a future run breaks that, the figure says so rather than labelling a measured
    difference "structural".
    """
    out = []
    for strat in strata_present(doc):
        # `seen` is separate from `same` so that a stratum carrying only one of the two arms is not
        # called a control by vacuous agreement
        seen, same = False, True
        for thr in thresholds_of(doc):
            for metric in PRIMARY_METRICS:
                for macro in (False, True):
                    a = value(doc, thr, strat, pair[0], metric, macro=macro)
                    b = value(doc, thr, strat, pair[1], metric, macro=macro)
                    if a is None or b is None:
                        continue
                    seen = True
                    if a != b:
                        same = False
        if seen and same:
            out.append(strat)
    return tuple(out)


def f_census(paths) -> dict[str, dict]:
    """Per stratum: how many pairs, how many at f exactly 1.000, and the smallest f in the slot.

    The caption for the control slot depends on this. "The oracle span is the called ORF" is exactly
    true only where f = 1.000; on route B that is 3 of the slot's 6 pairs and the other three are at
    0.976-0.983, so the honest sentence names both. It also cross-checks the join in script 45: these
    counts must equal the JSON's own n per stratum.
    """
    out: dict[str, dict] = {}
    for p in paths:
        with open(p, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if not row.get("f"):
                    continue
                f = float(row["f"])
                cell = out.setdefault(stratum(f), {"n": 0, "n_exact": 0, "min_f": f})
                cell["n"] += 1
                cell["n_exact"] += int(f == 1.0)
                cell["min_f"] = min(cell["min_f"], f)
    return out


def control_note(doc: dict, controls: tuple[str, ...], census: dict[str, dict]) -> str:
    """The one sentence the shaded slot needs, qualified by the census when one was read."""
    if not controls:
        return ("no stratum has a zero whole-to-oracle gap in this run: the f >= 0.95 control did not "
                "reproduce and should be investigated before the figure is quoted")
    names = ", ".join(FIG_LABEL.get(s, s) for s in controls)
    cell = census.get(controls[-1]) if len(controls) == 1 else None
    detail = ""
    if cell:
        detail = (": {} of {} of its pairs sit at f = 1.000 exactly and the rest at f >= {:.3f}, so the "
                  "two arms query the same residues or within a few percent of them"
                  .format(cell["n_exact"], cell["n"], cell["min_f"]))
    return ("shaded slot = the control ({}): with no dilution the oracle span is the called ORF, so both "
            "arms issue the same query and the gap is zero by construction, not by measurement{}"
            .format(names, detail))


# --------------------------------------------------------------------------------------- drawing


def _panel(ax, doc: dict, thr: str, metric: str, pair: tuple[str, str],
           controls: tuple[str, ...], *, annotate: bool) -> list[float]:
    """One holdout x one metric: a vertical paired connector per stratum per aggregation.

    Every `ax.plot` here is vertical by construction. That is the whole design constraint: a horizontal
    line in this panel would be a cross-stratum trend, which this project says is not a thing the data
    supports.
    """
    strata = strata_present(doc)
    drawn: list[float] = []
    for i, strat in enumerate(strata):
        if strat in controls:
            ax.axvspan(i - 0.5, i + 0.5, color=CONTROL_SHADE, zorder=0)
        if i:
            ax.axvline(i - 0.5, color="#dddddd", lw=0.8, zorder=0)
        n_pairs, n_payloads = counts(doc, thr, strat)
        for name, ls, filled, dx in AGGREGATIONS:
            macro = name == "macro"
            a = value(doc, thr, strat, pair[0], metric, macro=macro)
            b = value(doc, thr, strat, pair[1], metric, macro=macro)
            if a is None or b is None:
                continue
            n = (n_payloads if macro else n_pairs) or 8
            x = i + dx
            ax.plot([x, x], [a, b], ls=ls, color=CONNECTOR, lw=1.4, zorder=1)
            for arm, y, halo in zip(pair, (a, b), (HALO, 1.0)):
                st = arm_display(MONO[0])[arm]
                # `whole` is drawn as a wider halo for figstyle's reason: where the gap is zero the two
                # markers coincide, and a halo shows that as a ring rather than as a vanished arm
                ax.scatter([x], [y], s=(26 + 6.0 * n) * halo, marker="o", zorder=3, linewidths=1.5,
                           facecolor=st.colour if filled else "white", edgecolor=st.colour)
                drawn.append(y)
            if annotate and not macro:
                shown = [(a, "bottom", 9)] if a == b else [(a, "bottom", 9), (b, "top", -9)]
                for y, va, off in shown:
                    ax.annotate(f"{y:,.2f}", (x, y), textcoords="offset points", xytext=(-10, off),
                                ha="right", va=va, fontsize=7.2, color="#666666")
                r = fold(a, b)
                if r is not None:
                    # "no gap" is reserved for the control slot, where the two arms are the same query
                    # and the zero is structural. A measured coincidence, `whole` and `oracle` both
                    # at 3.46 where the payload's own entry is in the catalogue, is a ratio of 1.0
                    # and says so, because the two are not the same statement.
                    label = "no gap" if strat in controls else f"x{r:.1f}"
                    # parked at the foot of the slot: the one place in the panel no marker reaches, so
                    # it never has to be nudged around the data
                    ax.text(i, 0.035, label,
                            transform=blended_transform_factory(ax.transData, ax.transAxes),
                            ha="center", va="bottom", fontsize=8.6, color="#333333")
    m = METRICS[metric]
    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels([_tick(doc, thr, s, s in controls) for s in strata], fontsize=7.4)
    # asymmetric padding: the micro pair carries its values to its left and runs off a tight axis
    ax.set_xlim(-0.72, len(strata) - 0.45)
    if m.log:
        top = max(drawn or [1.0])
        ax.set_yscale("log")
        # limits reversed rather than inverted after the fact: 1 is the best rank, so it belongs at the
        # top, and an explicit pair keeps the annotation headroom off the autoscaler
        ax.set_ylim(top * 3.0, 0.74)
        ticks = [t for t in RANK_TICKS if t <= top * 2.2]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:,}" for t in ticks])
    elif metric == "recall10":
        ax.set_ylim(-4, 104)
    else:
        ax.set_ylim(-0.04, 1.04)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    return drawn


def _tick(doc: dict, thr: str, strat: str, control: bool) -> str:
    """The slot's name over its n, stacked: a 6-pair slot must not read like a 16-pair one."""
    n_pairs, n_payloads = counts(doc, thr, strat)
    return "{}{}\n{} pairs\n{} payloads".format(
        FIG_LABEL.get(strat, strat), "  (control)" if control else "",
        "?" if n_pairs is None else n_pairs, "?" if n_payloads is None else n_payloads)


def _legend(pair: tuple[str, str]) -> tuple[list, list[str]]:
    handles = [Line2D([], [], ls="", marker="o", ms=8, color=arm_display(MONO[0])[a].colour,
                      label=arm_display(MONO[0])[a].label) for a in pair]
    handles += [Line2D([], [], ls="-", color="#777777", marker="o", ms=6,
                       label="micro, over constructs"),
                Line2D([], [], ls="--", color="#777777", marker="o", ms=6, markerfacecolor="white",
                       label="macro, over distinct payloads (this one governs)")]
    return handles, [h.get_label() for h in handles]


def _column_title(thr: str) -> str:
    pct = float(thr) * 100
    if pct >= 100:
        return "100%, not a holdout: the payload's own entry is in the catalogue"
    if abs(pct - 25) < 1e-9:
        return "25% holdout, the twilight zone, where the claim lives"
    return f"{pct:.0f}% holdout"


def _save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / stem}.png / .pdf")


def figure(doc: dict, pair: tuple[str, str], controls: tuple[str, ...], notes: list[str], *,
           title: str, annotate: bool = True):
    """rows = the primary metrics, columns = the holdout sweep, each panel = paired contrasts."""
    thrs = thresholds_of(doc)
    fig, axes = plt.subplots(len(PRIMARY_METRICS), len(thrs), squeeze=False,
                             figsize=(4.9 * len(thrs), 3.5 * len(PRIMARY_METRICS)))
    for j, thr in enumerate(thrs):
        for i, metric in enumerate(PRIMARY_METRICS):
            ax = axes[i][j]
            _panel(ax, doc, thr, metric, pair, controls,
                   annotate=annotate and metric == "mean_rank")
            if j == 0:
                ax.set_ylabel(METRICS[metric].label, fontsize=8.5)
            if i == 0:
                ax.set_title(_column_title(thr), fontsize=9.5, loc="left")
            if i == len(PRIMARY_METRICS) - 1:
                ax.set_xlabel("payload fraction f = oracle span / called ORF", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.055, 1, 0.975))
    fig.suptitle(title, fontsize=12.5, x=0.0, y=0.995, ha="left")
    handles, labels = _legend(pair)
    fig.legend(handles, labels, fontsize=7.8, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.004))
    # each note is placed under the one above rather than at a fixed pitch: a note that wraps to three
    # lines would otherwise be overprinted by the next one, which is how the first draft read
    y = -0.010
    for note in notes:
        wrapped = textwrap.fill(note, 132)
        fig.text(0.0, y, wrapped, fontsize=7.5, color="#777777", ha="left", va="top")
        y -= 0.013 + 0.013 * wrapped.count("\n")
    return fig


# --------------------------------------------------------------------------------------- output


def table(doc: dict, pair: tuple[str, str], controls: tuple[str, ...]) -> list[dict]:
    """The drawn numbers, one row per holdout x stratum x metric x aggregation. Never pooled."""
    rows = []
    for thr in thresholds_of(doc):
        for strat in strata_present(doc):
            n_pairs, n_payloads = counts(doc, thr, strat)
            for name, _ls, _filled, _dx in AGGREGATIONS:
                macro = name == "macro"
                for metric in PRIMARY_METRICS:
                    a = value(doc, thr, strat, pair[0], metric, macro=macro)
                    b = value(doc, thr, strat, pair[1], metric, macro=macro)
                    if a is None or b is None:
                        continue
                    # a ratio is only meaningful on the lower-is-better metric; MRR and recall@10 get
                    # the difference instead, which is how the working record quotes them
                    r = fold(a, b) if not METRICS[metric].higher_is_better else None
                    rows.append({"holdout": thr, "f_stratum": strat, "is_control": strat in controls,
                                 "metric": metric, "aggregation": name,
                                 "n_pairs": n_pairs, "n_payloads": n_payloads,
                                 pair[0]: round(a, 4), pair[1]: round(b, 4),
                                 "delta": round(b - a, 4),
                                 "fold_better": None if r is None else round(r, 3)})
    return rows


def write_table(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows, one per holdout x stratum x metric x aggregation)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mono", action="store_true",
                    help="draw in greyscale. The encoding is unchanged: grey value "
                         "replaces hue as the channel saying which segments a line is drawn "
                         "from, and line style still says how they were aggregated. For print, "
                         "for photocopies, and for readers who cannot separate the oracle red "
                         "from the bhs green.")
    ap.add_argument("--strata", type=Path,
                    default=ROOT / "results/route_b_strata/route_b_strata.json")
    ap.add_argument("--manifest", type=Path, nargs="*", default=[
        ROOT / "data/manifest_route_b.csv", ROOT / "data/manifest_route_b_second.csv"],
        help="route B's benchmark manifests, read only for the f census in the control caption")
    ap.add_argument("--out", type=Path, default=ROOT / "results/route_b_figures")
    args = ap.parse_args(argv)
    MONO[0] = bool(args.mono)

    if not args.strata.exists():
        print(f"no strata report at {args.strata}: run scripts/45_route_b_strata.py first")
        return 2
    doc = json.loads(args.strata.read_text(encoding="utf-8"))

    strata, thrs = strata_present(doc), thresholds_of(doc)
    print(f"route B dose-response, from {args.strata}")
    print(f"  holdouts: {', '.join(thrs)}    strata: {', '.join(strata)}")
    for thr in thrs[-1:]:
        for strat in strata:
            n_pairs, n_payloads = counts(doc, thr, strat)
            print(f"        {STRATUM_LABEL.get(strat, strat):<18}{n_pairs} pairs, {n_payloads} payloads")
    bad = disagreeing_counts(doc)
    print(f"  BINDING, cells whose arms were scored on different n: {len(bad)}  {bad}")

    present = [p for p in args.manifest if p.exists()]
    census = f_census(present)
    if present:
        for strat in strata:
            cell = census.get(strat, {})
            n_json = counts(doc, thrs[-1], strat)[0]
            flag = "" if cell.get("n") == n_json else "  <: DISAGREES with the strata report"
            print("    f census {:<18}{} pairs, {} at f = 1.000, min f {:.4f}{}".format(
                STRATUM_LABEL.get(strat, strat), cell.get("n"), cell.get("n_exact"),
                cell.get("min_f", float("nan")), flag))
    else:
        print("  no manifest read, so the control caption states only the measured zero gap")

    controls = zero_gap_strata(doc)
    print(f"  zero whole-to-oracle gap, recovered from the numbers: {controls or '(none)'}")
    print(" NEVER pool these strata into one scalar, and never read a step between "
          "slots as an effect (51.1): the slots hold different payloads.")

    note_pairing = ("read the gap inside a slot, never the step between slots: each stratum holds "
                    "different payloads with different family densities, so a cross-stratum step mixes "
                    "an effect with a sample change: nothing here is connected "
                    "across strata")
    note_style = ("colour = which residues the query is, line style = how the pairs were aggregated "
                  "(solid, filled = micro, over constructs; dashed, hollow = macro, over distinct "
                  "payloads); marker area scales with the count behind it")
    def note_caveats(pair: tuple[str, str]) -> str:
        return ("top-1 is not drawn: it is a column and never a headline. The mean rank "
                "row is reported beside its count of rank >= 10,000 failures, summed over the cells "
                "drawn here: {}. Route B's catalogue is family-structured and about "
                "1.6x harder than its size suggests, so these absolute ranks are comparable between "
                "arms and not to any other catalogue's ".format(
                    rank_failures(doc, pair)))

    fig = figure(doc, PAIR, controls,
                 [note_pairing, control_note(doc, controls, census), note_style, note_caveats(PAIR)],
                 title="Route B: the whole-to-oracle gap grows with dilution, within each stratum "
                       "")
    _save(fig, args.out, "fig_d11_route_b_dose_response")

    fig = figure(doc, CEILING_PAIR, zero_gap_strata(doc, CEILING_PAIR),
                 [note_pairing,
                  "the label-free arm is not a scorer beating its own ground truth: `oracle` issues ONE "
                  "query, the labelled payload span, and `bhs_per_segment` issues one per claimed "
                  "segment, 2.28 per construct on this tier, up to 7, and takes the best rank among "
                  "them, so the arms are not level on queries issued ",
                  "its cost is real and concentrated: at 0.70 <= f < 0.95 the segmenter claimed one wrong "
                  "54-residue window on three GAL4 deposits and never queried the other 254 residues, "
                  "which is the whole difference in that slot ",
                  note_style, note_caveats(CEILING_PAIR)],
                 title="Route B: the label-free decomposition against the labelled ceiling "
                       "", annotate=False)
    _save(fig, args.out, "fig_d12_route_b_ceiling")

    write_table(table(doc, PAIR, controls), args.out / "route_b_dose_response.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
