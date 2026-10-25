"""How an arm and a metric are drawn, in one place, so the figure scripts cannot drift apart.

`scripts/07_figures.py` and `scripts/17_compare_tiers.py` both read `twilight_*_summary.json` and both
named the arms as literals: `["whole_rank", "auto_rank", "oracle_rank"]` in one and a tuple of
`("whole", "oracle")` in the other. That is the same defect `ARM_NAMES` was introduced to fix in
`10_twilight.py`, one file further downstream: when `auto_per_segment` and `bhs_per_segment` were added
to the benchmark they appeared in every CSV and in no figure, silently, because a literal cannot grow.

Two things live here, and nothing else:

* **`ARM_DISPLAY`**: label, colour and line style per arm. The encoding is deliberate: **colour says
  which segments a line is drawn from and line style says how they were aggregated.** `auto` and
  `auto_per_segment` are the same blue, solid against dashed; `bhs` and `bhs_per_segment` the same green.
  A reader can then see at a glance that the gap between a solid and a dashed line of one colour is the
  aggregation alone, identical segments, identical scorer, which is the finding (the working record, 25.5,
  28.1) and not a difference in method.
* **`METRICS`** and the readers that pull a metric out of a summary cell. Mean rank and MRR are primary,
  `recall@10` is the operational secondary and top-1 is a column, never a headline. Each
  metric is readable micro (per construct) and macro (per payload), because constructs are not
  independent trials and both are always reported.

**Older summaries are read, not rejected.** A `_summary.json` written before 17 Oct has no `arms` key,
no `*_mean_rank` and no `*_top10_macro`; the arm set is therefore discovered from the `by_threshold`
cells rather than from the `arms` key, and a metric a file does not carry comes back `None` so the
caller can say what it skipped instead of drawing a flat line at zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .arms import REPORT_ARM_NAMES

NAN = float("nan")


@dataclass(frozen=True)
class ArmStyle:
    label: str
    colour: str
    #: matplotlib fmt: marker + line style. Solid = max-pooled, dashed = ranked one query at a time.
    fmt: str
    lw: float
    ms: float
    #: hollow markers for the per-segment arms, so the pair reads as one arm under two aggregations
    hollow: bool = False


#: Drawn in `REPORT_ARM_NAMES` order wherever a legend is built.
ARM_DISPLAY: dict[str, ArmStyle] = {
    # `whole` is a wide halo: the motif arm tracks it almost exactly under alignment, and a same-width
    # line would hide it and read as a missing series.
    "whole": ArmStyle("ORF as called", "#b4b4b4", "-o", 4.5, 9.0),
    "auto": ArmStyle("motif segments, pooled", "#2f6fb3", "-D", 1.7, 5.0),
    "auto_per_segment": ArmStyle("motif segments, per-segment", "#2f6fb3", "--D", 1.7, 5.5, hollow=True),
    "bhs": ArmStyle("best-hit segments, pooled", "#2f9b6f", "-s", 1.7, 5.0),
    "bhs_per_segment": ArmStyle("best-hit segments, per-segment", "#2f9b6f", "--s", 1.9, 5.5, hollow=True),
    "oracle": ArmStyle("oracle payload span (ceiling)", "#b3402f", "-o", 1.9, 5.0),
    # The fourth source of segments, and the only one that consults nothing: fixed windows over the ORF.
    "tile": ArmStyle("fixed windows, pooled", "#7a4fb3", "-^", 1.7, 5.0),
    "tile_per_segment": ArmStyle("fixed windows, per-segment", "#7a4fb3", "--^", 1.9, 5.5, hollow=True),
}


#: The same arms in greyscale. **The encoding is preserved, not abandoned**: where `ARM_DISPLAY` uses
#: hue to say which segments a line is drawn from, this uses grey *value*, lightest for the whole frame
#: through to near-black for the labelled ceiling. Line style still says how the hits were aggregated and
#: hollow markers still mark the per-segment arms, so every reading rule in `legend_note` survives and
#: the paired arms still share a colour.
#:
#: Why it exists: a figure that depends on separating red from green fails when printed, photocopied, or
#: read by the ~8% of men with a red-green deficiency, and `oracle` red against `bhs` green is exactly
#: that pair. It is also what lets the portfolio deck embed the real figures without breaking a
#: monochrome rule. Greys are spaced by at least 0x2A so adjacent arms stay distinguishable on paper.
ARM_DISPLAY_MONO: dict[str, ArmStyle] = {
    "whole": ArmStyle("ORF as called", "#c8c8c8", "-o", 4.5, 9.0),
    "auto": ArmStyle("motif segments, pooled", "#8c8c8c", "-D", 1.7, 5.0),
    "auto_per_segment": ArmStyle("motif segments, per-segment", "#8c8c8c", "--D", 1.7, 5.5, hollow=True),
    "bhs": ArmStyle("best-hit segments, pooled", "#595959", "-s", 1.7, 5.0),
    "bhs_per_segment": ArmStyle("best-hit segments, per-segment", "#595959", "--s", 1.9, 5.5, hollow=True),
    "oracle": ArmStyle("oracle payload span (ceiling)", "#111111", "-o", 1.9, 5.0),
    # `tile` is the one arm whose grey is NOT 0x2A from its neighbours, and that is forced rather than
    # chosen: with `whole` 0xC8, `auto` 0x8C, `bhs` 0x59 and `oracle` 0x11 fixed by existing figures,
    # no value in 0x00-0xE9 is 0x2A from all four, and the only admissible band is 0xF2 and lighter,
    # which is invisible on white. 0x33 sits between `bhs` and `oracle` at gaps of 0x26 and 0x22, and
    # the separation that does the work on paper is the marker: `^` against `s` and `o`.
    # `test_no_fifth_grey_clears_the_spacing_rule` proves the impossibility rather than asserting it.
    "tile": ArmStyle("fixed windows, pooled", "#333333", "-^", 1.7, 5.0),
    "tile_per_segment": ArmStyle("fixed windows, per-segment", "#333333", "--^", 1.9, 5.5, hollow=True),
}


def arm_display(mono: bool = False) -> dict[str, ArmStyle]:
    """The palette to draw with. One accessor, so a script chooses once and cannot mix the two."""
    return ARM_DISPLAY_MONO if mono else ARM_DISPLAY


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    #: a rank axis is logarithmic and inverted, because 1 is the best value and the tail is long
    log: bool
    higher_is_better: bool
    micro: Callable[[dict, str], float | None]
    macro: Callable[[dict, str], float | None]


def _rate(cell: dict, arm: str, field: str) -> float | None:
    """A count over the arm's own n, as a percentage. Computable from every summary ever written."""
    n = cell.get(f"{arm}_n")
    c = cell.get(f"{arm}_{field}")
    if not n or c is None:
        return None
    return 100.0 * c / n


METRICS: dict[str, Metric] = {
    "mean_rank": Metric(
        "mean_rank", "mean rank of the payload's family (lower is better)", True, False,
        lambda cell, arm: cell.get(f"{arm}_mean_rank"),
        lambda cell, arm: cell.get(f"{arm}_mean_rank_macro")),
    "mrr": Metric(
        "mrr", "MRR (higher is better)", False, True,
        lambda cell, arm: cell.get(f"{arm}_mrr"),
        lambda cell, arm: cell.get(f"{arm}_mrr_macro")),
    "recall10": Metric(
        "recall10", "recall@10, an analyst reviewing ten hits sees it (%)", False, True,
        lambda cell, arm: _rate(cell, arm, "top10"),
        lambda cell, arm: cell.get(f"{arm}_top10_macro")),
    "top1": Metric(
        "top1", "retrieved at rank 1 (%), a column, not a headline", False, True,
        lambda cell, arm: _rate(cell, arm, "top1"),
        lambda cell, arm: cell.get(f"{arm}_top1_macro")),
}

#: The order a multi-metric figure stacks its rows: primary first, top-1 last and labelled as a column.
PRIMARY_METRICS: tuple[str, ...] = ("mean_rank", "mrr", "recall10")


def thresholds_of(agg: dict) -> list[float]:
    """The holdout sweep, loosest first, which is left-to-right into the twilight zone."""
    return sorted((float(k) for k in agg.get("by_threshold") or {}), reverse=True)


def cell(agg: dict, thr: float) -> dict:
    return (agg.get("by_threshold") or {}).get(f"{thr:.2f}") or {}


def arms_present(agg: dict) -> tuple[str, ...]:
    """Which report arms this summary actually carries, in `REPORT_ARM_NAMES` order.

    Discovered from the threshold cells, not from the `arms` key: every summary written before the key
    existed would otherwise come back empty and its figure would be blank rather than old.
    """
    cells = (agg.get("by_threshold") or {}).values()
    return tuple(a for a in REPORT_ARM_NAMES if any(c.get(f"{a}_n") for c in cells))


def value(agg: dict, thr: float, arm: str, metric: str, *, macro: bool = False) -> float | None:
    m = METRICS[metric]
    return (m.macro if macro else m.micro)(cell(agg, thr), arm)


def series(agg: dict, arm: str, metric: str, thresholds: list[float], *,
           macro: bool = False) -> list[float]:
    """One y value per threshold, `nan` where the summary does not carry it, so gaps stay gaps."""
    out = []
    for t in thresholds:
        v = value(agg, t, arm, metric, macro=macro)
        out.append(NAN if v is None else float(v))
    return out


def available(agg: dict, arm: str, metric: str, *, macro: bool = False) -> bool:
    """True when at least one threshold carries the metric for this arm."""
    return any(value(agg, t, arm, metric, macro=macro) is not None for t in thresholds_of(agg))


def missing_metrics(agg: dict, *, macro: bool = False) -> list[str]:
    """Which metrics this summary cannot supply for any arm it has: printed rather than drawn flat."""
    arms = arms_present(agg)
    return [k for k in METRICS
            if not any(available(agg, a, k, macro=macro) for a in arms)]


def legend_note(mono: bool = False) -> str:
    """The one sentence a reader needs to decode the colour/style encoding.

    It changes with the palette because the encoding does: in greyscale the distinguishing channel is
    value rather than hue, and a caption that said "colour" over a grey figure would be telling the
    reader to look for something that is not there.
    """
    which = "grey value (lighter = less of the frame removed)" if mono else "colour"
    return (f"{which} = which segments the query is, line style = how they were aggregated "
            "(solid = max-pooled into one ranking, dashed = each ranked in its own hit list)")
