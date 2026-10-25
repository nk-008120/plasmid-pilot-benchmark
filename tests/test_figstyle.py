"""`pilot.figstyle`: the table the figure scripts read their arms and metrics from.

The defect this module exists to prevent is specific and has already happened once. `07_figures.py` and
`17_compare_tiers.py` each named the arms as a literal, so when `auto_per_segment` and `bhs_per_segment`
were added to `10_twilight.py` they appeared in every CSV and in no figure: no error, no warning, just
two missing lines. The tests here pin the three properties that stop a repeat:

1. **Arms are discovered from the data, not from a literal and not from the `arms` key**, because every
   summary written before 17 Oct has no `arms` key at all and must still plot.
2. **A metric a summary does not carry comes back `None`**, so the caller can report what it skipped.
   A silent 0.0 would draw a flat line at the bottom of the axis and read as a measured result.
3. **Colour means "which segments" and line style means "which aggregation"**, which is what makes the
   gap between a solid and a dashed line of one colour readable as the aggregation alone.
"""
from __future__ import annotations

import math

import pytest

from pilot.arms import ARM_NAMES, PER_SEGMENT_ARMS, REPORT_ARM_NAMES
from pilot.figstyle import (ARM_DISPLAY, METRICS, PRIMARY_METRICS, arms_present, available,
                            legend_note, missing_metrics, series, thresholds_of, value)


def cell(arm: str, *, n=10, top1=5, top10=8, mean_rank=None, mrr=None,
         top1_macro=None, top10_macro=None, mean_rank_macro=None, mrr_macro=None) -> dict:
    d = {f"{arm}_n": n, f"{arm}_top1": top1, f"{arm}_top10": top10}
    for k, v in (("mean_rank", mean_rank), ("mrr", mrr), ("top1_macro", top1_macro),
                 ("top10_macro", top10_macro), ("mean_rank_macro", mean_rank_macro),
                 ("mrr_macro", mrr_macro)):
        if v is not None:
            d[f"{arm}_{k}"] = v
    return d


def summary(cells: dict[str, dict], **top) -> dict:
    return {"by_threshold": cells, **top}


# --------------------------------------------------------------- arms are discovered, not declared

def test_arms_come_from_the_cells_not_from_the_arms_key():
    """A summary written before the `arms` key existed must still plot every arm it holds."""
    agg = summary({"0.25": {**cell("whole"), **cell("oracle")}})
    assert "arms" not in agg
    assert arms_present(agg) == ("whole", "oracle")


def test_a_declared_arm_with_no_data_is_not_drawn():
    """`--no-bhs` writes the arm name into `arms` but no cells; a phantom line is worse than none."""
    agg = summary({"0.25": {**cell("whole"), "bhs_n": 0}}, arms=list(REPORT_ARM_NAMES))
    assert arms_present(agg) == ("whole",)


def test_arms_come_back_in_report_order():
    agg = summary({"0.25": {**cell("oracle"), **cell("bhs_per_segment"), **cell("whole"),
                            **cell("auto")}})
    got = arms_present(agg)
    assert got == tuple(a for a in REPORT_ARM_NAMES if a in got)
    assert got == ("whole", "auto", "oracle", "bhs_per_segment")


def test_an_arm_present_at_only_one_threshold_still_counts():
    agg = summary({"1.00": {**cell("whole")}, "0.25": {**cell("whole"), **cell("bhs")}})
    assert arms_present(agg) == ("whole", "bhs")


def test_every_report_arm_has_a_style():
    """A new arm must not be plottable-but-unstyled: that is a KeyError in the middle of a figure."""
    assert set(ARM_DISPLAY) == set(REPORT_ARM_NAMES)


# ------------------------------------------------------------------- missing metrics stay missing

def test_a_metric_the_summary_lacks_is_none_not_zero():
    agg = summary({"0.25": cell("whole")})  # no mean_rank, no mrr
    assert value(agg, 0.25, "whole", "mean_rank") is None
    assert value(agg, 0.25, "whole", "mrr") is None
    assert not available(agg, "whole", "mean_rank")


def test_missing_metrics_are_reported_as_nan_in_a_series():
    """nan leaves a gap in the line; 0.0 would draw a measured-looking floor."""
    agg = summary({"0.90": cell("whole", mean_rank=2.0), "0.25": cell("whole")})
    ys = series(agg, "whole", "mean_rank", [0.90, 0.25])
    assert ys[0] == 2.0
    assert math.isnan(ys[1])


def test_recall10_and_top1_are_computable_from_an_old_summary():
    """Counts and n were always written, so these two never need a re-run: only the macro forms do."""
    agg = summary({"0.25": cell("whole", n=20, top1=5, top10=16)})
    assert value(agg, 0.25, "whole", "top1") == pytest.approx(25.0)
    assert value(agg, 0.25, "whole", "recall10") == pytest.approx(80.0)
    assert value(agg, 0.25, "whole", "recall10", macro=True) is None


def test_missing_metrics_lists_exactly_what_cannot_be_drawn():
    old = summary({"0.25": cell("whole", n=20, top1=5, top10=16, top1_macro=30.0)})
    assert missing_metrics(old) == ["mean_rank", "mrr"]
    assert missing_metrics(old, macro=True) == ["mean_rank", "mrr", "recall10"]
    new = summary({"0.25": cell("whole", mean_rank=3.0, mrr=0.5, mean_rank_macro=4.0,
                                mrr_macro=0.4, top10_macro=70.0, top1_macro=30.0)})
    assert missing_metrics(new) == []
    assert missing_metrics(new, macro=True) == []


def test_a_zero_n_arm_does_not_divide_by_zero():
    agg = summary({"0.25": {"whole_n": 0, "whole_top1": 0, "whole_top10": 0}})
    assert value(agg, 0.25, "whole", "recall10") is None


def test_an_absent_threshold_is_not_an_error():
    agg = summary({"0.25": cell("whole", mean_rank=3.0)})
    assert value(agg, 0.90, "whole", "mean_rank") is None
    assert math.isnan(series(agg, "whole", "mean_rank", [0.90])[0])


# ------------------------------------------------------------------------- the encoding itself

def test_colour_is_the_segments_and_style_is_the_aggregation():
    """The pair in one colour is the same segments under the two aggregations: that is the finding."""
    for pooled, per_seg in PER_SEGMENT_ARMS.items():
        assert ARM_DISPLAY[pooled].colour == ARM_DISPLAY[per_seg].colour
        assert "--" not in ARM_DISPLAY[pooled].fmt
        assert "--" in ARM_DISPLAY[per_seg].fmt
        assert ARM_DISPLAY[per_seg].hollow and not ARM_DISPLAY[pooled].hollow


def test_the_query_arms_are_visually_distinct_from_each_other():
    colours = {a: ARM_DISPLAY[a].colour for a in ARM_NAMES}
    assert len(set(colours.values())) == len(ARM_NAMES)


def test_top1_is_not_one_of_the_primary_rows():
    """the working record point 3: a column, never a headline. It must not sit in the primary figure."""
    assert "top1" in METRICS
    assert "top1" not in PRIMARY_METRICS
    assert PRIMARY_METRICS == ("mean_rank", "mrr", "recall10")


def test_mean_rank_is_the_only_logarithmic_lower_is_better_metric():
    assert METRICS["mean_rank"].log and not METRICS["mean_rank"].higher_is_better
    for k in ("mrr", "recall10", "top1"):
        assert not METRICS[k].log and METRICS[k].higher_is_better


def test_thresholds_run_loosest_first():
    """Left to right is into the twilight zone, which is the direction every twilight figure reads."""
    agg = summary({"0.25": {}, "1.00": {}, "0.50": {}})
    assert thresholds_of(agg) == [1.0, 0.5, 0.25]


def test_legend_note_explains_both_channels():
    note = legend_note().lower()
    assert "colour" in note and "line style" in note
    assert "pooled" in note and "own hit list" in note
