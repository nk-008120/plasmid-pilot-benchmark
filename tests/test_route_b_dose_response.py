"""Guards for `scripts/46_route_b_dose_response.py`, the route B dose-response figure.

Two of these are the figure's whole argument rather than housekeeping:

* `test_no_line_in_the_figure_spans_two_strata` is the drawn form of this project's warning. `whole`
  has mean rank 3.00 at 0.70-0.95 and 19.33 at f >= 0.95, so a line through one arm across the slots
  would read as "dilution helps" when what changed is which payloads are in the slot. Every line in
  the figure must therefore be vertical: a contrast between two arms on the same constructs.
* `test_a_stratum_whose_arms_differ_is_not_called_a_control` keeps the shaded slot honest. The zero
  gap at f >= 0.95 is structural, and the script recovers it from the numbers instead of asserting it
  from the stratum's name, so a future run where it stops holding is reported and not relabelled.

The rest guard the joins and the units: recall is drawn in `pilot.figstyle`'s percent and not the
JSON's fraction, counts come from the arms rather than from an assumption, and no row of the written
table is ever a pooled scalar across strata.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(stem: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / stem)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dr = pytest.importorskip("matplotlib") and _load("46_route_b_dose_response.py", "rb_dose_response")


def _cell(mean, mrr, rec, n=10, n_payloads=5, macro_mean=None):
    return {"micro": {"n": n, "mean_rank": mean, "median_rank": 1, "mrr": mrr,
                      "recall_at_10": rec, "top1": 1, "n_ge_10000": 0},
            "macro": {"n_payloads": n_payloads, "mean_rank": macro_mean if macro_mean else mean,
                      "mrr": mrr, "recall_at_10": rec}}


def doc():
    """Two holdouts, two strata: one with a gap, one where the two arms are identical throughout.

    Each holdout gets its own copy of the cells. Sharing them would let a test that edits one holdout
    silently edit the other, and the tests below turn on exactly that distinction.
    """
    def diluted():
        return {"whole": _cell(81.19, 0.229, 0.562, n=16, n_payloads=7, macro_mean=68.19),
                "oracle": _cell(7.81, 0.761, 0.750, n=16, n_payloads=7, macro_mean=11.71),
                "bhs_per_segment": _cell(3.94, 0.814, 0.938, n=16, n_payloads=7, macro_mean=7.14)}

    def flat():
        return {"whole": _cell(19.33, 0.557, 0.833, n=6, n_payloads=4, macro_mean=27.25),
                "oracle": _cell(19.33, 0.557, 0.833, n=6, n_payloads=4, macro_mean=27.25),
                "bhs_per_segment": _cell(1.0, 1.0, 1.0, n=6, n_payloads=4)}

    return {"thresholds": {"0.25": {"<0.50": diluted(), ">=0.95": flat()},
                           "1.0": {"<0.50": diluted(), ">=0.95": flat()}}}


def test_the_strata_are_the_ones_the_report_script_wrote():
    """A figure that silently renamed a stratum would drop it, not fail: script 45 is the source."""
    rpt = _load("45_route_b_strata.py", "rb_strata_for_figure")
    assert dr.STRATUM_ORDER == rpt.STRATUM_NAMES
    assert dr.F_STRATA == rpt.F_STRATA
    for f in (0.0, 0.2267, 0.4999, 0.50, 0.6999, 0.70, 0.9499, 0.95, 0.9762, 1.0):
        assert dr.stratum(f) == rpt.stratum(f), f


def test_every_stratum_has_a_display_label():
    assert set(dr.FIG_LABEL) == set(dr.STRATUM_LABEL) == set(dr.STRATUM_ORDER)


def test_holdouts_run_loosest_first_into_the_twilight_zone():
    assert dr.thresholds_of(doc()) == ["1.0", "0.25"]
    assert dr.strata_present(doc()) == ("<0.50", ">=0.95")


def test_recall_is_scaled_into_figstyles_percent_and_the_others_are_not():
    """`figstyle.METRICS['recall10']` is labelled and axis-scaled in percent; the JSON is a fraction.
    Drawing 0.938 on a 0-100 axis would put every recall point on the floor."""
    d = doc()
    assert dr.value(d, "0.25", "<0.50", "oracle", "recall10") == pytest.approx(75.0)
    assert dr.value(d, "0.25", "<0.50", "oracle", "mrr") == pytest.approx(0.761)
    assert dr.value(d, "0.25", "<0.50", "whole", "mean_rank") == pytest.approx(81.19)


def test_macro_is_a_different_number_and_is_read_separately():
    d = doc()
    assert dr.value(d, "0.25", "<0.50", "whole", "mean_rank", macro=True) == pytest.approx(68.19)
    assert dr.value(d, "0.25", "<0.50", "oracle", "mean_rank", macro=True) == pytest.approx(11.71)


def test_a_metric_or_arm_the_run_does_not_carry_comes_back_none():
    d = doc()
    assert dr.value(d, "0.25", "<0.50", "auto_per_segment", "mrr") is None
    assert dr.value(d, "0.25", "0.70-0.95", "whole", "mrr") is None
    assert dr.value(d, "0.9", "<0.50", "whole", "mrr") is None


def test_counts_come_from_the_arms_and_a_disagreement_is_reported():
    d = doc()
    assert dr.counts(d, "0.25", "<0.50") == (16, 7)
    assert dr.counts(d, "0.25", ">=0.95") == (6, 4)
    assert dr.disagreeing_counts(d) == []
    d["thresholds"]["0.25"]["<0.50"]["oracle"]["micro"]["n"] = 15
    assert dr.counts(d, "0.25", "<0.50")[0] is None, "an unpaired cell has no single n"
    assert dr.disagreeing_counts(d) == [("0.25", "<0.50", [15, 16])]


def test_the_zero_gap_stratum_is_recovered_from_the_numbers():
    assert dr.zero_gap_strata(doc()) == (">=0.95",)


def test_a_stratum_whose_arms_differ_is_not_called_a_control():
    """One macro cell out of twelve is enough: the gap is claimed structural, so it is zero or it is
    not, and a near-zero gap must be drawn as a measurement."""
    d = doc()
    d["thresholds"]["1.0"][">=0.95"]["oracle"]["macro"]["mrr"] = 0.558
    assert dr.zero_gap_strata(d) == ()


def test_a_stratum_with_nothing_comparable_is_not_a_control_by_default():
    d = {"thresholds": {"0.25": {"<0.50": {"whole": _cell(3.0, 0.5, 0.5)}}}}
    assert dr.zero_gap_strata(d) == (), "one arm present is not a measured zero gap"


def test_the_rank_failure_count_is_summed_from_the_run_not_written_into_the_caption():
    """a mean rank is reportable only beside its count of rank >= 10,000 rows. The
    caption states that count, so it has to come from the numbers and change when they do."""
    d = doc()
    assert dr.rank_failures(d, dr.PAIR) == 0
    d["thresholds"]["0.25"]["<0.50"]["whole"]["micro"]["n_ge_10000"] = 2
    assert dr.rank_failures(d, dr.PAIR) == 2
    assert dr.rank_failures(d, dr.CEILING_PAIR) == 0, "only the drawn arms are counted"


def test_fold_is_the_ratio_and_survives_a_zero():
    assert dr.fold(81.19, 7.81) == pytest.approx(10.395, abs=1e-3)
    assert dr.fold(19.33, 19.33) == pytest.approx(1.0)
    assert dr.fold(1.0, 0) is None and dr.fold(None, 2.0) is None


def test_f_census_counts_the_pairs_at_f_exactly_one(tmp_path):
    """The control caption says "3 of 6 at f = 1.000"; it is read, not assumed."""
    p = tmp_path / "m.csv"
    p.write_text("source_id,payload_accession,f\na,P1,1.0\nb,P2,0.9762\nc,P3,0.45\nd,P4,\n",
                 encoding="utf-8-sig")
    census = dr.f_census([p])
    assert census[">=0.95"] == {"n": 2, "n_exact": 1, "min_f": 0.9762}
    assert census["<0.50"]["n"] == 1, "a row with no f is skipped, not counted as 0"


def test_the_control_note_says_what_it_measured_and_what_it_read():
    census = {">=0.95": {"n": 6, "n_exact": 3, "min_f": 0.9762}}
    note = dr.control_note(doc(), (">=0.95",), census)
    assert "3 of 6" in note and "0.976" in note
    assert "by construction" in note
    assert "did not reproduce" in dr.control_note(doc(), (), census)


def test_the_table_is_per_stratum_and_never_pools():
    rows = dr.table(doc(), dr.PAIR, (">=0.95",))
    assert {r["f_stratum"] for r in rows} == {"<0.50", ">=0.95"}
    assert all(r["n_pairs"] in (16, 6) for r in rows)
    assert len(rows) == 2 * 2 * 3 * 2, "holdout x stratum x metric x aggregation"
    ranks = [r for r in rows if r["metric"] == "mean_rank" and r["aggregation"] == "micro"
             and r["holdout"] == "0.25" and r["f_stratum"] == "<0.50"]
    assert ranks[0]["fold_better"] == 10.396, "the written ratio is rounded, not re-derived"
    assert ranks[0]["is_control"] is False
    mrr = [r for r in rows if r["metric"] == "mrr"][0]
    assert mrr["fold_better"] is None, "a ratio is meaningless on a higher-is-better metric"


def test_no_line_in_the_figure_spans_two_strata():
    """a cross-stratum step mixes an effect with a sample change. Every line drawn here
    is a vertical paired contrast, so every one has a single x."""
    fig = dr.figure(doc(), dr.PAIR, (">=0.95",), ["note"], title="t")
    lines = [ln for ax in fig.axes for ln in ax.lines]
    assert lines, "the figure drew no connectors at all"
    for ln in lines:
        xs = set(ln.get_xdata())
        assert len(xs) == 1, f"a line spans {sorted(xs)}: that is a cross-stratum trend"


def test_every_stratum_carries_its_n_on_the_axis():
    fig = dr.figure(doc(), dr.PAIR, (">=0.95",), ["note"], title="t")
    labels = [t.get_text() for ax in fig.axes for t in ax.get_xticklabels() if t.get_text()]
    assert labels
    assert all("pairs" in t and "payloads" in t for t in labels)
    assert any("16 pairs" in t and "7 payloads" in t for t in labels)
    assert any("6 pairs" in t and "4 payloads" in t for t in labels)


def test_the_control_slot_is_shaded_once_per_panel():
    fig = dr.figure(doc(), dr.PAIR, (">=0.95",), ["note"], title="t")
    panels = [ax for ax in fig.axes if ax.get_xticklabels() and ax.lines]
    assert panels
    for ax in panels:
        assert len(ax.patches) == 1, "one shaded slot per panel, the measured zero-gap control"


def test_main_writes_both_figures_and_the_table(tmp_path):
    import json
    strata = tmp_path / "route_b_strata.json"
    strata.write_text(json.dumps(doc()), encoding="utf-8")
    out = tmp_path / "figs"
    assert dr.main(["--strata", str(strata), "--out", str(out), "--manifest"]) == 0
    for stem in ("fig_d11_route_b_dose_response", "fig_d12_route_b_ceiling"):
        assert (out / f"{stem}.png").stat().st_size > 0
        assert (out / f"{stem}.pdf").stat().st_size > 0
    body = (out / "route_b_dose_response.csv").read_text(encoding="utf-8-sig")
    assert body.startswith("holdout,f_stratum,is_control,metric,aggregation")
    assert "whole,oracle" in body.splitlines()[0]


def test_main_says_so_rather_than_crashing_when_the_report_is_missing(tmp_path, capsys):
    assert dr.main(["--strata", str(tmp_path / "nope.json"), "--out", str(tmp_path)]) == 2
    assert "45_route_b_strata.py" in capsys.readouterr().out
