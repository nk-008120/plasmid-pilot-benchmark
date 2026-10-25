"""`scripts/30_bhs_strata.py`: the statistics that turn the segmenter's coverage into a claim.

Loaded by path because `scripts/` is not an importable package, as in `tests/test_compare_tiers.py`.

The arithmetic here is small and hand-rolled (no scipy in this project), which is exactly why it needs
tests: a Spearman that mishandles ties would report a confident correlation over a corpus that is
dominated by `f = 1.0`, and a confusion table with sensitivity and specificity transposed would read as
a usable triage signal when it is the opposite. Both failures look entirely plausible in the output.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("bhs_strata", ROOT / "scripts" / "30_bhs_strata.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_pearson_against_known_values(mod):
    assert mod.pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert mod.pearson([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)
    assert mod.pearson([1, 2, 3], [1, 1, 1]) is None  # zero variance, not a correlation of 0
    assert mod.pearson([1], [1]) is None


def test_pearson_is_not_fooled_by_a_monotone_but_nonlinear_pair(mod):
    """The reason Spearman is reported too: a perfect monotone relationship is not a Pearson 1.0."""
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]
    assert mod.pearson(x, y) < 0.99
    assert mod.spearman(x, y) == pytest.approx(1.0)


def test_ranks_with_ties_averages_tied_positions(mod):
    assert mod.ranks_with_ties([10, 20, 30]) == [1, 2, 3]
    assert mod.ranks_with_ties([10, 10, 30]) == [1.5, 1.5, 3]
    assert mod.ranks_with_ties([5, 5, 5, 5]) == [2.5, 2.5, 2.5, 2.5]


def test_spearman_over_a_point_mass_is_undefined_not_zero(mod):
    """The corpus's own shape: f is 1.0 for most constructs. If every f tied, the rank vector has no
    variance and there is no correlation to report: returning 0.0 would read as "measured, no effect"."""
    assert mod.spearman([0.3, 0.5, 0.9, 1.0], [1.0, 1.0, 1.0, 1.0]) is None


def test_confusion_counts_and_derived_rates(mod):
    pred = [True, True, False, False, False]
    truth = [True, False, True, False, False]
    c = mod.confusion(pred, truth)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 1, 2)
    assert c["sens"] == pytest.approx(0.5)
    assert c["spec"] == pytest.approx(2 / 3)
    assert c["ppv"] == pytest.approx(0.5)
    assert c["npv"] == pytest.approx(2 / 3)
    assert c["j"] == pytest.approx(0.5 + 2 / 3 - 1)


def test_confusion_rates_are_none_when_their_denominator_is_empty(mod):
    c = mod.confusion([False, False], [False, False])
    assert c["sens"] is None and c["j"] is None
    assert c["spec"] == pytest.approx(1.0)


def test_sweep_predicts_with_a_strict_less_than_so_the_threshold_is_not_double_counted(mod):
    """`cov < t`: a construct sitting exactly on t is predicted negative. The boundary has to be stated
    somewhere and a test is the only place it cannot drift from."""
    got = dict(mod.sweep([0.5], [True], [0.5, 0.55]))
    assert got[0.50]["tp"] == 0 and got[0.50]["fn"] == 1
    assert got[0.55]["tp"] == 1 and got[0.55]["fn"] == 0


def test_a_perfect_separator_scores_j_of_one(mod):
    cov = [0.2, 0.3, 0.9, 1.0]
    truth = [True, True, False, False]
    best = max(c["j"] for _, c in mod.sweep(cov, truth, [0.4, 0.5, 0.6]))
    assert best == pytest.approx(1.0)


def test_an_anticorrelated_predictor_scores_j_at_or_below_zero(mod):
    """The direction matters: cov is expected to be LOW when f is low. If the real relationship ran the
    other way, J must not come out positive and be read as a signal."""
    cov = [0.9, 1.0, 0.2, 0.3]
    truth = [True, True, False, False]
    assert max(c["j"] for _, c in mod.sweep(cov, truth, [0.4, 0.5, 0.6])) <= 0
