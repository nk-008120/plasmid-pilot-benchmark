"""Guards for `scripts/37_gap_closure.py`, which computes the project's headline number."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load():
    spec = importlib.util.spec_from_file_location(
        "s37", Path(__file__).resolve().parents[1] / "scripts/37_gap_closure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load()


def test_a_blank_rank_is_a_miss_and_not_rank_one():
    """The trap this reader exists to avoid: `int(x or 0)` promotes every miss to the best rank.

    A construct the arm never retrieved has an empty rank cell. Coercing that to 0 or 1 turns the worst
    possible outcome into the best one, silently, on exactly the constructs the benchmark is about.
    """
    for blank in ("", "   ", "NA", "none", "NaN", None):
        assert MOD.read_rank(blank) is None
    assert MOD.read_rank("1") == 1
    assert MOD.read_rank("40750") == 40750
    assert MOD.read_rank("3.0") == 3

    rows = [{"r": "1"}, {"r": ""}, {"r": "5"}, {"r": "11"}]
    assert MOD.hits(rows, "r") == 2                      # ranks 1 and 5; blank and 11 are not hits
    s = MOD.arm_summary(rows, "r")
    assert s["not_retrieved"] == 1
    assert s["mean_rank"] == pytest.approx((1 + 5 + 11) / 3, abs=0.005)  # the miss is excluded, not 0


def test_closure_is_undefined_rather_than_zero_when_a_tier_has_no_gap():
    """`synth_random` has oracle == whole. 0% would read as the arm failing; it is the tier."""
    no_gap = [{"rank_whole": "1", "rank_per_segment": "1", "rank_oracle": "1"}] * 5
    assert MOD.closure(no_gap)["closed_pct"] is None

    gap = ([{"rank_whole": "900", "rank_per_segment": "2", "rank_oracle": "2"}] * 4
           + [{"rank_whole": "1", "rank_per_segment": "1", "rank_oracle": "1"}])
    c = MOD.closure(gap)
    assert c["gap_oracle_minus_whole"] == 4
    assert c["closed_pct"] == 100.0
    assert c["residual_to_oracle"] == 0


def test_mcnemar_counts_only_discordant_pairs_and_is_two_sided():
    assert MOD.mcnemar_exact(0, 0) == 1.0           # no discordant pairs is no evidence
    assert MOD.mcnemar_exact(5, 5) == 1.0
    assert MOD.mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert MOD.mcnemar_exact(0, 10) == MOD.mcnemar_exact(10, 0)   # symmetric
    assert MOD.mcnemar_exact(141, 0) < 1e-30

    rows = [{"a": "900", "b": "1"}, {"a": "1", "b": "900"}, {"a": "1", "b": "1"}]
    p = MOD.paired(rows, "a", "b")
    assert (p["wins"], p["losses"]) == (1, 1)       # the concordant row is excluded


def test_the_headline_reproduces_from_the_committed_csvs():
    """this project's number, pinned to the files rather than to a run.

    If the detail CSVs are ever regenerated, this fails and the section has to be re-read rather than
    silently carrying a stale figure.
    """
    root = Path(__file__).resolve().parents[1]
    path = root / "results/synthetic_real/C_detail_238.csv"
    if not path.exists():
        pytest.skip("detail CSV not present")
    import csv as _csv
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(_csv.DictReader(fh))
    assert len(rows) == 238
    c = MOD.closure(rows)
    assert c["whole"] == 84 and c["oracle"] == 224 and c["bhs_per_segment"] == 225
    assert c["closed_pct"] >= 100.0
    p = MOD.paired(rows, "rank_whole", "rank_per_segment")
    assert (p["wins"], p["losses"]) == (141, 0)
    assert p["p_mcnemar"] < 1e-30
