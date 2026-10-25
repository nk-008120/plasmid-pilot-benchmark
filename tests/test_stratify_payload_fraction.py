"""`scripts/29_stratify_by_payload_fraction.py`: the stratification and the paired delta.

Loaded by path because `scripts/` is not an importable package, as in `tests/test_compare_tiers.py`.

Three things here are worth a test rather than a reading. The strata must partition: an `f` that lands
in two bands, or in none, would double-count or silently drop a construct and the stratum sizes would
still sum to something plausible. The macro average must weight each payload once: the corpus's 37
constructs sharing one payload are the reason the arm exists and getting it wrong is invisible in the
output. And the paired delta must report transitions, not a difference of two counts: `+12` net is
consistent with 12 won and 0 lost or with 20 won and 8 lost, which are different findings.
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "stratify", ROOT / "scripts" / "29_stratify_by_payload_fraction.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_strata_partition_the_unit_interval(mod):
    """Every f in [0, 1] lands in exactly one stratum, and the bands do not overlap."""
    for i in range(0, 1001):
        f = i / 1000
        hits = [name for name, lo, hi in mod.STRATA if lo <= f < hi]
        assert len(hits) == 1, (f, hits)
        assert mod.stratum_of(f) == hits[0]


def test_stratum_boundaries_are_lower_inclusive(mod):
    assert mod.stratum_of(0.4999) == "f < 0.50"
    assert mod.stratum_of(0.50) == "0.50-0.70"
    assert mod.stratum_of(0.6999) == "0.50-0.70"
    assert mod.stratum_of(0.70) == "0.70-0.95"
    assert mod.stratum_of(0.9499) == "0.70-0.95"
    assert mod.stratum_of(0.95) == "f >= 0.95"
    assert mod.stratum_of(1.0) == "f >= 0.95"


def test_an_f_above_one_is_an_error_not_a_category(mod):
    """oracle_len > orf_len cannot happen and must not be binned silently if it ever does."""
    with pytest.raises(ValueError):
        mod.stratum_of(1.5)


def test_macro_weights_each_payload_once(mod):
    """Four constructs, one payload in three of them. Micro top-1 is 1/4; macro is the mean of the two
    payloads' own rates, 1/3 and 1/1, which is 0.667: not 0.25."""
    ranks = {"a": 5, "b": 5, "c": 1, "d": 1}
    acc = {"a": "P1", "b": "P1", "c": "P1", "d": "P2"}
    got = mod.micro_macro(ranks, ["a", "b", "c", "d"], acc)
    assert got["n"] == 4 and got["n_payloads"] == 2
    assert got["top1"] == 2
    assert got["top1_micro"] == pytest.approx(50.0)
    assert got["top1_macro"] == pytest.approx(100 * (1 / 3 + 1) / 2)
    assert got["mean"] == pytest.approx(3.0)
    assert got["mean_macro"] == pytest.approx(((5 + 5 + 1) / 3 + 1) / 2)


def test_micro_macro_ignores_constructs_with_no_rank(mod):
    got = mod.micro_macro({"a": 1}, ["a", "b"], {"a": "P1", "b": "P2"})
    assert got["n"] == 1 and got["n_payloads"] == 1


def test_micro_macro_on_an_empty_set_is_empty_not_a_crash(mod):
    got = mod.micro_macro({}, [], {})
    assert got["n"] == 0 and got["top1_macro"] is None


def test_paired_reports_transitions_and_not_just_the_net(mod):
    """The case the net hides: 2 won and 1 lost is net +1, the same net as 1 won and 0 lost."""
    a = {"p": 3, "q": 4, "r": 1, "s": 1}
    b = {"p": 1, "q": 1, "r": 9, "s": 1}
    pr = mod.paired(a, b, ["p", "q", "r", "s"])
    assert pr["net"] == 1
    assert sorted(pr["won"]) == ["p", "q"]
    assert pr["lost"] == ["r"]
    assert pr["n_paired"] == 4
    assert pr["n_rank_changed"] == 3
    assert pr["d_mean"] == pytest.approx(((1 + 1 + 9 + 1) - (3 + 4 + 1 + 1)) / 4)


def test_paired_uses_only_constructs_scorable_under_both_arms(mod):
    pr = mod.paired({"p": 3, "q": 2}, {"p": 1}, ["p", "q"])
    assert pr["n_paired"] == 1 and pr["won"] == ["p"]


def test_payload_fractions_skips_a_missing_oracle_span(mod, tmp_path):
    p = tmp_path / "len.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["plasmid_id", "orf_len", "oracle_len"])
        w.writeheader()
        w.writerows([{"plasmid_id": "a", "orf_len": "400", "oracle_len": "200"},
                     {"plasmid_id": "b", "orf_len": "400", "oracle_len": ""}])
    got = mod.payload_fractions(p)
    assert set(got) == {"a"}
    assert got["a"] == (0.5, 400, 200)
