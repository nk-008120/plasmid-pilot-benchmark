"""Tests for `scripts/38_eba_metrics.py`.

The reader exists because `18_eba_sweep.py` prints top-1 only and top-1 pointed the wrong way on the
19 Oct probe. So the tests that matter are the ones pinning the quantities top-1 misses:
a blank rank is a miss and not rank 1, a miss contributes 0 to MRR, and the bimodality flag fires when
the mean detaches from the median.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "38_eba_metrics.py"


def _load():
    spec = importlib.util.spec_from_file_location("eba_metrics", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eba_metrics"] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load()


# --- read_rank: the trap 37_gap_closure.py records, pinned again here ------------------------------

@pytest.mark.parametrize("raw", ["", "   ", "NA", "none", "nan", None])
def test_a_blank_rank_is_a_miss_not_rank_zero_or_one(raw):
    assert m.read_rank(raw) is None


@pytest.mark.parametrize("raw,want", [("1", 1), ("7", 7), ("12.0", 12), ("29699", 29699)])
def test_a_present_rank_reads_back_as_an_int(raw, want):
    assert m.read_rank(raw) == want


# --- the metrics themselves -------------------------------------------------------------------------

def test_a_miss_contributes_zero_to_mrr_and_is_counted():
    s = m.summarise([1, 2, None, None])
    assert s["n"] == 4 and s["not_retrieved"] == 2
    # 1/1 + 1/2 + 0 + 0, over 4 constructs: not over the 2 that were retrieved.
    assert s["mrr"] == pytest.approx(0.375)
    assert s["recall_at_10"] == pytest.approx(50.0)


def test_mean_rank_is_computed_over_retrieved_only_and_median_is_reported_beside_it():
    s = m.summarise([1, 1, 1, 4000])
    assert s["mean_rank"] == pytest.approx(1000.75)
    assert s["median_rank"] == 1
    # The pair is the point: a mean 1000x its median is the 32.2 bimodality signature.
    assert s["mean_rank"] > 10 * s["median_rank"]


def test_recall_at_ten_is_inclusive_of_rank_ten():
    assert m.summarise([10])["recall_at_10"] == pytest.approx(100.0)
    assert m.summarise([11])["recall_at_10"] == pytest.approx(0.0)


def test_top_1_can_point_the_opposite_way_to_recall_at_10():
    """The probe's finding in miniature, so the reason this reader exists cannot rot."""
    sharp = m.summarise([1, 1, 5000, 5000])   # better top-1
    broad = m.summarise([2, 2, 2, 5000])      # better recall@10
    assert sharp["top_1"] > broad["top_1"]
    assert sharp["recall_at_10"] < broad["recall_at_10"]


# --- pairing ------------------------------------------------------------------------------------------

def test_mcnemar_is_one_when_nothing_is_discordant():
    assert m.mcnemar_exact(0, 0) == 1.0


def test_mcnemar_is_symmetric_and_bounded():
    assert m.mcnemar_exact(5, 1) == pytest.approx(m.mcnemar_exact(1, 5))
    assert 0.0 <= m.mcnemar_exact(9, 0) <= 1.0


def test_paired_counts_are_within_construct_and_separate_depth_from_rank():
    a = {"p1": 1, "p2": 50, "p3": None}
    b = {"p1": 3, "p2": 2, "p3": 4}
    pr = m.paired(a, b, ["p1", "p2", "p3"])
    # p2 and p3 cross into the top ten under b; nothing crosses out.
    assert (pr["wins"], pr["losses"]) == (2, 0)
    # p1 gets worse in rank while staying inside depth 10: a cost depth alone cannot see.
    assert (pr["rank_better"], pr["rank_worse"]) == (1, 1)


# --- end to end, against a written CSV -----------------------------------------------------------------

def _write(tmp_path: Path) -> Path:
    p = tmp_path / "sweep.csv"
    rows = [
        # one construct the control finds and the challenger misses, one the reverse
        ("align", "0.25", "pA", "P1", "whole", "1"),
        ("align", "0.25", "pB", "P2", "whole", ""),
        ("eba s=0.42", "0.25", "pA", "P1", "whole", "900"),
        ("eba s=0.42", "0.25", "pB", "P2", "whole", "2"),
        ("align", "0.25", "pA", "P1", "oracle", "1"),
        ("align", "0.25", "pB", "P2", "oracle", "1"),
        ("eba s=0.42", "0.25", "pA", "P1", "oracle", "1"),
        ("eba s=0.42", "0.25", "pB", "P2", "oracle", "1"),
    ]
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["scorer", "threshold", "plasmid_id", "payload_accession", "arm", "best_target_rank"])
        w.writerows(rows)
    return p


def test_main_runs_and_reports_both_directions(tmp_path, capsys):
    rc = m.main(["--sweep-csv", str(_write(tmp_path)), "--f-source", str(tmp_path / "absent.csv")])
    out = capsys.readouterr().out
    assert rc == 0
    # The pairing is one each way, which is exactly the case a marginal table would hide.
    assert "eba s=0.42-only 1" in out and "align-only 1" in out
    # With no f source, the binding count is stated rather than the strata silently vanishing.
    assert "have no f" in out


def test_a_missing_f_source_is_reported_not_silently_dropped(tmp_path, capsys):
    m.main(["--sweep-csv", str(_write(tmp_path)), "--f-source", str(tmp_path / "nope.csv")])
    out = capsys.readouterr().out
    assert "payload fraction known for 0 of 2 constructs" in out


def test_a_missing_sweep_csv_is_an_error_not_an_empty_table(tmp_path):
    assert m.main(["--sweep-csv", str(tmp_path / "nope.csv")]) == 2
