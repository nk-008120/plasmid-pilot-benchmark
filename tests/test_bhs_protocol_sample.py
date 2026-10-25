"""`scripts/26_bhs_protocol.py` construct selection: the sample has to be the sample it says it is.

Loaded by path because `scripts/` is not an importable package, as in `tests/test_compare_tiers.py`.

Two properties matter and neither is visible by inspection. First, the sampling **frame** is the
constructs scorable under every arm, so a construct the paired table would have dropped anyway is never
drawn: otherwise a "30-construct probe" silently measures fewer than 30. Second, the sample depends on
the seed and the two id sets alone, not on CSV row order or dict insertion order, or the probe would not
be reproducible and its extrapolation would be a measurement of nothing.
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("bhs_protocol", ROOT / "scripts" / "26_bhs_protocol.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def write_twilight(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["threshold", "plasmid_id", "arm", "best_target_rank"])
        w.writeheader()
        w.writerows(rows)


def test_frame_is_only_the_constructs_scorable_under_every_arm(mod, tmp_path):
    """An arm with an empty rank disqualifies the construct: the paired table would drop it."""
    p = tmp_path / "tw.csv"
    write_twilight(p, [
        {"threshold": "0.25", "plasmid_id": "a", "arm": "whole", "best_target_rank": "3"},
        {"threshold": "0.25", "plasmid_id": "a", "arm": "oracle", "best_target_rank": "1"},
        {"threshold": "0.25", "plasmid_id": "b", "arm": "whole", "best_target_rank": "2"},
        {"threshold": "0.25", "plasmid_id": "b", "arm": "oracle", "best_target_rank": ""},
        {"threshold": "0.50", "plasmid_id": "c", "arm": "whole", "best_target_rank": "1"},
    ])
    assert mod.scorable_frame(p, 0.25) == ["a"]
    assert mod.scorable_frame(p, 0.50) == ["c"]


def test_frame_threshold_match_is_numeric_not_textual(mod, tmp_path):
    """`0.25`, `0.250` and `.25` are the same holdout. A string compare would silently return [] and the
    run would then score the whole tier, which is the expensive direction to be wrong in."""
    p = tmp_path / "tw.csv"
    write_twilight(p, [{"threshold": "0.250", "plasmid_id": "a", "arm": "whole", "best_target_rank": "1"}])
    assert mod.scorable_frame(p, 0.25) == ["a"]


def test_no_filter_means_no_restriction_and_is_distinguishable_from_empty(mod):
    """None is "score everything"; [] is "score nothing". Collapsing them would make a filter that
    matched nothing run the full tier instead."""
    assert mod.select_constructs(["a", "b"]) is None
    assert mod.select_constructs(["a", "b"], only=["zz"]) == []


def test_sample_is_deterministic_in_the_seed_and_not_in_the_input_order(mod):
    pool = [f"p{i:02d}" for i in range(40)]
    a = mod.select_constructs(pool, frame=pool, sample=7, seed=11)
    b = mod.select_constructs(list(reversed(pool)), frame=list(reversed(pool)), sample=7, seed=11)
    assert a == b
    assert len(a) == 7 == len(set(a))
    assert mod.select_constructs(pool, frame=pool, sample=7, seed=12) != a


def test_sample_is_drawn_from_the_frame_intersected_with_what_the_run_has(mod):
    """A frame id the results dir does not carry cannot be scored, so it must not consume a draw."""
    got = mod.select_constructs(["a", "b", "c"], frame=["b", "c", "d", "e"], sample=2, seed=0)
    assert got == ["b", "c"]


def test_sample_at_or_above_the_pool_size_returns_the_whole_pool_sorted(mod):
    assert mod.select_constructs(["b", "a"], frame=["a", "b"], sample=5) == ["a", "b"]
    assert mod.select_constructs(["b", "a"], frame=["a", "b"], sample=None) == ["a", "b"]


def test_only_wins_over_frame_and_is_intersected_with_what_the_run_has(mod):
    assert mod.select_constructs(["a", "b"], only=["b", "a", "q"], frame=["a"], sample=1) == ["a", "b"]


def test_read_id_file_ignores_comments_and_blanks(mod, tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("# a comment\n\nalpha\nbeta  # trailing\n\n", encoding="utf-8")
    assert mod.read_id_file(p) == ["alpha", "beta"]


def test_counting_scorer_passes_through_and_counts_calls_and_residues(mod):
    class Inner:
        name = "fake"

        def score_all(self, q):
            return [len(q)]

    s = mod.CountingScorer(Inner())
    assert s.name == "fake"
    assert s.score_all("ABC") == [3]
    assert s.score_all("DE") == [2]
    assert (s.calls, s.residues) == (2, 5)
