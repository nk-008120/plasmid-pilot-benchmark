"""`scripts/10_twilight.py`'s two aggregations: what `auto`/`bhs` mean, and what the new arms mean.

Loaded by path because `scripts/` is not an importable package, as in `tests/test_bhs_protocol_sample.py`.

Three properties matter here and none of them is visible by inspection.

First, **the pooled path must stay max-over-the-stacked-rows.** Every comparison in the working record
reads `auto` and `bhs` as that quantity and the twilight CSVs are keyed on (threshold, plasmid_id, arm),
so a refactor that changed the aggregation would reinterpret stored results with no change of column
name: the one defect class in this repo that no output would have shown. `test_pooled_is_max_over_rows`
pins it.

Second, **the per-segment arms are additions, not redefinitions.** `REPORT_ARM_NAMES` puts them after
the four query arms, which is what makes "the existing arms did not move" a diff rather than an argument:
drop the rows whose arm is in `PER_SEGMENT_ARMS.values()` and the old file comes back.

Third, **the per-segment rank is the best rank any single query reaches in its own hit list**, which is
`scripts/26_bhs_protocol.py`'s `min` over the per-segment ranks. The two implementations have to agree or
one of them is wrong, so the aggregation is tested against hand-computed ranks rather than against itself.
"""
from __future__ import annotations

import importlib.util
import inspect
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from pilot.arms import ARM_NAMES, PER_SEGMENT_ARMS, REPORT_ARM_NAMES, Arms

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tw():
    spec = importlib.util.spec_from_file_location("twilight", ROOT / "scripts" / "10_twilight.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------- the pooled path's meaning

def test_pooled_is_max_over_rows(tw):
    """The guard against a later refactor quietly changing what `auto` and `bhs` are."""
    rows = [np.array([1.0, 9.0, 3.0], dtype=np.float32),
            np.array([4.0, 2.0, 3.0], dtype=np.float32),
            np.array([0.0, 0.0, 8.0], dtype=np.float32)]
    assert np.array_equal(tw.pool_scores(rows), np.max(np.vstack(rows), axis=0))
    assert np.array_equal(tw.pool_scores(rows), np.array([4.0, 9.0, 8.0], dtype=np.float32))


def test_pooled_single_query_is_the_row_itself(tw):
    """A one-query arm must not be perturbed by the stacking, in value or in dtype."""
    row = np.array([1.5, 2.5], dtype=np.float32)
    out = tw.pool_scores([row])
    assert out is row
    assert np.array_equal(out, np.max(np.vstack([row]), axis=0))
    assert out.dtype == row.dtype


def test_pooled_matches_a_hand_computed_max(tw):
    rng = np.random.default_rng(7)
    rows = [rng.random(50).astype(np.float32) for _ in range(4)]
    expected = np.array([max(r[i] for r in rows) for i in range(50)], dtype=np.float32)
    assert np.array_equal(tw.pool_scores(rows), expected)


# ------------------------------------------------------------------- the per-segment path's meaning

def test_per_segment_takes_the_best_rank_any_single_query_reaches(tw):
    """The arm's point: the payload's own segment ranks it 1 even though pooling buries it.

    Reference 0 is the target. Segment A scores a distractor family (refs 1-3) far above it; segment B
    scores the target top. Pooled, the distractor's scores are re-admitted and the target falls to 4.
    """
    seg_a = np.array([1.0, 9.0, 8.0, 7.0], dtype=np.float32)
    seg_b = np.array([5.0, 0.5, 0.4, 0.3], dtype=np.float32)
    tgt, mask = np.array([0]), np.array([], dtype=int)
    assert tw.best_target_rank(tw.pool_scores([seg_a, seg_b]), tgt, mask) == 4
    assert tw.per_segment_rank([seg_a, seg_b], tgt, mask) == 1


def test_per_segment_equals_pooled_when_there_is_one_query(tw):
    """The binding bound: a single-query construct cannot differ between the two aggregations."""
    rng = np.random.default_rng(11)
    row = rng.random(200).astype(np.float32)
    tgt = np.array([3, 57, 180])
    mask = np.array([7, 9])
    assert (tw.per_segment_rank([row], tgt, mask)
            == tw.best_target_rank(tw.pool_scores([row]), tgt, mask))


def test_per_segment_is_the_min_over_each_querys_own_rank(tw):
    """Agreement with `scripts/26_bhs_protocol.py`, whose per-segment value is `min` over the ranks."""
    rng = np.random.default_rng(3)
    rows = [rng.random(300).astype(np.float32) for _ in range(5)]
    tgt = np.array([11, 250])
    mask = np.array([0, 1, 2])
    each = [tw.best_target_rank(r, tgt, mask) for r in rows]
    assert tw.per_segment_rank(rows, tgt, mask) == min(each)


def test_per_segment_honours_the_holdout_mask(tw):
    """A masked entry is not a target: the per-segment path must mask exactly as the pooled one does."""
    row = np.array([9.0, 5.0, 1.0], dtype=np.float32)
    # ref 0 is held out, ref 1 is the target: masking ref 0 leaves the target top
    assert tw.per_segment_rank([row], np.array([1]), np.array([0])) == 1
    assert tw.per_segment_rank([row], np.array([1]), np.array([], dtype=int)) == 2


def test_per_segment_is_none_only_when_no_query_is_scorable(tw):
    """`None` has to mean the same thing it means for `best_target_rank`, or n shrinks silently."""
    row = np.array([1.0, 2.0], dtype=np.float32)
    assert tw.per_segment_rank([row], np.array([], dtype=int), np.array([], dtype=int)) is None
    # target held out: unscorable through either aggregation
    assert tw.per_segment_rank([row], np.array([1]), np.array([1])) is None
    assert tw.per_segment_rank([row, row], np.array([0]), np.array([], dtype=int)) == 2


def test_per_segment_skips_the_unscorable_queries_rather_than_failing(tw):
    """One query scoring -inf on the target must not discard the others' ranks."""
    good = np.array([5.0, 1.0], dtype=np.float32)
    dead = np.array([-np.inf, -np.inf], dtype=np.float32)
    assert tw.per_segment_rank([dead, good], np.array([0]), np.array([], dtype=int)) == 1


def test_per_segment_takes_no_scorer_at_all(tw):
    """The arm is free by construction: its inputs are score rows, so it cannot start a catalogue pass.

    Guards the one way this could stop being true: a signature that admits a scorer and re-scores the
    segments. If this assertion ever needs changing, the arm has stopped being free and the runtime
    comparison in the report has to be redone.
    """
    params = inspect.signature(tw.per_segment_rank).parameters
    assert list(params) == ["arm_rows", "target_idx", "mask_idx"]
    rows = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 0.0], dtype=np.float32)]
    assert tw.per_segment_rank(rows, np.array([0]), np.array([], dtype=int)) == 1


# ------------------------------------------------------------------------------- the arm-name contract

def test_report_arms_extend_the_query_arms_without_renaming_them():
    assert REPORT_ARM_NAMES[:len(ARM_NAMES)] == ARM_NAMES
    assert set(REPORT_ARM_NAMES) - set(ARM_NAMES) == set(PER_SEGMENT_ARMS.values())
    # the new names come last, which is what makes the "existing rows unchanged" check a diff
    assert all(a in PER_SEGMENT_ARMS.values() for a in REPORT_ARM_NAMES[len(ARM_NAMES):])


def test_per_segment_arms_are_new_names_not_redefinitions():
    for query_arm, seg_arm in PER_SEGMENT_ARMS.items():
        assert query_arm in ARM_NAMES
        assert seg_arm not in ARM_NAMES
        assert seg_arm == f"{query_arm}_per_segment"


def test_only_the_segment_arms_have_a_per_segment_counterpart():
    """`whole` and `oracle` are single-query by definition; a per-segment row for them would be a copy.

    Derived rather than enumerated, and the change is deliberate. This read
    `set(PER_SEGMENT_ARMS) == {"auto", "bhs"}`, which pinned the *list of segment arms as of the day it
    was written* while its own docstring states a *rule* about single-query arms. Adding `tile` in
    October broke the assertion without breaking the rule, which is the signature of a test that
    enumerates where it meant to derive. A segment arm is one whose `Arms` field holds a list of
    segments; `whole` and `oracle` are properties returning one string and can never acquire a
    counterpart. So the rule is now checked against the dataclass, and a fifth segment arm will pass
    while a per-segment row for `whole` still fails.
    """
    segment_arms = {f.name for f in fields(Arms) if f.name in ARM_NAMES}
    assert set(PER_SEGMENT_ARMS) == segment_arms
    assert {"whole", "oracle"}.isdisjoint(PER_SEGMENT_ARMS)
    assert segment_arms == {"auto", "bhs", "tile"}, "a segment arm was added or removed; intended?"


# ----------------------------------------------------------------------------- the macro average

def test_macro_weights_each_payload_once(tw):
    """37 constructs sharing a payload are one answer repeated, not 37 trials."""
    per_payload = [[1, 1, 1, 1], [5]]  # payload A top-1 four times, payload B at rank 5 once
    assert tw.macro_avg(per_payload, lambda v: 100 * sum(1 for x in v if x == 1) / len(v), 1) == 50.0
    assert tw.macro_avg(per_payload, lambda v: sum(v) / len(v), 2) == 3.0  # micro would be 1.8
    assert tw.macro_avg([], lambda v: 1.0, 2) is None
