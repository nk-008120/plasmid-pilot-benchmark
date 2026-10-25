"""`26_bhs_protocol.py` must not hand the segmenter the payload, and must not mask it from the scorer.

this project measured the leak: at `--threshold 1.0` the predicate `identity > thr` is empty, so the
segmenter was given an unmasked catalogue holding the payload at 100% identity. It claimed exactly the
payload's span in iteration 1 and reproduced `oracle` while still reading as label-free: on **28 of 43**
constructs, against 0 of 43 at 0.25 and 0.50.

The fix cannot be "mask the payload", because at that same threshold the payload is itself a *target*
(`identity <= thr`), so masking it for the scorer would delete the question instead of answering it. Two
masks are therefore needed where there was one, and this file pins the asymmetry, since nothing in the
output would show it: a run with both masks equal and a run with them correctly split differ only in
ranks, and ranks are what the script is for.

Four properties, one per failure this could have:

* the segmenter mask holds the payload's own entry and the scoring mask does not: the fix itself;
* below 1.0 the two masks are **equal**, so the fix provably changes no published 0.25 or 0.50 number;
* condition B's filler exclusions land in **both**, because a filler relative is gone from the database
  and so is neither segmentable nor retrievable;
* a payload absent from the catalogue adds nothing, because `idx_of[acc]` would raise.

Loaded by path because `scripts/` is not an importable package, as in `tests/test_filler_exclusions.py`.
"""
from __future__ import annotations

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


class _H:
    """One row of `homologs.csv` as the script reads it: an accession and its identity to the payload."""

    def __init__(self, accession: str, identity: float):
        self.accession, self.identity = accession, identity


# P00001 is the payload. Its own row is identity 1.0, which is what every real homolog table carries
# for a payload against itself, and is the whole reason the sub-1.0 case is a no-op.
IDX = {"P00001": 0, "P00002": 1, "P00003": 2, "P00004": 3, "F00001": 4}
HS = [_H("P00001", 1.0), _H("P00002", 0.80), _H("P00003", 0.30), _H("P00004", 0.10)]


def test_the_segmenter_mask_holds_the_payload_and_the_scoring_mask_does_not(mod):
    """At 1.00 the predicate is empty, so this is the only thing standing between the segmenter and the
    answer. The scoring mask must stay empty, or the payload stops being retrievable as its own target."""
    scoring, seg = mod.construct_masks(HS, {}, IDX, "P00001", 1.0)
    assert list(scoring) == []
    assert list(seg) == [IDX["P00001"]]


@pytest.mark.parametrize("thr", [0.25, 0.35, 0.50, 0.70, 0.90, 0.99])
def test_below_one_the_two_masks_are_identical(mod, thr):
    """The sweep this script is quoted at. If these two ever differ the fix has changed a published
    number, which is exactly what it must not do: `1.0 > thr` already puts the payload in both."""
    scoring, seg = mod.construct_masks(HS, {}, IDX, "P00001", thr)
    assert list(scoring) == list(seg)
    assert IDX["P00001"] in list(scoring)


def test_filler_exclusions_are_in_both_masks(mod):
    """Condition B removes a filler's relatives from the database, so they are neither claimable by the
    segmenter nor retrievable by the scorer. Only the payload's own entry is asymmetric."""
    scoring, seg = mod.construct_masks(HS, {"F00001": 0.9}, IDX, "P00001", 0.50)
    assert IDX["F00001"] in list(scoring)
    assert IDX["F00001"] in list(seg)
    assert list(scoring) == list(seg)


def test_a_filler_at_or_below_the_threshold_is_kept(mod):
    """`ident > thr`, not `>=`: a filler relative exactly at the threshold stays in the database, which
    is the same boundary the payload's own relatives use. Two different comparisons here would be the
    coverage-denominator bug again."""
    scoring, _ = mod.construct_masks(HS, {"F00001": 0.50}, IDX, "P00001", 0.50)
    assert IDX["F00001"] not in list(scoring)


def test_a_payload_absent_from_the_catalogue_adds_nothing(mod):
    """`Q99999` is not in `idx_of`, and indexing it would raise rather than mask the wrong row. The
    script already skips such constructs upstream; this pins that the mask builder is safe anyway."""
    scoring, seg = mod.construct_masks(HS, {}, IDX, "Q99999", 1.0)
    assert list(scoring) == []
    assert list(seg) == []


def test_both_masks_are_sorted_unique_integer_arrays(mod):
    """They index numpy rows and are handed to `best_hit_segments` and to `r[mask] = -inf`. A duplicate
    is harmless, an unsorted array is not obviously so, and a float dtype would silently fancy-index
    wrongly, so the contract is pinned rather than assumed."""
    scoring, seg = mod.construct_masks(HS, {"F00001": 0.9}, IDX, "P00001", 0.50)
    for m in (scoring, seg):
        assert m.dtype.kind == "i"
        assert list(m) == sorted(set(m.tolist()))
