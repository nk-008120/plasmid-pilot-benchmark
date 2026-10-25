"""Tests for `--bhs-scorer {run,align}` in `scripts/10_twilight.py` (open item 42).

The flag exists because `best_hit_segments` stops against `homology.score_floor`, which is the
significance gate inverted into **raw BLOSUM62 units**. A cosine scorer produces numbers in [-1, 1],
so under `--scorer esm2:` the segmenter compares them against a floor of 99.5-116.1, every construct
stops at k = 0, and the arm degenerates to the whole ORF without saying so. That is the mechanism
behind this project's `--no-bhs`, and `test_a_cosine_scale_scorer_stops_the_segmenter_at_k_zero` pins
it so the reason for the flag cannot rot.

The acceptance test the the run board names is a no-op: `--scorer align --bhs-scorer align` must reproduce
`--scorer align` cell for cell. It is tested here at the level where it is *guaranteed* rather than
merely observed: `resolve_bhs_scorer` returns the run scorer **itself** when that scorer already
aligns, so there is no second code path to drift along. The cell-for-cell run is still worth doing
once on real data, and is recorded in CONTEXT; identity is what makes it cheap to trust afterwards.
"""
from __future__ import annotations

import importlib.util
import inspect
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pilot.align import AlignmentScorer
from pilot.decompose import best_hit_segments

parasail = pytest.importorskip("parasail", reason="the alignment arm is an optional dependency")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "10_twilight.py"


def _load():
    spec = importlib.util.spec_from_file_location("twilight", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["twilight"] = mod
    spec.loader.exec_module(mod)
    return mod


tw = _load()


def _rand_prot(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(n))


class _CosineScorer:
    """Scores in [-1, 1], like every embedding arm and unlike the floor's units."""

    name = "cosine-stub"

    def __init__(self, subjects: list[str]):
        self.subjects = subjects

    def score_all(self, query: str) -> np.ndarray:
        qs = set(query)
        return np.array([len(qs & set(s)) / max(len(qs | set(s)), 1) for s in self.subjects])


@pytest.fixture
def world():
    rng = random.Random(11)
    payload = _rand_prot(200, rng)
    other = _rand_prot(200, rng)
    refs = [payload, other] + [_rand_prot(150, rng) for _ in range(8)]
    return {"orf": payload + other, "refs": refs, "db": sum(len(s) for s in refs)}


# --- the mechanism the flag exists to fix -------------------------------------------------------------

def test_a_cosine_scale_scorer_stops_the_segmenter_at_k_zero(world):
    """The bug, pinned. Nothing in [-1, 1] clears a floor of ~100, so the arm emits no segments."""
    res = best_hit_segments(world["orf"], world["refs"], _CosineScorer(world["refs"]), world["db"])
    assert res.segments == ()
    assert res.floor > 1.0, "if the floor ever lands inside cosine range this test proves nothing"


def test_the_alignment_scorer_the_resolver_hands_back_does_segment(world):
    """And the fix, measured against it on the same construct and the same catalogue."""
    sc = tw.resolve_bhs_scorer("align", _CosineScorer(world["refs"]), world["refs"])
    res = best_hit_segments(world["orf"], world["refs"], sc, world["db"])
    assert len(res.segments) >= 1


# --- resolution ---------------------------------------------------------------------------------------

def test_run_mode_hands_back_the_run_scorer_untouched(world):
    sc = _CosineScorer(world["refs"])
    assert tw.resolve_bhs_scorer("run", sc, world["refs"]) is sc


def test_align_mode_is_a_no_op_by_identity_when_the_run_scorer_already_aligns(world):
    """The the run board's acceptance condition, at the level where it is guaranteed rather than observed."""
    sc = AlignmentScorer(world["refs"])
    assert tw.resolve_bhs_scorer("align", sc, world["refs"]) is sc


def test_align_mode_replaces_a_non_alignment_scorer(world):
    sc = _CosineScorer(world["refs"])
    out = tw.resolve_bhs_scorer("align", sc, world["refs"])
    assert out is not sc
    assert isinstance(out, AlignmentScorer)


def test_the_replacement_scores_in_the_floors_units_and_not_in_cosine_range(world):
    """The whole point: the returned scorer's scores must be comparable with `score_floor`."""
    out = tw.resolve_bhs_scorer("align", _CosineScorer(world["refs"]), world["refs"])
    # A query identical to a catalogue entry must score far above anything a cosine could reach.
    assert float(np.max(out.score_all(world["refs"][0]))) > 100.0


# --- wiring, so the flag cannot exist while doing nothing ----------------------------------------------

def test_the_resolved_scorer_is_what_build_arms_is_given():
    """A structural test, because a flag that resolves correctly and is then ignored passes every
    behavioural test above while changing nothing in the run."""
    src = inspect.getsource(tw.main)
    assert "scorer=bhs_scorer" in src, "build_arms is still being handed the retrieval scorer"
    assert "resolve_bhs_scorer(" in src


def test_the_flag_is_exposed_with_both_choices_and_defaults_to_run():
    out = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True,
                         cwd=str(ROOT), timeout=300)
    assert out.returncode == 0, out.stderr
    text = " ".join(out.stdout.split())
    assert "--bhs-scorer" in text
    assert "run,align" in text.replace(" ", "")
    assert "the default" in text
