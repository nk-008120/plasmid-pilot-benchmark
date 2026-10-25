"""Tests for the derived fragment-length bound (`scripts/50_fragment_bound.py`).

The script answers a question with arithmetic rather than a run: what is the shortest fragment that
could clear a significance gate even against a perfect copy of itself? That makes it a ceiling on every
scorer, so it is worth being sure the arithmetic is the arithmetic claimed.

Three things are checked. The hard-coded BLOSUM62 diagonal is the real one, because a hand-copied
matrix is exactly the class of error this repository keeps finding. The crossing the script solves for
is a genuine crossing and is unique, so "the smallest m" is well defined rather than an artefact of
scanning upwards. And the bound moves the right way with catalogue size, since a bigger search space
can only demand a higher score.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from pilot.homology import score_floor

_SPEC = importlib.util.spec_from_file_location(
    "fragment_bound", Path(__file__).resolve().parents[1] / "scripts" / "50_fragment_bound.py")
fb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fb)


def test_hardcoded_diagonal_is_the_real_blosum62():
    """The table is written out so the script runs without parasail; it must still be right."""
    pytest.importorskip("parasail", reason="the cross-check needs the real matrix")
    assert fb._check_against_parasail().startswith("diagonal verified")


def test_diagonal_covers_the_twenty_standard_residues():
    assert len(fb.BLOSUM62_DIAGONAL) == 20
    assert set(fb.BLOSUM62_DIAGONAL) == set("ACDEFGHIKLMNPQRSTVWY")


def test_the_crossing_is_real_and_unique():
    """At the answer the best possible score clears the floor; one residue shorter it does not.

    Uniqueness is not an assumption: the best possible score grows linearly in m and the floor grows as
    log m, so the difference is increasing and crosses zero once. This pins that the implementation
    agrees with that reasoning rather than returning the first of several crossings.
    """
    s_diag, N = 5.151, 20_806_731
    m = fb.min_length(s_diag, N)
    assert m is not None
    assert m * s_diag >= score_floor(fb.MAX_EVALUE, m, N)
    assert (m - 1) * s_diag < score_floor(fb.MAX_EVALUE, m - 1, N)
    # and it never dips back below once cleared
    assert all(k * s_diag >= score_floor(fb.MAX_EVALUE, k, N) for k in range(m, m + 400))


def test_bound_rises_with_catalogue_size():
    """A larger search space can only demand a longer fragment, never a shorter one."""
    s_diag = 5.151
    sizes = [2_600_000, 7_800_000, 14_614_744, 20_806_731, 207_099_657]
    bounds = [fb.min_length(s_diag, n) for n in sizes]
    assert all(b is not None for b in bounds)
    assert bounds == sorted(bounds)


def test_a_generous_diagonal_gives_a_shorter_bound():
    """The two columns in the report must bracket, or the report is meaningless."""
    N = 20_806_731
    assert fb.min_length(11.0, N) < fb.min_length(5.151, N)


def test_composition_falls_back_to_a_flat_prior_without_a_reference():
    freq, src = fb.composition(Path("does/not/exist.fasta"))
    assert src.startswith("flat prior")
    assert pytest.approx(sum(freq.values()), abs=1e-9) == 1.0


def test_a_hopeless_scoring_system_returns_none_rather_than_a_number():
    """If the per-residue score is too small the floor is never reached, and that is an answer."""
    assert fb.min_length(0.001, 20_806_731, limit=5_000) is None
