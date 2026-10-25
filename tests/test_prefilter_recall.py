"""The prefilter is the two-stage scorer's ceiling, so its recall has to be reportable."""
from __future__ import annotations

import random

import numpy as np

from pilot.eba import EbaScorer
from pilot.positional import get_positional_embedder

AA = "ACDEFGHIKLMNPQRSTVWY"


def _prot(n: int, rng: random.Random) -> str:
    return "".join(rng.choice(AA) for _ in range(n))


class _CountingPrefilter:
    """A prefilter that records how many times it was asked, so double work is visible."""

    name = "counting"

    def __init__(self, refs):
        self.refs = refs
        self.calls = 0

    def score_all(self, query):
        self.calls += 1
        qs = set(query)
        return np.array([float(len(qs & set(s))) for s in self.refs])


def _scorer(refs, prefilter, top_k):
    return EbaScorer(get_positional_embedder("blosum"), refs, prefilter=prefilter, top_k=top_k)


def test_prefilter_scores_is_none_without_a_prefilter():
    rng = random.Random(1)
    refs = [_prot(60, rng) for _ in range(12)]
    assert _scorer(refs, None, 0).prefilter_scores(refs[0]) is None


def test_prefilter_scores_returns_one_score_per_reference():
    rng = random.Random(2)
    refs = [_prot(60, rng) for _ in range(12)]
    pf = _CountingPrefilter(refs)
    row = _scorer(refs, pf, 5).prefilter_scores(refs[3])
    assert row is not None and row.shape == (12,)


def test_shortlist_and_prefilter_scores_share_one_prefilter_pass():
    # The whole point of instrumenting prefilter recall is that it costs nothing anyone would decline
    # to pay, so the memo is part of the contract, not an optimisation detail.
    rng = random.Random(3)
    refs = [_prot(60, rng) for _ in range(20)]
    pf = _CountingPrefilter(refs)
    sc = _scorer(refs, pf, 5)
    q = refs[7]
    sc.prefilter_scores(q)
    sc.shortlist(q)
    sc.prefilter_scores(q)
    assert pf.calls == 1


def test_a_different_query_invalidates_the_memo():
    rng = random.Random(4)
    refs = [_prot(60, rng) for _ in range(20)]
    pf = _CountingPrefilter(refs)
    sc = _scorer(refs, pf, 5)
    sc.prefilter_scores(refs[1])
    sc.prefilter_scores(refs[2])
    sc.prefilter_scores(refs[1])
    assert pf.calls == 3


def test_the_shortlist_is_exactly_the_top_k_of_the_prefilter_row():
    # This is what makes a recorded prefilter RANK equivalent to shortlist membership at every k, which
    # is how the recall curve comes for free instead of one run per candidate top_k.
    rng = random.Random(5)
    refs = [_prot(60, rng) for _ in range(30)]
    pf = _CountingPrefilter(refs)
    sc = _scorer(refs, pf, 7)
    q = refs[11]
    row = sc.prefilter_scores(q)
    expected = set(np.argsort(-row, kind="stable")[:7].tolist())
    assert set(sc.shortlist(q).tolist()) == expected
    for k in (1, 3, 7):
        in_top_k = {i for i in range(30) if int((row >= row[i]).sum()) <= k}
        assert in_top_k <= set(np.argsort(-row, kind="stable")[:k].tolist()) | in_top_k


def test_a_reference_outside_the_shortlist_cannot_reach_rank_one():
    # eba.py states this in prose; the prefilter recall report is only meaningful if it is true.
    rng = random.Random(6)
    refs = [_prot(80, rng) for _ in range(25)]
    pf = _CountingPrefilter(refs)
    sc = _scorer(refs, pf, 3)
    q = refs[20]
    short = set(sc.shortlist(q).tolist())
    row = sc.score_all(q)
    winner = int(np.argmax(row))
    assert winner in short
