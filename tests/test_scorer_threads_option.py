"""`threads` is operational, so `get_scorer` must consume it for every arm, not only the alignment one.

`tests/test_align_threads.py` already proves the threaded score pass is bitwise identical to the serial
one. What it does not cover is the plumbing, and the plumbing had a hole: `get_scorer` read `threads`
out of `**kwargs` for the alignment arm and then forwarded the same `**kwargs` to `get_embedder` and
`EbaScorer`, which accept no such argument. So

    get_scorer("facebook/esm2_t33_650M_UR50D", refs, threads=4)

raised `TypeError`, and a script that offers a single `--threads` flag could run its alignment arm and
not its embedding arm: the exact portability `get_scorer` exists to provide, and the reason
`scripts/10_twilight.py` could not simply pass the flag through when one was added on 17 Oct.

The failure is loud, but it only fires on the arm nobody threads, so it would have waited until an
ESM-2 run was launched with the flag already in the command line.
"""
from __future__ import annotations

import inspect

import pytest

from pilot.align import AlignmentScorer, get_scorer

REFS = ["ACDEFGHIKLMNPQRSTVWY" * 5, "MKTAYIAKQRQISFVKSHFSRQ" * 4,
        "MSKGEELFTGVVPILVELDGDV" * 3, "GSHMLEDPRVPVEWLDGR" * 6]


def test_alignment_arm_takes_the_thread_count():
    s = get_scorer("align", REFS, cache_dir=None, threads=3)
    assert isinstance(s, AlignmentScorer)
    assert s.threads == 3


@pytest.mark.parametrize("spec", ["eba:blosum", "eba"])
def test_a_model_arm_accepts_threads_and_ignores_it(spec):
    """The call must succeed. Before the fix it raised TypeError on an unexpected keyword."""
    s = get_scorer(spec, REFS, cache_dir=None, threads=4)
    assert s.name  # constructed at all


def test_threads_is_consumed_rather_than_forwarded():
    """A model arm must never SEE `threads`, or a future constructor could silently act on it."""
    seen = {}

    class Spy:
        name = "spy"

        def __init__(self, **kw):
            seen.update(kw)

    import pilot.align as align

    original = align.EmbeddingScorer
    try:
        align.EmbeddingScorer = lambda emb, refs, cache_dir=None: Spy(embedder=emb)
        import pilot.embed as embed

        orig_get = embed.get_embedder
        embed.get_embedder = lambda spec, **kw: Spy(**kw)
        get_scorer("kmer3-1024", REFS, cache_dir=None, threads=7, batch_size=2)
    finally:
        align.EmbeddingScorer = original
        embed.get_embedder = orig_get
    assert "threads" not in seen
    assert seen.get("batch_size") == 2  # a real model option still gets through


def test_none_and_zero_fall_back_to_serial():
    """A script passing its flag's default through must not produce a 0-thread scorer."""
    assert get_scorer("align", REFS, cache_dir=None, threads=None).threads == 1
    assert get_scorer("align", REFS, cache_dir=None, threads=0).threads == 1
    assert get_scorer("align", REFS, cache_dir=None).threads == 1


def test_the_thread_count_stays_out_of_the_scorer_name():
    """The name is the cache key and labels every output column: two settings, one set of artefacts."""
    assert (get_scorer("align", REFS, cache_dir=None, threads=1).name
            == get_scorer("align", REFS, cache_dir=None, threads=4).name)


def test_twilight_offers_the_flag_and_does_not_route_it_through_model_kwargs():
    """`model_kwargs` drops options for the model-free arms, which would drop `threads` with them."""
    from pilot.align import model_kwargs

    assert "threads" not in model_kwargs("align", batch_size=8, max_length=1022, threads=4)
    assert "threads" not in inspect.signature(model_kwargs).parameters
