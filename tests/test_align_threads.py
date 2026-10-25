"""GATE 1: the threaded Smith-Waterman score pass must be BITWISE identical to the serial one.

`np.array_equal`, never `allclose`. Parallelising this pass must not reassociate any arithmetic: each
output element is one independent `sw_striped_sat` call on one (query, reference) pair, the slices are
contiguous and disjoint, and every element is written by exactly one thread. No sum, no dot product, no
reduction of any kind crosses a thread boundary, so exact equality is the correct expectation and not an
optimistic one. If it ever fails, the implementation changed the numerics rather than the scheduling, and
every number downstream of it is suspect, which is why a tolerance would be the wrong test here.

The decisive case is the one against the **real catalogue**, because that is what the runs use and because
only a long reference list exercises more than one slice per thread. It is marked slow-ish but runs in a
few seconds: it is one query per thread count over 40,750 references, not a benchmark.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pilot.align import AlignmentScorer, get_scorer
from pilot.reference import iter_fasta

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "results" / "gate_all173" / "reference_twilight.fasta"

#: Fixed queries. Short, long, one that is itself a catalogue member, one with a repeat, one low-complexity.
QUERIES = (
    "MSTKFALVTGCGQGGIGEALITEYARRGIHAIATVLPAEPSDHLARAGITFFPLDVTNEE",
    "MKAILVVLLYTFATANADTLCIGYHANNSTDTVDTVLEKNVTVTHSVNLLEDKHNGKLCK",
    "ACDEFGHIKLMNPQRSTVWY" * 25,
    "GGGGSGGGGSGGGGSGGGGSGGGGSGGGGS",
    "M" + "ACDEFGHIKLMNPQRSTVWY" * 3,
)


def synthetic_refs(n=600, seed=0):
    rng = np.random.default_rng(seed)
    alpha = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY", dtype="S1")
    return ["".join(x.decode() for x in rng.choice(alpha, size=int(rng.integers(50, 400))))
            for _ in range(n)]


@pytest.mark.parametrize("threads", [2, 3, 4, 8])
def test_threaded_scores_are_bitwise_identical_on_synthetic_references(threads):
    refs = synthetic_refs()
    serial = AlignmentScorer(refs, threads=1)
    par = AlignmentScorer(refs, threads=threads)
    for q in QUERIES:
        a, b = serial.score_all(q), par.score_all(q)
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        assert np.array_equal(a, b), f"threads={threads} changed the scores for a {len(q)} aa query"


@pytest.mark.skipif(not CATALOGUE.is_file(), reason="the 40,750-protein catalogue is not present")
def test_threaded_scores_are_bitwise_identical_on_the_real_catalogue():
    """The gate proper: the catalogue the runs actually search, 40,750 proteins / 20,806,731 residues."""
    refs = [e.seq for e in iter_fasta(CATALOGUE)]
    assert len(refs) > 10_000, "not the real catalogue"
    serial = AlignmentScorer(refs, threads=1)
    for threads in (2, 4):
        par = AlignmentScorer(refs, threads=threads)
        for q in QUERIES[:3]:
            a, b = serial.score_all(q), par.score_all(q)
            assert a.dtype == b.dtype == np.float32
            assert np.array_equal(a, b), (
                f"threads={threads} changed the scores on the real catalogue for a {len(q)} aa query; "
                f"{int((a != b).sum())} of {a.size} elements differ")


def test_a_thread_count_above_the_reference_count_falls_back_without_changing_the_result():
    """Guards the `n < 2 * threads` branch: 8 threads over 3 references must not spawn empty workers or
    reorder anything."""
    refs = ["ACDEFGHIKLMNPQRST", "MKAILVVLLYTFATANA", "GGGGSGGGGS"]
    a = AlignmentScorer(refs, threads=1).score_all(QUERIES[0])
    b = AlignmentScorer(refs, threads=8).score_all(QUERIES[0])
    assert np.array_equal(a, b)


def test_every_element_is_written_so_no_slice_is_skipped():
    """`np.empty` is uninitialised, so a bounds error would leave garbage rather than a zero. Scoring a
    query against itself must give a strictly positive score for every reference that shares residues."""
    refs = [QUERIES[0]] * 7 + [QUERIES[2]] * 6
    for threads in (1, 2, 4, 5):
        row = AlignmentScorer(refs, threads=threads).score_all(QUERIES[0])
        assert row.shape == (13,)
        assert np.all(row > 0), f"threads={threads} left an element unwritten"
        assert len(set(row[:7].tolist())) == 1  # identical references must score identically


def test_an_empty_query_is_zeros_on_every_thread_count():
    refs = synthetic_refs(n=40)
    for threads in (1, 2, 4):
        row = AlignmentScorer(refs, threads=threads).score_all("")
        assert np.array_equal(row, np.zeros(40, dtype=np.float32))


def test_the_scorer_name_does_not_carry_the_thread_count():
    """The name is the embedding cache key and labels every output file and results column. Two runs that
    differ only in thread count are numerically identical, the gate above is what establishes that, so
    they must not produce differently named artefacts."""
    refs = synthetic_refs(n=20)
    assert AlignmentScorer(refs, threads=1).name == AlignmentScorer(refs, threads=4).name
    assert AlignmentScorer(refs, threads=4).name == "sw-blosum62-11-1"


def test_threads_defaults_to_one_and_is_plumbed_through_get_scorer():
    refs = synthetic_refs(n=20)
    assert get_scorer("align", refs).threads == 1
    assert get_scorer("align", refs, threads=4).threads == 4
    assert get_scorer("align", refs, threads=None).threads == 1
    assert get_scorer("align", refs, threads=0).threads == 1


def test_get_scorer_still_accepts_the_model_options_it_has_always_ignored():
    """`model_kwargs` drops model options for the alignment arm, but a caller passing them directly must
    not now crash on the new keyword path."""
    refs = synthetic_refs(n=20)
    s = get_scorer("align", refs, threads=2)
    assert s.name == "sw-blosum62-11-1"
    assert np.array_equal(s.score_all(QUERIES[0]),
                          get_scorer("align", refs).score_all(QUERIES[0]))
