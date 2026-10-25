"""`scripts/31_recall_at_depth.py`: recall at review depth, and what a crossing count is not.

Loaded by path because `scripts/` is not an importable package, as in `tests/test_compare_tiers.py`
and `tests/test_stratify_payload_fraction.py`.

This script is applied to the corpus, so this project's working rules requires it to carry tests. Three things here are
worth a test rather than a reading.

`crossings` must be paired. An id scorable under one arm and not the other has to be dropped rather
than counted as a loss, which is the trap this project recorded: a sample change and an effect change
summed into one number.

**A crossing count is not a harm count, and the difference is the whole.** A construct
moving 2 -> 3 crosses no depth in `DEPTHS`, both ranks are outside 1 and inside 5, so it appears in
no won/lost list at any depth while being, in rank, worse. That is exactly `ncbi_AF146395_1` in the
corpus at the 25% holdout, and it is why the full corpus segmenter run was first reported as harming 2
constructs when by rank it harms 4. The test pins the behaviour so the next reader meets it here rather
than in a number they then have to explain.

And `recall`'s macro average must weight each payload once. The corpus's 37 constructs sharing one
payload are the reason the macro arm exists, and getting it wrong is invisible in the output.

The regression test at the end reads finished result files, which are gitignored (`results/*`), so it
skips when they are absent rather than failing a fresh clone.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORPUS_TWILIGHT = ROOT / "results" / "gate173_control_corpus" / "twilight_sw-blosum62-11-1_corpus.csv"
CORPUS_LENGTHS = ROOT / "results" / "bhs_corpus_probe" / "frame_orf_lengths.csv"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "recall_at_depth", ROOT / "scripts" / "31_recall_at_depth.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_strata_partition_the_unit_interval(mod):
    """Every f in [0, 1] lands in exactly one stratum. f = 1.000 is 180 of 286 corpus constructs."""
    for i in range(0, 1001):
        f = i / 1000
        hits = [name for name, lo, hi in mod.STRATA if lo <= f < hi]
        assert len(hits) == 1, (f, hits)
        assert mod.stratum_of(f) == hits[0]


def test_crossings_are_paired(mod):
    """An id missing from either arm is dropped, not counted as a loss."""
    a = {"p1": 20, "p2": 3, "only_in_a": 50}
    b = {"p1": 2, "p2": 3, "only_in_b": 1}
    out = mod.crossings(a, b, ["p1", "p2", "only_in_a", "only_in_b"], 10)
    assert out["n"] == 2
    assert out["won"] == ["p1"]
    assert out["lost"] == []
    assert out["net"] == 1


def test_net_is_won_minus_lost_not_a_count_difference(mod):
    """+1 net must distinguish 1 won / 0 lost from 3 won / 2 lost."""
    a = {"w1": 20, "w2": 30, "w3": 40, "l1": 2, "l2": 3}
    b = {"w1": 1, "w2": 1, "w3": 1, "l1": 99, "l2": 99}
    ids = list(a)
    out = mod.crossings(a, b, ids, 10)
    assert len(out["won"]) == 3
    assert len(out["lost"]) == 2
    assert out["net"] == 1
    hits_a = sum(1 for p in ids if a[p] <= 10)
    hits_b = sum(1 for p in ids if b[p] <= 10)
    assert hits_b - hits_a == out["net"]  # equal here, and the lists are what say why


def test_a_rank_regression_can_cross_no_depth_at_all(mod):
    """2 -> 3 is worse by rank and appears in no won/lost list at any depth.

    This is `ncbi_AF146395_1` (P01834, f=0.547), the one construct in either measured tier where
    locating the payload makes retrieval worse. A report that counts only crossings will call it
    unaffected. the working record is why mean rank is primary and crossings are secondary.
    """
    a = {"regressed": 2}
    b = {"regressed": 3}
    for k in mod.DEPTHS:
        out = mod.crossings(a, b, ["regressed"], k)
        assert out["won"] == [] and out["lost"] == [], k
        assert out["net"] == 0, k
    assert b["regressed"] > a["regressed"]


def test_recall_macro_weights_each_payload_once(mod):
    """Three constructs of one payload and one of another: macro is 50%, micro is 75%."""
    ranks = {"c1": 1, "c2": 1, "c3": 1, "c4": 7}
    acc = {"c1": "P1", "c2": "P1", "c3": "P1", "c4": "P2"}
    micro, macro, hits, n, n_pay = mod.recall(ranks, list(ranks), acc, 1)
    assert (hits, n, n_pay) == (3, 4, 2)
    assert micro == pytest.approx(75.0)
    assert macro == pytest.approx(50.0)


def test_recall_is_monotone_in_depth(mod):
    """recall@k cannot fall as k grows; the depth sweep depends on it."""
    ranks = {"a": 1, "b": 4, "c": 9, "d": 22, "e": 400}
    acc = {k: f"P{k}" for k in ranks}
    got = [mod.recall(ranks, list(ranks), acc, k)[2] for k in mod.DEPTHS]
    assert got == sorted(got)
    assert got == [1, 2, 3, 4]


@pytest.mark.skipif(not (CORPUS_TWILIGHT.is_file() and CORPUS_LENGTHS.is_file()),
                    reason="finished corpus results are gitignored; regression runs where they exist")
def test_corpus_depth10_crossing_set_is_exactly_three_constructs(mod):
    """The depth-10 operating point admits every mechanism event and no residue-trim artefact.

    At depth 1, six of the twelve corpus winners are f >= 0.93: ORFs that already are the payload,
    crossing because 4 to 14 residues were trimmed. At depth 10 none of them qualifies and the three
    survivors are all f < 0.57. That separation is the evidence for the metric chosen in the working record,
    and it is a property of this data rather than of the code, so it is pinned here.
    """
    ranks, _acc = mod.read_ranks(CORPUS_TWILIGHT)
    frac = mod.read_f(CORPUS_LENGTHS, "lengths")
    whole, oracle = ranks[0.25]["whole"], ranks[0.25]["oracle"]
    ids = sorted(set(whole) & set(oracle))
    assert len(ids) == 286

    at10 = mod.crossings(whole, oracle, ids, 10)
    assert set(at10["won"]) == {"ncbi_AY254174_1", "ncbi_AF114752_1", "ncbi_AF416988_1"}
    assert at10["lost"] == []
    assert all(frac[p][0] < 0.57 for p in at10["won"])

    at1 = mod.crossings(whole, oracle, ids, 1)
    assert len(at1["won"]) == 12 and at1["lost"] == []
    assert sum(1 for p in at1["won"] if frac[p][0] >= 0.70) == 6

    # depths 10 and 25 do not discriminate: the choice of 10 rests on review practice, not on data
    assert set(mod.crossings(whole, oracle, ids, 25)["won"]) == set(at10["won"])
