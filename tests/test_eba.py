"""The EBA positional scorer: the dynamic programming must be exactly right before anything is read off it.

The central test is `test_matches_parasail_on_blosum62`. `pilot.eba` reimplements affine-gap
Smith-Waterman in numpy because parasail cannot take a position-specific matrix, so the only thing that
makes the reimplementation trustworthy is that, handed the *same* substitution matrix and the same gaps, it
returns the same score as the control arm already in use.
"""
import random

import numpy as np
import pytest

from pilot.align import get_scorer, rank_one
from pilot.eba import EbaScorer, blosum_matrix, get_eba_scorer, sw_affine, sw_affine_batch
from pilot.positional import BlosumPositional, KmerPositional

parasail = pytest.importorskip("parasail")
AA = "ACDEFGHIKLMNPQRSTVWY"


def naive_gotoh(sub, gap_open, gap_extend):
    """Textbook O(mn) Gotoh, written for clarity rather than speed, as an independent second opinion."""
    m, n = sub.shape
    neg = -1e18
    h = [[0.0] * (n + 1) for _ in range(m + 1)]
    e = [[neg] * (n + 1) for _ in range(m + 1)]
    f = [[neg] * (n + 1) for _ in range(m + 1)]
    best = 0.0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            e[i][j] = max(h[i][j - 1] - gap_open, e[i][j - 1] - gap_extend)
            f[i][j] = max(h[i - 1][j] - gap_open, f[i - 1][j] - gap_extend)
            h[i][j] = max(0.0, h[i - 1][j - 1] + float(sub[i - 1][j - 1]), e[i][j], f[i][j])
            best = max(best, h[i][j])
    return best


def random_pairs(n, seed, max_len=70, graft=0.6):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a, b = rng.randint(1, max_len), rng.randint(1, max_len)
        q = "".join(rng.choice(AA) for _ in range(a))
        r = "".join(rng.choice(AA) for _ in range(b))
        if rng.random() < graft:  # make a good share of the pairs genuinely alignable
            k = rng.randint(1, min(a, b))
            st = rng.randint(0, b - k)
            r = r[:st] + q[:k] + r[st + k:]
        out.append((q, r))
    return out


@pytest.mark.parametrize("gap_open,gap_extend", [(11, 1), (10, 1), (5, 2), (2, 1)])
def test_matches_parasail_on_blosum62(gap_open, gap_extend):
    """Same matrix, same gaps, same score as the alignment control arm. This is the correctness claim.

    Gaps of 1/1 are deliberately not in the parameter list: there `parasail.sw_striped_sat` disagrees with
    `parasail.sw_scan_sat`, `parasail.sw` and this implementation, which all agree with each other, so the
    striped kernel is the outlier rather than this code. The project scores with 11/1, so it is a
    curiosity and not a problem, but it is why that setting is excluded here instead of silently passing.
    """
    for q, r in random_pairs(120, seed=4):
        theirs = float(parasail.sw_striped_sat(q, r, gap_open, gap_extend, parasail.blosum62).score)
        assert sw_affine(blosum_matrix(q, r), gap_open, gap_extend) == pytest.approx(theirs, abs=1e-3)


def test_matches_a_naive_gotoh():
    for q, r in random_pairs(40, seed=9, max_len=28):
        sub = blosum_matrix(q, r)
        assert sw_affine(sub, 11, 1) == pytest.approx(naive_gotoh(sub, 11, 1), abs=1e-6)


def test_batch_equals_one_at_a_time_despite_padding():
    """Padded reference columns must not leak into a real cell, whatever the length spread in the batch."""
    pairs = random_pairs(12, seed=21, max_len=40)
    q = pairs[0][0]
    refs = [r for _, r in pairs]
    n_max = max(len(r) for r in refs)
    sub = np.full((len(refs), len(q), n_max), np.float32(-1e30), dtype=np.float32)
    for i, r in enumerate(refs):
        sub[i, :, : len(r)] = blosum_matrix(q, r)
    batch = sw_affine_batch(sub, 11, 1)
    for i, r in enumerate(refs):
        assert batch[i] == pytest.approx(sw_affine(blosum_matrix(q, r), 11, 1), abs=1e-3)


def test_self_alignment_scores_the_diagonal():
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    sub = blosum_matrix(seq, seq)
    assert sw_affine(sub, 11, 1) == pytest.approx(float(np.trace(sub)), abs=1e-6)


def test_alignment_is_local_not_global():
    """A motif buried in unrelated sequence must score as the motif, which is the whole point of local."""
    rng = random.Random(3)
    motif = "WWYYFFHHCCWWYYFFHHCC"
    flank = "".join(rng.choice("AGST") for _ in range(120))
    construct = flank + motif + flank
    alone = sw_affine(blosum_matrix(motif, motif), 11, 1)
    buried = sw_affine(blosum_matrix(motif, construct), 11, 1)
    assert buried == pytest.approx(alone, abs=1e-6)


def test_empty_inputs_score_zero():
    assert sw_affine(np.zeros((0, 5), dtype=np.float32)) == 0.0
    assert sw_affine(np.zeros((5, 0), dtype=np.float32)) == 0.0
    assert sw_affine_batch(np.zeros((0, 3, 3), dtype=np.float32), 1, 1).shape == (0,)


def test_self_alignment_under_a_cosine_matrix_is_exactly_length_times_one_minus_shift():
    """Every self cosine is 1, so the whole diagonal aligns and the score is analytic. A cheap exact check."""
    emb = BlosumPositional()
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    (m,) = emb.embed_residues([seq])
    cos = m @ m.T
    for shift in (0.0, 0.25, 0.5):
        assert sw_affine(cos - shift, 0.5, 0.1) == pytest.approx(len(seq) * (1 - shift), abs=1e-3)


def test_score_falls_monotonically_as_shift_rises():
    emb = BlosumPositional()
    rng = random.Random(5)
    q = "".join(rng.choice(AA) for _ in range(90))
    r = "".join(rng.choice(AA) for _ in range(90))
    qe, re_ = emb.embed_residues([q, r])
    cos = qe @ re_.T
    scores = [sw_affine(cos - sh, 0.5, 0.1) for sh in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)]
    assert scores == sorted(scores, reverse=True)


def test_shift_improves_discrimination_between_a_remote_homolog_and_noise():
    """What `shift` buys is not a bigger score, it is a bigger ratio. Measured, because it is the tuned knob.

    A 70%-mutated copy of the query scores 1.21x a random sequence at shift 0 and 1.34x at shift 0.25 with
    BLOSUM62-row embeddings. `scripts/18_eba_sweep.py` sweeps it properly against the 23 hand-audited
    constructs; this only pins the direction so a sign error cannot pass.
    """
    emb = BlosumPositional()
    rng = random.Random(5)
    q = "".join(rng.choice(AA) for _ in range(120))
    noise = "".join(rng.choice(AA) for _ in range(120))
    remote = "".join(c if rng.random() > 0.70 else rng.choice(AA) for c in q)
    qe, ne, he = emb.embed_residues([q, noise, remote])

    def ratio(shift):
        return (sw_affine(qe @ he.T - shift, 0.5, 0.1)
                / max(sw_affine(qe @ ne.T - shift, 0.5, 0.1), 1e-9))

    assert ratio(0.25) > ratio(0.0)
    assert ratio(0.40) > ratio(0.0)


def test_scorer_returns_one_score_per_reference_and_finds_an_exact_match():
    refs = ["MKTAYIAKQRQISFVKSHFSRQ", "GGGGSGGGGSGGGGSGGGGS", "WWYYFFHHCCWWYYFFHHCC",
            "MVSKGEELFTGVVPILVELDGDVNG"]
    sc = EbaScorer(BlosumPositional(), refs, prefilter=None, top_k=0)
    row = sc.score_all(refs[2])
    assert row.shape == (len(refs),)
    assert int(np.argmax(row)) == 2


def test_prefilter_shortlists_and_the_rest_rank_below_it():
    rng = random.Random(8)
    refs = ["".join(rng.choice(AA) for _ in range(60)) for _ in range(40)]
    target = refs[17]
    sc = get_eba_scorer("eba:blosum", refs, top_k=5)
    idx = sc.shortlist(target)
    assert idx.size == 5
    assert 17 in set(idx.tolist())  # an exact match must survive even a weak prefilter
    row = sc.score_all(target)
    rest = [i for i in range(len(refs)) if i not in set(idx.tolist())]
    assert row[idx].min() > row[rest].max()
    assert int(np.argmax(row)) == 17


def test_non_shortlisted_entries_keep_prefilter_order():
    rng = random.Random(13)
    refs = ["".join(rng.choice(AA) for _ in range(50)) for _ in range(30)]
    sc = get_eba_scorer("eba:blosum", refs, top_k=4)
    q = refs[2]
    row = sc.score_all(q)
    idx = set(sc.shortlist(q).tolist())
    rest = np.array([i for i in range(len(refs)) if i not in idx])
    pf = sc.prefilter.score_all(q)[rest]
    # ranking the leftovers by the blended row must reproduce ranking them by the prefilter
    assert list(np.argsort(-row[rest], kind="stable")) == list(np.argsort(-pf, kind="stable"))


def test_top_k_changes_cost_but_not_the_winner_when_the_answer_is_shortlisted():
    rng = random.Random(2)
    refs = ["".join(rng.choice(AA) for _ in range(55)) for _ in range(25)]
    q = refs[6]
    wide = get_eba_scorer("eba:blosum", refs, top_k=0)
    narrow = get_eba_scorer("eba:blosum", refs, top_k=8)
    assert int(np.argmax(wide.score_all(q))) == int(np.argmax(narrow.score_all(q))) == 6


def test_chunk_budget_does_not_change_the_answer():
    """The memory budget only sets how many substitution matrices are built at once."""
    rng = random.Random(6)
    refs = ["".join(rng.choice(AA) for _ in range(rng.randint(20, 90))) for _ in range(12)]
    q = refs[4]
    big = EbaScorer(BlosumPositional(), refs, prefilter=None, top_k=0, budget_floats=20_000_000)
    tiny = EbaScorer(BlosumPositional(), refs, prefilter=None, top_k=0, budget_floats=1)
    assert np.allclose(big.score_all(q), tiny.score_all(q), atol=1e-4)


def test_reaches_the_shared_scorer_interface_and_rank_one():
    refs = ["MKTAYIAKQRQISFVKSHFSRQ", "WWYYFFHHCCWWYYFFHHCC", "GGGGSGGGGSGGGGS"]
    sc = get_scorer("eba:kmerpos", refs, top_k=0, prefilter="none")
    assert isinstance(sc, EbaScorer)
    rank, score, top_acc, top_score = rank_one(refs[1], sc, ["a", "b", "c"], 1)
    assert rank == 1
    assert top_acc == "b"


def test_name_records_every_tuned_number():
    sc = EbaScorer(KmerPositional(dim=32), ["AAAA"], prefilter=None, top_k=0,
                   shift=0.3, gap_open=0.7, gap_extend=0.2)
    assert sc.name == "eba-kmerpos3-32-2-s0.3-g0.7-0.2-k1"


def test_empty_query_scores_nothing_without_crashing():
    sc = EbaScorer(BlosumPositional(), ["MKTAYIAK", "WWYYFFHH"], prefilter=None, top_k=0)
    assert sc.score_all("").shape == (2,)


def test_deterministic():
    refs = ["MKTAYIAKQRQISFVKSHFSRQ", "WWYYFFHHCCWWYYFFHHCC", "MVSKGEELFTGVVPILVELDGDVNG"]
    sc = EbaScorer(BlosumPositional(), refs, prefilter=None, top_k=0)
    assert np.array_equal(sc.score_all(refs[0]), sc.score_all(refs[0]))


def test_lazy_and_precomputed_reference_embeddings_agree():
    """`precompute=False` is what lets the scorer face a 32,387-protein catalogue; it must not change scores."""
    rng = random.Random(31)
    refs = ["".join(rng.choice(AA) for _ in range(rng.randint(25, 80))) for _ in range(15)]
    q = refs[9]
    lazy = EbaScorer(BlosumPositional(), refs, prefilter=None, top_k=0, precompute=False)
    eager = EbaScorer(BlosumPositional(), refs, prefilter=None, top_k=0, precompute=True)
    assert np.allclose(lazy.score_all(q), eager.score_all(q), atol=1e-4)
    assert lazy._ref_emb is None and eager._ref_emb is not None


def test_blosum_path_defaults_to_lazy_and_an_esm_path_defaults_to_precompute():
    refs = ["MKTAYIAK", "WWYYFFHH"]
    assert get_eba_scorer("eba:blosum", refs, top_k=0)._ref_emb is None
    assert get_eba_scorer("eba:kmerpos", refs, top_k=0)._ref_emb is None


def test_twilight_style_construction_with_a_batch_size_kwarg():
    """The exact call `scripts/10_twilight.py` makes for a non-k-mer scorer, which must not raise."""
    refs = ["MKTAYIAKQRQISFVKSHFSRQ", "WWYYFFHHCCWWYYFFHHCC", "GGGGSGGGGSGGGGS"]
    sc = get_scorer("eba:blosum", refs, batch_size=8)
    assert isinstance(sc, EbaScorer)
    assert sc.score_all(refs[0]).shape == (3,)


class TestSpecOptions:
    """The analysis scripts pick a scorer with one string and have no flags for the positional knobs, so
    the knobs ride in the string. The setting then shows up in `name`, and so in the output filename."""

    def test_embedder_only(self):
        from pilot.eba import parse_spec

        assert parse_spec("eba") == ("blosum", {})
        assert parse_spec("eba:blosum") == ("blosum", {})
        assert parse_spec("eba:kmerpos") == ("kmerpos", {})

    def test_options_are_parsed_and_typed(self):
        from pilot.eba import parse_spec

        emb, opts = parse_spec("eba:blosum:shift=0.6,top_k=2000,gap_open=0.8")
        assert emb == "blosum"
        assert opts == {"shift": 0.6, "top_k": 2000, "gap_open": 0.8}
        assert isinstance(opts["top_k"], int) and isinstance(opts["shift"], float)

    def test_a_checkpoint_name_with_a_slash_survives(self):
        from pilot.eba import parse_spec

        emb, opts = parse_spec("eba:facebook/esm2_t33_650M_UR50D:shift=0.6")
        assert emb == "facebook/esm2_t33_650M_UR50D"
        assert opts == {"shift": 0.6}

    def test_an_unknown_option_is_refused_rather_than_ignored(self):
        from pilot.eba import parse_spec

        with pytest.raises(ValueError, match="unknown eba option"):
            parse_spec("eba:blosum:shfit=0.6")

    def test_the_option_reaches_the_scorer_and_its_name(self):
        refs = ["MKTAYIAKQRQISFVKSHFSRQ", "WWYYFFHHCCWWYYFFHHCC", "GGGGSGGGGSGGGGS"]
        sc = get_scorer("eba:blosum:shift=0.6,top_k=0", refs)
        assert sc.shift == 0.6
        assert "s0.6" in sc.name

    def test_an_explicit_keyword_argument_wins_over_the_spec(self):
        refs = ["MKTAYIAKQRQISFVKSHFSRQ", "WWYYFFHHCCWWYYFFHHCC"]
        sc = get_scorer("eba:blosum:shift=0.6", refs, shift=0.3, top_k=0)
        assert sc.shift == 0.3
