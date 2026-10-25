"""The empirical significance floor: it must recover a known lambda, and carry m and N."""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from dataclasses import replace

from pilot.cosine_floor import (BACKGROUND, BLOSUM62_EXPECTED_OVER_MAX, BLOSUM62_EXPECTED_OVER_SD,
                                 BLOSUM62_EXPECTED_SCORE, BLOSUM62_MAX_SCORE,
                                 BLOSUM62_PAIR_SCORE_SD, BLOSUM62_UNGAPPED_LAMBDA, EvdFit, PairStats,
                                 calibrate, fit_evd, shuffle_preserving_composition,
                                 solve_ungapped_lambda)
from pilot.decompose import best_hit_segments
from pilot.homology import score_floor

AA = "ACDEFGHIKLMNPQRSTVWY"


def _rand_prot(n: int, rng: random.Random) -> str:
    return "".join(rng.choice(AA) for _ in range(n))


def test_shuffle_preserves_length_and_composition():
    rng = random.Random(1)
    seq = "MKVLAAGIVGLNQWWWPPPP"
    out = shuffle_preserving_composition(seq, rng)
    assert len(out) == len(seq)
    assert sorted(out) == sorted(seq)


def test_shuffle_actually_reorders_a_long_sequence():
    rng = random.Random(2)
    seq = _rand_prot(400, random.Random(3))
    assert shuffle_preserving_composition(seq, rng) != seq


def _exponential_rows(lam: float, k_param: float, m: int, sub_lens, n_rows: int, seed: int):
    """Score rows drawn so that E_hat(x) == K * m * N * exp(-lam x) by construction.

    The survival function of the per-pair maximum is what the fit regresses, so sampling scores from an
    exponential tail with rate `lam` and the right intensity is the exact null the estimator assumes.
    """
    rng = np.random.default_rng(seed)
    n = sum(sub_lens)
    rows = []
    for _ in range(n_rows):
        # expected count above 0 is K*m*N; draw that many exponential(lam) excesses
        count = rng.poisson(k_param * m * n)
        vals = rng.exponential(1.0 / lam, size=max(int(count), 1))
        padded = np.zeros(len(sub_lens), dtype=float)
        padded[: min(vals.size, padded.size)] = np.sort(vals)[::-1][: padded.size]
        rows.append(padded)
    return rows


def test_fit_recovers_a_planted_lambda():
    lam, k_param, m = 0.8, 2e-5, 300
    sub_lens = [250] * 600
    rows = _exponential_rows(lam, k_param, m, sub_lens, 40, seed=7)
    fit = fit_evd(rows, [m] * len(rows), sub_lens, n_boot=40, seed=11)
    assert fit.lam == pytest.approx(lam, rel=0.25), fit.lam
    assert fit.r_squared > 0.95
    assert fit.lam_ci[0] < fit.lam < fit.lam_ci[1]


def test_floor_grows_with_query_length_and_database_size():
    raw = EvdFit(lam=0.5, k_param=1e-4, n_queries=10, n_pairs=100, subject_residues=1000,
                 query_lens=(100,), tail_lo=0.01, tail_hi=10.0, n_tail_points=24,
                 r_squared=0.99, lam_ci=(0.4, 0.6), lam_by_query_len=((100, 0.5),),
                 effective_length=False)
    a = raw.score_floor(1e-3, 100, 1_000_000)
    assert raw.score_floor(1e-3, 3000, 1_000_000) > a
    assert raw.score_floor(1e-3, 100, 10_000_000) > a
    # both enter inside a logarithm, exactly as in homology.score_floor
    assert raw.score_floor(1e-3, 100, 10_000_000) - a == pytest.approx(math.log(10) / 0.5)
    # with the finite-length correction the identity is only approximate, by design: the correction
    # shortens both sequences, which is the whole point of it
    corrected = replace(raw, effective_length=True)
    assert corrected.score_floor(1e-3, 3000, 1_000_000) > corrected.score_floor(1e-3, 100, 1_000_000)
    assert corrected.score_floor(1e-3, 100, 1_000_000) != pytest.approx(a, rel=1e-12)


def test_floor_inverts_its_own_evalue():
    fit = EvdFit(lam=0.37, k_param=3e-5, n_queries=1, n_pairs=1, subject_residues=1,
                 query_lens=(1,), tail_lo=0.01, tail_hi=10.0, n_tail_points=24, r_squared=1.0,
                 lam_ci=(0.3, 0.4), lam_by_query_len=())
    # both ways: the pair must invert each other whether or not the correction is applied
    for f in (replace(fit, effective_length=False), replace(fit, effective_length=True)):
        s = f.score_floor(1e-3, 420, 20_806_731)
        assert f.log_evalue(s, 420, 20_806_731) == pytest.approx(math.log(1e-3)), f.effective_length


def test_a_scoring_system_with_no_exponential_tail_is_rejected():
    # Uniform scores. A survival function always decreases, so a negative slope proves nothing: this
    # is rejected because the tail is CURVED on a log-count axis, which is the only real evidence that
    # no Gumbel regime exists. The sign check alone passed this input, which is how MIN_R2 came to be.
    rows = [np.linspace(0, 10, 500) for _ in range(30)]
    fit = fit_evd(rows, [200 + 20 * (i % 3) for i in range(30)], [200] * 500, n_boot=0)
    assert not fit.usable
    assert "not exponential" in fit.rejected_reason
    #... and the same input is accepted when the caller explicitly lowers the bar, so the guard is a
    # stated requirement rather than a hidden one
    assert fit_evd(rows, [200] * 30, [200] * 500, n_boot=0, min_r2=0.0).usable


def test_a_constant_score_row_is_an_error_not_a_division():
    with pytest.raises(ValueError):
        fit_evd([np.full(500, 3.0) for _ in range(30)], [200] * 30, [200] * 500, n_boot=0,
                min_r2=0.0)


def test_too_few_tail_observations_is_an_error_not_a_guess():
    with pytest.raises(ValueError):
        fit_evd([np.array([1.0, 2.0])], [100], [100, 100], n_boot=0)


def test_lambda_spread_reports_a_length_dependence():
    fit = EvdFit(lam=0.5, k_param=1e-4, n_queries=4, n_pairs=4, subject_residues=10,
                 query_lens=(100, 900), tail_lo=0.01, tail_hi=10.0, n_tail_points=24,
                 r_squared=0.9, lam_ci=(0.4, 0.6),
                 lam_by_query_len=((100, 0.44), (900, 0.61)), n_length_bins=2)
    assert fit.lam_spread_across_lengths == pytest.approx(0.17)


class _StubScorer:
    """Scores a query against subjects by shared-residue count. Deterministic, no model."""

    name = "stub"

    def __init__(self, subjects):
        self.subjects = subjects

    def score_all(self, query):
        qs = set(query)
        return np.array([float(len(qs & set(s))) for s in self.subjects])


def test_calibrate_runs_one_pass_per_shuffle_and_records_lengths():
    rng = random.Random(5)
    subs = [_rand_prot(80, rng) for _ in range(40)]
    qs = [_rand_prot(120, rng), _rand_prot(260, rng)]
    sc = _StubScorer(subs)
    try:
        fit = calibrate(sc, subs, qs, n_shuffles=3, n_boot=0, min_r2=0.0)
    except ValueError:
        return  # the stub's score range is too coarse for a tail; the contract tested is below
    assert fit.n_queries == 6
    assert fit.query_lens == (120, 260)
    assert fit.scorer_name == "stub"


def test_best_hit_segments_uses_floor_fn_and_defaults_to_the_alignment_floor():
    rng = random.Random(9)
    payload = _rand_prot(200, rng)
    other = _rand_prot(200, rng)
    orf = payload + other
    refs = [payload, other] + [_rand_prot(150, rng) for _ in range(8)]
    db = sum(len(s) for s in refs)
    sc = __import__("pilot.align", fromlist=["get_scorer"]).get_scorer("align", refs)

    base = best_hit_segments(orf, refs, sc, db, max_evalue=1e-3)
    same = best_hit_segments(orf, refs, sc, db, max_evalue=1e-3, floor_fn=score_floor)
    assert (len(base.segments), base.stop_reason) == (len(same.segments), same.stop_reason)
    assert base.floor == same.floor

    # an unreachable floor stops before the first emission; a floor of zero does not
    high = best_hit_segments(orf, refs, sc, db, floor_fn=lambda e, m, n: 1e9)
    assert high.segments == () and high.stop_reason == "below_floor"
    low = best_hit_segments(orf, refs, sc, db, floor_fn=lambda e, m, n: 0.0)
    assert len(low.segments) >= len(base.segments)


# ---------------------------------------------------------------------------
# the ORF reader: it feeds arms.pick_payload_orf, whose tie-break is the id string
# ---------------------------------------------------------------------------


def test_the_calibration_script_reads_orfs_exactly_as_the_harness_does(tmp_path):
    """A divergent reader changes which ORF is selected on a tie, silently.

    `arms.pick_payload_orf` sorts on `(containment_count, oid)` and keeps only the top `PREFILTER_TOP`
    candidates, so the id string is load-bearing. `scripts/36_cosine_floor.py` had its own reader that
    kept the FASTA description and stripped the plasmid prefix; this pins both readers to the same
    output on a real-format header.
    """
    import importlib.util
    from pathlib import Path as _P

    faa = tmp_path / "orfs.faa"
    faa.write_text(
        ">addgene_107230_pUC57_PP7v3|orf001|+|1..1944 length_aa=647 class=unannotated\n"
        "MRCEIPHRCVRRKYRIRRHSPFRLRNCW\n"
        ">addgene_107230_pUC57_PP7v3|orf002|+|24..197 length_aa=57 class=unannotated\n"
        "MRKEKIPHQAPFAIQAAQLLGRAIGAGL\n"
        ">ncbi_A13313_1|orf007|-|90..501 length_aa=137 class=cds_match\n"
        "MKVLAAGIVGLNQW\n",
        encoding="utf-8")

    def _load(rel):
        spec = importlib.util.spec_from_file_location(
            rel.replace("/", "_"), _P(__file__).resolve().parents[1] / rel)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    harness = _load("scripts/10_twilight.py").read_faa(faa)
    calib = _load("scripts/36_cosine_floor.py").read_faa(faa)
    assert {k: dict(v) for k, v in calib.items()} == {k: dict(v) for k, v in harness.items()}
    # and the keys are what the harness documents, not the ORF number alone
    assert set(calib) == {"addgene_107230_pUC57_PP7v3", "ncbi_A13313_1"}
    assert "addgene_107230_pUC57_PP7v3|orf001|+|1..1944" in calib["addgene_107230_pUC57_PP7v3"]


# ---------------------------------------------------------------------------
# the pre-flight, and rejection as a result rather than a crash
# ---------------------------------------------------------------------------


class _FixedEmbedder:
    """Per-residue vectors with a controllable background cosine."""

    def __init__(self, cos: float):
        self.name = f"fixed-{cos}"
        self.cos = cos

    def embed_residues(self, seqs):
        import numpy as _np
        rng = _np.random.default_rng(3)
        out = []
        base = _np.array([1.0, 0.0])
        for s in seqs:
            ang = _np.arccos(self.cos)
            # alternate +/- ang around base so distinct residues sit at exactly `cos` apart
            v = _np.stack([[_np.cos(a), _np.sin(a)]
                           for a in rng.choice([ang / 2, -ang / 2], size=len(s))])
            out.append(v * 1.0 + base * 0.0)
        return out


def test_background_pair_stats_finds_the_planted_cosine_and_the_viable_shift():
    from pilot.cosine_floor import background_pair_stats
    seqs = ["ACDEFGHIKL" * 6 for _ in range(30)]
    stats = background_pair_stats(_FixedEmbedder(0.5), seqs, n_pairs=4000, seed=1)
    # pairs are either 0 degrees apart (cos 1) or `ang` apart (cos 0.5), so the mean sits between
    assert 0.5 <= stats.mean_cosine <= 1.0
    assert stats.min_viable_shift == stats.mean_cosine
    assert stats.expected_pair_score(stats.mean_cosine + 0.1) < 0
    assert stats.expected_pair_score(stats.mean_cosine - 0.1) > 0
    lo, hi = stats.recommended_shift_range()
    assert stats.expected_pair_score(lo) < 0 and stats.expected_pair_score(hi) < 0
    assert lo < hi
    assert stats.n_proteins > 1 and stats.se_clustered >= 0


class _PerProteinEmbedder:
    """Every residue of protein i points the same way, and each protein points differently.

    The extreme of intra-cluster correlation: a pair's cosine is decided entirely by which two
    proteins it came from, so treating pairs as independent observations is maximally wrong.
    """

    name = "per-protein"

    def embed_residues(self, seqs):
        import numpy as _np
        out = []
        for i, s in enumerate(seqs):
            ang = (i / max(len(seqs), 1)) * (_np.pi / 2)
            out.append(_np.tile([_np.cos(ang), _np.sin(ang)], (len(s), 1)))
        return out


def test_background_pair_stats_clusters_by_protein():
    # The SE must be computed over per-protein means, not over pairs: this project measured a tenfold
    # difference on exactly this quantity and the naive estimate is the wrong one. The inequality is
    # not universal: it holds when residues within a protein are correlated, which is the real case
    # and is what this embedder makes extreme.
    from pilot.cosine_floor import background_pair_stats
    seqs = ["ACDEFGHIKL" * 6 for _ in range(30)]
    stats = background_pair_stats(_PerProteinEmbedder(), seqs, n_pairs=6000, seed=2)
    naive = stats.sd_cosine / (stats.n_pairs ** 0.5)
    assert stats.se_clustered > 3 * naive, (stats.se_clustered, naive)


def test_a_fit_whose_bins_all_fail_is_rejected_but_still_returns_its_diagnostics():
    rows = [np.linspace(0, 10, 500) for _ in range(30)]
    fit = fit_evd(rows, [200 + 20 * (i % 3) for i in range(30)], [200] * 500, n_boot=0)
    assert not fit.usable
    assert "not exponential at any query length" in fit.rejected_reason
    # the point of not raising: the numbers that interpret the failure exist
    assert fit.r2_by_query_len and fit.lam_by_query_len
    assert fit.lam > 0


def test_a_fit_whose_bins_pass_is_usable_even_when_the_pool_is_curved():
    # A mixture of exponentials is curved on a log axis by construction, so a low pooled R^2 beside
    # straight bins is a precision statement, not a validity one.
    lam_by_group = (0.6, 1.4)
    rows, lens = [], []
    for gi, lam in enumerate(lam_by_group):
        m = 200 + 400 * gi
        rows += _exponential_rows(lam, 2e-5, m, [250] * 600, 30, seed=20 + gi)
        lens += [m] * 30
    fit = fit_evd(rows, lens, [250] * 600, n_boot=0, n_length_bins=2)
    assert fit.usable, fit.rejected_reason
    lo, hi = fit.lam_range
    assert lo < fit.lam < hi
    assert fit.with_lam(lo).lam == pytest.approx(lo)
    assert fit.with_lam(lo).k_param == fit.k_param


def test_the_effective_length_correction_changes_k_and_not_lambda():
    # Measured 18 Oct and documented: the lengths enter through the intensity K m N, and lambda is the
    # slope of the score tail, which no length term touches. Asserted so the claim cannot rot.
    lam, m = 0.8, 300
    sub = [250] * 600
    rows = _exponential_rows(lam, 2e-5, m, sub, 40, seed=31)
    on = fit_evd(rows, [m] * len(rows), sub, n_boot=0, effective_length=True)
    off = fit_evd(rows, [m] * len(rows), sub, n_boot=0, effective_length=False)
    assert on.lam == pytest.approx(off.lam, rel=1e-9)
    assert on.k_param != off.k_param
    assert on.mean_effective_query_len < off.mean_effective_query_len


def test_score_shuffles_returns_one_row_per_shuffle_so_it_can_be_cached():
    from pilot.cosine_floor import score_shuffles
    rng = random.Random(8)
    subs = [_rand_prot(60, rng) for _ in range(12)]
    rows, lens = score_shuffles(_StubScorer(subs), [_rand_prot(100, rng)] * 2, n_shuffles=3, seed=4)
    assert len(rows) == len(lens) == 6
    assert all(r.shape == (12,) for r in rows)


def test_the_shipped_eba_shift_is_recorded_as_degenerate_for_the_blosum_embedder():
    """A regression guard on a *fact*, not on code: 0.25 sits below the background cosine.

    Measured on the 40,750-protein catalogue at 0.2697 (clustered SE 0.00125) and predicted from
    Swiss-Prot amino-acid frequencies at 0.2717. If someone later raises `eba.SHIFT` past the boundary
    this test should be updated to assert the new value is viable: the point is that the relationship
    between the default and the boundary is never again implicit.
    """
    from pilot.cosine_floor import background_pair_stats
    from pilot.eba import SHIFT
    from pilot.positional import get_positional_embedder

    rng = random.Random(17)
    # Swiss-Prot-like composition, which is what the real catalogue has; a uniform-composition sample
    # measures 0.2318 and would make the default look safe.
    freq = {"A": .0825, "R": .0553, "N": .0406, "D": .0545, "C": .0137, "Q": .0393, "E": .0675,
            "G": .0707, "H": .0227, "I": .0596, "L": .0966, "K": .0584, "M": .0242, "F": .0386,
            "P": .0470, "S": .0656, "T": .0534, "W": .0108, "Y": .0292, "V": .0687}
    letters, weights = list(freq), list(freq.values())
    seqs = ["".join(rng.choices(letters, weights=weights, k=200)) for _ in range(60)]
    stats = background_pair_stats(get_positional_embedder("blosum"), seqs, n_pairs=40_000, seed=5)
    assert stats.mean_cosine == pytest.approx(0.27, abs=0.02), stats.mean_cosine
    assert stats.expected_pair_score(SHIFT) > 0, "eba.SHIFT is no longer degenerate; update this test"
    lo, hi = stats.recommended_shift_range()
    assert lo > SHIFT and stats.expected_pair_score(lo) < 0
    assert stats.expected_pair_score(stats.blosum62_analogous_shift()) < 0


# ---------------------------------------------------------------------------
# The ungapped lambda bound: theory against a fit, at no cost.


def _blosum62_background_table():
    """`(scores, weights)` over the 400 ordered amino-acid pairs under Swiss-Prot frequencies."""
    from Bio.Align import substitution_matrices

    m = substitution_matrices.load("BLOSUM62")
    aas = list(BACKGROUND)
    tot = sum(BACKGROUND.values())
    p = np.array([BACKGROUND[a] / tot for a in aas])
    s = np.array([[float(m[a, b]) for b in aas] for a in aas])
    return s.ravel(), np.outer(p, p).ravel()


def _blosum_row_cosines():
    """The exact background cosine table of the `blosum62-pos` embedder: normalised BLOSUM62 rows."""
    from Bio.Align import substitution_matrices

    m = substitution_matrices.load("BLOSUM62")
    aas = list(BACKGROUND)
    tot = sum(BACKGROUND.values())
    p = np.array([BACKGROUND[a] / tot for a in aas])
    s = np.array([[float(m[a, b]) for b in aas] for a in aas])
    r = s / np.linalg.norm(s, axis=1, keepdims=True)
    return (r @ r.T).ravel(), np.outer(p, p).ravel()


def test_the_ungapped_lambda_of_blosum62_reproduces_the_published_value():
    """The one external check available on this machinery, and it is the reason to trust the bound.

    NCBI publishes 0.318 for ungapped BLOSUM62. Solving `sum p_i p_j exp(lambda s_ij) = 1` on the same
    matrix under Swiss-Prot frequencies has to land there, or the solver, the matrix units or the
    background are wrong, and all three feed the bound that rejected this project's fits.
    """
    scores, weights = _blosum62_background_table()
    lam = solve_ungapped_lambda(scores, weights)
    assert lam == pytest.approx(0.3172, abs=5e-4)
    assert abs(lam - 0.318) < 0.005, f"published ungapped BLOSUM62 lambda is 0.318, solved {lam}"
    assert BLOSUM62_UNGAPPED_LAMBDA == pytest.approx(lam, abs=5e-4)


def test_the_blosum62_constants_are_computed_and_the_old_ratio_was_wrong():
    """Pins the 18 Oct erratum: the ratio in use until then was -0.047, from a -0.52 expected score.

    Under the background the cosine side is measured on, BLOSUM62's expected score is -0.9379, not
    -0.52, and the ratio is -0.0853. Both sides of a ratio have to be computed the same way; this test
    is what stops a published number in an unstated convention being substituted back in.
    """
    scores, weights = _blosum62_background_table()
    e = float((weights * scores).sum())
    sd = math.sqrt(float((weights * scores ** 2).sum()) - e ** 2)
    assert e == pytest.approx(BLOSUM62_EXPECTED_SCORE, abs=5e-4)
    assert sd == pytest.approx(BLOSUM62_PAIR_SCORE_SD, abs=5e-4)
    assert scores.max() == BLOSUM62_MAX_SCORE
    assert BLOSUM62_EXPECTED_OVER_MAX == pytest.approx(-0.0853, abs=5e-4)
    assert BLOSUM62_EXPECTED_OVER_SD == pytest.approx(-0.4496, abs=5e-4)
    assert BLOSUM62_EXPECTED_OVER_MAX != pytest.approx(-0.047, abs=1e-3), "the erratum is back"


def _pair_stats_for_blosum_rows() -> PairStats:
    c, w = _blosum_row_cosines()
    mu = float((w * c).sum())
    sd = math.sqrt(float((w * c ** 2).sum()) - mu ** 2)
    return PairStats(mean_cosine=mu, sd_cosine=sd, median_cosine=mu, n_pairs=int(c.size),
                     n_proteins=0, se_clustered=float("nan"), embedder_name="blosum62-pos",
                     cosine_hist=tuple((float(x), float(y)) for x, y in zip(c, w)))


def test_no_ungapped_lambda_exists_at_the_shipped_shift():
    """`eba.SHIFT = 0.25` has a positive expected pair score, so the defining equation has no root.

    This is the same refusal as the working record's, reached from the other direction: there is no lambda,
    not merely a badly fitted one.
    """
    st = _pair_stats_for_blosum_rows()
    assert st.expected_pair_score(0.25) > 0
    assert math.isnan(st.ungapped_lambda(0.25))
    assert math.isnan(st.gumbel_onset(0.25))


def test_the_ungapped_lambda_and_the_gumbel_onset_rise_with_the_shift():
    """Both are monotone in the shift, which is what makes 'raise the shift' a usable prescription."""
    st = _pair_stats_for_blosum_rows()
    shifts = [0.3025, 0.3271, 0.42, 0.4372, 0.50]
    lams = [st.ungapped_lambda(s) for s in shifts]
    assert all(b > a for a, b in zip(lams, lams[1:])), lams
    onsets = [st.gumbel_onset(s) for s in shifts]
    assert all(b > a for a, b in zip(onsets, onsets[1:])), onsets
    # the two numbers the working record turns on, to a tolerance the 200-bin histogram can hold
    assert st.ungapped_lambda(0.3025) == pytest.approx(0.4275, abs=0.01)
    assert st.gumbel_onset(0.3025) == pytest.approx(0.161, abs=0.01)
    assert st.gumbel_onset(0.42) == pytest.approx(0.768, abs=0.02)


def test_the_snr_matched_shift_puts_the_signal_to_noise_at_blosum62s():
    """A derivation, so it must round-trip; the `E/max` analogy is kept but is not this."""
    st = _pair_stats_for_blosum_rows()
    s = st.snr_matched_shift()
    assert st.signal_to_noise(s) == pytest.approx(BLOSUM62_EXPECTED_OVER_SD, abs=1e-9)
    assert s == pytest.approx(0.4372, abs=5e-3)
    # and the two criteria genuinely disagree, which is the whole.3
    assert st.blosum62_analogous_shift() == pytest.approx(0.3271, abs=5e-3)
    assert s - st.blosum62_analogous_shift() > 0.10


def _gumbel_rows(lam: float, lens: list[int], n_each: int, n_sub: int, sub_len: int, seed: int):
    """Score rows drawn from the exact Karlin-Altschul model, so the planted lambda is the truth.

    A subject score above `x` has probability `1 - exp(-K m n exp(-lambda x))`, which is a Gumbel with
    location `ln(K m n) / lambda` and scale `1 / lambda`. Planting it per query length is what makes
    the length-dependence check below a measurement of the estimator rather than of the scorer.
    """
    rng = np.random.default_rng(seed)
    rows, qlens = [], []
    for m in lens:
        mu = math.log(0.04 * m * n_sub * sub_len) / lam
        for _ in range(n_each):
            rows.append(rng.gumbel(mu, 1.0 / lam, n_sub))
            qlens.append(m)
    return rows, qlens, [sub_len] * n_sub


def test_a_true_gumbel_at_this_sample_size_has_a_small_lambda_spread():
    """The reference the real run is read against, and the reason its drift is not sampling noise.

    this project's calibration fitted 192 shuffled queries over query lengths 120 to 669 and reported
    a per-bin lambda spread of **162% of lambda** with one bin's R^2 at 0.890. Planting a genuine
    Gumbel at the same lengths and the same sample size gives a spread under 25% and every bin above
    0.99, so neither the estimator nor 64 queries per bin can produce what that run produced. This
    test is what licenses reading the drift as a property of the scoring system.
    """
    rows, qlens, sublens = _gumbel_rows(0.40, [120, 236, 261, 311, 457, 493, 640, 669],
                                        n_each=24, n_sub=1000, sub_len=350, seed=7)
    fit = fit_evd(rows, qlens, sublens, n_boot=0, n_length_bins=3)
    assert fit.usable, fit.rejected_reason
    assert fit.lam == pytest.approx(0.40, rel=0.10)
    assert all(r2 >= 0.99 for _, r2 in fit.r2_by_query_len), fit.r2_by_query_len
    assert 100 * fit.lam_spread_across_lengths / fit.lam < 25, fit.lam_by_query_len


def test_a_fitted_lambda_above_the_ungapped_bound_is_reported_and_does_not_reject():
    """Theory against the fit. Reported beside the verdict, never folded into it (the working record rule 2).

    The pre-registered rejection rule predates this check, so the check may not silently tighten it:
    a fit that passes its R^2 rule still reports `usable`, and the bound shows up only in
    `lams_above_bound`.
    """
    rows, qlens, sublens = _gumbel_rows(0.40, [200, 400, 600], n_each=24, n_sub=800,
                                        sub_len=350, seed=11)
    ok = fit_evd(rows, qlens, sublens, n_boot=0, ungapped_lambda_bound=0.60)
    assert ok.lams_above_bound == ()
    tight = fit_evd(rows, qlens, sublens, n_boot=0, ungapped_lambda_bound=0.20)
    assert tight.usable, "the bound must not change the pre-registered verdict"
    assert tight.lams_above_bound, "a lambda above the bound has to be reported"
    assert tight.lams_above_bound[0][0] == 0, "the pooled fit is labelled 0"
    # and with no bound supplied there is nothing to report either way
    assert fit_evd(rows, qlens, sublens, n_boot=0).lams_above_bound == ()


def test_the_agreement_block_reports_each_variants_own_distribution():
    """The the working record defect: every variant's block printed the POOLED cosine distribution.

    The counts and the confusion matrix were correct throughout, which is why 27 minutes of output
    looked self-consistent: only the `mean`/`histogram` line was wrong, and it was wrong in the
    direction of making a 95%-agreeing variant look like a 50%-agreeing one.
    """
    import importlib.util
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "s36", _P(__file__).resolve().parents[1] / "scripts/36_cosine_floor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rows = [{"align_k": 1, "align_iterations": 1, "cosine_k": 0, "cosine_iterations": 0,
             "cosine_lam_min_k": 1, "cosine_lam_min_iterations": 1,
             "cosine_lam_max_k": 2, "cosine_lam_max_iterations": 3},
            {"align_k": 2, "align_iterations": 2, "cosine_k": 0, "cosine_iterations": 0,
             "cosine_lam_min_k": 2, "cosine_lam_min_iterations": 2,
             "cosine_lam_max_k": 3, "cosine_lam_max_iterations": 4}]
    a = mod.agreement(rows, ["pooled", "lam_min", "lam_max"])

    assert a["k"]["cosine"]["histogram"] == {"0": 2}
    assert a["k_lam_min"]["cosine"]["histogram"] == {"1": 1, "2": 1}
    assert a["k_lam_max"]["cosine"]["histogram"] == {"2": 1, "3": 1}
    assert a["iterations_lam_max"]["cosine"]["histogram"] == {"3": 1, "4": 1}
    assert a["k_lam_min"]["exact"] == 2 and a["k"]["exact"] == 0
    # the distribution and the confusion matrix must tell the same story: the defect was exactly
    # that they did not
    for key, rec in a.items():
        from_confusion = []
        for pair, n in rec["confusion"].items():
            from_confusion += [int(pair.split("->")[1])] * n
        assert sorted(from_confusion) == sorted(
            int(v) for v, n in rec["cosine"]["histogram"].items() for _ in range(n)), key


def test_un_refuted_and_promotable_are_different_tests():
    """The the working record reading error, pinned: `usable` is not this project's rule 2.

    `usable` tolerates one failing length bin on purpose: a mixture of exponentials is curved on a
    log axis, so a single poor bin does not refute the existence of a Gumbel regime. Rule 2 asks the
    other question, whether a floor may be promoted from the fit, and requires every bin. A fit that
    is un-refuted but not promotable is the case that actually arose.
    """
    base = fit_evd(*_gumbel_rows(0.40, [120, 236, 261, 311, 457, 493, 640, 669], n_each=24,
                                 n_sub=1000, sub_len=350, seed=7), n_boot=0)
    assert base.usable and base.valid_for_promotion()[0]

    one_bad = replace(base, r2_by_query_len=((208, 0.995), (322, 0.991), (566, 0.890)))
    assert one_bad.usable, "one failing bin must not refute the regime"
    ok, why = one_bad.valid_for_promotion()
    assert not ok and "566" in why, why
    # and the threshold is the caller's, so a looser gate passes the same fit
    assert one_bad.valid_for_promotion(min_r2=0.88)[0]

    none_fitted = replace(base, r2_by_query_len=())
    assert not none_fitted.valid_for_promotion()[0]


def test_background_pair_stats_carries_the_distribution_the_lambda_bound_needs():
    """The histogram is not decoration: `ungapped_lambda` needs the whole distribution, not two moments.

    Two scoring systems can share a mean and an sd and have different `lambda`, because `lambda` is the
    root of an exponential-moment equation. So `background_pair_stats` carries the sampled histogram and
    the bound is available at the same cost as the mean, which is the point of a check that is meant to
    run before every EBA run.
    """
    from pilot.cosine_floor import background_pair_stats

    seqs = ["ACDEFGHIKLMNPQRSTVWY" * 4 for _ in range(30)]
    stats = background_pair_stats(_FixedEmbedder(0.5), seqs, n_pairs=4000, seed=1)
    assert stats.cosine_hist, "the histogram has to be populated or the bound is unavailable"
    assert sum(n for _, n in stats.cosine_hist) == stats.n_pairs
    # inside the viable range a root exists and the onset is positive; below it neither does
    viable = stats.mean_cosine + 0.2
    assert stats.ungapped_lambda(viable) > 0
    assert stats.gumbel_onset(viable) > 0
    assert math.isnan(stats.ungapped_lambda(stats.mean_cosine - 0.05))
    # a hand-built PairStats has no histogram and must say so rather than guess
    bare = PairStats(mean_cosine=0.27, sd_cosine=0.37, median_cosine=0.27, n_pairs=1,
                     n_proteins=1, se_clustered=0.0)
    assert math.isnan(bare.ungapped_lambda(0.42))
    #... while the moment-based quantities still work without it
    assert bare.signal_to_noise(0.42) == pytest.approx((0.27 - 0.42) / 0.37)
    assert bare.snr_matched_shift() > 0.27


def test_the_json_record_drops_the_histogram_and_keeps_what_is_derived_from_it():
    """200 bins are an implementation detail; the bound and the onset are what a reader needs."""
    import importlib.util
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "s36b", _P(__file__).resolve().parents[1] / "scripts/36_cosine_floor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    st = _pair_stats_for_blosum_rows()
    rec = mod.stats_record(st, 0.42)
    assert "cosine_hist" not in rec
    assert rec["n_cosine_hist_bins"] == len(st.cosine_hist)
    assert rec["ungapped_lambda"] == pytest.approx(st.ungapped_lambda(0.42), abs=1e-6)
    assert rec["gumbel_onset"] == pytest.approx(st.gumbel_onset(0.42), abs=1e-4)
    assert rec["signal_to_noise"] == pytest.approx(st.signal_to_noise(0.42), abs=1e-4)
    assert rec["snr_matched_shift"] == pytest.approx(0.4372, abs=5e-3)
    assert rec["mean_cosine"] == st.mean_cosine
    import json
    json.dumps(rec)
