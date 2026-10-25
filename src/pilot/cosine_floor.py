"""An empirical significance floor, in whatever units a scorer returns.

The best-hit segmenter stops when no remaining hit clears a score floor, and that floor is
`homology.score_floor`: Karlin-Altschul inverted, `S_min = (ln(K m N) + ln(1/E)) / lambda`, with `K`
and `lambda` the published BLOSUM62 gapped constants. Those constants are properties of *that* scoring
system. Hand the segmenter a cosine-substitution scorer and the floor is meaningless, so the project's
best method could not run on an embedding at all (open item 25).

**What this module does not do: invent a new rule.** It fits the *same* equation to the scorer actually
in use. The asymptotic distribution of an optimal local alignment score is Gumbel for any scoring scheme
with a negative expected score per aligned pair, which is a property of the alignment, not of where the
substitution scores came from. So `E = K m N exp(-lambda S)` still holds; only `K` and `lambda` are
unknown, and they are estimable from scores against shuffled queries. That keeps both of the floor's
dependencies, query length `m` and database size `N`, which a chosen cosine threshold carries
neither of, and it is why the replacement has the same signature as the thing it replaces.

**Why shuffled queries and not shuffled subjects.** The null wanted is "this query against a database
with no relative of it in it". Shuffling the query destroys its own structure while preserving its
length and composition, so the length dependence being fitted is the real one. Shuffling subjects would
also destroy the *catalogue's* compositional structure, which the real search does see.

**The estimate is clustered, because the observations are not independent.** One shuffled query scored
against 2,000 subjects gives 2,000 scores that share a query, exactly as this project's 100,000 residue
pairs shared proteins and understated the standard error tenfold. `fit_evd` bootstraps over *shuffled
queries*, never over pairs, and reports the interval it gets.

**Calibrate with no prefilter.** `EbaScorer` maps non-shortlisted references into a band below the
reranked ones, so with `top_k` set the score row is partly synthetic and the tail being fitted is the
band rather than the scorer. Build the calibration scorer with `top_k=0`.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

import numpy as np

# The tail the regression is fitted on, as an expected count of subject hits per query. Below ~0.01 the
# counts are too sparse to be stable; above ~10 the asymptotic form has not taken hold, because the bulk
# of a local-alignment score distribution is not its tail. Reported with every fit rather than buried.
TAIL_LO, TAIL_HI = 0.01, 10.0
N_TAIL_POINTS = 24
# A survival function always decreases, so "the slope is negative" is not evidence of anything: a
# uniform score distribution passes it. The test that a Gumbel regime exists at all is that the tail is
# *straight* on a log-count axis, so the fit is rejected on R^2 rather than on sign. Reported with every
# fit, never silently met.
MIN_R2 = 0.97


# Swiss-Prot background amino-acid frequencies. Used only to weight the expected pair score, which is a
# property of the scoring system and not of any particular query.
BACKGROUND = {"A": .0825, "R": .0553, "N": .0406, "D": .0545, "C": .0137, "Q": .0393, "E": .0675,
              "G": .0707, "H": .0227, "I": .0596, "L": .0966, "K": .0584, "M": .0242, "F": .0386,
              "P": .0470, "S": .0656, "T": .0534, "W": .0108, "Y": .0292, "V": .0687}
# BLOSUM62's own moments under the background above, recomputed here rather than quoted. **The value
# used until 18 Oct was -0.047, from a textbook expected score of -0.52, and it was wrong** (CONTEXT
# 35.2): under these frequencies the expected score is -0.9379 half-bits, so the ratio is -0.0853. The
# defect was not the arithmetic but the method: the cosine side of the ratio was measured on the
# catalogue while the BLOSUM62 side was a published number in an unstated convention, which is the
# coverage-denominator bug of 18.4 in a new costume: both sides of a ratio must be computed the same way.
# The internal check that settles it: solving `sum p_i p_j exp(lambda s_ij) = 1` on the same table gives
# lambda = 0.3172 against NCBI's published ungapped BLOSUM62 lambda of 0.318, so the moments are right.
BLOSUM62_EXPECTED_SCORE = -0.9379
BLOSUM62_PAIR_SCORE_SD = 2.0864
BLOSUM62_MAX_SCORE = 11.0
BLOSUM62_UNGAPPED_LAMBDA = 0.3172
BLOSUM62_EXPECTED_OVER_MAX = BLOSUM62_EXPECTED_SCORE / BLOSUM62_MAX_SCORE          # -0.0853
# **The ratio that transfers between scoring systems is this one, not the one above.** What decides
# whether a local alignment terminates, and how long an alignment has to be before the Gumbel
# asymptotics assert themselves, is how many standard deviations of pair-score noise the per-pair
# drift buys, not how the drift compares with a single extreme matrix entry. `max` is one cell of a
# 20x20 table and carries no information about the distribution's scale, so matching `E/max` across two
# scoring systems with different dynamic ranges matches nothing. the working record is what that cost.
BLOSUM62_EXPECTED_OVER_SD = BLOSUM62_EXPECTED_SCORE / BLOSUM62_PAIR_SCORE_SD       # -0.4496
VIABLE_EXPECTED_SCORE_RANGE = (-0.25, -0.02)


def solve_ungapped_lambda(scores, weights, hi: float = 60.0) -> float:
    """The `lambda` of a scoring system from its pair-score distribution alone: no alignment, no fits.

    Karlin-Altschul defines `lambda` as the unique positive root of `sum_ij p_i p_j exp(lambda s_ij) = 1`,
    which exists exactly when the expected pair score is negative. This is the *ungapped* lambda, and it
    is an **upper bound on the gapped one**: allowing gaps can only raise the optimal score of any pair
    of sequences, which shifts the score distribution right and flattens its exponential tail. BLAST's
    own constants show the size of the gap: BLOSUM62 ungapped 0.318 against gapped (11,1) 0.267.

    That makes it the cheapest hard check on a fitted lambda there is. A fitted value **above** this
    bound is not a noisy estimate of lambda; it is evidence that whatever was fitted is not lambda. It
    costs no dynamic programming and no shuffled queries, so it belongs in the pre-flight beside the
    expected pair score (where it rejected all three per-bin fits of a 27-minute run).

    Returns `nan` when no root exists, which is the degenerate case `E[pair score] >= 0`.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if s.size != w.size or s.size == 0:
        raise ValueError("one weight per pair score")
    tot = float(w.sum())
    if tot <= 0:
        raise ValueError("pair-score weights must be positive")
    w = w / tot

    def f(lam: float) -> float:
        return float((w * np.exp(lam * s)).sum()) - 1.0

    if float((w * s).sum()) >= 0 or f(hi) < 0:
        return float("nan")
    lo = 1e-12
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class PairStats:
    """The background cosine of one embedder, and the shift range that admits local alignment at all."""

    mean_cosine: float
    sd_cosine: float
    median_cosine: float
    n_pairs: int
    n_proteins: int
    se_clustered: float
    embedder_name: str = ""
    # `(bin centre, weight)` over the sampled cosines. The first two moments fix admissibility
    # (`E[pair score] < 0`) but not `lambda`, which needs the whole distribution, so the histogram is
    # carried here and `ungapped_lambda` is available at the same cost as the mean. Empty on a
    # hand-built `PairStats`, in which case `ungapped_lambda` returns `nan` rather than guessing.
    cosine_hist: tuple[tuple[float, float], ...] = ()

    def expected_pair_score(self, shift: float) -> float:
        return self.mean_cosine - float(shift)

    @property
    def min_viable_shift(self) -> float:
        """Below this, `E[pair score] >= 0` and there is no local optimum to be extreme-valued about.

        Smith-Waterman's local optimum exists because extending an alignment through unrelated residues
        costs score on average. If the expected pair score is positive, the optimal local alignment
        grows with the sequences instead of terminating, the score measures length rather than
        similarity, and the score distribution is Gaussian rather than Gumbel, so there is no E-value
        to invert and no floor to derive. **This is a property of the scoring system, computable in
        milliseconds, and it bounds every shift a sweep may legitimately visit.**
        """
        return self.mean_cosine

    def recommended_shift_range(self) -> tuple[float, float]:
        lo, hi = VIABLE_EXPECTED_SCORE_RANGE
        return (self.mean_cosine - hi, self.mean_cosine - lo)

    def blosum62_analogous_shift(self) -> float:
        """The shift putting `E[pair score] / max` at BLOSUM62's ratio. One point inside the range.

        **Kept, and no longer the recommendation.** With the corrected ratio (-0.0853, not the -0.047
        used until 18 Oct) this lands at 0.3271 for the BLOSUM62-row embedder rather than 0.3025. Both
        sit where the Gumbel regime has not asserted itself at the query lengths this project uses, which
        is why `snr_matched_shift` supersedes it: and the comment on
        `BLOSUM62_EXPECTED_OVER_SD`.
        """
        return (self.mean_cosine - BLOSUM62_EXPECTED_OVER_MAX) / (1 - BLOSUM62_EXPECTED_OVER_MAX)

    def signal_to_noise(self, shift: float) -> float:
        """`E[pair score] / sd[pair score]`: the drift per aligned pair in units of its own noise."""
        return self.expected_pair_score(shift) / self.sd_cosine if self.sd_cosine > 0 else float("nan")

    def snr_matched_shift(self) -> float:
        """The shift putting the per-pair signal-to-noise at BLOSUM62's own, which is -0.4496.

        A derivation rather than an analogy to a single matrix cell: it matches the quantity that sets
        how quickly the extreme-value asymptotics take hold, so a scorer tuned to it reaches a Gumbel
        regime over the same alignment lengths BLOSUM62 does. For the BLOSUM62-row embedder it gives
        0.4372.
        """
        return self.mean_cosine - BLOSUM62_EXPECTED_OVER_SD * self.sd_cosine

    def ungapped_lambda(self, shift: float) -> float:
        """`lambda` for the cosine scoring system at `shift`, from the sampled pair distribution.

        An upper bound on any gapped `lambda` fitted against shuffled queries: see
        `solve_ungapped_lambda`. Needs `cosine_hist`; `nan` without it.
        """
        if not self.cosine_hist:
            return float("nan")
        c = np.asarray([x for x, _ in self.cosine_hist], dtype=np.float64)
        w = np.asarray([y for _, y in self.cosine_hist], dtype=np.float64)
        return solve_ungapped_lambda(c - float(shift), w)

    def gumbel_onset(self, shift: float) -> float:
        """`lambda_ungapped * sd[pair score]`: how fast the tail decays per unit of pair-score noise.

        **The diagnostic that explains the working record.** A small value means an alignment must be long
        before its score distribution looks Gumbel, so a fit over short and middling queries measures
        the approach to the asymptote rather than `lambda`, and the fitted value then depends on query
        length, which `lambda` cannot. At shift 0.3025 this is 0.161 for the BLOSUM62-row embedder and
        the fitted `lambda` drifted 2.2-fold across three length bins; at 0.42 it is 0.768.
        """
        lam = self.ungapped_lambda(shift)
        return lam * self.sd_cosine if lam == lam else float("nan")


def background_pair_stats(embedder, seqs: list[str], n_pairs: int = 100_000, seed: int = 20261018,
                          max_proteins: int = 400) -> PairStats:
    """Cosine between residues drawn from *different* proteins, which is the unrelated-pair background.

    Costs one embedding pass over at most `max_proteins` sequences and no dynamic programming at all, so
    it is affordable before every run and **should be**: the working record is 2 h 45 m of calibration thrown
    away by a scoring system this check rejects in under a second.

    The standard error is **clustered by protein**, because residues drawn from one protein are not
    independent observations: this project measured a tenfold difference between the naive and
    clustered estimates on exactly this quantity.
    """
    rng = random.Random(seed)
    pool = list(seqs)
    rng.shuffle(pool)
    pool = [s for s in pool[:max_proteins] if len(s) >= 2]
    if len(pool) < 2:
        raise ValueError("need at least two sequences to sample cross-protein residue pairs")
    embs = embedder.embed_residues(pool)
    norms = []
    for e in embs:
        e = np.asarray(e, dtype=np.float64)
        n = np.linalg.norm(e, axis=1, keepdims=True)
        norms.append(e / np.where(n == 0, 1.0, n))
    per_cluster: dict[int, list[float]] = {}
    for _ in range(n_pairs):
        i = rng.randrange(len(norms))
        j = rng.randrange(len(norms))
        if i == j:
            continue
        a = norms[i][rng.randrange(norms[i].shape[0])]
        b = norms[j][rng.randrange(norms[j].shape[0])]
        per_cluster.setdefault(i, []).append(float(a @ b))
    vals = [v for xs in per_cluster.values() for v in xs]
    if not vals:
        raise ValueError("no cross-protein pairs were sampled")
    arr = np.asarray(vals)
    means = np.asarray([float(np.mean(xs)) for xs in per_cluster.values() if xs])
    se = float(np.std(means, ddof=1) / np.sqrt(means.size)) if means.size > 1 else float("nan")
    # 200 bins over the observed range: enough resolution that `solve_ungapped_lambda` lands within a
    # per-mille of the root solved on the raw sample, and small enough to carry in a JSON record.
    counts, edges = np.histogram(arr, bins=200)
    centres = 0.5 * (edges[:-1] + edges[1:])
    hist = tuple((float(c), float(n)) for c, n in zip(centres, counts) if n > 0)
    return PairStats(mean_cosine=float(arr.mean()), sd_cosine=float(arr.std()),
                     median_cosine=float(np.median(arr)), n_pairs=int(arr.size),
                     n_proteins=len(per_cluster), se_clustered=se,
                     embedder_name=getattr(embedder, "name", ""), cosine_hist=hist)


def shuffle_preserving_composition(seq: str, rng: random.Random) -> str:
    """A permutation of `seq`: same length, same residue counts, no remaining order."""
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


@dataclass(frozen=True)
class EvdFit:
    """`lambda` and `K` for one scorer, plus everything needed to judge the fit.

    `score_floor` has the same signature as `homology.score_floor`, so an `EvdFit` can be handed
    straight to `best_hit_segments(..., floor_fn=fit.score_floor)`.
    """

    lam: float
    k_param: float
    n_queries: int
    n_pairs: int
    subject_residues: int
    query_lens: tuple[int, ...]
    tail_lo: float
    tail_hi: float
    n_tail_points: int
    r_squared: float
    lam_ci: tuple[float, float]
    lam_by_query_len: tuple[tuple[int, float], ...]
    n_length_bins: int = 3
    scorer_name: str = ""
    # A fit that fails its goodness test is still a measurement, and the per-bin numbers are what say
    # *why* it failed: a pooled R^2 can be low either because no Gumbel regime exists or because
    # lambda varies across the lengths that were pooled. An earlier version raised before computing
    # them, which discarded 2 h 45 m of scoring and left the question unanswerable.
    rejected_reason: str = ""
    r2_by_query_len: tuple[tuple[int, float], ...] = ()
    effective_length: bool = True
    mean_effective_query_len: float = 0.0
    # The ungapped `lambda` of the same scoring system, from `PairStats.ungapped_lambda`. Carried so the
    # fit can be judged against theory and not only against its own R^2: a gapped `lambda` must be
    # *below* this, so anything above it is not an estimate of `lambda` at all.
    ungapped_lambda_bound: float = float("nan")

    @property
    def usable(self) -> bool:
        return not self.rejected_reason

    def valid_for_promotion(self, min_r2: float = MIN_R2) -> tuple[bool, str]:
        """the working record rule 2: **every** query-length bin at or above `MIN_R2`. Not the same test as
        `usable`, and the distinction matters.

        `usable` is a *refutation* flag: it asks whether a Gumbel regime exists at all, and so it
        tolerates one bin failing, because a mixture of exponentials is curved on a log axis by
        construction. Rule 2 is an *acceptance* gate: whether this fit is good enough to promote a
        floor from. A fit can be un-refuted and still not promotable, which is exactly what the
        shift-0.3025 calibration turned out to be, and reading `usable` as the
        acceptance test is what made that run look like a pass.
        """
        if not self.r2_by_query_len:
            return (False, "no per-bin fit succeeded, so rule 2 cannot be evaluated")
        bad = [(m, v) for m, v in self.r2_by_query_len if v < min_r2]
        if bad:
            return (False, f"per-bin R^2 below {min_r2} at query lengths {bad}")
        return (True, "")

    @property
    def lams_above_bound(self) -> tuple[tuple[int, float], ...]:
        """The `(query length, lambda)` pairs that exceed the ungapped bound, pooled fit labelled 0.

        Empty when no bound was supplied. **Reported, never used to reject**: the rejection rule is
        pre-registered in the working record and this check arrived after it, so it is evidence to read
        beside the verdict rather than a change to the verdict.
        """
        b = self.ungapped_lambda_bound
        if b != b:
            return ()
        out = [(0, round(self.lam, 6))] if self.lam > b else []
        out += [(m, v) for m, v in self.lam_by_query_len if v > b]
        return tuple(out)

    def score_floor(self, max_evalue: float, query_len: int, db_residues: int) -> float:
        """The score at which `E == max_evalue`, in this scorer's units. Inverse of `log_evalue`.

        Signature-compatible with `homology.score_floor`, so it drops into
        `best_hit_segments(floor_fn=...)`. When the fit used the finite-length correction the same
        correction is applied here, or the floor would be evaluated under a different model from the
        one that produced `lambda`.
        """
        m, n = float(max(query_len, 1)), float(max(db_residues, 1))
        if self.effective_length:
            em, en = effective_lengths([m], [n], self.lam, self.k_param)
            m, n = float(em[0]), en
        return ((math.log(self.k_param) + math.log(m) + math.log(n)
                 - math.log(max_evalue)) / self.lam)

    def log_evalue(self, score: float, query_len: int, db_residues: int) -> float:
        """`ln E` for this score. The exact inverse of `score_floor`, correction and all.

        The two must apply the same finite-length correction or they stop inverting each other:
        caught by `test_floor_inverts_its_own_evalue` the moment the correction was added to one of
        them.
        """
        m, n = float(max(query_len, 1)), float(max(db_residues, 1))
        if self.effective_length:
            em, en = effective_lengths([m], [n], self.lam, self.k_param)
            m, n = float(em[0]), en
        return math.log(self.k_param) + math.log(m) + math.log(n) - self.lam * score

    @property
    def lam_range(self) -> tuple[float, float]:
        """The per-bin lambda extremes, or the point estimate twice. The floor's sensitivity band.

        A lambda that varies across query-length bins is a precision problem, not a validity one, so
        the honest response is to carry the band into the acceptance test rather than to report one
        floor as if it were exact. `scripts/36_cosine_floor.py` segments at all three.
        """
        if not self.lam_by_query_len:
            return (self.lam, self.lam)
        vals = [v for _, v in self.lam_by_query_len]
        return (min(vals), max(vals))

    def with_lam(self, lam: float) -> "EvdFit":
        """The same fit at a different lambda, for the sensitivity check."""
        return replace(self, lam=float(lam))

    @property
    def lam_spread_across_lengths(self) -> float:
        """Max minus min of the per-length-bin `lambda`. The headline goodness check.

        `lambda` is a property of the scoring system and must not depend on query length. If this is
        large relative to `lam`, the fit has absorbed a length effect into a constant that is supposed
        not to have one, and the floor will be wrong at the lengths furthest from the calibration set.

        **Read it with `n_queries` in hand.** The check is only informative once each bin's tail has
        enough observations to fit: at a handful of shuffled queries per bin it reports sampling noise
        and will read as a large spread whatever the scorer does. Bins, not exact lengths, exist for
        that reason: one bin per distinct ORF length is never affordable.
        """
        if len(self.lam_by_query_len) < 2:
            return 0.0
        vals = [v for _, v in self.lam_by_query_len]
        return max(vals) - min(vals)


def effective_lengths(query_lens, subject_lens, lam: float, k_param: float,
                      n_iter: int = 5) -> tuple[np.ndarray, float]:
    """BLAST's finite-length correction: subtract the expected alignment length from both sequences.

    An optimal alignment needs room: a hit cannot start within `l` residues of either end, so the
    number of places it can start is `(m - l)(n - l)` rather than `m n`. The expected length of an
    optimal local alignment is `l = ln(K m n) / lambda`, which depends on the parameters being fitted,
    so this iterates to a fixed point, as every Karlin-Altschul based tool does.

    **It corrects `K`, not `lambda`, and it does not fix a drifting lambda.** Measured 18 Oct: the fit
    is identical to six decimal places with the correction on and off, because the lengths enter the
    E-value through the intensity `K m N` and `lambda` is the *slope* of the score tail, which no
    length term touches. The correction still belongs here, it is what makes the floor right at
    lengths far from the calibration set, but an apparent lambda that varies across query-length bins
    is a different problem and is not addressed by this.

    Returns `(corrected query lengths, corrected total subject residues)`.
    """
    m = np.asarray(query_lens, dtype=np.float64)
    n = np.asarray(subject_lens, dtype=np.float64)
    m_eff, n_eff = m.copy(), n.copy()
    for _ in range(max(n_iter, 1)):
        ell = (math.log(max(k_param, 1e-300)) + np.log(np.maximum(m_eff.mean(), 1.0))
               + np.log(np.maximum(n_eff.sum(), 1.0))) / max(lam, 1e-12)
        # Clipped at one residue, NOT at BLAST's `1/K`. BLAST can use `1/K` because its `K` is of
        # order 0.04, so `1/K` is about 25 residues; a *fitted* K here can be any magnitude, and when
        # `1/K` exceeded every query length the clip silently disabled the whole correction: found by
        # `test_the_effective_length_correction_changes_k_and_not_lambda`, which saw the corrected and
        # uncorrected K come back bit-identical.
        m_eff = np.maximum(m - ell, 1.0)
        n_eff = np.maximum(n - ell, 1.0)
    return m_eff, float(n_eff.sum())


def _tail_regression(scores: np.ndarray, n_queries: int, mean_m: float, subject_residues: int,
                     tail_lo: float, tail_hi: float, n_points: int,
                     min_r2: float = 0.0) -> tuple[float, float, float]:
    """Regress `log E_hat(x)` on `x` over the tail. Returns `(lambda, K, r_squared)`.

    `E_hat(x)` is the observed mean number of subjects per query scoring at or above `x`, which
    Karlin-Altschul predicts is `K * m * N_subjects * exp(-lambda x)`. So the slope is `-lambda` and the
    intercept gives `K` once `m` and `N` are divided out. Fitting the *tail* of the survival function
    rather than the mode is the point: the mode of a local-alignment score distribution is not Gumbel
    and never was.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    counts = np.arange(1, s.size + 1, dtype=np.float64) / n_queries  # E_hat at x = s[i]
    keep = (counts >= tail_lo) & (counts <= tail_hi)
    if keep.sum() < 4:
        raise ValueError(f"only {int(keep.sum())} observations in the tail "
                         f"[{tail_lo}, {tail_hi}] per query; raise the shuffle count")
    x, y = s[keep], np.log(counts[keep])
    # Thin to `n_points` evenly across the kept range so a dense bulk cannot outvote a sparse tail.
    if x.size > n_points:
        pick = np.unique(np.linspace(0, x.size - 1, n_points).round().astype(int))
        x, y = x[pick], y[pick]
    if np.ptp(x) == 0:
        raise ValueError("every tail score is identical; there is no distribution to fit")
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if slope >= 0:
        raise ValueError(f"fitted slope {slope:.4g} is not negative, which should be impossible for a "
                         "survival function; the input is not a set of score rows")
    if r2 < min_r2:
        raise ValueError(f"the tail is not exponential: R^2 = {r2:.4f} against a required {min_r2}. "
                         "This scoring system has no Gumbel regime over the fitted range, so there is "
                         "no E-value to invert and no floor to derive. Check the shift: a scorer "
                         "whose expected pair score is positive aligns end to end and has no local "
                         "optimum to be extreme-valued about.")
    lam = float(-slope)
    k_param = float(math.exp(intercept) / (max(mean_m, 1.0) * max(subject_residues, 1)))
    return lam, k_param, r2


def fit_evd(score_rows: list[np.ndarray], query_lens: list[int], subject_lens: list[int],
            *, n_boot: int = 200, seed: int = 20261018, scorer_name: str = "",
            tail_lo: float = TAIL_LO, tail_hi: float = TAIL_HI,
            n_tail_points: int = N_TAIL_POINTS, min_r2: float = MIN_R2,
            n_length_bins: int = 3, effective_length: bool = True,
            ungapped_lambda_bound: float = float("nan")) -> EvdFit:
    """Fit `lambda` and `K` from shuffled-query score rows.

    `score_rows[i]` is one shuffled query's score against every subject; `query_lens[i]` is that
    query's length. Scores from one row share a query, so the bootstrap resamples **rows**, which is
    what keeps the interval honest (the working record is the precedent for why this matters).
    """
    if len(score_rows) != len(query_lens):
        raise ValueError("one query length per score row")
    if not score_rows:
        raise ValueError("no score rows")
    rows = [np.asarray(r, dtype=np.float64).ravel() for r in score_rows]
    subject_residues = int(sum(subject_lens))
    mean_m = float(sum(query_lens)) / len(query_lens)
    eff_m = np.asarray(query_lens, dtype=np.float64)
    eff_n = float(subject_residues)
    if effective_length:
        # Two passes: an uncorrected fit supplies the (lambda, K) the correction needs, then the
        # corrected lengths are used for the real fit. Cheap: the regression is milliseconds.
        try:
            l0, k0, _ = _tail_regression(np.concatenate(rows), len(rows), mean_m, subject_residues,
                                         tail_lo, tail_hi, n_tail_points)
            eff_m, eff_n = effective_lengths(query_lens, subject_lens, l0, k0)
            mean_m = float(eff_m.mean())
            subject_residues = max(int(eff_n), 1)
        except ValueError:
            effective_length = False
    flat = np.concatenate(rows)
    # min_r2 is NOT passed here: the pooled fit is computed unconditionally and judged at the end, after
    # the per-bin diagnostics that interpret a failure have been computed.
    lam, k_param, r2 = _tail_regression(flat, len(rows), mean_m, subject_residues,
                                        tail_lo, tail_hi, n_tail_points)

    # lambda per query-length BIN: the goodness check that matters, since lambda must not depend on m.
    # Binned by quantile rather than by exact length, because one bin per distinct ORF length leaves a
    # handful of shuffles in each and the check then measures sampling noise instead of the scorer.
    order = sorted(range(len(rows)), key=lambda i: query_lens[i])
    nb = max(min(n_length_bins, len(order)), 1)
    lam_by_len: list[tuple[int, float]] = []
    r2_by_len: list[tuple[int, float]] = []
    for b in range(nb):
        idx = order[b * len(order) // nb : (b + 1) * len(order) // nb]
        if not idx:
            continue
        ms = [float(eff_m[i]) for i in idx]
        try:
            lm, _, br2 = _tail_regression(np.concatenate([rows[i] for i in idx]), len(idx),
                                          sum(ms) / len(ms), subject_residues,
                                          tail_lo, tail_hi, n_tail_points)
        except ValueError:
            continue
        label = int(round(sum(float(query_lens[i]) for i in idx) / len(idx)))
        lam_by_len.append((label, round(lm, 6)))
        r2_by_len.append((label, round(br2, 5)))

    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(max(n_boot, 0)):
        pick = rng.integers(0, len(rows), len(rows))
        try:
            lm, _, _ = _tail_regression(np.concatenate([rows[int(i)] for i in pick]), len(rows),
                                        mean_m, subject_residues, tail_lo, tail_hi, n_tail_points)
        except ValueError:
            continue
        boots.append(lm)
    ci = ((float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if len(boots) >= 20
          else (float("nan"), float("nan")))

    # Judged last, so every diagnostic exists whatever the verdict. A low POOLED R^2 beside high
    # per-bin R^2 means lambda varies with length and the pooling is wrong; a low R^2 in every bin
    # means the scoring system has no Gumbel regime, which `background_pair_stats` would have said in
    # under a second.
    # **The bins decide, not the pool.** A Gumbel regime is a property of the scoring system, and the
    # evidence for it is that each query group's own tail is straight. A low POOLED R^2 beside high
    # per-bin R^2 means only that one lambda does not serve every group, a mixture of exponentials is
    # curved on a log axis by construction, which is a statement about precision, not about whether
    # an E-value exists. So the pool's R^2 becomes `lam_spread_across_lengths`, reported as the floor's
    # uncertainty and carried into a sensitivity check, and rejection is reserved for the case the
    # guard was written for: no bin fits, which is what a positive expected pair score looks like.
    reason = ""
    bins_ok = [v for _, v in r2_by_len if v >= min_r2]
    if r2_by_len and len(bins_ok) < max(len(r2_by_len) - 1, 1):
        reason = (f"the tail is not exponential at any query length: per-bin R^2 {r2_by_len}, only "
                  f"{len(bins_ok)} of {len(r2_by_len)} bins at or above {min_r2} (pooled "
                  f"{r2:.4f}). There is no Gumbel regime for this scoring system, so no E-value to "
                  f"invert and no floor to derive. Check `background_pair_stats`: a scorer whose "
                  f"expected pair score is positive aligns end to end and has no local optimum to be "
                  f"extreme-valued about.")
    elif not r2_by_len and r2 < min_r2:
        reason = (f"pooled tail R^2 = {r2:.4f} against a required {min_r2} and no per-bin fit "
                  f"succeeded, so nothing distinguishes a missing Gumbel regime from a mixture.")

    return EvdFit(lam=lam, k_param=k_param, n_queries=len(rows), n_pairs=int(flat.size),
                  subject_residues=subject_residues,
                  query_lens=tuple(sorted({int(m) for m in query_lens})),
                  tail_lo=tail_lo, tail_hi=tail_hi, n_tail_points=n_tail_points,
                  r_squared=round(r2, 5), lam_ci=ci, lam_by_query_len=tuple(lam_by_len),
                  n_length_bins=nb, scorer_name=scorer_name, rejected_reason=reason,
                  r2_by_query_len=tuple(r2_by_len), effective_length=effective_length,
                  mean_effective_query_len=round(float(np.mean(eff_m)), 2),
                  ungapped_lambda_bound=float(ungapped_lambda_bound))


def score_shuffles(scorer, query_seqs: list[str], *, n_shuffles: int = 8, seed: int = 20261018,
                   progress=None) -> tuple[list[np.ndarray], list[int]]:
    """The expensive half of calibration, separated so its output can be cached.

    Returns `(score_rows, query_lens)`. **Callers should persist this before fitting.** The fit is
    milliseconds and the scoring is hours; coupling them meant one rejected fit threw away the hours.
    """
    rng = random.Random(seed)
    rows, lens = [], []
    total = len(query_seqs) * n_shuffles
    for qi, q in enumerate(query_seqs):
        for si in range(n_shuffles):
            rows.append(np.asarray(scorer.score_all(shuffle_preserving_composition(q, rng))))
            lens.append(len(q))
            if progress is not None:
                progress(qi * n_shuffles + si + 1, total)
    return rows, lens


def calibrate(scorer, subject_seqs: list[str], query_seqs: list[str], *, n_shuffles: int = 8,
              seed: int = 20261018, progress=None, **fit_kwargs) -> EvdFit:
    """Shuffle each query `n_shuffles` times, score every shuffle against `subject_seqs`, and fit.

    `scorer` must be built over exactly `subject_seqs` and, if it is two-stage, with `top_k=0`: see
    the module docstring. Cost is `len(query_seqs) * n_shuffles` full catalogue passes, which is why
    calibration runs on a sampled catalogue and a cheap embedder first.
    """
    rows, lens = score_shuffles(scorer, query_seqs, n_shuffles=n_shuffles, seed=seed,
                                progress=progress)
    return fit_evd(rows, lens, [len(s) for s in subject_seqs], seed=seed,
                   scorer_name=getattr(scorer, "name", ""), **fit_kwargs)
