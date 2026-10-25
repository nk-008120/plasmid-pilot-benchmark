"""EBA-style positional scoring: per-residue embeddings, a cosine substitution matrix, Smith-Waterman.

This is the arm the working record identified as the one the literature says should win, and the reason the
rest of the retrieval code is in the wrong regime. The published ordering, in the twilight zone, is

    embeddings + alignment (0.92 Spearman) > profile alignment (0.82) > plain embedding cosine (0.56)

and `pilot.embed` + `pilot.retrieval` implement the 0.56 variant: one mean-pooled vector per chain, one
cosine. Johnson et al. (eLife 2023;12:RP91415) state the same problem independently, that whole-protein
embeddings cannot discriminate domain-level homology, and reach for positional embeddings in a local
search as the remedy.

**What this module does.** For a query of m residues and a reference of n, the per-residue embeddings give
a cosine matrix `C = Q @ R.T` of shape (m, n), because `pilot.positional` returns L2-normalised rows. That
matrix is used as a **position-specific substitution matrix** and aligned with the same affine-gap
Smith-Waterman recurrence the `align` arm uses. The substitution score is

    s(i, j) = (cos(i, j) - shift) * scale

`shift` is the one tuned number, and it is the whole ballgame: it sets where "unrelated" starts. Below it a
residue pair is a mismatch and the local alignment is willing to end rather than extend through. With
shift = 0 every pair with a non-negative cosine looks like a match and the alignment runs away to the full
length of both sequences, which is global alignment wearing a local alignment's clothes. this project's working rules says to
sweep a tuned threshold and show the plateau instead of reporting the best value; `scripts/18_eba_sweep.py`
does that against the 23 hand-audited constructs.

**Why `parasail` cannot be used here.** parasail's kernels take a 20x20 integer matrix over the amino-acid
alphabet. A position-specific matrix is a different shape of problem, so the dynamic programming is
implemented here in numpy. `sw_affine_batch` is checked against `parasail` on integer BLOSUM62 matrices in
`tests/test_eba.py`, which is what makes it trustworthy: with the same substitution matrix and the same
gaps it must return exactly the same score as the control arm.

**Cost, stated plainly.** Exhaustive EBA against the 32,387-protein twilight reference is not tractable on
a CPU: the substitution matrices alone are O(m * n * dim) per reference. So `EbaScorer` is a **two-stage**
scorer, which is what every deployed embedding search does: a cheap global prefilter shortlists `top_k`
references and the dynamic programming reranks only those. That makes the prefilter part of the method and
a possible source of failure in its own right, so `shortlist()` is public and `18_eba_sweep.py` reports
prefilter recall separately from rerank accuracy. Conflating the two would hide both.
"""
from __future__ import annotations

import numpy as np

from .align import Scorer
from .positional import PositionalEmbedder, get_positional_embedder

NEG = np.float32(-1e30)

# **0.25 is DEGENERATE for the BLOSUM62-row embedder and must not be used with it.** Measured 18 Oct
# on the 40,750-protein catalogue: the background cosine between residues of different proteins is
# 0.2697 (clustered SE 0.00125, 99,760 pairs), so `cosine - 0.25` has expected value **+0.0197**. A
# local alignment has a local optimum only because extending through unrelated residues costs score on
# average; above zero the optimal alignment grows with the sequences, the score measures length rather
# than similarity, and there is no E-value. The constant is left as it is so that
# sections 19-21's provenance stays readable, changing it would silently redate every historical
# number, but every new run passes `shift=` explicitly, from
# `pilot.cosine_floor.background_pair_stats`, which answers it per embedder in under a second.
SHIFT = 0.25
SCALE = 1.0
GAP_OPEN = 0.5
GAP_EXTEND = 0.1
TOP_K = 200
BUDGET_FLOATS = 20_000_000  # ~80 MB of float32 per chunk of substitution matrices


def sw_affine_batch(sub: np.ndarray, gap_open: float, gap_extend: float) -> np.ndarray:
    """Local (Smith-Waterman) alignment score for each of `r` position-specific substitution matrices.

    `sub` is (r, m, n). Returns (r,) best local scores. Affine gaps follow the same convention as
    `parasail` and EMBOSS: a gap of length k costs `gap_open + (k - 1) * gap_extend`.

    The recurrence is Gotoh's, evaluated **anti-diagonal by anti-diagonal**, because H[i][j] depends on
    E[i][j-1] within its own row and that dependency is what stops a row-wise formulation from
    vectorising. On diagonal d = i + j every cell is independent, so one numpy expression does the whole
    diagonal for every reference in the batch at once.

    H, E and F live in preallocated ring buffers, three deep for H and two for E and F, which is as far
    back as the recurrence reaches. Only the band and the two cells just outside it are written each step;
    those two are the only out-of-band entries the next diagonal can read, so nothing else needs clearing.
    That removes the per-diagonal allocation and roughly doubles the cell rate, to about 25 M cells/s on
    one core. Skewing the substitution matrices so a diagonal becomes a contiguous slice was tried and is
    slower: the strided slice it produces has worse locality than the gather it replaces.
    """
    sub = np.ascontiguousarray(sub, dtype=np.float32)
    r, m, n = sub.shape
    if r == 0 or m == 0 or n == 0:
        return np.zeros(r, dtype=np.float32)
    go, ge = np.float32(gap_open), np.float32(gap_extend)
    width = m + 2  # index 0 is the i = 0 boundary, index m + 1 keeps the i + 1 shifts in range
    h = np.zeros((3, r, width), dtype=np.float32)
    e = np.full((2, r, width), NEG, dtype=np.float32)
    f = np.full((2, r, width), NEG, dtype=np.float32)
    best = np.zeros(r, dtype=np.float32)

    for d in range(2, m + n + 1):
        lo, hi = max(1, d - n), min(m, d - 1)
        if lo > hi:
            continue
        cur, prev1, prev2 = d % 3, (d - 1) % 3, (d - 2) % 3
        ce, pe = d % 2, (d - 1) % 2
        band = slice(lo, hi + 1)
        up = slice(lo - 1, hi)  # the same cells shifted one row back, i.e. index i - 1
        rows = np.arange(lo - 1, hi)
        ee = np.maximum(h[prev1][:, band] - go, e[pe][:, band] - ge)
        ff = np.maximum(h[prev1][:, up] - go, f[pe][:, up] - ge)
        hh = np.maximum(h[prev2][:, up] + sub[:, rows, d - 2 - rows], np.maximum(ee, ff))
        np.maximum(hh, 0.0, out=hh)
        h[cur][:, band], e[ce][:, band], f[ce][:, band] = hh, ee, ff
        edges = [lo - 1] + ([hi + 1] if hi + 1 < width else [])
        h[cur][:, edges] = 0.0
        e[ce][:, edges] = NEG
        f[ce][:, edges] = NEG
        np.maximum(best, hh.max(axis=1), out=best)
    return best


def sw_affine(sub: np.ndarray, gap_open: float = GAP_OPEN, gap_extend: float = GAP_EXTEND) -> float:
    """`sw_affine_batch` for a single (m, n) substitution matrix."""
    return float(sw_affine_batch(np.asarray(sub, dtype=np.float32)[None], gap_open, gap_extend)[0])


def blosum_matrix(query: str, ref: str) -> np.ndarray:
    """The (m, n) BLOSUM62 substitution matrix for one pair, for checking the DP against `parasail`."""
    from Bio.Align import substitution_matrices  # noqa: PLC0415

    m = substitution_matrices.load("BLOSUM62")
    alphabet = {a: i for i, a in enumerate(m.alphabet)}
    arr = np.asarray(m, dtype=np.float32)
    qi = np.array([alphabet[c] for c in query.upper()], dtype=int)
    ri = np.array([alphabet[c] for c in ref.upper()], dtype=int)
    return arr[np.ix_(qi, ri)]


class EbaScorer:
    """Per-residue embeddings plus Smith-Waterman, behind the `pilot.align.Scorer` interface.

    Drops into `scripts/06_decompose.py` and `scripts/10_twilight.py` unchanged, because `score_all` has
    the same contract as `AlignmentScorer.score_all`: one score per reference, higher is better.

    `prefilter` is any other `Scorer` (in practice a mean-pooled `EmbeddingScorer`, which is cheap). The
    top `top_k` references by prefilter score get the dynamic programming. The rest keep their prefilter
    *order* but are mapped into a band strictly below the lowest reranked score, so the returned row is a
    well-defined ranking end to end and matches what a two-stage pipeline would actually hand back: the
    reranked shortlist first, everything else behind it. A reference the prefilter never shortlisted
    cannot reach rank 1, which is a real property of the method rather than an artifact of the harness.
    """

    def __init__(self, embedder: PositionalEmbedder, ref_seqs: list[str], prefilter: Scorer | None = None,
                 shift: float = SHIFT, scale: float = SCALE, gap_open: float = GAP_OPEN,
                 gap_extend: float = GAP_EXTEND, top_k: int = TOP_K,
                 budget_floats: int = BUDGET_FLOATS, precompute: bool = False) -> None:
        self.embedder = embedder
        self.ref_seqs = ref_seqs
        self.prefilter = prefilter
        self.shift, self.scale = float(shift), float(scale)
        self.gap_open, self.gap_extend = float(gap_open), float(gap_extend)
        self.top_k = min(top_k, len(ref_seqs)) if top_k else len(ref_seqs)
        self.budget_floats = budget_floats
        self.name = (f"eba-{embedder.name}-s{self.shift:g}-g{self.gap_open:g}-{self.gap_extend:g}"
                     f"-k{self.top_k}")
        # Per-residue vectors for the whole catalogue do not fit: the 32,387-protein twilight reference is
        # about 14 M residues, which is 1.1 GB even at BLOSUM62's 20 dimensions and 70 GB at ESM-2 650M's
        # 1,280. With a prefilter only `top_k` of them are ever aligned, so they are embedded per chunk and
        # discarded. `precompute=True` is for a model expensive enough that embedding twice costs more than
        # storing the result, which is the GPU case.
        self._ref_emb = embedder.embed_residues(ref_seqs) if precompute else None
        self._ref_len = np.array([len(s) for s in ref_seqs], dtype=int)
        # One-entry memo so `prefilter_scores` and `shortlist` share a single prefilter pass per query.
        # Without it, instrumenting prefilter recall would double the prefilter cost of every query:
        # cheap against the dynamic programming, but not free, and the point of the instrumentation is
        # that it should cost nothing anyone would decline to pay.
        self._pf_memo: tuple[str, np.ndarray] | None = None

    def prefilter_scores(self, query: str) -> np.ndarray | None:
        """The prefilter's score for every reference, or `None` when there is no prefilter.

        Public because `top_k` is a **limit that binds**, and this project's working rules asks for the binding count rather
        than the setting. From this row a caller derives the prefilter's rank of the true answer, and from
        that rank the shortlist recall at *every* `top_k` at once, so the curve that a derived `top_k`
        needs costs one prefilter pass, not one run per candidate value. `shortlist` answers the same
        question at one `k` and loses the curve.
        """
        if self.prefilter is None:
            return None
        if self._pf_memo is not None and self._pf_memo[0] == query:
            return self._pf_memo[1]
        pf = np.asarray(self.prefilter.score_all(query))
        self._pf_memo = (query, pf)
        return pf

    def shortlist(self, query: str) -> np.ndarray:
        """Indices the dynamic programming will score, best prefilter score first.

        Exposed so a caller can report how often the true answer was dropped before the reranker ever saw
        it, which is a different failure from the reranker getting it wrong.
        """
        n = len(self.ref_seqs)
        if self.prefilter is None or self.top_k >= n:
            return np.arange(n)
        pf = self.prefilter_scores(query)
        return np.argsort(-pf, kind="stable")[: self.top_k]

    def _embed_refs(self, idx) -> list[np.ndarray]:
        if self._ref_emb is not None:
            return [self._ref_emb[int(j)] for j in idx]
        return self.embedder.embed_residues([self.ref_seqs[int(j)] for j in idx])

    def _dp_scores(self, q_emb: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """EBA score for each reference in `idx`, chunked so the substitution matrices fit in memory."""
        m = q_emb.shape[0]
        out = np.zeros(idx.size, dtype=np.float32)
        order = idx[np.argsort(self._ref_len[idx], kind="stable")]  # similar lengths chunk together
        pos = {int(j): p for p, j in enumerate(idx)}
        start = 0
        while start < order.size:
            size = 1
            # grow the chunk while the padded (size, m, n_max) block stays inside the float budget
            while start + size < order.size:
                nn = int(self._ref_len[order[start + size]])
                if m * max(nn, 1) * (size + 1) > self.budget_floats:
                    break
                size += 1
            chunk = order[start : start + size]
            start += size
            embs = self._embed_refs(chunk)
            n_max = max((e.shape[0] for e in embs), default=0)
            if n_max == 0 or m == 0:
                continue
            sub = np.full((len(embs), m, n_max), NEG, dtype=np.float32)
            for row, r in enumerate(embs):
                if r.shape[0]:
                    sub[row, :, : r.shape[0]] = (q_emb @ r.T - self.shift) * self.scale
            scores = sw_affine_batch(sub, self.gap_open, self.gap_extend)
            for row, j in enumerate(chunk):
                out[pos[int(j)]] = scores[row]
        return out

    def score_all(self, query: str) -> np.ndarray:
        n = len(self.ref_seqs)
        row = np.full(n, NEG, dtype=np.float32)
        if not query:
            return row
        q_emb = self.embedder.embed_residues([query])[0]
        idx = self.shortlist(query)
        if idx.size == 0 or q_emb.shape[0] == 0:
            return row
        row[idx] = self._dp_scores(q_emb, idx)
        rest = np.setdiff1d(np.arange(n), idx, assume_unique=False)
        if rest.size and self.prefilter is not None:
            pf = self.prefilter.score_all(query)[rest]
            top = float(row[idx].min()) - 1e-6
            lo, hi = float(pf.min()), float(pf.max())
            row[rest] = (top - 1.0 + (pf - lo) / (hi - lo) if hi > lo else np.full(rest.size, top - 1.0))
        return row


_SPEC_TYPES = {"shift": float, "scale": float, "gap_open": float, "gap_extend": float,
               "top_k": int, "budget_floats": int, "prefilter": str, "precompute": lambda v: v == "True"}


def parse_spec(spec: str) -> tuple[str, dict]:
    """`'eba:<embedder>[:k=v,k=v]'` -> (embedder spec, options).

    The analysis scripts choose a scorer by one string and have no flags of their own for the positional
    arm's knobs, so the knobs ride in the string: `--scorer eba:blosum:shift=0.6,top_k=2000`. That keeps
    `06_decompose.py` and `10_twilight.py` untouched while still letting a run pin the one tuned number,
    and it means the setting is recorded in the scorer's `name` and so in the output filename.

    Options passed as real keyword arguments win, since they are the more specific request.
    """
    parts = spec.split(":")
    emb_spec = parts[1] if len(parts) > 1 and parts[1] else "blosum"
    opts: dict = {}
    if len(parts) > 2:
        for item in ":".join(parts[2:]).split(","):
            if "=" not in item:
                continue
            key, _, raw = item.partition("=")
            key, raw = key.strip(), raw.strip()
            cast = _SPEC_TYPES.get(key)
            if cast is None:
                raise ValueError(f"unknown eba option {key!r} in {spec!r}; "
                                 f"known: {', '.join(sorted(_SPEC_TYPES))}")
            opts[key] = cast(raw)
    return emb_spec, opts


def get_eba_scorer(spec: str, ref_seqs: list[str], **kwargs) -> EbaScorer:
    """`'eba'` or `'eba:<embedder>'` -> an `EbaScorer`; extra keyword arguments go to the scorer.

    The prefilter defaults to the mean-pooled k-mer embedder, which is deliberately the weakest sensible
    choice: a prefilter that is itself a foundation model would make the two-stage result hard to
    attribute. `prefilter='none'` runs exhaustive dynamic programming, which is only sane on a small
    reference set.
    """
    emb_spec, spec_opts = parse_spec(spec)
    for key, value in spec_opts.items():
        kwargs.setdefault(key, value)
    prefilter = kwargs.pop("prefilter", "kmer")
    top_k = kwargs.pop("top_k", TOP_K)
    cache_dir = kwargs.pop("cache_dir", None)
    emb_kwargs = {k: kwargs.pop(k) for k in ("batch_size", "device", "max_length") if k in kwargs}
    kwargs.setdefault("precompute", emb_spec not in ("blosum", "blosum62") and not emb_spec.startswith("kmerpos"))
    pf: Scorer | None = None
    if prefilter not in (None, "none", ""):
        from .align import EmbeddingScorer  # noqa: PLC0415 (avoids a circular import at module level)
        from .embed import get_embedder  # noqa: PLC0415

        pf = EmbeddingScorer(get_embedder(prefilter), ref_seqs, cache_dir=cache_dir)
    return EbaScorer(get_positional_embedder(emb_spec, **emb_kwargs), ref_seqs,
                     prefilter=pf, top_k=top_k, **kwargs)
