"""Smith-Waterman scoring, so the embedding result has an alignment control to be compared against.

The proposal's premise is that alignment-based screening misses things an embedding can catch. That premise
is untested unless alignment is actually run on the same queries, against the same reference set, scored the
same way. This module provides that arm.

Ranking is by **raw Smith-Waterman score** under BLOSUM62 with BLASTP's default affine gaps (open 11,
extend 1). For a fixed scoring system and a fixed database, BLASTP's bit score is a monotone function of the
raw score, and its E-value is monotone in the bit score, so ranking by raw score reproduces the order BLASTP
would give. What it does *not* reproduce is BLASTP's heuristic seeding: this is a full, exact
Smith-Waterman over every reference, so it is strictly more sensitive than BLASTP, not less. That matters
for how the comparison is read: a protein this arm misses is one BLASTP would also miss.

`parasail` is an optional dependency (`pip install -e ".[align]"`). It is imported lazily so the rest of the
package keeps working without it.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Protocol

import numpy as np

GAP_OPEN = 11
GAP_EXTEND = 1


class Scorer(Protocol):
    """Scores one query against every reference, higher is better. Both arms implement this."""

    name: str

    def score_all(self, query: str) -> np.ndarray: ...


MODEL_FREE = ("align", "sw", "alignment")


def model_kwargs(spec: str, **options) -> dict:
    """Model-only options for `get_scorer`, dropped for the arms that have no model.

    `--batch-size` and `--max-length` mean something to ESM-2 and nothing to a k-mer table or to
    Smith-Waterman, and passing them anyway is a `TypeError`. Both `06_decompose.py` and
    `10_twilight.py` need the same rule, so it lives here rather than being written out twice and
    drifting. Options left at `None` are dropped, so a script can offer a flag without forcing a value.

    `eba:...` counts as a model arm: the factory routes these to the positional embedder, which keeps
    them for an ESM-2 checkpoint and ignores them for the BLOSUM62 and k-mer-context backends.
    """
    if spec in MODEL_FREE or spec.startswith("kmer"):
        return {}
    return {k: v for k, v in options.items() if v is not None}


def release_cached_gpu_memory() -> None:
    """Return torch's cached-but-unused GPU blocks to the driver.

    Embedding the catalogue is a run's **peak** GPU allocation, and torch's caching allocator keeps those
    blocks reserved for the rest of the process even though the single-query passes that follow need a
    fraction of them. That is free on a card with one tenant and decisive on a card with two: two
    concurrent ESM-2 650M runs on the Omen's 8 GB fit only if the one past its build is not still holding
    its build peak.

    Deliberately a no-op when torch was never imported: the k-mer and BLOSUM paths never load it, and
    importing it here to free memory it is not using would cost seconds on every CPU run.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 (a torch without CUDA, or a driver that refuses; never fatal)
        return


def embedder_config(embedder) -> dict:
    """The embedder's configuration, read from its own `__init__` signature.

    Taken from the signature rather than from a hand-written list so that a configuration field added to
    any embedder later is picked up without anyone remembering to update the cache key. That is the
    failure mode worth engineering against here: forgetting costs a silently wrong matrix, not a crash.

    Only `__init__` parameters are read, and only where the instance exposes a same-named attribute
    holding a simple value. That deliberately skips mutable runtime state such as `Esm2Embedder.dim`,
    which is 0 before the model loads and the hidden size afterwards; keying on it would make the cache
    miss depending on whether the embedder had been used yet.
    """
    import inspect  # noqa: PLC0415 (only needed when a cache key is built)

    try:
        params = inspect.signature(type(embedder).__init__).parameters
    except (TypeError, ValueError):  # a C-level or otherwise uninspectable __init__
        return {}
    out = {}
    for name in params:
        if name in ("self", "args", "kwargs", "kw"):
            continue
        value = getattr(embedder, name, None)
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[name] = value
    return out


class EmbeddingScorer:
    """Cosine similarity against a reference catalogue embedded once and cached on disk.

    Embedding the catalogue dominates the cost of every embedding run: 32,387 proteins through ESM-2 650M
    is tens of minutes, while the few hundred queries that follow are seconds. Each script invocation used
    to redo it, so running the three tiers of the twilight benchmark paid that cost three times for an
    identical matrix, and a run that died in the last tier threw away everything.

    The cache key is the embedder's name, **its configuration**, and a digest of the reference sequences
    themselves, not the file path. A reference set that is rebuilt (as the twilight set was, when
    DIAMOND's default sensitivity turned out to be the wrong setting) therefore gets a new key
    automatically, instead of silently scoring new sequences against an old matrix, which would be a
    very quiet way to be wrong.

    Configuration has to be in the key because the name does not carry it. `Esm2Embedder.name` is just
    the checkpoint, so two runs differing only in `max_length` share a name while producing different
    vectors for every reference longer than the shorter limit: 155 of the 5,016 in `reference.fasta`
    and 1,360 of the 32,387 in `reference_twilight.fasta`. A run at `max_length=2048` would have loaded
    the matrix computed at 1022 and reported it as its own.
    """

    def __init__(self, embedder, ref_seqs: list[str], cache_dir: Path | str | None = None,
                 verbose: bool = True) -> None:
        self.embedder = embedder
        self.name = embedder.name
        self.matrix = self._load_or_embed(ref_seqs, cache_dir, verbose)

    def _cache_path(self, ref_seqs: list[str], cache_dir: Path) -> Path:
        h = hashlib.blake2b(digest_size=12)
        h.update(self.name.encode())
        h.update(repr(embedder_config(self.embedder)).encode())
        h.update(str(len(ref_seqs)).encode())
        for s in ref_seqs:
            h.update(s.encode())
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in self.name)
        return cache_dir / f"refmatrix_{safe}_{h.hexdigest()}.npy"

    def _load_or_embed(self, ref_seqs: list[str], cache_dir: Path | str | None, verbose: bool) -> np.ndarray:
        if cache_dir is None:
            m = self.embedder.embed(ref_seqs)
            release_cached_gpu_memory()
            return m
        cache_dir = Path(cache_dir)
        path = self._cache_path(ref_seqs, cache_dir)
        if path.is_file():
            m = np.load(path)
            if m.shape[0] == len(ref_seqs):
                if verbose:
                    print(f"  reference matrix loaded from cache: {path.name}  {m.shape}", flush=True)
                return m
            if verbose:  # a stale file that somehow collided, recompute rather than trust it
                print(f"  cache {path.name} has {m.shape[0]} rows, expected {len(ref_seqs)}; recomputing",
                      flush=True)
        m = self.embedder.embed(ref_seqs)
        release_cached_gpu_memory()
        tmp = None
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            # the temp name must itself end in .npy: np.save appends the suffix when it is absent, so a
            # ".npy.tmp" target becomes ".npy.tmp.npy" and the rename below finds nothing to move.
            # It also carries the pid, because the cache path is a pure function of the embedder and the
            # reference set: two runs building the SAME matrix at the same time, two command prompts on
            # one GPU, would otherwise share one temp file and interleave their writes into it, and the
            # row-count check on load cannot see a torn file whose row count is right.
            tmp = path.with_name(f"{path.stem}.tmp{os.getpid()}.npy")
            np.save(tmp, m)
            tmp.replace(path)  # atomic, so an interrupted run cannot leave a half-written cache
            if verbose:
                print(f"  reference matrix cached: {path.name} ({m.nbytes / 1e6:.0f} MB)", flush=True)
        except OSError as exc:  # caching is an optimisation, never a requirement
            if tmp is not None:
                # never leave a partial write behind to be tidied by hand, and never let the tidying
                # itself be what takes the run down
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            if verbose:
                print(f"  could not write cache ({exc}); continuing", flush=True)
        return m

    def score_all(self, query: str) -> np.ndarray:
        return (self.embedder.embed([query]) @ self.matrix.T)[0]


class AlignmentScorer:
    """Exact Smith-Waterman against every reference. See the module docstring for why raw score is used.

    `threads` splits the pass across OS threads, the same device `homology.find_homologs` already uses for
    its score pass: parasail's kernels release the GIL, so this is real parallelism, and the threads share
    the one copy of the catalogue rather than needing 20.8 M residues each.

    **The result is bitwise identical to the single-threaded path, by construction rather than by
    tolerance.** Each output element is one independent `sw_striped_sat` call on one (query, reference)
    pair; the slices are contiguous and disjoint, and every element is written by exactly one thread. No
    sum, no dot product, no reduction of any kind crosses a thread boundary, so there is nothing for
    parallelism to reassociate and no floating-point arithmetic is reordered. The scores are integers from
    parasail cast to float32 in both paths. `tests/test_align_threads.py` asserts `np.array_equal` against
    the real catalogue, not `allclose`: if exact equality ever fails, the scheduling is not all that
    changed and the arrays must not be trusted.
    """

    def __init__(self, ref_seqs: list[str], gap_open: int = GAP_OPEN, gap_extend: int = GAP_EXTEND,
                 threads: int = 1) -> None:
        import parasail  # noqa: PLC0415 (optional dependency, imported on use)

        self._p = parasail
        self._matrix = parasail.blosum62
        self.ref_seqs = ref_seqs
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.threads = max(1, int(threads))
        # The name deliberately does NOT carry the thread count. It is the cache key for the embedding
        # arms and it labels every output file and every results column, so making it depend on a purely
        # operational setting would fork the artefacts of two runs that are numerically identical, and
        # the test above is what entitles us to say they are.
        self.name = f"sw-blosum62-{gap_open}-{gap_extend}"

    def score_all(self, query: str) -> np.ndarray:
        n = len(self.ref_seqs)
        if not query:
            return np.zeros(n, dtype=np.float32)
        out = np.empty(n, dtype=np.float32)
        # `_sat` picks the widest integer width that avoids saturation; the 16-bit kernels silently clip on
        # long, highly similar pairs, which here would mean a 1,368 aa Cas9 scoring the same as a short tag.
        f = self._p.sw_striped_sat
        refs, go, ge, m = self.ref_seqs, self.gap_open, self.gap_extend, self._matrix

        def fill(lo: int, hi: int) -> None:
            for i in range(lo, hi):
                out[i] = f(query, refs[i], go, ge, m).score

        if self.threads <= 1 or n < 2 * self.threads:
            fill(0, n)
            return out
        import threading  # noqa: PLC0415 (only needed on the parallel path)

        bounds = [(n * k // self.threads, n * (k + 1) // self.threads) for k in range(self.threads)]
        workers = [threading.Thread(target=fill, args=b, daemon=True) for b in bounds]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        return out


def rank_one(query: str, scorer: Scorer, ref_accessions: list[str], target: int | None):
    """(rank, score-to-target, top-1 accession, top-1 score). Ranks are 1-based; ties take the worst place.

    Shared by both arms so the two are never ranked by subtly different rules.
    """
    if not query:
        return None, None, "", 0.0
    row = scorer.score_all(query)
    top = int(np.argmax(row))
    if target is None:
        return None, None, ref_accessions[top], float(row[top])
    return int((row >= row[target]).sum()), float(row[target]), ref_accessions[top], float(row[top])


def best_target_rank(row: np.ndarray, target_idx: np.ndarray, mask_idx: np.ndarray) -> int | None:
    """Rank of the best-scoring true homolog once the held-out relatives are removed from the catalogue.

    Masked entries are set to -inf rather than deleted so that index bookkeeping stays aligned with the
    reference list; ranks then count only what a screen would actually have had available.

    Lives here, beside `rank_one`, because it is the rule by which **every** benchmark in this repository
    turns a score row into a rank: `scripts/10_twilight.py` and `scripts/51_fragment_probe.py` both call
    this one function. A probe that ranked by its own subtly different rule would produce numbers that
    look comparable to the twilight CSVs and are not, which is the class of defect this repo has been
    bitten by more than once and never sees in an output.
    """
    if target_idx.size == 0:
        return None
    r = row.astype(np.float64, copy=True)
    r[mask_idx] = -np.inf
    best = float(np.max(r[target_idx]))
    if not np.isfinite(best):
        return None
    return int((r >= best).sum())


def get_scorer(spec: str, ref_seqs: list[str], cache_dir: Path | str | None = None, **kwargs) -> Scorer:
    """The three arms, behind one call, so a script written against one of them runs the others unchanged.

    * `'align'` / `'sw'` -> exact Smith-Waterman (`AlignmentScorer`), optionally over `threads`
    * `'eba'` / `'eba:<embedder>'` -> the positional scorer (`pilot.eba.EbaScorer`)
    * anything else -> a mean-pooled embedder (`pilot.embed.get_embedder`)

    `cache_dir` is ignored by the alignment arm, which has no up-front cost to amortise. The positional
    arm passes it on to its prefilter, which is a mean-pooled `EmbeddingScorer` over the same reference
    set and so hits exactly the same cache entry the mean-pooled arm writes.
    """
    # `threads` is operational, not a model option, so it does not travel through `model_kwargs`
    # (which drops everything for the model-free arms). It is read here and ignored by the others:
    # which means it has to be REMOVED here rather than merely unused, or it reaches `get_embedder`
    # and `EbaScorer` as an unexpected keyword and the call raises. A caller that threads its
    # alignment arm would then be unable to run the same line with an ESM-2 checkpoint, which is
    # exactly the portability this factory exists to provide.
    threads = int(kwargs.pop("threads", None) or 1)
    if spec in ("align", "sw", "alignment"):
        return AlignmentScorer(ref_seqs, threads=threads)
    if spec == "eba" or spec.startswith("eba:"):
        from .eba import get_eba_scorer  # noqa: PLC0415 (keeps the import graph acyclic)

        return get_eba_scorer(spec, ref_seqs, cache_dir=cache_dir, **kwargs)
    from .embed import get_embedder  # noqa: PLC0415 (avoids importing torch for the alignment arm)

    return EmbeddingScorer(get_embedder(spec, **kwargs), ref_seqs, cache_dir=cache_dir)
