"""Tests for the Smith-Waterman control arm."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pilot.align import AlignmentScorer, EmbeddingScorer, get_scorer, rank_one  # noqa: F401
from pilot.embed import get_embedder

parasail = pytest.importorskip("parasail")

A = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDE"
B = "MSDNEKLIAFIEQVNQLSPAERAELVQQLTAHWDGQRWQAFADGLQNHLDGQAYTLDMLKQAQRHGVDVSTLSPEQRAAFEALLHRHQGDEWSAIAQRLAG"
C = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
REFS, ACC = [A, B, C], ["refA", "refB", "refC"]


class TestAlignmentScorer:
    def test_a_sequence_scores_highest_against_itself(self):
        s = AlignmentScorer(REFS)
        for i, q in enumerate(REFS):
            assert int(np.argmax(s.score_all(q))) == i

    def test_rank_one_finds_the_exact_match_first(self):
        s = AlignmentScorer(REFS)
        rank, score, top, _ = rank_one(B, s, ACC, 1)
        assert rank == 1 and top == "refB" and score > 0

    def test_a_fragment_still_ranks_its_parent_first(self):
        """Local alignment is the point: a decomposed segment is a fragment of something longer."""
        s = AlignmentScorer(REFS)
        assert rank_one(A[20:80], s, ACC, 0)[0] == 1

    def test_empty_query_scores_nothing_rather_than_raising(self):
        s = AlignmentScorer(REFS)
        assert rank_one("", s, ACC, 0) == (None, None, "", 0.0)
        assert s.score_all("").shape == (len(REFS),)

    def test_score_vector_has_one_entry_per_reference(self):
        assert AlignmentScorer(REFS).score_all(A).shape == (len(REFS),)

    def test_long_similar_pairs_do_not_saturate(self):
        """The 16-bit kernels clip around 32k; a 2,000 aa self-match must still beat a short one."""
        long_ref = A * 20
        s = AlignmentScorer([long_ref, C])
        row = s.score_all(long_ref)
        assert row[0] > row[1]
        assert row[0] > 3000


class TestScorerInterchangeability:
    def test_both_arms_satisfy_the_same_interface(self):
        for sc in (AlignmentScorer(REFS), EmbeddingScorer(get_embedder("kmer"), REFS)):
            row = sc.score_all(A)
            assert row.shape == (len(REFS),)
            assert isinstance(sc.name, str) and sc.name

    def test_get_scorer_dispatches_on_the_spec(self):
        assert isinstance(get_scorer("align", REFS), AlignmentScorer)
        assert isinstance(get_scorer("kmer", REFS), EmbeddingScorer)

    def test_ties_take_the_worst_rank_in_both_arms(self):
        """Two identical references must not let the target claim rank 1 by tie-break luck."""
        refs, acc = [A, A, C], ["x", "y", "z"]
        for sc in (AlignmentScorer(refs), EmbeddingScorer(get_embedder("kmer"), refs)):
            assert rank_one(A, sc, acc, 0)[0] == 2


class TestReferenceCaching:
    """The reference matrix is the expensive part of any embedding run; caching it is what makes three
    twilight tiers affordable. These tests exist because the first version silently failed to cache:
    `np.save` appends `.npy`, so the temp file never matched the rename target and every run recomputed."""

    def test_the_matrix_is_written_and_reused(self, tmp_path):
        emb = get_embedder("kmer")
        first = EmbeddingScorer(emb, REFS, cache_dir=tmp_path, verbose=False)
        files = list(tmp_path.glob("refmatrix_*.npy"))
        assert len(files) == 1, "nothing was cached"
        assert not list(tmp_path.glob("*.tmp*")), "a temp file was left behind"
        second = EmbeddingScorer(emb, REFS, cache_dir=tmp_path, verbose=False)
        assert np.array_equal(first.matrix, second.matrix)

    def test_a_changed_reference_set_gets_a_different_key(self, tmp_path):
        """A rebuilt reference set must not be scored against the old matrix: a very quiet way to be
        wrong. The key digests the sequences, not the file path."""
        emb = get_embedder("kmer")
        EmbeddingScorer(emb, REFS, cache_dir=tmp_path, verbose=False)
        EmbeddingScorer(emb, [*REFS, C * 2], cache_dir=tmp_path, verbose=False)
        assert len(list(tmp_path.glob("refmatrix_*.npy"))) == 2

    def test_cached_scores_match_uncached_ones(self, tmp_path):
        emb = get_embedder("kmer")
        plain = EmbeddingScorer(emb, REFS, cache_dir=None, verbose=False)
        EmbeddingScorer(emb, REFS, cache_dir=tmp_path, verbose=False)
        cached = EmbeddingScorer(emb, REFS, cache_dir=tmp_path, verbose=False)
        assert np.allclose(plain.score_all(A), cached.score_all(A))

    def test_caching_off_by_default_writes_nothing(self, tmp_path):
        EmbeddingScorer(get_embedder("kmer"), REFS, verbose=False)
        assert not list(tmp_path.iterdir())

    def test_an_unwritable_cache_dir_does_not_break_the_run(self, tmp_path):
        """Caching is an optimisation; a read-only disk must not take the experiment down with it."""
        blocked = tmp_path / "file_not_a_dir"
        blocked.write_text("x")
        sc = EmbeddingScorer(get_embedder("kmer"), REFS, cache_dir=blocked / "sub", verbose=False)
        assert sc.matrix.shape[0] == len(REFS)


class TestCacheKeyCoversConfiguration:
    """A cache key that ignores configuration is the quiet kind of wrong: the run loads someone else's
    matrix and reports it as its own. `Esm2Embedder.name` is only the checkpoint, so `max_length` has to
    be in the key: it changes the vectors of every reference longer than the shorter of the two limits.
    """

    def test_two_embedders_differing_only_in_max_length_get_different_paths(self, tmp_path):
        from pilot.embed import Esm2Embedder

        short = EmbeddingScorer.__new__(EmbeddingScorer)
        long_ = EmbeddingScorer.__new__(EmbeddingScorer)
        short.embedder, long_.embedder = Esm2Embedder(max_length=1022), Esm2Embedder(max_length=2048)
        short.name = long_.name = short.embedder.name
        assert short.embedder.name == long_.embedder.name, "the bug only bites when the names match"
        assert short._cache_path(REFS, tmp_path) != long_._cache_path(REFS, tmp_path)

    def test_the_same_configuration_still_hits_the_same_path(self, tmp_path):
        from pilot.embed import Esm2Embedder

        a, b = EmbeddingScorer.__new__(EmbeddingScorer), EmbeddingScorer.__new__(EmbeddingScorer)
        a.embedder, b.embedder = Esm2Embedder(max_length=2048), Esm2Embedder(max_length=2048)
        a.name = b.name = a.embedder.name
        assert a._cache_path(REFS, tmp_path) == b._cache_path(REFS, tmp_path)

    def test_kmer_width_is_in_the_key_even_though_the_name_already_carries_it(self, tmp_path):
        a, b = EmbeddingScorer.__new__(EmbeddingScorer), EmbeddingScorer.__new__(EmbeddingScorer)
        a.embedder, b.embedder = get_embedder("kmer", dim=256), get_embedder("kmer", dim=512)
        a.name, b.name = a.embedder.name, b.embedder.name
        assert a._cache_path(REFS, tmp_path) != b._cache_path(REFS, tmp_path)

    def test_config_is_read_from_the_init_signature_so_new_fields_are_picked_up(self):
        """The point of reading the signature is that nobody has to remember to update the key."""
        from pilot.align import embedder_config

        class Future:
            name = "future"

            def __init__(self, width: int = 7, mode: str = "a"):
                self.width, self.mode = width, mode

        assert embedder_config(Future()) == {"width": 7, "mode": "a"}
        assert embedder_config(Future(width=9)) == {"width": 9, "mode": "a"}

    def test_mutable_runtime_state_stays_out_of_the_key(self):
        """`Esm2Embedder.dim` is 0 before the model loads and the hidden size after; keying on it would
        make the cache hit or miss depending on whether the embedder had been used yet."""
        from pilot.align import embedder_config
        from pilot.embed import Esm2Embedder

        emb = Esm2Embedder()
        before = embedder_config(emb)
        emb.dim = 1280  # what _load() would set
        assert embedder_config(emb) == before
        assert "dim" not in before

    def test_a_changed_max_length_does_not_reuse_the_cached_matrix(self, tmp_path):
        """End to end on the k-mer stub, which is the only embedder that runs without torch."""
        first = EmbeddingScorer(get_embedder("kmer", dim=256), REFS, cache_dir=tmp_path, verbose=False)
        second = EmbeddingScorer(get_embedder("kmer", dim=512), REFS, cache_dir=tmp_path, verbose=False)
        assert len(list(tmp_path.glob("refmatrix_*.npy"))) == 2
        assert first.matrix.shape[1] != second.matrix.shape[1]


class TestModelKwargs:
    """Which arms take model options. Both analysis scripts share this rule, so it is tested once here
    rather than twice by inspection."""

    def test_the_alignment_arm_takes_none(self):
        from pilot.align import model_kwargs

        for spec in ("align", "sw", "alignment"):
            assert model_kwargs(spec, batch_size=8, max_length=2048) == {}

    def test_the_kmer_arm_takes_none(self):
        from pilot.align import model_kwargs

        assert model_kwargs("kmer", batch_size=8, max_length=2048) == {}
        assert model_kwargs("kmer:5", batch_size=8, max_length=2048) == {}

    def test_an_esm_checkpoint_takes_them(self):
        from pilot.align import model_kwargs

        assert model_kwargs("facebook/esm2_t33_650M_UR50D", batch_size=4, max_length=2048) == {
            "batch_size": 4, "max_length": 2048}

    def test_the_positional_arm_takes_them_too(self):
        from pilot.align import model_kwargs

        assert model_kwargs("eba:facebook/esm2_t33_650M_UR50D", max_length=2048) == {"max_length": 2048}

    def test_none_values_are_dropped(self):
        from pilot.align import model_kwargs

        assert model_kwargs("facebook/esm2_t12_35M_UR50D", batch_size=8, max_length=None) == {
            "batch_size": 8}

    def test_max_length_reaches_the_embedder_and_the_cache_key(self, tmp_path):
        from pilot.align import model_kwargs
        from pilot.embed import Esm2Embedder

        kw = model_kwargs("facebook/esm2_t12_35M_UR50D", batch_size=8, max_length=2048)
        emb = Esm2Embedder(model_name="facebook/esm2_t12_35M_UR50D", **kw)
        assert emb.max_length == 2048
        sc = EmbeddingScorer.__new__(EmbeddingScorer)
        sc.embedder, sc.name = emb, emb.name
        other = EmbeddingScorer.__new__(EmbeddingScorer)
        other.embedder = Esm2Embedder(model_name="facebook/esm2_t12_35M_UR50D", max_length=1022)
        other.name = other.embedder.name
        assert sc._cache_path(REFS, tmp_path) != other._cache_path(REFS, tmp_path)

    def test_the_arms_that_ignore_it_still_build(self):
        """`--max-length` on a blosum or k-mer-context positional run must be dropped, not raise."""
        sc = get_scorer("eba:blosum", REFS, batch_size=8, max_length=2048, top_k=0)
        assert sc.score_all(REFS[0]).shape == (3,)
        sc = get_scorer("eba:kmerpos", REFS, batch_size=8, max_length=2048, top_k=0, prefilter="none")
        assert sc.score_all(REFS[0]).shape == (3,)


class TestTwoRunsShareOneCard:
    """Two command prompts, one GPU: the Omen's 8 GB holds two ESM-2 650M runs, and the only state they
    share is `data/embed_cache/`. These tests pin the two things that made that safe."""

    def test_concurrent_builds_of_one_matrix_do_not_share_a_temp_file(self, tmp_path, monkeypatch):
        """The cache path is a pure function of the embedder and the reference set, so two runs building
        the same matrix target the same file by design. The temp file they stage it through must not be
        the same file, or their writes interleave into it, and `_load_or_embed`'s row-count check
        cannot see a torn matrix whose row count is right."""
        import pilot.align as al

        seen = []
        real_save = al.np.save
        monkeypatch.setattr(al.np, "save", lambda p, m: (seen.append(Path(p)), real_save(p, m))[1])
        for pid in (1111, 2222):
            monkeypatch.setattr(al.os, "getpid", lambda pid=pid: pid)
            EmbeddingScorer(get_embedder("kmer"), REFS, cache_dir=tmp_path, verbose=False)
            for stale in tmp_path.glob("refmatrix_*.npy"):
                stale.unlink()  # or the second run is a cache hit and never reaches np.save
        assert len(seen) == 2
        assert seen[0] != seen[1], "both builds staged through one temp file"
        assert all(p.suffix == ".npy" for p in seen), "np.save would append .npy and break the rename"

    def test_a_failed_rename_leaves_no_temp_file_behind(self, tmp_path, monkeypatch):
        """A half-written cache that nobody tidies is the next run's mystery, and the run itself must
        still finish: caching is an optimisation."""
        monkeypatch.setattr(Path, "replace", lambda self, other: (_ for _ in ()).throw(OSError("nope")))
        sc = EmbeddingScorer(get_embedder("kmer"), REFS, cache_dir=tmp_path, verbose=False)
        assert sc.matrix.shape[0] == len(REFS)
        assert not list(tmp_path.glob("*.tmp*")), "a temp file survived the failure"

    def test_the_build_returns_its_peak_gpu_memory(self, monkeypatch):
        """The catalogue pass is a run's peak allocation and torch's allocator keeps it reserved for the
        rest of the process. On a shared card the second run's build needs those blocks back."""
        import sys
        import types

        calls = []
        fake = types.ModuleType("torch")
        fake.cuda = types.SimpleNamespace(is_available=lambda: True,
                                          empty_cache=lambda: calls.append(1))
        monkeypatch.setitem(sys.modules, "torch", fake)
        EmbeddingScorer(get_embedder("kmer"), REFS, cache_dir=None, verbose=False)
        assert calls, "the build did not release its cached GPU blocks"

    def test_releasing_is_a_no_op_without_a_loaded_torch(self, monkeypatch):
        """The k-mer and BLOSUM paths never import torch, and importing it here to free memory it is not
        using would cost seconds on every CPU run."""
        import sys

        from pilot.align import release_cached_gpu_memory

        monkeypatch.delitem(sys.modules, "torch", raising=False)
        release_cached_gpu_memory()  # must not raise, and must not import torch
        assert "torch" not in sys.modules

    def test_a_driver_that_refuses_does_not_take_the_run_down(self, monkeypatch):
        import sys
        import types

        from pilot.align import release_cached_gpu_memory

        fake = types.ModuleType("torch")
        fake.cuda = types.SimpleNamespace(
            is_available=lambda: (_ for _ in ()).throw(RuntimeError("no driver")),
            empty_cache=lambda: None)
        monkeypatch.setitem(sys.modules, "torch", fake)
        release_cached_gpu_memory()
