"""Per-residue embedders. The contract `pilot.eba` relies on is one row per residue, each L2-normalised."""
import numpy as np
import pytest

from pilot.positional import (
    AA,
    BlosumPositional,
    Esm2Positional,
    KmerPositional,
    get_positional_embedder,
)


def test_blosum_rows_are_normalised_and_one_per_residue():
    emb = BlosumPositional()
    seq = "MKTAYIAKQRQ"
    (m,) = emb.embed_residues([seq])
    assert m.shape == (len(seq), 20)
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)


def test_blosum_cosine_tracks_chemical_similarity():
    """Conservative substitutions must score above non-conservative ones, or the matrix is not wired up."""
    emb = BlosumPositional()
    (m,) = emb.embed_residues([AA])
    cos = m @ m.T
    at = {a: i for i, a in enumerate(AA)}
    assert cos[at["I"], at["I"]] == pytest.approx(1.0, abs=1e-5)
    assert cos[at["I"], at["L"]] > cos[at["I"], at["D"]]
    assert cos[at["K"], at["R"]] > cos[at["K"], at["W"]]
    assert cos[at["F"], at["Y"]] > cos[at["F"], at["P"]]


def test_blosum_unknown_residues_are_zero_not_guessed():
    emb = BlosumPositional()
    (m,) = emb.embed_residues(["AXBZU*"])
    assert np.linalg.norm(m[0]) == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(m[1:], 0.0)


def test_kmer_positional_shape_and_normalisation():
    emb = KmerPositional(k=3, dim=64, window=2)
    seq = "MKTAYIAKQRQISFVKSHF"
    (m,) = emb.embed_residues([seq])
    assert m.shape == (len(seq), 64)
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)


def test_kmer_positional_is_deterministic_and_context_sensitive():
    emb = KmerPositional(k=3, dim=256)
    a, b = emb.embed_residues(["MKTAYIAKQRQISFVK", "MKTAYIAKQRQISFVK"])
    assert np.array_equal(a, b)
    same, other = emb.embed_residues(["WWWWWWWWWW", "MKTAYIAKQR"])
    # a residue in a repeat has identical context everywhere, so its own cosines are 1
    assert (same @ same.T).min() == pytest.approx(1.0, abs=1e-5)
    assert (same @ other.T).max() < 0.5


def test_empty_sequence_gives_an_empty_block():
    for emb in (BlosumPositional(), KmerPositional(dim=32)):
        (m,) = emb.embed_residues([""])
        assert m.shape[0] == 0
        assert m.shape[1] == emb.dim


def test_several_sequences_come_back_in_order():
    emb = BlosumPositional()
    seqs = ["MK", "MKTAY", "W"]
    out = emb.embed_residues(seqs)
    assert [m.shape[0] for m in out] == [2, 5, 1]


def test_dispatch():
    assert isinstance(get_positional_embedder("blosum"), BlosumPositional)
    assert isinstance(get_positional_embedder("blosum62"), BlosumPositional)
    assert isinstance(get_positional_embedder("kmerpos"), KmerPositional)
    assert get_positional_embedder("kmerpos:5").k == 5
    esm = get_positional_embedder("facebook/esm2_t12_35M_UR50D")
    assert isinstance(esm, Esm2Positional)
    assert esm.name == "esm2_t12_35M_UR50D-pos"


def test_esm2_constructs_without_torch_installed():
    """Construction must stay lazy: the twilight script builds a scorer before it knows it has a GPU."""
    emb = Esm2Positional(max_length=64)
    assert emb.dim == 0
    assert emb.n_truncated == 0


def test_model_only_arguments_are_dropped_for_backends_that_cannot_use_them():
    """`10_twilight.py` passes batch_size to every non-k-mer scorer; blosum and kmerpos must survive it."""
    assert isinstance(get_positional_embedder("blosum", batch_size=8), BlosumPositional)
    emb = get_positional_embedder("kmerpos:4", batch_size=8, dim=64)
    assert isinstance(emb, KmerPositional)
    assert (emb.k, emb.dim) == (4, 64)
