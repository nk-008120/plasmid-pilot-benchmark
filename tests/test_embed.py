import numpy as np
import pytest

from pilot.embed import KmerEmbedder, get_embedder


def test_rows_are_l2_normalised_and_deterministic():
    e = KmerEmbedder()
    a = e.embed(["MKVLAAGIVGLNLGGK", "MGGSHHHHHHSSGLVPRGSH"])
    assert a.shape == (2, 1024)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.allclose(a, e.embed(["MKVLAAGIVGLNLGGK", "MGGSHHHHHHSSGLVPRGSH"]))


def test_identical_sequences_have_cosine_one_and_a_shared_prefix_scores_higher():
    e = KmerEmbedder()
    base = "MKVLAAGIVGLNLGGKQWERTYIPASDFGHKLCVNM" * 3
    v = e.embed([base, base, base + "WWWWWWWWWWWW" * 3, "PPPPPPPPPPPP" * 9])
    assert v[0] @ v[1] == pytest.approx(1.0, abs=1e-5)
    assert v[0] @ v[2] > v[0] @ v[3]          # shares composition vs shares nothing


def test_non_standard_residues_are_dropped_not_crashed():
    e = KmerEmbedder()
    assert np.isfinite(e.embed(["MKVX*ZBJUO-LAAG"])).all()
    assert e.embed([""]).shape == (1, 1024)   # empty sequence -> zero vector, no divide-by-zero


def test_k_changes_the_representation():
    s = ["MKVLAAGIVGLNLGGKQWERTY" * 4]
    assert not np.allclose(KmerEmbedder(k=3).embed(s), KmerEmbedder(k=5).embed(s))


def test_get_embedder_dispatch():
    assert isinstance(get_embedder("kmer"), KmerEmbedder)
    assert get_embedder("kmer:5").k == 5
    esm = get_embedder("facebook/esm2_t12_35M_UR50D")   # constructed, not loaded
    assert esm.name == "esm2_t12_35M_UR50D" and esm._model is None
