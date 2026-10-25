"""`Esm2Positional.embed_residues_pooled` must return the retrieval arm's own pooled vector.

the working record compares a per-residue effect against a pooled effect and calls a mechanism from the ratio.
That is only a statement about `10_twilight.py`'s embedding arm if the pooled vector measured here is
**the same vector** `pilot.embed.Esm2Embedder.embed` ranks on. The two implementations are deliberately
separate: `data/embed_cache/` holds matrices built by the second one and its name is the cache key
, so refactoring it into a shared helper risks a last-float32-place disagreement between a
cached matrix and a fresh one, which for a ranking measurement is a tie-break rather than a rounding
error. The duplication is therefore paid for with this file: identical hidden states are driven through
both poolers and the outputs must agree.

A fake tokenizer and a fake model stand in for a 2.6 GB download. They are not a stub in the weak sense:
the hidden states depend on both token id and position, the batch has sequences of different lengths so
padding is exercised, and `batch_size` is smaller than the batch so the concatenation path runs. Every
one of those is a way the two implementations could disagree.

Four things are pinned:

* the pooled vectors are equal, to float32 tolerance;
* the residue rows exclude `<cls>` and `<eos>` while the pooled vector includes them, which is the
  asymmetry `embed_residues_pooled`'s docstring claims and the reason the working record reports the size of the
  difference rather than assuming it away;
* the l2 rows are exactly `embed_residues`' rows, so the new method did not quietly change the per-residue
  quantity the EBA arm depends on;
* the raw rows are NOT normalised, because the whole reason the method exists is that a mean of
  normalised rows is a different vector from a normalised mean of raw rows.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pilot.embed import Esm2Embedder  # noqa: E402
from pilot.positional import Esm2Positional  # noqa: E402

DIM = 8
AA = "ACDEFGHIKLMNPQRSTVWY"


class _Enc(dict):
    """What `AutoTokenizer.__call__` returns, reduced to the two things both embedders use."""

    def to(self, device):  # noqa: D102 (device is irrelevant to a fake)
        return self


class _FakeTok:
    """`<cls>` + residues + `<eos>`, right-padded, with the attention mask that implies.

    **`max_length` caps TOKENS, not residues**, exactly as the real ESM-2 tokenizer does: a cap of `n`
    admits `n - 2` residues. Reproducing that here rather than capping residues is what makes
    `test_truncation_does_not_return_the_eos_row_as_a_residue` a test of this project's erratum instead of
    a test of a convenient fiction.
    """

    def __call__(self, batch, return_tensors=None, padding=None, truncation=None, max_length=None):
        bodies = [[ord(c) % 29 + 4 for c in s] for s in batch]
        if truncation and max_length:
            bodies = [b[: max_length - 2] for b in bodies]
        widest = max(len(b) for b in bodies)
        ids, am = [], []
        for b in bodies:
            pad = widest - len(b)
            ids.append([0] + b + [2] + [1] * pad)
            am.append([1] * (len(b) + 2) + [0] * pad)
        return _Enc(input_ids=torch.tensor(ids), attention_mask=torch.tensor(am))


class _FakeModel:
    """Hidden states that depend on token id AND position, so neither pooler can agree by accident.

    The padded positions carry non-zero values on purpose: a pooler that ignored the attention mask would
    pass a test whose padding was zeros, and ignoring the mask is exactly the bug this file exists to
    rule out.
    """

    def __init__(self, dim: int = DIM) -> None:
        self.config = types.SimpleNamespace(hidden_size=dim)
        self.dim = dim

    def __call__(self, input_ids=None, attention_mask=None):
        b, t = input_ids.shape
        g = torch.arange(self.dim, dtype=torch.float32)
        pos = torch.arange(t, dtype=torch.float32).view(1, t, 1)
        h = torch.sin(input_ids.unsqueeze(-1).float() * 0.37 + g * 1.11) + 0.1 * pos
        return types.SimpleNamespace(last_hidden_state=h)


def _wire(obj, model, tok):
    """Pre-load the lazy fields so `_load` returns early and no checkpoint is fetched."""
    obj._torch = torch
    obj._tok = tok
    obj._model = model
    obj._device = "cpu"
    obj.dim = model.dim
    return obj


SEQS = ["MKVLAAG", "MKVLAAGYTT", "MW", "ACDEFGHIKLMNPQRSTVWY"]


@pytest.fixture
def pair():
    model, tok = _FakeModel(), _FakeTok()
    pos = _wire(Esm2Positional(batch_size=3, max_length=64), model, tok)
    flat = _wire(Esm2Embedder(batch_size=3, max_length=64), model, tok)
    return pos, flat


def test_pooled_vector_is_the_retrieval_arms_own_vector(pair):
    pos, flat = pair
    _, _, pooled = pos.embed_residues_pooled(SEQS)
    arm = flat.embed(SEQS)
    assert pooled.shape == arm.shape == (len(SEQS), DIM)
    assert np.allclose(pooled, arm, atol=1e-6), np.abs(pooled - arm).max()


def test_pooled_includes_the_special_tokens_and_the_residue_rows_do_not(pair):
    """The asymmetry is real, it shrinks like 2 / (L + 2), and the working record reports its size.

    Two facts, not one: the residue-only mean is never *equal* to the pooled vector, so the script cannot
    quietly swap one for the other; and the gap closes as the sequence lengthens, which is what makes the
    arm's quantity and the residue-only quantity interchangeable for a 705 aa payload and not for a
    dipeptide. Asserting a flat tolerance would have hidden the length dependence and invited exactly the
    short-sequence surprise that produced this docstring.
    """
    pos, _ = pair
    lengths = [4, 16, 60]
    seqs = ["".join(AA[(i * 7) % 20] for i in range(n)) for n in lengths]
    l2_rows, raw, pooled = pos.embed_residues_pooled(seqs)
    assert [m.shape[0] for m in l2_rows] == lengths
    gaps = []
    for raw_rows, p in zip(raw, pooled):
        residue_only = raw_rows.mean(axis=0)
        residue_only = residue_only / np.linalg.norm(residue_only)
        assert not np.allclose(residue_only, p, atol=1e-6)
        gaps.append(1.0 - float(np.dot(residue_only, p)))
    assert gaps[0] > gaps[1] > gaps[2], gaps
    assert gaps[-1] < 0.02, gaps


def test_l2_rows_match_embed_residues_exactly(pair):
    """The new method must not have changed the per-residue quantity `pilot.eba` scores on."""
    pos, _ = pair
    a = pos.embed_residues(SEQS)
    l2_rows, _, _ = pos.embed_residues_pooled(SEQS)
    assert len(a) == len(l2_rows)
    for x, y in zip(a, l2_rows):
        assert np.array_equal(x, y)


def test_raw_rows_are_not_normalised(pair):
    """If they were, the pooled vector could have been reconstructed from `embed_residues` and this
    method would not need to exist."""
    pos, _ = pair
    _, raw, _ = pos.embed_residues_pooled(SEQS)
    norms = np.concatenate([np.linalg.norm(m, axis=1) for m in raw])
    assert not np.allclose(norms, 1.0)


def test_truncation_is_counted_not_silent(pair):
    pos, _ = pair
    pos.max_length = 5
    l2_rows, _, _ = pos.embed_residues_pooled(["ACDEFGHIK", "MW"])
    # 5 tokens admit 3 residues, not 5: see the erratum test below.
    assert [m.shape[0] for m in l2_rows] == [3, 2]
    assert pos.n_truncated == 1


def test_truncation_does_not_return_the_eos_row_as_a_residue(pair):
    """this project's erratum: `max_length` counts TOKENS, not residues.

    `<cls>` takes one of them, so a cap of `n` admits `n - 2` residues, and the old
    `[1: 1 + len(seq)]` window, where `len(seq)` is the sequence already sliced to `n`, ran one row
    past the last residue and returned the `<eos>` embedding as if it were one. The fake tokenizer
    reproduces that arithmetic exactly, which is why this is a test and not an assertion about ESM-2.

    The property that makes the fix safe is in the second half: for every sequence at or under the cap
    the two expressions are the same slice, so nothing that fitted has moved.
    """
    pos, _ = pair
    pos.max_length = 10
    seqs = ["ACDEFGHIKLMNPQ", "ACDEFGH"]  # one over the cap, one under
    l2_rows, raw, _ = pos.embed_residues_pooled(seqs)
    # 10 tokens = <cls> + 8 residues + <eos>. Eight, not nine, and not ten.
    assert l2_rows[0].shape[0] == 8
    assert raw[0].shape[0] == 8
    # The sequence that fits is untouched by the fix.
    assert l2_rows[1].shape[0] == 7
    assert np.array_equal(pos.embed_residues(seqs)[1], l2_rows[1])


def test_the_fix_is_a_no_op_for_every_sequence_that_fits(pair):
    """What licenses changing shared code mid-project: the slice only moves where it was wrong."""
    pos, _ = pair
    pos.max_length = 64
    rows = pos.embed_residues(SEQS)
    assert [m.shape[0] for m in rows] == [len(s) for s in SEQS]
    for s, m in zip(SEQS, rows):
        assert m.shape[0] == len(s)


def test_the_truncation_count_counts_sequences_that_lost_residues(pair):
    """`<cls>` and `<eos>` take a token slot each, so the band `max_length - 2 < len <= max_length` is
    truncated too and used to be reported as untruncated. A limit's binding count has to count the
    limit binding."""
    pos, _ = pair
    pos.max_length = 10
    l2_rows, _, _ = pos.embed_residues_pooled(["A" * 11, "A" * 9, "A" * 8, "A" * 7])
    # A 10-token cap admits exactly 8 residues, so 11 and 9 lose some and 8 and 7 do not. The old
    # `len(s) > max_length` test called 9 untruncated, which is the row that matters: it is the band
    # just over the real cap, and it is where a quiet clip would look like a clean run.
    assert [m.shape[0] for m in l2_rows] == [8, 8, 8, 7]
    assert pos.n_truncated == 2
