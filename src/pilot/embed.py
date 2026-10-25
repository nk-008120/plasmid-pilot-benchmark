"""Pluggable protein-embedding backends.

The retrieval logic must be testable without a 2 GB model download, so the backend is an interface with
two implementations:

* `KmerEmbedder`: k-mer composition. Deterministic, instant, no dependencies beyond numpy. It is the
  stub the tests run against, and it doubles as a legitimate *baseline*: if ESM-2 embeddings do not beat
  k-mer composition at retrieval, the foundation model is adding nothing and that is worth knowing.
* `Esm2Embedder`: mean-pooled ESM-2 hidden states. torch/transformers are imported lazily so the rest
  of the package runs without them.

Both return L2-normalised vectors, so cosine similarity is a plain dot product.
"""
from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, seqs: list[str]) -> np.ndarray:  # (len(seqs), dim), rows L2-normalised
        ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class KmerEmbedder:
    """Hashed k-mer frequency vector. Similar sequences get similar vectors; identical ones get identical.

    Not a language model: it knows nothing about structure or homology beyond shared short substrings.
    That is exactly why it makes a fair baseline: it measures how much of retrieval performance is
    explained by raw composition alone.
    """

    def __init__(self, k: int = 3, dim: int = 1024):
        self.k, self.dim, self.name = k, dim, f"kmer{k}-{dim}"

    def _bucket(self, kmer: str) -> int:
        return int(hashlib.blake2b(kmer.encode(), digest_size=8).hexdigest(), 16) % self.dim

    def embed(self, seqs: list[str]) -> np.ndarray:
        out = np.zeros((len(seqs), self.dim), dtype=np.float32)
        for i, s in enumerate(seqs):
            s = "".join(c for c in s.upper() if c in AA)
            for j in range(len(s) - self.k + 1):
                out[i, self._bucket(s[j : j + self.k])] += 1.0
        return _l2(out)


class Esm2Embedder:
    """Mean-pooled last-hidden-state embeddings from an ESM-2 checkpoint.

    Validate locally with `facebook/esm2_t12_35M_UR50D`; run `facebook/esm2_t33_650M_UR50D` on a GPU.
    Sequences longer than `max_length` are truncated: report how many that affected, because several
    called ORFs in this pilot exceed 1,000 residues.
    """

    def __init__(self, model_name: str = "facebook/esm2_t12_35M_UR50D", device: str | None = None,
                 batch_size: int = 8, max_length: int = 1022):
        self.model_name, self.batch_size, self.max_length = model_name, batch_size, max_length
        self.name = model_name.split("/")[-1]
        self._device, self._tok, self._model, self.dim = device, None, None, 0

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        # see pilot.positional.Esm2Positional._load for why the mapped load is worth the fallback
        try:
            self._model = AutoModel.from_pretrained(self.model_name, low_cpu_mem_usage=True)
        except (ImportError, ValueError, TypeError):
            self._model = AutoModel.from_pretrained(self.model_name)
        self._model = self._model.to(self._device).eval()
        self.dim = int(self._model.config.hidden_size)

    def embed(self, seqs: list[str]) -> np.ndarray:
        self._load()
        torch = self._torch
        vecs = []
        for i in range(0, len(seqs), self.batch_size):
            batch = [s[: self.max_length] for s in seqs[i : i + self.batch_size]]
            enc = self._tok(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=self.max_length).to(self._device)
            with torch.no_grad():
                h = self._model(**enc).last_hidden_state
            # Mean over the attention mask, which excludes padding and **includes `<cls>` and `<eos>`**:
            # the mask is 1 for every real token and both special tokens are real tokens. The comment
            # here used to claim the special tokens were excluded, and they never were, so every
            # retrieval vector in this project, and every row of the cached reference matrices, is a mean
            # over `L + 2` positions rather than `L`. Measured, that is worth a median 0.00046 of cosine
            # against the residue-only mean and at most 0.00298, so it is small, not zero,
            # and it is stated rather than assumed away.
            #
            # **Do not "fix" this to exclude them.** The embedder's name is the cache key
            # and `data/embed_cache/` holds 40,750-row matrices built by this exact arithmetic; changing
            # it silently invalidates them while the key still matches. The behaviour is pinned by
            # `tests/test_positional_pooled.py::test_pooled_vector_is_the_retrieval_arms_own_vector` and
            # `::test_pooled_includes_the_special_tokens_and_the_residue_rows_do_not`.
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            vecs.append(pooled.cpu().numpy())
        return _l2(np.concatenate(vecs).astype(np.float32))


def get_embedder(spec: str, **kw) -> Embedder:
    """`kmer` / `kmer:5` for the baseline, anything else is treated as an ESM-2 checkpoint name."""
    if spec == "kmer" or spec.startswith("kmer:"):
        k = int(spec.split(":", 1)[1]) if ":" in spec else 3
        return KmerEmbedder(k=k, **kw)
    return Esm2Embedder(model_name=spec, **kw)
