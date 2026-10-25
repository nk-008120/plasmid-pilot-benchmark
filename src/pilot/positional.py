"""Per-residue protein embeddings, for the EBA-style positional scorer in `pilot.eba`.

`pilot.embed` produces **one vector per sequence**. the working record is the finding that this is the wrong
regime: Pantolini's 0.92 Spearman against TM-score is per-residue embeddings plus a dynamic-programming
alignment, and the ablation that mean-pools and takes a cosine instead scores 0.56, below HH-align's 0.82.
Johnson et al. (eLife 2023;12:RP91415) report the same thing from the other direction. So the positional
arm needs an embedder that keeps the residue axis, which is what this module adds.

Three implementations, for three different jobs:

* `BlosumPositional`: each residue becomes its BLOSUM62 row. This is not a foundation model and is not
  meant to be: it is the **validation harness**. A cosine over BLOSUM62 rows is a smooth surrogate for
  BLOSUM62 itself, so running the EBA scorer with it must land close to the exact Smith-Waterman control
  on the 23 hand-audited constructs. If it does not, the bug is in the dynamic programming rather than in
  the embedding, and that is worth being able to tell apart before a GPU is involved.
* `KmerPositional`: hashed k-mer context per residue. Dependency-free, instant, so the tests can
  exercise the whole path without biopython's matrices or a model download.
* `Esm2Positional`: the real thing: ESM-2 last-hidden-state vectors, special tokens stripped. torch and
  transformers are imported lazily, as in `pilot.embed`.

All three return rows that are **L2-normalised per residue**, so a cosine substitution matrix is a plain
matrix product `Q @ R.T` and `pilot.eba` never has to re-normalise.
"""
from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"


class PositionalEmbedder(Protocol):
    """One vector per residue, rows L2-normalised. The positional counterpart of `pilot.embed.Embedder`."""

    name: str
    dim: int

    def embed_residues(self, seqs: list[str]) -> list[np.ndarray]:  # each (len(seq), dim)
        ...


def _l2_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


class BlosumPositional:
    """Each residue is its BLOSUM62 row, L2-normalised. The harness the dynamic programming is checked on.

    Why this is a fair surrogate and not a circular test: the cosine of two normalised BLOSUM62 rows is
    not the BLOSUM62 score, it is the correlation between how the two residues substitute for everything
    else. Related residues have similar rows, so the cosine matrix behaves like a rescaled BLOSUM62 but is
    not equal to it. Reproducing the alignment arm's *ranking* with it therefore tests the alignment, and
    recovering the exact scores is tested separately in `tests/test_eba.py` against `parasail`.

    Unknown residues (X, B, Z, U, and anything else) get the zero vector, so they score 0 against
    everything rather than silently aligning to whatever the matrix happens to list first.
    """

    name = "blosum62-pos"

    def __init__(self) -> None:
        from Bio.Align import substitution_matrices  # noqa: PLC0415 (biopython is a core dependency)

        m = substitution_matrices.load("BLOSUM62")
        alphabet = list(m.alphabet)
        keep = [alphabet.index(a) for a in AA]
        rows = np.asarray(m, dtype=np.float32)[np.ix_(keep, keep)]
        self.dim = len(AA)
        self._table = {a: v for a, v in zip(AA, _l2_rows(rows))}
        self._zero = np.zeros(self.dim, dtype=np.float32)

    def embed_residues(self, seqs: list[str]) -> list[np.ndarray]:
        out = []
        for s in seqs:
            if not s:
                out.append(np.zeros((0, self.dim), dtype=np.float32))
                continue
            out.append(np.stack([self._table.get(c, self._zero) for c in s.upper()]))
        return out


class KmerPositional:
    """Hashed k-mer context around each residue. The stub the unit tests run against.

    Residue `i` gets the buckets of every k-mer starting within `window` of it, so two positions score
    highly when their local context agrees and the vectors degrade smoothly rather than being one-hot.
    It knows nothing about homology beyond shared substrings, which is the same honest limitation
    `pilot.embed.KmerEmbedder` carries.
    """

    def __init__(self, k: int = 3, dim: int = 128, window: int = 2) -> None:
        self.k, self.dim, self.window = k, dim, window
        self.name = f"kmerpos{k}-{dim}-{window}"

    def _bucket(self, kmer: str) -> int:
        return int(hashlib.blake2b(kmer.encode(), digest_size=8).hexdigest(), 16) % self.dim

    def embed_residues(self, seqs: list[str]) -> list[np.ndarray]:
        out = []
        for s in seqs:
            s = s.upper()
            n = len(s)
            v = np.zeros((n, self.dim), dtype=np.float32)
            if n == 0:
                out.append(v)
                continue
            buckets = [self._bucket(s[j : j + self.k]) for j in range(max(0, n - self.k + 1))]
            for i in range(n):
                for j in range(max(0, i - self.window), min(len(buckets), i + self.window + 1)):
                    v[i, buckets[j]] += 1.0
            out.append(_l2_rows(v))
        return out


class Esm2Positional:
    """ESM-2 per-residue hidden states, special tokens stripped and each residue L2-normalised.

    The same checkpoint advice as `pilot.embed.Esm2Embedder`: validate with `facebook/esm2_t12_35M_UR50D`,
    run `facebook/esm2_t33_650M_UR50D` on a GPU. Sequences longer than `max_length` are truncated and the
    count of truncated sequences is kept in `n_truncated`, because several called ORFs in this pilot run
    past 1,000 residues and a silently clipped query would be scored as if it were a shorter protein.
    """

    def __init__(self, model_name: str = "facebook/esm2_t12_35M_UR50D", device: str | None = None,
                 batch_size: int = 8, max_length: int = 1022) -> None:
        self.model_name, self.batch_size, self.max_length = model_name, batch_size, max_length
        self.name = model_name.split("/")[-1] + "-pos"
        self._device, self._tok, self._model, self.dim = device, None, None, 0
        self.n_truncated = 0

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        # `low_cpu_mem_usage` memory-maps the safetensors weights instead of materialising a second copy
        # while loading. On a CPU box that matters more than it sounds: ESM-2 650M is about 2.6 GB in
        # float32, the default path peaks at roughly twice that, and mapped weights are file-backed, so
        # the OS can evict them under pressure rather than forcing the process into the page file. It
        # needs `accelerate` on some transformers versions, so a failure falls back to the plain load
        # rather than taking the run down.
        try:
            self._model = AutoModel.from_pretrained(self.model_name, low_cpu_mem_usage=True)
        except (ImportError, ValueError, TypeError):
            self._model = AutoModel.from_pretrained(self.model_name)
        self._model = self._model.to(self._device).eval()
        self.dim = int(self._model.config.hidden_size)

    def embed_residues(self, seqs: list[str]) -> list[np.ndarray]:
        self._load()
        torch = self._torch
        out: list[np.ndarray] = []
        for i in range(0, len(seqs), self.batch_size):
            raw = seqs[i : i + self.batch_size]
            # Residues are lost once the sequence exceeds `max_length - 2`, not `max_length`:
            # <cls> and <eos> each take a token slot. Same erratum as the slice below; this is
            # the reported COUNT of it, and it used to undercount by the sequences that land in
            # that two-residue band.
            self.n_truncated += sum(1 for s in raw if len(s) > self.max_length - 2)
            batch = [s[: self.max_length] for s in raw]
            enc = self._tok(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=self.max_length).to(self._device)
            with torch.no_grad():
                h = self._model(**enc).last_hidden_state.cpu().numpy()
            mask = enc["attention_mask"].cpu().numpy().astype(bool)
            for row, keep, seq in zip(h, mask, batch):
                # ESM-2 prepends <cls> and appends <eos>; both are inside the attention mask and neither
                # is a residue, so taking the mask at face value would shift every position by one.
                #
                # **ERRATUM, 20 Oct.** This used to read `[1: 1 + len(seq)]`, which is the
                # same slice for every sequence that FITS and off by one for every sequence that does
                # not. `max_length` counts tokens, not residues, so at `max_length = 2048` the tokenizer
                # emits <cls> + 2,046 residues + <eos>, while `len(seq)` is 2,048 after the slice above
                #, so the window ran to 2,047 and its LAST ROW WAS THE <eos> EMBEDDING, returned as if
                # it were a residue. `[1:-1]` is what was meant in both cases: `row[keep]` has already
                # dropped the padding, so what remains is exactly <cls>, the residues, <eos>. Identical
                # output for every sequence at or under the cap, which `tests/test_positional_pooled.py`
                # pins; one fewer, correct row for every sequence over it. It bit 504 of the
                # 40,750-protein catalogue and 9 of the 24 constructs this project measures.
                res = row[keep][1:-1]
                out.append(_l2_rows(res))
        return out

    def embed_residues_pooled(
        self, seqs: list[str]
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """`(l2 rows, raw rows, mean-pooled vectors)` for each sequence, from ONE forward pass.

        `embed_residues` normalises every residue row, which is what `pilot.eba` needs and which makes the
        retrieval arm's pooled vector **unrecoverable** from its output: `pilot.embed.Esm2Embedder`
        averages the *raw* hidden states and normalises once at the end, and a mean of normalised rows is
        a different vector. this project has to put the per-residue effect and the pooled effect on one
        scale, so this method hands back all three quantities rather than paying for two passes.

        Two things about it are deliberate and load-bearing:

        * **It pools over the attention mask including `<cls>` and `<eos>`**, because that is what the
          retrieval arm does. The residue rows beside it exclude both, as `embed_residues` does. The two
          special tokens are 2 of `L + 2` positions, 0.3% of the mean for a 705 aa payload, which is
          small and not zero, so it is stated rather than assumed away.
        * **The pooling arithmetic is duplicated from `pilot.embed.Esm2Embedder.embed` rather than
          shared.** `data/embed_cache/` holds ESM-2 reference matrices built by that code and the
          embedder's name is the cache key, so refactoring it into a common helper risks
          a last-float32-place disagreement between a cached matrix and a fresh one, for a ranking
          measurement that is a tie-break, not a rounding error. `tests/test_positional_pooled.py` drives
          identical hidden states through both implementations and asserts they agree, which makes
          "this is the retrieval arm's pooled vector" a checked fact instead of an intention.
        """
        self._load()
        torch = self._torch
        l2_out: list[np.ndarray] = []
        raw_out: list[np.ndarray] = []
        pooled: list[np.ndarray] = []
        for i in range(0, len(seqs), self.batch_size):
            raw = seqs[i : i + self.batch_size]
            # Residues are lost once the sequence exceeds `max_length - 2`, not `max_length`:
            # <cls> and <eos> each take a token slot. Same erratum as the slice below; this is
            # the reported COUNT of it, and it used to undercount by the sequences that land in
            # that two-residue band.
            self.n_truncated += sum(1 for s in raw if len(s) > self.max_length - 2)
            batch = [s[: self.max_length] for s in raw]
            enc = self._tok(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=self.max_length).to(self._device)
            with torch.no_grad():
                h = self._model(**enc).last_hidden_state
            # ---- the retrieval arm's pooled vector, verbatim from pilot.embed.Esm2Embedder.embed ----
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
            # ---- and the residue rows, as embed_residues produces them ----
            hn = h.cpu().numpy()
            mask = enc["attention_mask"].cpu().numpy().astype(bool)
            for row, keep, seq in zip(hn, mask, batch):
                res = row[keep][1:-1]  # see the erratum in `embed_residues`: NOT `[1: 1 + len(seq)]`
                raw_out.append(res.astype(np.float32))
                l2_out.append(_l2_rows(res))
        return l2_out, raw_out, _l2_rows(np.concatenate(pooled).astype(np.float32))

def get_positional_embedder(spec: str, **kw) -> PositionalEmbedder:
    """`'blosum'`, `'kmerpos'` / `'kmerpos:5'`, or an ESM-2 checkpoint name.

    Model-only arguments are dropped for the backends that have no use for them, because the callers are
    generic: `scripts/10_twilight.py` hands every non-k-mer scorer a `batch_size`, which is meaningful to
    ESM-2 and meaningless to a BLOSUM table. Silently ignoring it here is what lets `--scorer eba:blosum`
    run through those scripts unchanged.
    """
    if spec in ("blosum", "blosum62"):
        return BlosumPositional()
    if spec == "kmerpos" or spec.startswith("kmerpos:"):
        k = int(spec.split(":", 1)[1]) if ":" in spec else 3
        allowed = {n: v for n, v in kw.items() if n in ("dim", "window")}
        return KmerPositional(k=k, **allowed)
    return Esm2Positional(model_name=spec, **kw)
