"""Can a model identify the payload from the ORFs a naive pipeline produces?

For each plasmid the manifest names the correct answer (`payload_accession`). Every ORF called from that
plasmid is embedded and compared against the reference catalogue; the payload's **rank** in that ranked
list is the score. Rank 1 means the model, handed that ORF, would name the right protein.

Two numbers per plasmid, and the gap between them is the finding:

* `best_rank`: the best rank any of the plasmid's ORFs achieves. "Could the payload be identified at
                   all from this construct?"
* `payload_orf_rank`: the rank achieved by the ORF that matches the stated insert size, i.e. the one a
                   decomposition step is supposed to hand over. "Is the *right* ORF identifiable?"

A fusion degrades the second: the ORF containing the payload also contains a tag, a 2A peptide or a second
gene, and that contamination drags its embedding away from the payload. That degradation is the quantity
the whole project exists to measure, so it is reported per archetype rather than as one average.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embed import Embedder


@dataclass
class OrfHit:
    orf_id: str
    payload_rank: int | None  # 1-based rank of the true payload; None when the plasmid has no payload
    top1_accession: str
    top1_score: float
    payload_score: float | None


@dataclass
class PlasmidResult:
    plasmid_id: str
    archetype: str
    payload_accession: str
    n_orfs: int
    best_rank: int | None
    best_rank_orf: str
    payload_orf_rank: int | None  # rank for the insert-matching ORF, when one is identified
    payload_orf_id: str
    hits: list[OrfHit]

    @property
    def found_top1(self) -> bool:
        return self.best_rank == 1


def rank_against(
    orf_seqs: dict[str, str],
    ref_seqs: list[str],
    ref_accessions: list[str],
    embedder: Embedder,
    ref_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Cosine similarity of every ORF against every reference. Returns (n_orfs, n_refs) and the ORF order.

    Pass `ref_matrix` from `embed_reference` when looping over plasmids. Without it the reference set is
    re-embedded for every plasmid: harmless for the k-mer baseline, but with ESM-2 650M and 5,000
    references that is 23x the GPU work, turning a ~20 minute job into most of a day.
    """
    ids = list(orf_seqs)
    if not ids:
        return np.zeros((0, len(ref_accessions)), dtype=np.float32), []
    q = embedder.embed([orf_seqs[i] for i in ids])
    r = embedder.embed(ref_seqs) if ref_matrix is None else ref_matrix
    return q @ r.T, ids  # both are L2-normalised, so the dot product is the cosine


def embed_reference(ref_seqs: list[str], embedder: Embedder) -> np.ndarray:
    """Embed the reference catalogue once, to be reused across every plasmid."""
    return embedder.embed(ref_seqs)


def evaluate(
    sims: np.ndarray,
    orf_ids: list[str],
    ref_accessions: list[str],
    payload_accession: str,
    payload_orf_id: str = "",
) -> tuple[list[OrfHit], int | None, str, int | None]:
    """Rank the payload within each ORF's similarity row. Ranks are 1-based; ties take the worst position."""
    try:
        target = ref_accessions.index(payload_accession) if payload_accession else None
    except ValueError:  # the manifest names an accession the reference set does not contain
        target = None
    hits, best, best_orf, payload_orf_rank = [], None, "", None
    for i, oid in enumerate(orf_ids):
        row = sims[i]
        top = int(np.argmax(row))
        rank = score = None
        if target is not None:
            score = float(row[target])
            # worst-case rank under ties: count everything scoring at least as high
            rank = int((row >= row[target]).sum())
        hits.append(OrfHit(oid, rank, ref_accessions[top], float(row[top]), score))
        if rank is not None and (best is None or rank < best):
            best, best_orf = rank, oid
        if oid == payload_orf_id:
            payload_orf_rank = rank
    return hits, best, best_orf, payload_orf_rank


def summarise(results: list[PlasmidResult], top_k: tuple[int, ...] = (1, 5, 10)) -> dict:
    """Aggregate overall and per archetype. Plasmids with no protein payload are excluded, not counted as failures."""
    scored = [r for r in results if r.best_rank is not None]
    out: dict = {
        "n_plasmids": len(results),
        "n_scored": len(scored),
        "n_no_payload": len(results) - len(scored),
    }
    for k in top_k:
        out[f"best_rank_top{k}"] = sum(1 for r in scored if r.best_rank <= k)
        out[f"payload_orf_top{k}"] = sum(
            1 for r in scored if r.payload_orf_rank is not None and r.payload_orf_rank <= k
        )
    by: dict[str, dict] = {}
    for r in scored:
        d = by.setdefault(r.archetype, {"n": 0, **{f"best_top{k}": 0 for k in top_k},
                                        **{f"orf_top{k}": 0 for k in top_k}})
        d["n"] += 1
        for k in top_k:
            if r.best_rank <= k:
                d[f"best_top{k}"] += 1
            if r.payload_orf_rank is not None and r.payload_orf_rank <= k:
                d[f"orf_top{k}"] += 1
    out["by_archetype"] = by
    return out
