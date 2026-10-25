"""Synthetic plasmids with exact ground truth, used by the tests and by the no-download smoke test.

Everything here is random sequence built from codon rules; it contains no real genes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

SENSE = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT" if a + b + c not in ("TAA", "TAG", "TGA")]


def random_dna(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def random_codons(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(SENSE) for _ in range(n))


def random_cds(rng: random.Random, n_codons: int) -> str:
    """ATG + n sense codons + TAA."""
    return "ATG" + random_codons(rng, n_codons) + "TAA"


def rc(s: str) -> str:
    return str(Seq(s).reverse_complement())


@dataclass
class Planted:
    label: str
    strand: int
    protein: str  # expected translation, no stop, first residue M
    start: int  # 0-based forward coordinate of the lowest end, after rotation
    length_nt: int
    wraps: bool


def _protein(cds_nt: str) -> str:
    body = cds_nt[:-3] if cds_nt[-3:] in ("TAA", "TAG", "TGA") else cds_nt
    return "M" + str(Seq(body[3:]).translate())


def build_plasmid(seed: int, cds_specs: list[dict], rotate_to_wrap: bool = False, bg: int = 400):
    """Return (sequence, [Planted]). Each CDS is preceded in-frame by TAA so its ORF start is unambiguous.

    cds_specs: dicts with keys label, strand (+1/-1), n_codons.
    rotate_to_wrap: rotate the circle so that the first CDS crosses the origin.
    """
    rng = random.Random(seed)
    construct = random_dna(rng, bg)
    placed = []  # (label, strand, protein, construct_start, length)
    for spec in cds_specs:
        cds = random_cds(rng, spec["n_codons"])
        block = "TAA" + cds
        if spec["strand"] == 1:
            start = len(construct) + 3
            construct += block
        else:
            start = len(construct)  # rc(block) puts rc(cds) first, then rc("TAA") = "TTA"
            construct += rc(block)
        placed.append((spec["label"], spec["strand"], _protein(cds), start, len(cds)))
        construct += random_dna(rng, bg)
    return _rotate(construct, placed, rotate_to_wrap)


def build_fusion_plasmid(seed: int, tag_codons: int = 60, gene_codons: int = 150, bg: int = 400):
    """A tag-like part fused in frame to a gene: one long ORF, two annotated CDSs (the tag has no stop).

    A naive pipeline sees a single ORF containing both parts, which is exactly what decomposition must undo.
    """
    rng = random.Random(seed)
    tag = "ATG" + random_codons(rng, tag_codons)  # no stop: it runs straight into the gene
    gene = random_cds(rng, gene_codons)
    construct = random_dna(rng, bg) + "TAA"
    a = len(construct)
    construct += tag + gene + random_dna(rng, bg)
    placed = [
        ("tag_part", 1, _protein(tag), a, len(tag)),
        ("gene_part", 1, _protein(gene), a + len(tag), len(gene)),
    ]
    return _rotate(construct, placed, False)


def _rotate(construct: str, placed: list[tuple], rotate_to_wrap: bool):
    L = len(construct)
    k = 0
    if rotate_to_wrap:
        _, _, _, a, n = placed[0]
        k = a + n // 2
    seq = construct[k:] + construct[:k]
    planted = []
    for label, strand, protein, a, n in placed:
        s = (a - k) % L
        planted.append(Planted(label, strand, protein, s, n, s + n > L))
    return seq, planted


def write_genbank(path: Path, seq: str, planted: list[Planted], name: str = "synth", topology: str = "circular"):
    L = len(seq)
    feats = []
    for p in planted:
        if p.wraps:
            a = FeatureLocation(p.start, L, p.strand)
            b = FeatureLocation(0, p.start + p.length_nt - L, p.strand)
            loc = CompoundLocation([a, b] if p.strand == 1 else [b, a])  # parts in transcription order
        else:
            loc = FeatureLocation(p.start, p.start + p.length_nt, p.strand)
        feats.append(SeqFeature(loc, type="CDS", qualifiers={"label": [p.label]}))
    rec = SeqRecord(Seq(seq), id=name, name=name, description="synthetic test plasmid", features=feats)
    rec.annotations["molecule_type"] = "DNA"
    if topology:
        rec.annotations["topology"] = topology
    SeqIO.write(rec, str(path), "genbank")


def write_demo_set(out_dir: Path) -> list[Path]:
    """Four synthetic plasmids covering the cases the pipeline must handle. Returns the files written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = {
        "demo_single_fwd": build_plasmid(101, [{"label": "geneA", "strand": 1, "n_codons": 250}]),
        "demo_single_rev_wrap": build_plasmid(102, [{"label": "geneB", "strand": -1, "n_codons": 220}], rotate_to_wrap=True),
        "demo_two_gene": build_plasmid(
            103, [{"label": "geneC", "strand": 1, "n_codons": 180}, {"label": "geneD", "strand": -1, "n_codons": 140}]
        ),
        "demo_fusion": build_fusion_plasmid(104),
    }
    paths = []
    for name, (seq, planted) in cases.items():
        path = out_dir / f"{name}.gb"
        write_genbank(path, seq, planted, name=name[:16])
        paths.append(path)
    return paths
