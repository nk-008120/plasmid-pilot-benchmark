"""Classify a construct's architecture from sequence evidence rather than from its annotation.

`archetype.py` reads GenBank feature labels. That works on curated Addgene deposits and fails on a bulk
NCBI corpus: those records carry a median of two CDS features, so tags and 2A peptides are simply not
labelled. Classified that way, only 15 of 789 corpus constructs came out as `tagged_fusion` against 4 of
23 in the hand-picked set: an annotation artefact, not a fact about the constructs, and the same finding
as the working record 12.1 reappearing at scale.

This module reclassifies from what is actually in the translated ORFs: 2A peptides, protease sites, tags
and linkers (via `decompose.find_motifs`), plus how many distinct non-vector payloads the construct
carries (via `autolabel`). A P2A peptide is visible in the protein sequence whether or not a depositor
annotated it.

**Two outputs, not one.** The old classifier forced one exclusive label, which made `backbone_heavy`
collide with everything else: a plasmid can perfectly well be a single-CDS construct *and* mostly
backbone, and #134898 (PCNA, payload 12% of the plasmid) is exactly that. So `archetype` carries the
construct's architecture and `flags` carries orthogonal facts that used to be forced into the same slot.
This closes open item 9 in the working record.
"""
from __future__ import annotations

from dataclasses import dataclass

from .autolabel import Hit, is_vector_part
from .decompose import find_motifs

BACKBONE_FRACTION = 0.15
# An ORF this much longer than the protein it matches is carrying something else. Addgene #17551's
# FOXO1-EGFP ORF is 899 aa against a 652 aa FOXO1: 0.27 dilution, and no motif anywhere to find it by.
FUSION_DILUTION = 0.20
# 2A peptides make two *proteins*; an engineered protease site marks a removable *tag*. Treating both as
# cassette evidence labelled Addgene #187828 (GST-TEV-NDP52) a cassette when it is a tagged fusion.
CASSETTE_MOTIFS = frozenset({"2A_skip"})
ARCHETYPES = ("no_payload", "multi_gene_cassette", "tagged_fusion", "single_cds")


@dataclass(frozen=True)
class SeqArchetypeCall:
    archetype: str
    flags: frozenset[str]
    n_payload_orfs: int
    n_cleavage_motifs: int
    n_tag_motifs: int
    payload_fraction: float
    evidence: tuple[str, ...]


def classify(
    orf_seqs: dict[str, str],
    hits: list[Hit],
    plasmid_length_bp: int,
    backbone_fraction: float = BACKBONE_FRACTION,
    min_identity: float = 0.85,
    min_coverage: float = 0.80,
) -> SeqArchetypeCall:
    """Archetype plus flags for one construct.

    Precedence is decided by the hand labels, not by taste. Cassette evidence outranks tag evidence:
    Addgene #85400 carries a FLAG tag *and* a P2A-linked second gene, and the depositor's description,
    and the hand label, call it a cassette. The old classifier gave tags priority and got it wrong, a
    disagreement the manifest recorded at the time ("expect auto=tagged_fusion (tag wins priority)").
    """
    payload_hits = [
        h for h in hits
        if not is_vector_part(h) and h.identity >= min_identity and h.coverage >= min_coverage
    ]
    # One best hit per ORF. Counting distinct *accessions* counted homolog families as separate payloads:
    # #134898 carries one untagged PCNA and DIAMOND returns twelve PCNA orthologues, which read as a
    # twelve-gene cassette. Distinct payload *loci* is the signal; distinct accessions is noise.
    best_per_orf: dict[str, Hit] = {}
    for h in payload_hits:
        if h.orf_id not in best_per_orf or h.bitscore > best_per_orf[h.orf_id].bitscore:
            best_per_orf[h.orf_id] = h
    payload_orf_ids = set(best_per_orf)
    payload_accessions = {h.accession for h in best_per_orf.values()}
    max_dilution = max((h.dilution for h in best_per_orf.values()), default=0.0)
    evidence: list[str] = []
    flags: set[str] = set()

    # Motifs are counted only inside ORFs that carry a payload. A 2A peptide in a spurious ORF on the
    # opposite strand is noise; the same motif inside the construct's coding sequence is architecture.
    n_cassette = n_protease = n_tag = 0
    for oid in payload_orf_ids:
        for m in find_motifs(orf_seqs.get(oid, "")):
            if m.name in CASSETTE_MOTIFS:
                n_cassette += 1
            elif m.kind == "cleave":
                n_protease += 1
            else:
                n_tag += 1
    n_cleave = n_cassette + n_protease

    longest = max((len(orf_seqs.get(o, "")) for o in payload_orf_ids), default=0) * 3
    fraction = round(longest / plasmid_length_bp, 4) if plasmid_length_bp else 0.0

    if not payload_hits:
        return SeqArchetypeCall("no_payload", frozenset(), 0, 0, 0, 0.0,
                                ("no ORF carries a confident non-vector hit",))

    if n_cassette:
        flags.add("has_2a")
        evidence.append(f"{n_cassette} ribosome-skipping motif(s) inside a payload ORF")
    if n_protease:
        flags.add("has_protease_site")
        evidence.append(f"{n_protease} engineered protease site(s)")
    if n_tag:
        flags.add("has_tag")
        evidence.append(f"{n_tag} tag/linker motif(s) on a payload ORF")
    if len(payload_orf_ids) > 1 and len(payload_accessions) > 1:
        flags.add("multi_payload")
        evidence.append(f"{len(payload_orf_ids)} payload ORFs carrying {len(payload_accessions)} proteins")
    if max_dilution > FUSION_DILUTION:
        flags.add("diluted_payload")
        evidence.append(f"payload ORF is {max_dilution:.0%} longer than the protein it matches")
    if fraction < backbone_fraction:
        flags.add("backbone_heavy")
        evidence.append(f"payload is {fraction:.1%} of the plasmid")

    if n_cassette or (len(payload_orf_ids) > 1 and len(payload_accessions) > 1):
        archetype = "multi_gene_cassette"
    elif n_tag or n_protease or max_dilution > FUSION_DILUTION:
        archetype = "tagged_fusion"
    else:
        archetype = "single_cds"
    return SeqArchetypeCall(archetype, frozenset(flags), len(payload_orf_ids), n_cleave, n_tag,
                            fraction, tuple(evidence))
