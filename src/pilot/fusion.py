"""Lay out what sits inside the ORF that corresponds to a plasmid's insert.

The audit question for a fusion or cassette is "what is the extra sequence either side of the payload?".
Answering it by eye means opening each plasmid in a viewer and reading off coordinates. This module does
the mechanical part: it picks the called ORF that best matches the stated insert size, then lists every
annotated feature that overlaps that ORF, positioned in amino-acid offsets from the ORF's own start.

What is left for a human is the part that needs judgement: confirming the payload's identity against the
repository page, and naming each extra piece (tag / linker / second gene / read-through / backbone partner
/ domain). The tool deliberately does not guess those labels.
"""
from __future__ import annotations

from dataclasses import dataclass

from .genbank import Plasmid
from .insert import check_insert
from .intervals import overlap_bp
from .orfs import ORF

SKIP_TYPES = frozenset({"source", "gene"})


@dataclass
class FeatureHit:
    label: str
    type: str
    strand: int
    same_strand: bool
    start_aa: int  # offset from the ORF's start codon, in residues (may be negative if it starts upstream)
    end_aa: int
    length_aa: int
    covered_fraction: float  # share of the feature that lies inside the ORF
    n_segments: int = 1  # >1 means a compound feature (origin-crossing, spliced, or several merged sites)


def _orf_offset_aa(orf: ORF, pos: int, plasmid_length: int) -> int:
    """Translate a forward-strand coordinate into residues from the ORF's start codon.

    Negative for a position upstream of the start codon. The modulo needed for circular plasmids would
    otherwise wrap an upstream position to a huge positive offset: a Kozak site just before a 1,656 aa
    ORF was reported at residue 2,559, i.e. past the end of the ORF it supposedly sits in.
    """
    if orf.strand == 1:
        nt = (pos - orf.start) % plasmid_length
    else:
        nt = (orf.start + orf.length_nt - pos) % plasmid_length
    if nt > orf.length_nt:
        nt -= plasmid_length  # upstream of the start codon, not most of the way round the plasmid
    return nt // 3


def features_inside(plasmid: Plasmid, orf: ORF) -> list[FeatureHit]:
    """Every labelled feature overlapping `orf`, in order along the ORF."""
    L = plasmid.length
    segs = orf.segments(L)
    hits: list[FeatureHit] = []
    for f in plasmid.features:
        if f.type in SKIP_TYPES or not f.label:
            continue
        shared = overlap_bp(segs, f.segments)
        if shared == 0:
            continue
        f_len = sum(e - s for s, e in f.segments)
        # Map only the segments that actually overlap, one at a time. Taking min(start)/max(end) across all
        # segments misreports two real cases seen in Addgene files: a feature crossing the origin (both ends
        # map to the same offset, so a 367 aa MBP collapsed to 1 aa) and several sites merged into one
        # multi-part feature (a 10 bp Kozak spanned 2,559 residues to reach the last copy).
        offsets = [
            _orf_offset_aa(orf, pos, L)
            for seg in f.segments
            if overlap_bp(segs, [seg])
            for pos in seg
        ]
        if not offsets:
            continue
        hits.append(
            FeatureHit(
                label=f.label,
                type=f.type,
                strand=f.strand,
                same_strand=(f.strand == orf.strand),
                start_aa=min(offsets),
                end_aa=max(offsets),
                length_aa=max(1, f_len // 3),
                covered_fraction=round(shared / max(1, f_len), 3),
                n_segments=len(f.segments),
            )
        )
    hits.sort(key=lambda h: (h.start_aa, h.end_aa))
    return hits


def payload_orf(plasmid: Plasmid, orfs: list[ORF], insert_size_bp: int | None, tolerance: float = 0.10):
    """The ORF taken to carry the insert, plus the insert check that chose it. (None, check) if no size."""
    chk = check_insert(orfs, insert_size_bp, tolerance)
    if chk.best_orf_index is None:
        return None, chk
    return orfs[chk.best_orf_index], chk
