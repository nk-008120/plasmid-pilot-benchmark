"""A second, annotation-independent ground truth: the insert size the repository states.

The CDS annotations in public GenBank files label *known vector parts* (resistance markers, origins,
common tags) because they come from matching against a parts catalogue. A depositor's own insert matches
nothing in that catalogue, so it is usually not annotated at all. Scoring only against annotated CDSs
therefore measures recovery of the backbone, not of the cargo.

Addgene states an "Insert Size (bp)" on every plasmid page. Recording it in the manifest gives a cheap
check that is independent of the annotation: does the ORF caller produce *something* the length of the
insert? It is coarse (one number per plasmid, and the stated size includes cloning junctions), so it
answers "was the cargo recoverable at all", not "were its boundaries right".
"""
from __future__ import annotations

from dataclasses import dataclass

from .orfs import ORF

DEFAULT_TOLERANCE = 0.10  # +/- 10% of the stated insert size counts as recovered


@dataclass
class InsertCheck:
    insert_size_bp: int | None
    best_orf_index: int | None
    best_orf_nt: int | None  # ORF length including its stop codon
    ratio: float | None  # best_orf_nt / insert_size_bp
    recovered: bool | None  # ratio within the tolerance band
    tolerance: float = DEFAULT_TOLERANCE

    @property
    def verdict(self) -> str:
        if self.insert_size_bp is None:
            return ""  # no stated size in the manifest
        if self.recovered:
            return "recovered"
        if self.best_orf_nt is None:
            return "no_orfs"
        return "short" if self.ratio < 1 else "long"


def parse_insert_size(value: str | None) -> int | None:
    """Read the manifest cell. Blank, '-', 'NA' and anything unparseable mean 'not stated'."""
    if value is None:
        return None
    text = value.strip().replace(",", "").replace("bp", "").strip()
    if not text or text.upper() in {"NA", "N/A", "-", "?"}:
        return None
    try:
        n = int(float(text))
    except ValueError:
        return None
    return n if n > 0 else None


def check_insert(orfs: list[ORF], insert_size_bp: int | None, tolerance: float = DEFAULT_TOLERANCE) -> InsertCheck:
    """Find the called ORF whose length is closest to the stated insert size.

    Closest-match rather than longest: in a small vector the resistance marker is often the longest ORF,
    so 'longest' would pick the backbone. Closest-match can of course land on a coincidence, which is why
    the caller also reports the matched ORF's class: an `unannotated` match is the expected signature of
    a real payload, a `cds_match` is more likely a known part of the right size.
    """
    if insert_size_bp is None:
        return InsertCheck(None, None, None, None, None, tolerance)
    if not orfs:
        return InsertCheck(insert_size_bp, None, None, None, False, tolerance)
    idx = min(range(len(orfs)), key=lambda i: abs(orfs[i].length_nt - insert_size_bp))
    nt = orfs[idx].length_nt
    # Compare lengths directly rather than |ratio - 1|: the latter is off by a float ulp exactly on the
    # tolerance boundary (1100/1000 - 1 > 0.10), which would silently reject edge cases.
    recovered = abs(nt - insert_size_bp) <= tolerance * insert_size_bp
    return InsertCheck(insert_size_bp, idx, nt, round(nt / insert_size_bp, 4), recovered, tolerance)
