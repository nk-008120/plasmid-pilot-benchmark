"""Score called ORFs against the annotated CDSs of a plasmid.

For every usable annotated CDS (protein A) we look at called ORFs (protein P) on the same strand that
overlap it:

  exact      P == A
  contained  A is inside a longer P  (upstream/downstream extension, typical of fusions and tags)
  truncated  P is inside a longer A  (ORF finder started later than the annotation, e.g. non-ATG start)
  missed     no overlapping same-strand ORF relates to A in any of those ways

Matching uses the UN-substituted translations (``protein_raw``). A reported protein has its first
residue forced to Met, which is right biologically but breaks substring matching: a gene with a GTG
start reads "M..." on its own and "V..." inside a longer ORF, so the two would never match and a gene
that was plainly found would score as ``missed``.

Two other statuses keep denominators honest:
  below_min_aa  A is shorter than the ORF-calling threshold, so it cannot be recovered by design
  invalid the annotation cannot serve as ground truth (partial, internal stop, bad length,...)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .genbank import AnnotatedCDS, Plasmid
from .intervals import overlap_bp
from .orfs import ORF

SHADOW_FRACTION = 0.5  # an unmatched ORF with >= this share of its length inside annotated CDSs is a "shadow"
CONTEXT_FRACTION = 0.5  # non-CDS features covering >= this share of an ORF are reported as its context


@dataclass
class CDSResult:
    cds: AnnotatedCDS
    status: str
    orf_index: int | None = None
    orf_minus_cds_aa: int | None = None
    n_extra_aa: int | None = None
    c_extra_aa: int | None = None
    coverage: float | None = None  # len(shorter) / len(longer) of the matched pair


@dataclass
class ORFResult:
    index: int
    orf: ORF
    klass: str  # cds_match | cds_shadow | unannotated
    matched_cds: list[str] = field(default_factory=list)
    feature_context: str = ""


def _raw(x) -> str:
    """The translation to match on; falls back to `protein` for objects built without `protein_raw`."""
    return getattr(x, "protein_raw", "") or x.protein


def _relation(a: str, p: str) -> str | None:
    if p == a:
        return "exact"
    if a in p:
        return "contained"
    if p in a:
        return "truncated"
    return None


def _rank(rel: str, p_len: int) -> tuple[int, int]:
    if rel == "exact":
        return (0, 0)
    if rel == "contained":
        return (1, p_len)  # prefer the tightest container
    return (2, -p_len)  # prefer the longest fragment


def _extras(a: str, p: str) -> tuple[int, int]:
    """(N-terminal, C-terminal) residues of P outside A, for P containing A."""
    if p.endswith(a):
        return len(p) - len(a), 0
    if p.startswith(a):
        return 0, len(p) - len(a)
    idx = p.find(a)
    return idx, len(p) - idx - len(a)


def analyse(
    plasmid: Plasmid, cdss: list[AnnotatedCDS], orfs: list[ORF], min_aa: int
) -> tuple[list[CDSResult], list[ORFResult]]:
    L = plasmid.length
    orf_segs = [o.segments(L) for o in orfs]

    cds_results: list[CDSResult] = []
    matched_by_orf: dict[int, list[str]] = defaultdict(list)
    for cds in cdss:
        if cds.invalid_reason:
            cds_results.append(CDSResult(cds, "invalid"))
            continue
        if len(cds.protein) < min_aa:
            cds_results.append(CDSResult(cds, "below_min_aa"))
            continue
        best = None
        for idx, orf in enumerate(orfs):
            if orf.strand != cds.strand or overlap_bp(cds.segments, orf_segs[idx]) == 0:
                continue
            rel = _relation(_raw(cds), _raw(orf))
            if rel is None:
                continue
            key = _rank(rel, len(orf.protein))
            if best is None or key < best[0]:
                best = (key, idx, rel)
        if best is None:
            cds_results.append(CDSResult(cds, "missed"))
            continue
        _, idx, rel = best
        a, p = _raw(cds), _raw(orfs[idx])
        res = CDSResult(
            cds,
            rel,
            orf_index=idx,
            orf_minus_cds_aa=len(p) - len(a),
            coverage=round(min(len(a), len(p)) / max(len(a), len(p)), 4),
        )
        if rel == "contained":
            res.n_extra_aa, res.c_extra_aa = _extras(a, p)
        cds_results.append(res)
        matched_by_orf[idx].append(cds.label or "unlabelled")

    non_cds = [f for f in plasmid.features if f.type not in ("CDS", "source", "gene")]
    orf_results: list[ORFResult] = []
    for idx, orf in enumerate(orfs):
        segs = orf_segs[idx]
        if idx in matched_by_orf:
            klass = "cds_match"
        else:
            inside = sum(overlap_bp(segs, c.segments) for c in cdss)
            klass = "cds_shadow" if min(inside, orf.length_nt) >= SHADOW_FRACTION * orf.length_nt else "unannotated"
        context = []
        for f in non_cds:
            if f.label and overlap_bp(segs, f.segments) >= CONTEXT_FRACTION * orf.length_nt:
                context.append(f"{f.type}:{f.label}")
        orf_results.append(ORFResult(idx, orf, klass, matched_by_orf.get(idx, []), "; ".join(context[:3])))
    return cds_results, orf_results
