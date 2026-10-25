"""Heuristic archetype labels (single CDS / multi-gene cassette / tagged fusion / backbone-heavy).

These labels exist to stratify and sanity-check the pilot set. They are rule-based guesses from the
GenBank annotations, so the manifest has an `archetype_manual` column that should override them after
you have looked at each plasmid.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .genbank import AnnotatedCDS, Plasmid
from .intervals import gap_bp, union_bp

# Selection markers and replication/backbone functions: real CDSs, but not the "payload" of a construct.
_BACKBONE_LABELS = [
    r"amp(r|icillin.*|\(r\))?", r"bla", r"aph\(3'\)-i\w*", r"kan(r|amycin.*|\(r\))?", r"nptii",
    r"neo(r|mycin.*)?", r"neor/kanr", r"cm(r)?", r"cat", r"chloramphenicol.*", r"tet(r|a)?",
    r"spec(r)?", r"spectinomycin.*", r"aada", r"strep(r|tomycin.*)", r"hyg(r|romycin.*)?", r"hph",
    r"puro(r|mycin.*)?", r"pac", r"bsd", r"blasticidin.*", r"zeo(r|cin.*)?", r"ble", r"sh ble",
    r"laci", r"lacz.*", r"ccdb", r"ccda", r"sacb", r"rop", r"rep[a-z]?", r"pir", r"mob\w*", r"trfa",
]
_BACKBONE = re.compile("|".join(f"(?:{p})" for p in _BACKBONE_LABELS), re.I)

# Purification/solubility tags, protease sites, linkers, localization signals.
_TAG = re.compile(
    r"(?<![a-z0-9])("
    r"6?x?his(?:6|8|10)?(?:[- ]?tag)?|polyhis\w*|flag|3x ?flag|ha(?:[- ]?tag)?|c-?myc|myc|v5|"
    r"strep(?:[- ]?tag)?(?: ?ii)?|twin-?strep|gst|mbp|sumo|trx|thioredoxin|avi(?:[- ]?tag)?|"
    r"s[- ]?tag|t7[- ]?tag|tev(?: site| cleavage.*)?|3c|prescission.*|thrombin|factor ?xa|"
    r"enterokinase|linker|\(?g[gs]{2,}.*|(?:sv40 )?nls|nes"
    r")(?![a-z0-9])",
    re.I,
)
# Large solubility partners that are legitimately long proteins but still count as tags.
_LONG_TAG = re.compile(r"^(gst|mbp|sumo|trx|thioredoxin)$", re.I)
_TWO_A = re.compile(r"(?<![a-z0-9])([ptef]2a|ires\d*|2a peptide)(?![a-z0-9])", re.I)

TAG_PROXIMITY_BP = 30
MAX_TAG_LEN_AA = 60  # a "tag"-labelled CDS longer than this is treated as payload unless it is a known big partner


def is_backbone_marker(label: str) -> bool:
    return bool(_BACKBONE.fullmatch(label.strip()))


def is_tag_label(label: str, protein_len: int | None = None) -> bool:
    if not _TAG.search(label):
        return False
    if protein_len is not None and protein_len > MAX_TAG_LEN_AA and not _LONG_TAG.fullmatch(label.strip()):
        return False  # e.g. an influenza "HA" gene is a payload, not an HA tag
    return True


@dataclass
class ArchetypeCall:
    archetype: str  # tagged_fusion | multi_gene_cassette | single_cds | backbone_heavy | no_payload
    n_payload_cds: int
    payload_fraction: float
    tag_near_cds: bool
    two_a_or_ires: bool
    backbone_heavy: bool
    tag_labels: list[str] = field(default_factory=list)


def classify_archetype(
    plasmid: Plasmid,
    cdss: list[AnnotatedCDS],
    min_aa: int = 50,
    backbone_threshold: float = 0.15,
) -> ArchetypeCall:
    L = plasmid.length
    payload = [
        c
        for c in cdss
        if not c.invalid_reason
        and len(c.protein) >= min_aa
        and not is_backbone_marker(c.label)
        and not is_tag_label(c.label, len(c.protein))
    ]
    payload_fraction = union_bp([c.segments for c in payload]) / max(1, L)

    tag_hits, tag_near, two_a = [], False, False
    for f in plasmid.features:
        if f.type in ("source", "gene"):
            continue
        near = lambda: any(  # noqa: E731
            gap_bp(f.segments, c.segments, L, plasmid.circular) <= TAG_PROXIMITY_BP for c in payload
        )
        if f.label and _TWO_A.search(f.label) and near():
            two_a = True
        if f.label and is_tag_label(f.label, None if f.type != "CDS" else sum(e - s for s, e in f.segments) // 3):
            if near():
                tag_near = True
                tag_hits.append(f.label)

    backbone_heavy = bool(payload) and payload_fraction < backbone_threshold
    if tag_near:
        archetype = "tagged_fusion"
    elif len(payload) >= 2 or (payload and two_a):
        archetype = "multi_gene_cassette"
    elif len(payload) == 1:
        archetype = "backbone_heavy" if backbone_heavy else "single_cds"
    else:
        archetype = "no_payload"
    return ArchetypeCall(
        archetype=archetype,
        n_payload_cds=len(payload),
        payload_fraction=round(payload_fraction, 4),
        tag_near_cds=tag_near,
        two_a_or_ires=two_a,
        backbone_heavy=backbone_heavy,
        tag_labels=sorted(set(tag_hits)),
    )
