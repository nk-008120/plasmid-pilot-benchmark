"""Load annotated plasmids from GenBank files, keeping strand and origin-crossing features intact."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from Bio import SeqIO
from Bio.Data import CodonTable
from Bio.Seq import Seq

GENBANK_SUFFIXES = (".gb", ".gbk", ".genbank", ".gbff")

_TABLE_11 = CodonTable.unambiguous_dna_by_id[11]
START_CODONS_T11 = frozenset(_TABLE_11.start_codons)
STOP_CODONS = frozenset(_TABLE_11.stop_codons)  # TAA, TAG, TGA (identical in table 1)


@dataclass
class Feature:
    type: str
    label: str
    strand: int  # +1, -1, or 0 (no strand / mixed strands)
    segments: list[tuple[int, int]]  # 0-based half-open; two segments if it crosses the origin
    location_str: str
    partial: bool = False  # fuzzy '<' or '>' end in the annotation
    nt_seq: str | None = None  # filled for CDS features only (None if extraction failed)


@dataclass
class Plasmid:
    id: str
    source_file: str
    name: str
    seq: str
    topology: str  # 'circular' or 'linear'
    topology_assumed: bool
    features: list[Feature]
    warnings: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.seq)

    @property
    def gc_fraction(self) -> float:
        return (self.seq.count("G") + self.seq.count("C")) / max(1, self.length)

    @property
    def n_nonacgt(self) -> int:
        return self.length - sum(self.seq.count(c) for c in "ACGT")

    @property
    def circular(self) -> bool:
        return self.topology == "circular"


@dataclass
class AnnotatedCDS:
    label: str
    strand: int
    segments: list[tuple[int, int]]
    protein: str  # translation without the terminal stop; first residue forced to M for valid starts
    nt_len: int
    has_terminal_stop: bool
    invalid_reason: str  # "" when the CDS is usable as ground truth
    protein_raw: str = ""  # same as protein but WITHOUT the Met substitution; used for matching only


def _label(qualifiers: dict) -> str:
    for key in ("label", "gene", "product", "standard_name", "note"):
        value = qualifiers.get(key)
        if value:
            return str(value[0]).strip()
    return ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plasmids(path: str | Path, assume_topology: str = "circular") -> list[Plasmid]:
    """Parse every record in a GenBank file. Raises on unreadable files; the caller decides what to do."""
    path = Path(path)
    records = list(SeqIO.parse(str(path), "genbank"))
    plasmids = []
    for n, rec in enumerate(records, start=1):
        seq = str(rec.seq).upper()  # raises if the record has no sequence
        topology = rec.annotations.get("topology")
        assumed = topology not in ("circular", "linear")
        if assumed:
            topology = assume_topology
        plasmid_id = path.stem if len(records) == 1 else f"{path.stem}#{n}"
        warnings: list[str] = []
        features: list[Feature] = []
        for f in rec.features:
            try:
                loc = f.location
                parts = loc.parts if hasattr(loc, "parts") else [loc]
                segs = [(int(p.start), int(p.end)) for p in parts]
                nt_seq = None
                if f.type == "CDS":
                    try:
                        nt_seq = str(f.extract(rec.seq)).upper()
                    except Exception as exc:  # external references, unknown positions,...
                        warnings.append(f"CDS extract failed ({_label(f.qualifiers) or 'unlabelled'}): {exc}")
                features.append(
                    Feature(
                        type=f.type,
                        label=_label(f.qualifiers),
                        strand=loc.strand or 0,
                        segments=segs,
                        location_str=str(loc),
                        partial=("<" in str(loc) or ">" in str(loc)),
                        nt_seq=nt_seq,
                    )
                )
            except Exception as exc:
                warnings.append(f"feature skipped ({f.type}): {exc}")
        plasmids.append(
            Plasmid(
                id=plasmid_id,
                source_file=path.name,
                name=rec.name or rec.id,
                seq=seq,
                topology=topology,
                topology_assumed=assumed,
                features=features,
                warnings=warnings,
            )
        )
    return plasmids


def annotated_cds(plasmid: Plasmid) -> list[AnnotatedCDS]:
    """Interpret CDS features as ground-truth proteins, flagging the ones that cannot be used as such."""
    out = []
    for f in plasmid.features:
        if f.type != "CDS":
            continue
        nt = f.nt_seq
        reason, protein, raw, has_stop = "", "", "", False
        if nt is None:
            reason = "extract_failed"
        elif f.partial:
            reason = "partial_location"
        elif f.strand == 0:
            reason = "mixed_or_no_strand"
        elif len(nt) % 3:
            reason = "length_not_multiple_of_3"
        else:
            translated = str(Seq(nt).translate(table=11))
            has_stop = translated.endswith("*")
            body = translated[:-1] if has_stop else translated
            if "*" in body:
                reason = "internal_stop"
            elif not body:
                reason = "empty"
            else:
                raw = body
                protein = body
                if nt[:3] in START_CODONS_T11:
                    protein = "M" + protein[1:]  # alternative start codons are read as Met by the ribosome
        out.append(
            AnnotatedCDS(
                label=f.label,
                strand=f.strand,
                segments=f.segments,
                protein=protein,
                nt_len=len(nt or ""),
                has_terminal_stop=has_stop,
                invalid_reason=reason,
                protein_raw=raw,
            )
        )
    return out
