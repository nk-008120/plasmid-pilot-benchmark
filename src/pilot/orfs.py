"""Six-frame ORF calling that treats circular plasmids as circular.

Definition used throughout the pilot (state it in the write-up):
  * An ORF runs from a start codon to the first in-frame stop codon (stop included in the length).
  * Within a stop-free stretch the *earliest* start codon is used (the "longest ORF" convention).
  * On a circular plasmid an ORF may cross the origin, but it may not be longer than the plasmid.
  * Reported protein: stop removed, first residue forced to Met (matters for alternative starts).
"""
from __future__ import annotations

from dataclasses import dataclass

from Bio.Seq import Seq

from .genbank import STOP_CODONS
from .intervals import Segment, segments as _segments


@dataclass(frozen=True)
class ORF:
    strand: int  # +1 or -1
    start: int  # 0-based; lowest-coordinate end of the ORF in the plasmid's forward coordinates
    length_nt: int  # includes the stop codon
    wraps: bool  # crosses the origin of a circular plasmid
    protein: str  # stop removed, first residue forced to Met (what the ribosome makes)
    protein_raw: str = ""  # same, but WITHOUT the Met substitution, used for matching only

    @property
    def length_aa(self) -> int:
        return len(self.protein)

    def segments(self, plasmid_length: int) -> list[Segment]:
        return _segments(self.start, self.length_nt, plasmid_length)

    def end(self, plasmid_length: int) -> int:
        """Exclusive end in forward coordinates, wrapped into 1..L."""
        return (self.start + self.length_nt - 1) % plasmid_length + 1


def _scan_strand(s: str, circular: bool, starts: frozenset[str], min_aa: int) -> list[tuple[int, int]]:
    """Return (start, length_nt) of ORFs on the forward reading of `s`."""
    L = len(s)
    if L < 3:
        return []
    t = s * 3 if circular else s
    # On a circle, only starts in the middle copy are reported: the first copy supplies the preceding
    # stop-codon context and the third copy supplies room for the ORF to finish.
    lo, hi = (L, 2 * L) if circular else (0, L)
    found: list[tuple[int, int]] = []
    for frame in range(3):
        run: list[int] = []  # start codons seen since the last in-frame stop
        for i in range(frame, len(t) - 2, 3):
            codon = t[i : i + 3]
            if codon in starts:
                run.append(i)
            elif codon in STOP_CODONS:
                for st in run:  # earliest start that gives a legal length
                    n = i + 3 - st
                    if n <= L:
                        if lo <= st < hi and n // 3 - 1 >= min_aa:
                            found.append((st - lo, n))
                        break
                run = []
    return found


def find_orfs(
    seq: str,
    circular: bool = True,
    min_aa: int = 50,
    start_codons: tuple[str, ...] = ("ATG",),
) -> list[ORF]:
    seq = seq.upper()
    L = len(seq)
    starts = frozenset(start_codons)
    orfs: list[ORF] = []
    for strand in (1, -1):
        s = seq if strand == 1 else str(Seq(seq).reverse_complement())
        doubled = s + s if circular else s
        for p, n in _scan_strand(s, circular, starts, min_aa):
            nt = doubled[p : p + n]
            raw = str(Seq(nt).translate(table=11))[:-1]
            protein = raw if nt[:3] == "ATG" else "M" + raw[1:]
            start = p if strand == 1 else (L - p - n) % L
            orfs.append(ORF(strand, start, n, start + n > L, protein, raw))
    orfs.sort(key=lambda o: (o.start, o.strand, o.length_nt))
    return orfs
