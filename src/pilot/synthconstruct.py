"""Build synthetic constructs from real proteins, with exact boundaries and dilution as a swept variable.

Section 16.6 established what the NCBI corpus cannot do: it contains **zero 2A peptides** across 12,647
ORFs and only 33 tagged fusions in 789 constructs. Those are precisely the architectures the project is
about, Addgene #73037's payload moves from rank 50 to rank 1 purely on its two 2A peptides, so the
corpus supplies breadth on simple constructs and almost no power where it matters.

This module supplies the rest. Each construct is assembled from a **real Swiss-Prot payload** placed in a
realistic architecture, so the retrieval task is genuine, while the boundaries are exact by construction
rather than inferred. That buys three things the observational tiers cannot:

* **Dilution becomes an independent variable.** Section 15.4 fitted a dose-response to 19 observed points
  whose dilutions happened to fall where they fell. Here a payload is built at 0.0, 0.2, 0.4, 0.6 and 0.8
  dilution, so the curve is measured rather than fitted to convenience samples.
* **Ground truth needs no oracle alignment.** `oracle_span` infers the payload region by aligning against
  the answer. Here the span is recorded when the construct is written.
* **The archetypes the corpus lacks**, at any n required.

**The payload is deliberately left unannotated.** Real deposits annotate the vector parts a cloning
catalogue recognises and leave the depositor's insert unlabelled (section 12.1), so these files annotate
markers only. A pipeline reading them faces the same problem it faces on real data; a construct whose
payload is helpfully labelled would make the benchmark easier than reality.

Nothing here is a sequence of concern: payloads are drawn from whatever reference set the caller supplies,
which in this project is reviewed Swiss-Prot entries filtered through `autolabel.is_vector_part`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .decompose import CLEAVAGE, TAGS

# Codon table 11, sense codons grouped by the residue they encode. Reverse translation picks uniformly
# among synonyms, which deliberately produces a codon usage unlike any real organism: the benchmark is
# about protein-level retrieval, and a realistic codon bias would add a confound without adding a question.
CODONS: dict[str, list[str]] = {}
for _a in "TCAG":
    for _b in "TCAG":
        for _c in "TCAG":
            _cod = _a + _b + _c
            from Bio.Seq import Seq as _Seq

            _aa = str(_Seq(_cod).translate(table=11))
            if _aa != "*":
                CODONS.setdefault(_aa, []).append(_cod)
STOPS = ("TAA", "TGA", "TAG")

# Real peptides used as joins. These are the sequences the decomposition rules look for, so a construct
# built with one is a genuine test of those rules rather than a restatement of them.
P2A = "ATNFSLLKQAGDVEENPGP"
T2A = "EGRGSLLTCGDVEENPGP"
TEV_SITE = "ENLYFQG"
FLAG3 = "MDYKDHDGDYKDHDIDYKDDDDK"  # the real tag starts with Met, as in Addgene #47948
HIS6 = "HHHHHH"
NLS = "PKKKRKV"
GS_LINKER = "GGGGSGGGGS"

ARCHETYPES = ("single_cds", "tagged_fusion", "motifless_fusion", "multi_gene_cassette", "backbone_heavy")


@dataclass(frozen=True)
class Part:
    """One module of the assembled protein, in 1-based inclusive ORF coordinates."""

    name: str
    kind: str  # payload | tag | linker | cleavage | filler | passenger
    start_aa: int
    end_aa: int


@dataclass
class Construct:
    construct_id: str
    archetype: str
    dna: str
    payload_accession: str
    payload_aa: int
    orf_start_nt: int  # 0-based start of the payload-bearing ORF in `dna`
    orf_aa: int
    payload_start_aa: int  # 1-based, within the ORF
    payload_end_aa: int
    dilution: float  # fraction of the ORF that is not the payload, exact, not inferred
    parts: list[Part] = field(default_factory=list)
    markers: list[tuple[str, int, int]] = field(default_factory=list)  # (label, start_nt, end_nt), annotated

    @property
    def payload_fraction(self) -> float:
        return round(self.payload_aa * 3 / len(self.dna), 4) if self.dna else 0.0


def reverse_translate(protein: str, rng: random.Random) -> str:
    """Protein -> DNA, choosing uniformly among synonymous codons. Unknown residues become alanine."""
    return "".join(rng.choice(CODONS.get(aa, CODONS["A"])) for aa in protein)


def random_protein(rng: random.Random, n: int) -> str:
    """Filler with no relationship to anything in the reference set.

    Residue frequencies are roughly those of real proteomes so that filler does not stand out from payload
    by composition alone: a classifier that separated them on amino-acid frequency would score well
    without doing the task.
    """
    alphabet = "AAAALLLLLGGGGSSSSVVVVEEEEKKKKIIIITTTTDDDDRRRRPPPNNNQQQFFFYYHHMMCCWW"
    return "".join(rng.choice(alphabet) for _ in range(n))


def _assemble(blocks: list[tuple[str, str, str]]) -> tuple[str, list[Part]]:
    """Concatenate (name, kind, sequence) blocks and record each one's 1-based inclusive span."""
    protein, parts, pos = [], [], 1
    for name, kind, seq in blocks:
        if not seq:
            continue
        protein.append(seq)
        parts.append(Part(name, kind, pos, pos + len(seq) - 1))
        pos += len(seq)
    return "".join(protein), parts


def _pad_to_dilution(payload_len: int, dilution: float) -> int:
    """Residues of non-payload needed so the payload is `1 - dilution` of the chain."""
    if dilution <= 0:
        return 0
    return max(0, round(payload_len * dilution / (1 - dilution)))


def design(
    payload: str,
    accession: str,
    archetype: str,
    dilution: float,
    rng: random.Random,
    *,
    filler=random_protein,
) -> tuple[str, list[Part]]:
    """The protein chain for one construct, as blocks, with the payload's span recorded exactly.

    Padding is distributed according to the architecture: a tagged fusion carries it as terminal tags and
    linkers, a cassette as a 2A-linked passenger gene, a motifless fusion as a directly joined partner
    with no join to find. The last of those is the case section 15.5 showed motif rules are blind to, and
    it exists here so that blindness can be measured at n > 2 instead of argued from two examples.

    `filler` is the source of every diluting block: `(rng, n) -> str`, defaulting to `random_protein`,
    which is condition A of this project's rebuild. the working record is why it is a parameter: random
    sequence has no family in the catalogue, so it never competes for rank, and the tier's whole-to-oracle
    delta is +0 at every holdout even at a dilution of 0.8. Swapping in real Swiss-Prot proteins
    (`pilot.fillers.RealProteinFiller`) is the one change that makes those 216 low-f constructs
    informative. It is injected rather than written in so that **A, B and C stay paired
    construct-for-construct with only the filler differing**: the padding budget, the architecture, the
    part boundaries and the ORF length are all computed before the filler is consulted, so the query
    length is identical across conditions and the `m` term of the significance floor cannot differ
    between them.
    """
    if archetype not in ARCHETYPES:
        raise ValueError(f"unknown archetype {archetype!r}; expected one of {ARCHETYPES}")
    pad = _pad_to_dilution(len(payload), dilution)

    if archetype in ("single_cds", "backbone_heavy") or pad == 0:
        return _assemble([("payload", "payload", payload)])

    if archetype == "tagged_fusion":
        # Tags are small, so a high dilution needs a large carrier as well, which is what a real
        # GST- or MBP-tagged construct looks like (Addgene #187828 is 33% GST by length).
        n_tag = FLAG3 + GS_LINKER
        c_tag = GS_LINKER + HIS6 + NLS
        # The padding budget is everything that is not payload, so the fixed motifs come out of it rather
        # than being added on top: otherwise the realised dilution overshoots its target by the length
        # of whatever motifs the architecture happens to require.
        fixed = len(n_tag) + len(c_tag) + len(TEV_SITE)
        carrier = pad - fixed
        if carrier < 0:  # too little padding for the full tag stack; drop the carrier and the TEV site
            return _assemble([
                ("3xFLAG+linker", "tag", n_tag[: max(1, pad)]),
                ("payload", "payload", payload),
                ("linker+His6+NLS", "tag", c_tag[: max(0, pad - len(n_tag))]),
            ])
        return _assemble([
            ("3xFLAG+linker", "tag", n_tag),
            ("carrier", "filler", filler(rng, carrier)),
            ("TEV site", "cleavage", TEV_SITE),
            ("payload", "payload", payload),
            ("linker+His6+NLS", "tag", c_tag),
        ])

    if archetype == "motifless_fusion":
        # A reporter-style partner joined directly: no tag, no linker, no cleavage site, nothing for a
        # motif scan to find. Addgene #17551 (FOXO1-EGFP) is the real instance.
        return _assemble([
            ("payload", "payload", payload),
            ("fusion partner", "filler", filler(rng, pad)),
        ])

    # multi_gene_cassette: payload - P2A - passenger [- T2A - passenger]. The 2A peptides are paid for
    # out of the padding budget, and a second passenger is only added when what remains would make a
    # plausible gene rather than a stub.
    if pad >= len(P2A) + len(T2A) + 120:
        body = pad - len(P2A) - len(T2A)
        first = body // 2
        return _assemble([
            ("payload", "payload", payload), ("P2A", "cleavage", P2A),
            ("passenger 1", "passenger", filler(rng, first)),
            ("T2A", "cleavage", T2A),
            ("passenger 2", "passenger", filler(rng, body - first)),
        ])
    first = pad - len(P2A)
    if first < 1:  # not enough padding to carry a 2A peptide and anything after it
        return _assemble([("payload", "payload", payload), ("P2A", "cleavage", P2A[:pad])])
    return _assemble([
        ("payload", "payload", payload), ("P2A", "cleavage", P2A),
        ("passenger 1", "passenger", filler(rng, first)),
    ])


def _encode_orf(protein: str, rng: random.Random) -> str:
    """In-frame stop + ATG + the chain + a stop codon.

    The leading stop is what makes this tier's coordinates exact. Without it the ORF caller walks back to
    any upstream in-frame ATG in the spacer and returns a longer chain than the one that was designed:
    correct behaviour on its part, observed at up to +15 residues, but it silently invalidates every
    recorded payload boundary. A stop immediately upstream fixes the ORF's start where it was put.

    A leading Met is encoded by the start codon rather than duplicated, so a chain beginning with M has
    exactly as many residues as the ORF the caller reports.
    """
    body = protein[1:] if protein[:1] == "M" else protein
    return rng.choice(STOPS) + "ATG" + reverse_translate(body, rng) + rng.choice(STOPS)


def build_construct(
    construct_id: str,
    payload: str,
    accession: str,
    archetype: str,
    dilution: float,
    markers: list[tuple[str, str]],
    rng: random.Random,
    backbone_pad_bp: int = 0,
    *,
    filler=random_protein,
) -> Construct:
    """Assemble a whole circular plasmid around one payload-bearing ORF.

    `markers` are real marker proteins (label, sequence) supplied by the caller: beta-lactamase and the
    like, pulled from Swiss-Prot. Real ones matter: the benchmark should make the pipeline exclude an
    actual resistance gene via `autolabel.is_vector_part`, not a random sequence standing in for one.

    Only the markers get CDS features. The payload is left unannotated, as it is in real deposits.
    """
    protein, parts = design(payload, accession, archetype, dilution, rng, filler=filler)
    payload_part = next(p for p in parts if p.kind == "payload")

    pieces: list[str] = []
    marker_spans: list[tuple[str, int, int]] = []

    def add(seq: str) -> int:
        start = sum(len(p) for p in pieces)
        pieces.append(seq)
        return start

    add(random_spacer(rng, rng.randint(250, 600)))  # promoter-like spacer
    orf_start = add(_encode_orf(protein, rng)) + 3  # skip the guard stop; the ORF begins at its ATG
    add(random_spacer(rng, rng.randint(150, 400)))  # terminator-like spacer
    for label, mseq in markers:
        s = add(_encode_orf(mseq, rng))
        marker_spans.append((label, s, s + (len(mseq) + 1) * 3))
        add(random_spacer(rng, rng.randint(150, 400)))
    if backbone_pad_bp > 0:
        add(random_spacer(rng, backbone_pad_bp))

    dna = "".join(pieces)
    orf_aa = len(protein)
    return Construct(
        construct_id=construct_id, archetype=archetype, dna=dna, payload_accession=accession,
        payload_aa=len(payload), orf_start_nt=orf_start, orf_aa=orf_aa,
        payload_start_aa=payload_part.start_aa, payload_end_aa=payload_part.end_aa,
        dilution=round(1.0 - len(payload) / orf_aa, 4), parts=parts, markers=marker_spans,
    )


def random_spacer(rng: random.Random, n: int) -> str:
    """Non-coding filler: plain random DNA.

    An earlier version built spacers from sense codons only, reasoning that a spacer should not terminate
    a neighbouring reading frame. That was backwards. Every coded region here already carries its own stop
    codon, so stop-free spacers protect nothing: they only let an upstream ATG read straight through the
    spacer and the marker beyond it, which turned 545 of 1,008 annotated markers from `exact` into
    `contained` and manufactured run-on ORFs no real plasmid would have. Random DNA stops roughly every 20
    codons per frame, as intergenic sequence does.
    """
    return "".join(rng.choice("ACGT") for _ in range(n))


def write_genbank(construct: Construct, path, molecule_name: str | None = None) -> None:
    """Write a circular GenBank record annotating the markers only."""
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqFeature import FeatureLocation, SeqFeature
    from Bio.SeqRecord import SeqRecord

    rec = SeqRecord(Seq(construct.dna), id=(molecule_name or construct.construct_id)[:15],
                    name=(molecule_name or construct.construct_id)[:15],
                    description=f"synthetic construct {construct.archetype}  "
                                f"dilution={construct.dilution}")
    rec.annotations.update(molecule_type="DNA", topology="circular", data_file_division="SYN",
                           organism="synthetic construct")
    for label, start, end in construct.markers:
        rec.features.append(SeqFeature(FeatureLocation(start, end, strand=1), type="CDS",
                                       qualifiers={"label": [label], "note": ["selection marker"]}))
    SeqIO.write(rec, path, "genbank")
