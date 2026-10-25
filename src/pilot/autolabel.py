"""Assign a payload to a construct automatically, from its ORFs' Swiss-Prot hits.

Hand-curating 23 plasmids took hours; a few hundred cannot be done that way. This module replaces the
hand step, reading a repository page to learn what the depositor actually put in the vector, with a
rule over alignment hits, and makes the rule's error rate measurable against the hand-audited set.

The problem is not finding hits. It is deciding which hit is the *payload*. Every plasmid returns
confident hits for its resistance marker, its repressor and its purification tag, and all of those are
real proteins that are not what the construct is for. So the work is in `is_vector_part` (exclude what
the vector supplies) and `assign_payload` (choose among what is left).

Three selection rules are implemented rather than one, because which is right is an empirical question.
`scripts/13_validate_autolabel.py` scores all three against the 23 hand-audited constructs and reports
agreement, so the rule is chosen by measurement instead of by argument.

**Known limitation, stated up front:** a payload assigned this way is defined *by alignment*. Any benchmark
that keeps the payload in the catalogue at full identity is therefore partly circular: alignment cannot
fail to find what alignment chose. The twilight holdouts in `scripts/10_twilight.py` are where this set
carries real information, and the hand-audited 23 remain the anchor. See the working record.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Selection markers, repressors and replication functions. Real proteins, supplied by the vector, never
# the reason an order was placed. Matched against the Swiss-Prot description, which is far more stable
# across records than a GenBank feature label (`archetype.py` handles the label side).
MARKER_DESC = re.compile(
    r"beta-lactamase|aminoglycoside.{0,30}(phosphotransferase|transferase|adenylyltransferase)|"
    r"kanamycin|neomycin|chloramphenicol acetyltransferase|puromycin|blasticidin|bleomycin|"
    r"hygromycin|spectinomycin|streptomycin adenylyltransferase|tetracycline (repressor|resistance)|"
    r"lactose operon repressor|lac repressor|beta-galactosidase|"
    r"repressor of primer|regulatory protein rop|protein rop|"
    r"ccdb|cytotoxic protein ccd|levansucrase|"
    r"mobilization protein|relaxase|replication (initiation )?protein|plasmid replication|"
    r"transcriptional repressor tetr|antitoxin|toxin-antitoxin|"
    # Cryptic backbone ORFs. Swiss-Prot carries plasmid reading frames that were never characterised;
    # they score highly against vector sequence and are never what an order is for. Before this line,
    # "Uncharacterized 9.2 kDa protein" (P03853) was the chosen payload on four unrelated constructs.
    r"uncharacteri[sz]ed.{0,25}protein|hypothetical.{0,25}protein|"
    r"protein of unknown function|putative.{0,25}uncharacteri[sz]ed|"
    # Retroviral structural and accessory genes. In a 3rd-generation transfer vector these survive as
    # vestigial backbone, the cPPT/CTS element sits inside *pol*, and RRE inside *env*, so a hit to
    # one describes the delivery vehicle, not the cargo. Confirmed on Addgene #85400 and #51024.
    r"\bgag\b|gag-pol|pol polyprotein|protein nef|protein rev|protein tat|protein vif|protein vpr|"
    r"protein vpu|envelope glycoprotein gp160|reverse transcriptase/ribonuclease|"
    # Mobile-element and cloning machinery. The 23 Addgene deposits used a narrow set of markers; a bulk
    # NCBI corpus uses a much wider one, and these were the proteins left standing as "payloads" on 128
    # constructs that are in fact empty vectors. Each is something a vector supplies, not cargo.
    r"streptomycin.{0,20}(kinase|phosphotransferase)|aminocyclitol|"
    r"\bintegrase\b|recombinase|\btransposase\b|transposon|insertion sequence|resolvase|"
    r"large t antigen|small t antigen|\bt antigen\b|"
    r"lactose permease|lac operon|galactoside.{0,20}permease|galactoside o-acetyltransferase|"
    r"single-strand DNA-binding|origin-binding|primase",
    re.I,
)

# Purification and solubility carriers. Legitimately large proteins, but in a construct they are the
# handle, not the cargo: Addgene #187828's GST is the clearest case in the pilot set. Excluded from
# payload selection but recorded separately, because a construct whose *point* is one of these exists and
# would be mislabelled rather than merely skipped.
CARRIER_DESC = re.compile(
    r"glutathione s-transferase|maltose.{0,40}binding|maltodextrin.{0,40}binding|thioredoxin|"
    r"small ubiquitin-related modifier|ubiquitin-like protein smt3|"
    r"chitin.{0,20}binding|calmodulin-binding",
    re.I,
)

# Fluorescent proteins and luciferases. Not excluded by default: in a reporter construct the reporter IS
# the payload (Addgene #17448 is pLenti-CMV-GFP). Exposed so a caller can exclude them when building a
# set that deliberately avoids reporters.
REPORTER_DESC = re.compile(
    r"green fluorescent protein|fluorescent protein|luciferase|luciferin|photoprotein", re.I
)

ORF_ID = re.compile(r"^(?P<plasmid>.+?)\|orf(?P<n>\d+)\|(?P<strand>[+-])\|(?P<start>\d+)\.\.(?P<end>\d+)$")
SP_ID = re.compile(r"^(?:sp|tr)\|(?P<acc>[^|]+)\|")

RULES = ("best_bitscore", "longest_orf", "first_in_construct")


@dataclass(frozen=True)
class Hit:
    """One ORF-to-reference alignment, from DIAMOND or BLAST+ tabular output."""

    orf_id: str
    accession: str
    description: str
    identity: float  # 0-1, over the aligned region
    coverage: float  # aligned length / reference protein length
    bitscore: float
    orf_aa: int      # length of the query ORF
    ref_aa: int = 0  # length of the reference protein

    @property
    def dilution(self) -> float:
        """Fraction of the ORF that the reference protein does not account for.

        An ORF much longer than the protein it matches is carrying something else: a tag, a reporter, a
        second domain. This is the same quantity `decompose.dilution` measures against an oracle span,
        computed here from an alignment instead, so it is available without knowing the answer.
        """
        if not self.orf_aa or not self.ref_aa:
            return 0.0
        return round(max(0.0, 1.0 - self.ref_aa / self.orf_aa), 4)

    @property
    def plasmid_id(self) -> str:
        m = ORF_ID.match(self.orf_id)
        return m.group("plasmid") if m else self.orf_id.split("|")[0]

    @property
    def orf_start(self) -> int:
        m = ORF_ID.match(self.orf_id)
        return int(m.group("start")) if m else 0

    @property
    def is_marker(self) -> bool:
        return bool(MARKER_DESC.search(self.description))

    @property
    def is_carrier(self) -> bool:
        return bool(CARRIER_DESC.search(self.description))

    @property
    def is_reporter(self) -> bool:
        return bool(REPORTER_DESC.search(self.description))


@dataclass
class AutoLabel:
    plasmid_id: str
    payload_accession: str
    payload_description: str
    identity: float
    coverage: float
    bitscore: float
    orf_id: str
    rule: str
    n_candidates: int          # distinct non-vector accessions available
    n_excluded_marker: int
    n_excluded_carrier: int
    passed_gate: bool
    reason: str = ""


def accession_of(sseqid: str) -> str:
    """'sp|P42212|GFP_AEQVI' -> 'P42212'; anything else is returned unchanged."""
    m = SP_ID.match(sseqid)
    return m.group("acc") if m else sseqid


def is_vector_part(hit: Hit, exclude_reporters: bool = False) -> bool:
    """Is this hit something the vector supplies rather than the construct's cargo?"""
    return hit.is_marker or hit.is_carrier or (exclude_reporters and hit.is_reporter)


# Gate defaults, chosen by sweeping against the 23 hand-audited constructs (the working record 16.2). Agreement
# is 16/20 everywhere from identity 0.85 down to 0.50 and is unaffected by coverage between 0.80 and 0.60:
# a plateau, not a peak, so the exact value is not load-bearing. 0.85 sits at the plateau's upper edge:
# low enough to accept an engineered reporter against its reviewed parent (mCherry is 0.86 identical to
# DsRed, NanoLuc ~0.90 to the Oplophorus subunit), high enough to stay conservative on the corpus, where
# there are no hand labels to catch a false one. Coverage stays at 0.80 because it is what correctly
# rejects a payload matched over 8% of a polyprotein (#171782) or 41% of a half-called ORF (#40306).
MIN_IDENTITY = 0.85
MIN_COVERAGE = 0.80


def assign_payload(
    hits: list[Hit],
    rule: str = "best_bitscore",
    min_identity: float = MIN_IDENTITY,
    min_coverage: float = MIN_COVERAGE,
    exclude_reporters: bool = False,
) -> AutoLabel | None:
    """Pick one payload accession for a plasmid, or None when nothing clears the gate.

    The gate is deliberately strict. A set of several hundred constructs with a stated, measured
    confidence floor is worth more to a reviewer than a larger set whose labels are unexamined, and the
    constructs that fail the gate are not lost: they are reported as the fraction this method cannot
    label, which is itself a result.
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
    if not hits:
        return None
    pid = hits[0].plasmid_id
    markers = {h.accession for h in hits if h.is_marker}
    carriers = {h.accession for h in hits if h.is_carrier}
    good = [h for h in hits if not is_vector_part(h, exclude_reporters)]
    gated = [h for h in good if h.identity >= min_identity and h.coverage >= min_coverage]
    n_cand = len({h.accession for h in good})

    if not gated:
        return AutoLabel(pid, "", "", 0.0, 0.0, 0.0, "", rule, n_cand, len(markers), len(carriers),
                         False, "no non-vector hit cleared the identity/coverage gate")
    if rule == "best_bitscore":
        best = max(gated, key=lambda h: h.bitscore)
    elif rule == "longest_orf":
        best = max(gated, key=lambda h: (h.orf_aa, h.bitscore))
    else:  # first_in_construct, the 5'-most ORF, which in a cassette is usually the gene of interest
        best = min(gated, key=lambda h: (h.orf_start, -h.bitscore))
    return AutoLabel(pid, best.accession, best.description, best.identity, best.coverage, best.bitscore,
                     best.orf_id, rule, n_cand, len(markers), len(carriers), True)


def read_tabular(path, min_bitscore: float = 50.0) -> dict[str, list[Hit]]:
    """Parse DIAMOND / BLAST+ `--outfmt 6 qseqid sseqid pident length qlen slen bitscore evalue stitle`.

    Both tools emit the same 'outfmt 6' layout, so one parser serves either. Coverage is computed against
    the *subject* length: a 1,400 aa fusion ORF matching a 240 aa reference over 240 columns has covered
    that reference completely, and judging it by query length would score it as a 17% match.
    """
    out: dict[str, list[Hit]] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 8 or line.startswith("#"):
                continue
            try:
                pident, length, _qlen, slen, bits = float(f[2]), int(f[3]), int(f[4]), int(f[5]), float(f[6])
            except ValueError:
                continue
            if bits < min_bitscore or slen <= 0:
                continue
            hit = Hit(f[0], accession_of(f[1]), f[8] if len(f) > 8 else "",
                      round(pident / 100.0, 4), round(min(length / slen, 1.0), 4), bits, _qlen, slen)
            out.setdefault(hit.plasmid_id, []).append(hit)
    return out
