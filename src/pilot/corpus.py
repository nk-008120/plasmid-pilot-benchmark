"""Screen a bulk-fetched set of GenBank records down to a usable, non-redundant corpus.

A query against NCBI returns what the query asked for, not a benchmark. Three things have to be removed
before several thousand records become a few hundred constructs worth measuring on:

1. **Records that cannot carry the experiment**: no CDS annotation, out of the size window, or heavy
   with ambiguous bases.
2. **Exact duplicates.** The same vector is deposited many times under different accessions.
3. **Near-duplicates.** More damaging than exact ones and invisible to a hash: pEGFP-N1 and pEGFP-C1
   differ by a few hundred bases out of 4,700. A benchmark where a third of the entries are the same
   backbone reports the retrieval accuracy of that backbone, with a confidence interval that lies.

Near-duplicate detection uses a MinHash sketch over canonical k-mers. Sketching is used rather than
alignment because all-pairs alignment on a few thousand plasmids is hours of work to answer a question
that a 256-value sketch answers in seconds, and because the threshold here is deliberately high: the
aim is to drop reruns of the same construct, not to thin out genuine homologs.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

MIN_BP, MAX_BP = 2_000, 25_000
MAX_AMBIGUOUS = 0.01
SKETCH_K = 16
SKETCH_SIZE = 256
NEAR_DUP_JACCARD = 0.90

_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


@dataclass
class Record:
    """One candidate construct, with everything the screen needs and nothing else."""

    accession: str
    filename: str
    seq: str
    n_cds: int
    topology: str
    description: str
    sketch: frozenset[int] = field(default_factory=frozenset)

    @property
    def length(self) -> int:
        return len(self.seq)

    @property
    def ambiguous_fraction(self) -> float:
        if not self.seq:
            return 1.0
        return sum(1 for c in self.seq.upper() if c not in "ACGT") / len(self.seq)


def canonical_kmers(seq: str, k: int = SKETCH_K) -> set[str]:
    """K-mers with each one collapsed to the lexicographically smaller of itself and its reverse
    complement, so a plasmid deposited in the opposite orientation sketches identically."""
    s = seq.upper()
    rc = s.translate(_COMPLEMENT)[::-1]
    out = set()
    n = len(s)
    for i in range(n - k + 1):
        a = s[i : i + k]
        if "N" in a:
            continue
        b = rc[n - k - i : n - i]
        out.add(min(a, b))
    return out


def sketch(seq: str, k: int = SKETCH_K, size: int = SKETCH_SIZE) -> frozenset[int]:
    """Bottom-`size` MinHash sketch. Jaccard between two sketches estimates k-mer Jaccard."""
    hashes = sorted(
        int.from_bytes(hashlib.blake2b(km.encode(), digest_size=8).digest(), "big")
        for km in canonical_kmers(seq, k)
    )
    return frozenset(hashes[:size])


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def sequence_hash(seq: str) -> str:
    """Orientation-independent exact-identity hash: a plasmid and its reverse complement agree."""
    s = seq.upper()
    rc = s.translate(_COMPLEMENT)[::-1]
    return hashlib.blake2b(min(s, rc).encode(), digest_size=16).hexdigest()


@dataclass
class ScreenResult:
    kept: list[Record]
    rejected: dict[str, list[str]]  # reason -> accessions

    @property
    def counts(self) -> dict[str, int]:
        d = {"kept": len(self.kept)}
        d.update({k: len(v) for k, v in self.rejected.items()})
        return d


def screen(
    records: list[Record],
    min_bp: int = MIN_BP,
    max_bp: int = MAX_BP,
    require_cds: bool = True,
    max_ambiguous: float = MAX_AMBIGUOUS,
    near_dup_threshold: float = NEAR_DUP_JACCARD,
) -> ScreenResult:
    """Apply every filter in order, recording why each record was dropped.

    Order matters for cost, not for outcome: the cheap scalar tests run before any sketching, so the
    expensive comparison only ever sees records that have already earned it. Records are sorted by CDS
    count before deduplication so that when two near-duplicates are found, the better-annotated one is
    the survivor rather than whichever happened to be fetched first.
    """
    rejected: dict[str, list[str]] = {
        "too_short": [], "too_long": [], "no_cds": [], "ambiguous_bases": [],
        "exact_duplicate": [], "near_duplicate": [],
    }
    passed: list[Record] = []
    for r in records:
        if r.length < min_bp:
            rejected["too_short"].append(r.accession)
        elif r.length > max_bp:
            rejected["too_long"].append(r.accession)
        elif require_cds and r.n_cds == 0:
            rejected["no_cds"].append(r.accession)
        elif r.ambiguous_fraction > max_ambiguous:
            rejected["ambiguous_bases"].append(r.accession)
        else:
            passed.append(r)

    passed.sort(key=lambda r: (-r.n_cds, r.accession))  # best-annotated copy wins a duplicate pair
    seen: dict[str, str] = {}
    unique: list[Record] = []
    for r in passed:
        h = sequence_hash(r.seq)
        if h in seen:
            rejected["exact_duplicate"].append(r.accession)
        else:
            seen[h] = r.accession
            unique.append(r)

    kept: list[Record] = []
    for r in unique:
        s = r.sketch or sketch(r.seq)
        r.sketch = s
        if any(jaccard(s, k.sketch) >= near_dup_threshold for k in kept):
            rejected["near_duplicate"].append(r.accession)
        else:
            kept.append(r)
    return ScreenResult(kept, rejected)


def stratify(records: list[Record], archetype_of, per_archetype: int, seed: int = 0) -> list[Record]:
    """Take up to `per_archetype` records of each archetype, deterministically.

    Stratifying rather than taking the first N matters because NCBI returns records in an order that
    correlates with submission date and therefore with construct fashion: an unstratified prefix would be
    dominated by whatever vector family was popular in whichever year the query happens to start in.
    """
    import random

    buckets: dict[str, list[Record]] = {}
    for r in records:
        buckets.setdefault(archetype_of(r), []).append(r)
    rng = random.Random(seed)
    out: list[Record] = []
    for name in sorted(buckets):
        group = sorted(buckets[name], key=lambda r: r.accession)
        rng.shuffle(group)
        out.extend(group[:per_archetype])
    return sorted(out, key=lambda r: r.accession)
