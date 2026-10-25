"""Real-protein fillers for the synthetic tier, and the one rule that keeps the benchmark from inverting.

the working record identified the defect this module fixes. Every diluting block in `synthconstruct.design`
came from `random_protein()`, and random sequence **has no family in the catalogue**, so it contributes no
competing hit and the tier's whole-to-oracle delta is +0 at every holdout even at a dilution of 0.8.
the working record counted the cost: 216 of 336 constructs sit below f = 0.70, against the corpus's 15, and all
216 are uninformative for that one reason. The tier is correctly positioned with the active ingredient
removed.

`RealProteinFiller` is that ingredient. It is a `(rng, n) -> str` callable, so it drops into
`design(..., filler=...)` without touching `random_protein` or any existing expected value.

**The correctness requirement, and it is the only one here.** A filler must never be a relative of its own
construct's payload. If it were, the filler's family would sit in the catalogue as a *true* homolog of the
answer, retrieving it would score as success, and the benchmark would measure the opposite of what it
claims: a construct would look easier the more diluted it got. Everything else in this module is a
measurement choice that can be argued either way; this one cannot. It is enforced in `eligible_pool` by
removing, for each payload, every accession that appears anywhere under that payload in
`results/gate_all173/homologs.csv` **at any identity**, plus every payload in that file, plus every
payload used by the tier itself.

That rule is weaker than the requirement and the gap is stated rather than hidden: `homologs.csv` is the
E <= 1e-3 **gated** file, so a sub-threshold relative of the payload can still be drawn.
The residual is measurable once each filler's own relatives have been found, the intersection of a
filler's catalogue relatives with its payload's, and `scripts/32_filler_homologs.py` reports it. The
inversion risk is therefore a number, not an assurance.

**Two measurement choices, declared where they are made.** A filler shorter than the padding budget is
tiled and a longer one truncated; the counts are reported, per this project's "report binding, not settings".
And one accession is fixed per (payload, slot), constant across archetypes and dilution levels, so that
along the dilution axis only the *amount* of filler changes and never its identity.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .autolabel import Hit, is_vector_part

#: Slots per construct. `design` emits at most two filler blocks (a cassette's two passengers); every
#: other architecture emits one. Fixed rather than discovered so the slot assignment is reproducible.
SLOTS = 2


@dataclass(frozen=True)
class FillerDraw:
    """One filler block actually placed, recorded so condition B can mask it and the manifest can carry it."""

    accession: str
    slot: int
    requested_aa: int
    source_aa: int
    mode: str  # "exact" | "tiled" | "truncated"


def load_homolog_sets(path: Path) -> tuple[dict[str, set[str]], set[str]]:
    """`payload -> {every accession recorded as its homolog, at any identity}`, and the payload set.

    Identity is deliberately ignored. A filler must not be a relative of the payload at *any* holdout
    threshold, so the strictest reading of the file is the right one: presence, not degree.
    """
    rel: dict[str, set[str]] = defaultdict(set)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            rel[r["payload_accession"]].add(r["homolog_accession"])
    return dict(rel), set(rel)


def is_eligible_filler(accession: str, description: str, *, forbidden: set[str]) -> bool:
    """Would this Swiss-Prot entry be an acceptable filler at all, before any payload-specific check?

    `is_vector_part` classifies on the description, not the accession, which is why `iter_swissprot` has
    to be read with `with_description=True`. Excluding vector parts here is the same filter the payload
    selection is documented to use, and it matters for a filler as much as for a payload: a filler that
    is beta-lactamase would put the construct's own resistance marker's family into the competition, so
    the arm would be measuring the backbone rather than the insert.
    """
    if accession in forbidden:
        return False
    hit = Hit(orf_id="", accession=accession, description=description,
              identity=0.0, coverage=0.0, bitscore=0.0, orf_aa=0)
    return not is_vector_part(hit)


def eligible_pool(
    entries: list[tuple[str, str, str]],
    *,
    payload_relatives: dict[str, set[str]],
    tier_payloads: set[str],
    catalogue_payloads: set[str],
    restrict_to: set[str] | None = None,
) -> tuple[list[tuple[str, str]], dict[str, set[str]]]:
    """The global filler pool, and the per-payload forbidden set applied on top of it.

    Returns `(pool, forbidden_by_payload)`. The pool is filtered once on description; the payload-specific
    exclusion is kept separate so that a draw can be audited as "excluded because it is a vector part"
    versus "excluded because it is this payload's relative", which are different failures.

    **`restrict_to` is what makes condition C mean what it says, and it was added after the first build
    showed the pool was wrong.** The requirement is "real fillers *with their families present in the
    catalogue*". The catalogue is not Swiss-Prot: it is the 40,750 gated homologs of this project's 173
    payloads, so a filler drawn uniformly from Swiss-Prot's 467,049 eligible entries has, with high
    probability, **no relative in the search database at all**: measured, 2 of 48 on the first build.
    Such a filler competes for rank exactly as little as `random_protein` does, and condition C would have
    reproduced condition A's null for an entirely new reason while looking like a real-protein tier.
    Passing the catalogue's own accessions here guarantees every filler has at least one database entry at
    100% identity, which is also what a real construct looks like: the GFP in a fusion is ~100%
    identical to a database entry (this project found iteration 1 claiming GFP at identity 0.98).
    """
    always = set(tier_payloads) | set(catalogue_payloads)
    pool = [(a, s) for a, s, d in entries
            if (restrict_to is None or a in restrict_to) and is_eligible_filler(a, d, forbidden=always)]
    forbidden = {p: (payload_relatives.get(p, set()) | always) for p in tier_payloads}
    return pool, forbidden


def fit_to_length(seq: str, n: int) -> tuple[str, str]:
    """`(sequence_of_exactly_n, mode)`: tile a short filler, truncate a long one.

    Tiling is the weaker of the two choices and its defence is specific: Smith-Waterman takes the best
    *single* local alignment, so an n-fold tandem repeat scores as one copy against the filler's family.
    Tiling therefore inflates the query's length without inflating the competing score, which is the
    direction that does not manufacture an effect. It is still artificial, so the counts are reported.
    """
    if n <= 0:
        return "", "exact"
    if len(seq) == n:
        return seq, "exact"
    if len(seq) > n:
        return seq[:n], "truncated"
    return (seq * (n // len(seq) + 1))[:n], "tiled"


class RealProteinFiller:
    """A `(rng, n) -> str` filler drawing real Swiss-Prot proteins, logging every block it places.

    Usage is two calls per construct:

        filler.begin(payload_accession)          # fixes this construct's slot assignment
        build_construct(..., filler=filler)      # design() consumes slot 0, then slot 1
        filler.last_draws                        # what was placed, for the manifest

    The slot assignment is drawn from `random.Random(f"{seed}:{payload}")` rather than from the caller's
    `rng`, which is what keeps a payload's fillers **identical across archetypes and dilution levels**:
    the caller's `rng` has been advanced a different number of times by the time each construct is built,
    so using it would give the same payload a different filler at every dilution and confound the amount
    of filler with its identity.
    """

    def __init__(self, pool: list[tuple[str, str]], forbidden_by_payload: dict[str, set[str]],
                 seed: int = 0, slots: int = SLOTS) -> None:
        if not pool:
            raise ValueError("empty filler pool")
        self.pool = pool
        self.forbidden_by_payload = forbidden_by_payload
        self.seed = seed
        self.slots = slots
        self.draws: list[FillerDraw] = []
        self.last_draws: list[FillerDraw] = []
        self._assignment: list[tuple[str, str]] = []
        self._slot = 0
        self._payload = ""
        self.pool_sizes: dict[str, int] = {}

    def assignment_for(self, payload_accession: str) -> list[tuple[str, str]]:
        """The `slots` (accession, sequence) fillers this payload always uses, in slot order."""
        bad = self.forbidden_by_payload.get(payload_accession, set())
        ok = [(a, s) for a, s in self.pool if a not in bad]
        self.pool_sizes[payload_accession] = len(ok)
        if len(ok) < self.slots:
            raise ValueError(f"only {len(ok)} eligible fillers for payload {payload_accession}")
        return random.Random(f"{self.seed}:{payload_accession}").sample(ok, self.slots)

    def begin(self, payload_accession: str) -> None:
        self._payload = payload_accession
        self._assignment = self.assignment_for(payload_accession)
        self._slot = 0
        self.last_draws = []

    def __call__(self, rng: random.Random, n: int) -> str:
        if not self._assignment:
            raise RuntimeError("RealProteinFiller.begin(payload_accession) must be called per construct")
        if n <= 0:
            return ""
        acc, seq = self._assignment[self._slot % self.slots]
        self._slot += 1
        out, mode = fit_to_length(seq, n)
        draw = FillerDraw(acc, self._slot - 1, n, len(seq), mode)
        self.draws.append(draw)
        self.last_draws.append(draw)
        return out
