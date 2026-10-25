"""Reference-side mean-pool dilution: how much of the catalogue is too long to be one vector.

**Why this exists.** This project's thesis is that a mean-pooled embedding mixes a *construct*, so the
query is diluted and decomposition repairs it. this project found the mirror image and it inverts the
direction: `addgene_171782_pET28_MBP_sTEV` sits at rank **12,136 of 40,750** on the `oracle` arm at the
**100% holdout**: the clean payload span as the query, the answer present at 100% identity, nothing
masked. Its payload P04517 is the TEV genome polyprotein at 3,054 aa and the construct carries only its
~240 aa protease domain. The *reference* is the diluted thing, and no decomposition of the query can
repair it.

That finding has n = 1. This module turns it into a distribution: dilution is a property of whichever
side is mean-pooled, and the catalogue had never been examined for it.

**The quantity, and why it is the mirror of `f`.** the working record stratifies every benchmark on the
query-side payload fraction `f = oracle_span / called_ORF`. The reference-side analogue is
`f_ref = aligned_columns / subject_length`: how much of a catalogue entry the correct answer actually
occupies. Both are label-using descriptions of a benchmark item and neither can enter a screening rule.

**`f_ref` is measured, not proxied, for the homolog targets.** `homologs.csv` stores `coverage` as
aligned columns over *payload* length, capped at 1.0 (`homology.coverage_of`), and `length` as the
*subject* length. So `aligned ~ coverage * payload_aa` and `f_ref = min(aligned, L) / L`. Where coverage
saturates the cap, `f_ref` is a **lower bound** rather than an estimate, and the count of such rows is
reported on every run: the cap is a limit and limits get binding counts.

**For the catalogue at large there is no query, so a proxy is needed, and the proxy has a blind spot
that happens to be the motivating case.** `family_length_ratio` is an entry's length over the median
length of its own homolog family, on the reasoning that domain architecture is conserved within a family
so a member twice the family median carries material its relatives do not. It detects a fusion inside a
family of single-domain proteins. It **cannot** detect a family that is uniformly multi-domain: P09866
sits at ratio 0.993 against a family median of 3,411 aa, and P04517's family median is 3,084 aa. Both are
polyproteins in families of polyproteins, so the proxy calls the two worst entries in the catalogue
clean. That is stated here rather than discovered later; see `report.md` of any run.

`repeat_fraction` is the second proxy and it is a *disqualifier*, not a detector: a long entry can be
long because it repeats rather than because it carries several domains, and a repeat-driven entry is not
evidence for the mechanism. It is the duplicate fraction among an entry's k-mers, which costs one pass.

Nothing here downloads, aligns or calls a domain. A crisp proxy with stated limits beats a fancy one.
"""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field

# The project's own payload-fraction strata, reused unchanged so a reference-side number
# is read against the same boundaries as a query-side one. They are boundaries of a description, never
# of a rule.
F_STRATA = (0.50, 0.70, 0.95)

# Lengths that matter to a mean-pooled embedding arm, in the order they bind.
#   1022 - ESM-2's position budget at the setting this project's truncation row records (3,889 of 40,750).
#   2048 - the current `--max-length`; its binding count is recorded nowhere in CONTEXT.
#   5000 - `homology.MAX_LEN`, the catalogue's own window; nothing should exceed it.
LENGTH_THRESHOLDS = (1022, 2048, 5000)

REPEAT_K = 6
# Above this an entry's length is substantially repeat, so "long" is not evidence of several domains.
# Chosen as an order-of-magnitude line and not swept: it is used to *exclude* entries from a claim, so a
# loose value costs the claim nothing and a tuned one would be chosen on the data it is tested against
#.
REPEAT_DISQUALIFY = 0.30


def repeat_fraction(seq: str, k: int = REPEAT_K) -> float:
    """Fraction of an entry's k-mer positions that are repeats of an earlier k-mer.

    0.0 for a sequence whose every k-mer is unique, approaching 1.0 for a pure tandem repeat. Returns
    0.0 for a sequence shorter than k, which is the honest answer: there is no evidence of repetition.
    """
    n = len(seq) - k + 1
    if n <= 0:
        return 0.0
    distinct = len({seq[i:i + k] for i in range(n)})
    return 1.0 - distinct / n


def family_length_ratio(subject_len: int, family_lengths: list[int]) -> float | None:
    """`subject_len / median(family_lengths)`, or None when the family cannot support the comparison.

    None, not 1.0, for a family of one: a singleton's ratio is 1.0 by construction and carries no
    information, and returning the number would let an uninformative case be counted as evidence that
    the entry is clean. The count of Nones is a binding count and is reported.
    """
    if len(family_lengths) < 2:
        return None
    med = statistics.median(family_lengths)
    return subject_len / med if med > 0 else None


def reference_payload_fraction(coverage: float, payload_aa: int,
                               subject_len: int) -> tuple[float, bool]:
    """`(f_ref, is_lower_bound)` for one `homologs.csv` row.

    `coverage` is aligned columns over *payload* length capped at 1.0, so aligned columns are recovered
    as `coverage * payload_aa` and that recovery is exact only while the cap is not reached. At
    `coverage >= 1.0` the true aligned count is at least `payload_aa`, so the returned `f_ref` is a
    lower bound and the flag says so. Both sides of the ratio are computed the same way: the
    coverage-denominator bug is exactly the failure to do that.
    """
    if subject_len <= 0:
        raise ValueError("subject_len must be positive")
    aligned = min(coverage * payload_aa, float(subject_len))
    return aligned / subject_len, coverage >= 1.0


def stratify(values: list[float], bounds: tuple[float, ...] = F_STRATA) -> dict[str, int]:
    """Counts in `[0, b0) [b0, b1)... [b_last, inf)`, keyed by a printable interval.

    Ascending bounds are required rather than sorted silently: a caller passing them out of order has a
    different bug than a mis-labelled stratum, and sorting would hide it.
    """
    if list(bounds) != sorted(bounds):
        raise ValueError("bounds must be ascending")
    keys = ([f"<{bounds[0]:.2f}"]
            + [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(bounds, bounds[1:])]
            + [f">={bounds[-1]:.2f}"])
    out = {k: 0 for k in keys}
    for v in values:
        idx = sum(1 for b in bounds if v >= b)
        out[keys[idx]] += 1
    return out


@dataclass
class LengthCensus:
    """A catalogue's length distribution with every limit's binding count attached."""

    n_entries: int
    n_residues: int
    min_len: int
    max_len: int
    median_len: float
    mean_len: float
    # threshold -> (entries above, residues in them)
    above: dict[int, tuple[int, int]] = field(default_factory=dict)
    # entries outside the project's own reference window, which the catalogue's reader drops
    n_below_window: int = 0
    n_above_window: int = 0
    residues_in_window: int = 0

    def fraction_entries(self, threshold: int) -> float:
        return self.above[threshold][0] / self.n_entries if self.n_entries else 0.0

    def fraction_residues(self, threshold: int) -> float:
        return self.above[threshold][1] / self.n_residues if self.n_residues else 0.0


def length_census(lengths: list[int], thresholds: tuple[int, ...] = LENGTH_THRESHOLDS,
                  window: tuple[int, int] | None = None) -> LengthCensus:
    """Entry and residue counts above each threshold, plus the window's own binding count.

    The residue share is reported beside the entry share because they are not the same statement and the
    gap between them is the point: long entries are a small minority of a catalogue and a large minority
    of its residues, so "1% of entries" and "7% of the thing being embedded" describe one fact.
    """
    if not lengths:
        raise ValueError("no lengths")
    total = sum(lengths)
    cen = LengthCensus(
        n_entries=len(lengths), n_residues=total, min_len=min(lengths), max_len=max(lengths),
        median_len=statistics.median(lengths), mean_len=statistics.fmean(lengths))
    for t in thresholds:
        above = [L for L in lengths if L > t]
        cen.above[t] = (len(above), sum(above))
    if window is not None:
        lo, hi = window
        cen.n_below_window = sum(1 for L in lengths if L < lo)
        cen.n_above_window = sum(1 for L in lengths if L > hi)
        cen.residues_in_window = sum(L for L in lengths if lo <= L <= hi)
    return cen


def macro_micro(per_row: list[tuple[str, float]], predicate) -> tuple[float, float, int]:
    """`(micro, macro, n_groups)` for a predicate over `(group, value)` rows.

    this project's working rules: catalogue rows are not independent trials, 51,093 homolog rows carry 173 payloads, one
    with 4,059 of them, so a row average is a statement about the sampling frame and a per-payload
    average is a statement about payloads. Neither is reported alone.
    """
    if not per_row:
        return 0.0, 0.0, 0
    micro = sum(1 for _, v in per_row if predicate(v)) / len(per_row)
    by_group: dict[str, list[float]] = {}
    for g, v in per_row:
        by_group.setdefault(g, []).append(v)
    macro = statistics.fmean(
        sum(1 for v in vs if predicate(v)) / len(vs) for vs in by_group.values())
    return micro, macro, len(by_group)


def family_lengths_from_rows(rows, length_of) -> dict[str, list[int]]:
    """`{payload: [subject lengths]}` from `homologs.csv`-shaped rows.

    Built from the homolog table rather than from the FASTA's `homolog_of:` headers, because a payload's
    own entry is written into the catalogue under its Swiss-Prot header and carries no tag, so a
    header-built family omits the one member whose length the question is about. P04517 is exactly that
    case: 34 tagged relatives and no self.
    """
    out: dict[str, list[int]] = {}
    for r in rows:
        out.setdefault(r["payload_accession"], []).append(length_of(r))
    return out


def summarise_repeats(fractions: list[float],
                      disqualify: float = REPEAT_DISQUALIFY) -> dict[str, float | int]:
    """Repeat-fraction distribution and how many entries the disqualifier removes."""
    if not fractions:
        return {"n": 0, "n_disqualified": 0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {"n": len(fractions),
            "n_disqualified": sum(1 for f in fractions if f > disqualify),
            "median": statistics.median(fractions),
            "p90": sorted(fractions)[int(0.9 * (len(fractions) - 1))],
            "max": max(fractions)}


def counter_summary(c: Counter) -> dict[str, int]:
    """A Counter as a plain dict with deterministic ordering, for JSON."""
    return {str(k): v for k, v in sorted(c.items())}
