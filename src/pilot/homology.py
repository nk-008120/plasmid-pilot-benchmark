"""Find each payload's homologs in Swiss-Prot, so the benchmark can be made hard on purpose.

The Step 2 benchmark put each payload *into* the reference set, so the correct answer was present at ~100%
identity and finding it was string matching. Exact Smith-Waterman scores 19/20 on it, which is what any
benchmark of that shape should produce and is not evidence about screening.

The regime the proposal is actually about is the **twilight zone**: a submitted sequence whose nearest known
relative is 20-35% identical. To build that regime from real data, each payload is aligned against all of
Swiss-Prot, its homologs are recorded with their percent identity, and the benchmark is then run with a
**holdout threshold**: every entry more than X% identical to the payload is removed from the reference set,
and the task becomes "find a true homolog of this protein when nothing closer than X% is available".

Sweeping X from 90% down to 25% turns one saturated number into a degradation curve, which is the shape the
proposal's claim actually has.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

MIN_LEN, MAX_LEN = 50, 5000

# Karlin-Altschul parameters for BLOSUM62 with affine gaps 11/1, as published by NCBI for BLASTP.
LAMBDA, K_PARAM = 0.267, 0.041


def bit_score(raw: float) -> float:
    """`(lambda * S - ln K) / ln 2`.

    Included to be explicit that **it cannot fix a length bias**: for a fixed scoring system this is an
    affine function of the raw score, so ranking by bit score reproduces ranking by raw score exactly.
    Only the E-value introduces length, through its `K * m * n` term.
    """
    import math  # noqa: PLC0415

    return (LAMBDA * raw - math.log(K_PARAM)) / math.log(2)


def log_evalue(raw: float, query_len: int, subject_len: int) -> float:
    """`ln E` for one query-subject pair. **For ranking within one query only, never for gating.**

    `E = K * m * n * exp(-lambda * S)` with `n` the subject length, so `ln E = ln(K m n) - lambda S`.
    The `n` term is what makes this useful for ordering: a longer subject accumulates a higher raw score
    by chance, so two hits with equal raw score are not equally surprising.

    What it is **not** is a significance statement, because it carries no search-space term. Comparing
    one pair in isolation ignores that the hit was the best of half a million tries. A threshold applied
    to this number would mean something different for every database, and would not change at all when
    the database grew, which is the opposite of what an E-value is for. Use `log_evalue_search` to gate.
    """
    import math  # noqa: PLC0415

    return math.log(K_PARAM) + math.log(max(query_len, 1)) + math.log(max(subject_len, 1)) - LAMBDA * raw


def log_evalue_search(raw: float, query_len: int, db_residues: int) -> float:
    """`ln E` against the whole search space. **This is the one to gate on.**

    BLASTP's convention: the search space is the query length times the summed residue count of the
    database actually searched, so `E = K * m * N * exp(-lambda * S)`. Doubling the database doubles E,
    which is the property a significance threshold needs and the pairwise form lacks.

    No effective-length correction is applied (BLAST subtracts the expected HSP length from `m` and from
    each sequence). **This is not uniformly conservative and the asymmetry is worth stating.** The
    expected HSP length is about `ln(K m N) / lambda`, which against this database is 79 to 90 residues,
    so the effective query length falls 40.6% for a 196 aa payload and 2.9% for a 3,054 aa one. Omitting
    the correction uses the larger `m`, making `ln E` larger and the gate stricter, by 0.226 log10 units
    at 196 aa against 0.013 at 3,054 aa. That is a small length bias in the *same direction* as the one
    this gate exists to remove, and it falls hardest on exactly the short payloads the gate eliminates.

    It is left out anyway, for two reasons that were checked rather than assumed: 0.23 log10 against a
    plateau five orders of magnitude wide (E <= 1 to E <= 1e-5 give identical results at every holdout)
    cannot change any outcome, and omitting it keeps the E-value reproducible from quantities already
    recorded in `homologs.csv`, which is what lets the threshold be swept post-hoc in seconds instead of
    by re-searching. Revisit if the plateau ever narrows.

    Why a third gate at all: `min_coverage` is a fraction of payload length, so it asks for 78 aligned
    columns from a 196 aa payload and 1,222 from a 3,054 aa one. Statistical significance does not scale
    that way. 78 columns at 20% identity is about 16 identical residues and is not distinguishable from
    chance, while 1,222 columns at 20% identity is overwhelming. Identity and coverage stay because
    coverage still does real work against low-complexity local matches; the E-value is what makes the
    stringency comparable across the length range.
    """
    import math  # noqa: PLC0415

    return math.log(K_PARAM) + math.log(max(query_len, 1)) + math.log(max(db_residues, 1)) - LAMBDA * raw


def score_floor(max_evalue: float, query_len: int, db_residues: int) -> float:
    """The raw Smith-Waterman score at which `E == max_evalue`. The exact inverse of `log_evalue_search`.

    `E = K m N exp(-lambda S)` has no free variable but `S` once the query and database are fixed, so a
    significance gate is a score floor:

        S_min = (ln(K m N) + ln(1 / max_evalue)) / lambda

    Query length enters only inside a logarithm, so the floor is nearly flat across the length range: at
    `max_evalue = 1e-3` against a 207 M residue database it is 105.4 for a 196 aa payload and 115.7 for a
    3,054 aa one.

    This is what makes a candidate cap unnecessary. Selecting every subject at or above the floor cannot
    truncate, whereas ranking and taking the top N silently dropped 2,975 qualifying homologs for 5 of 16
    audited payloads at N = 400. A floor derived from the identity and coverage gates would have been
    useless, those admit alignments whose expected score is negative, so the implied floor is below
    zero and excludes nothing, but a floor derived from a significance gate is exact.
    """
    import math  # noqa: PLC0415

    return (math.log(K_PARAM) + math.log(max(query_len, 1)) + math.log(max(db_residues, 1))
            - math.log(max_evalue)) / LAMBDA


@dataclass(frozen=True)
class Homolog:
    accession: str
    score: float
    identity: float  # identical columns / aligned columns
    coverage: float  # aligned columns / length of the payload
    length: int


def iter_swissprot(path: Path, min_len: int = MIN_LEN, max_len: int = MAX_LEN,
                   with_description: bool = False):
    """(accession, sequence) from a UniProt FASTA, plain or gzipped.

    With `with_description=True` the yield becomes `(accession, sequence, description)` instead. The
    description is needed by `autolabel.is_vector_part`, which classifies on it rather than on the
    accession, and the filler selection has to apply that same filter. Default is off so
    that every existing two-tuple caller is unchanged.

    Length-filtered on the way through. The **lower** bound stays at 50: below that an alignment has too
    few columns to support a meaningful percent-identity estimate, so a short peptide would enter the
    catalogue with a confident-looking identity that means nothing.

    The **upper** bound was 2,000, which was wrong in a way that was invisible until something depended
    on it. The stated reason, that titins and polyproteins win raw-score rankings on length rather than
    homology, is real, but the fix for a length-biased score is not to delete the long proteins from
    biology. Polyproteins are exactly what several of these constructs encode: P04517 (3,054 aa) and
    P09866 (3,387 aa) are payload accessions in this project's own manifests, and at 2,000 they could
    never be retrieved, never appear as a segment's answer, and never be scored against. The bound is now
    5,000, which admits every payload in the manifests and costs about 4% more residues to search.
    `pilot.reference.build` used 4,000 for the same job, so the two filters also disagreed.
    """
    op = gzip.open if path.suffix == ".gz" else open
    acc, desc, cur = None, "", []
    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                seq = "".join(cur)
                if acc and min_len <= len(seq) <= max_len:
                    yield (acc, seq, desc) if with_description else (acc, seq)
                head = line[1:].rstrip("\n")
                acc = head.split("|")[1] if head.count("|") >= 2 else head.split()[0]
                # UniProt FASTA: >db|ACC|ENTRY_NAME Description OS=...: the description is everything
                # after the first space, which is what `autolabel`'s regexes are written against.
                desc = head.split(" ", 1)[1] if " " in head else ""
                cur = []
            else:
                cur.append(line.strip())
    seq = "".join(cur)
    if acc and min_len <= len(seq) <= max_len:
        yield (acc, seq, desc) if with_description else (acc, seq)


def find_homologs(
    payload: str,
    entries: list[tuple[str, str]],
    top_n: int = 400,
    min_identity: float = 0.20,
    min_coverage: float = 0.40,
    gap_open: int = 11,
    gap_extend: int = 1,
    threads: int = 1,
    rank_by: str = "evalue",
    stats: dict | None = None,
    max_evalue: float | None = None,
) -> list[Homolog]:
    """Smith-Waterman `payload` against every entry; return the real homologs with percent identity.

    Two passes on purpose. The first is score-only over the whole database, which is fast; the second runs
    the traceback for just the top `top_n` by score, because the traceback is the expensive part and
    identity is only needed where it could matter. `min_coverage` is what keeps a short, locally-similar
    stretch, a coiled coil, a low-complexity run, from being recorded as a homolog of a 1,368 aa
    protein at high identity over 30 residues.

    `threads` splits the score-only pass across OS threads rather than processes. parasail's kernels
    release the GIL, so this is real parallelism (measured 2.0x on two threads), and threads share the one
    copy of Swiss-Prot. That matters more than it sounds: 559,680 entries is about 280 MB, and processes
    would need a copy each. The slices are disjoint and each writes only its own range of `scores`, so no
    lock is needed. The traceback pass stays serial because it runs over `top_n` hits, not the database.
    """
    import math

    import numpy as np
    import parasail

    m = parasail.blosum62
    n_entries = len(entries)
    scores = np.zeros(n_entries, dtype=np.float32)

    def score_slice(lo: int, hi: int) -> None:
        f = parasail.sw_striped_sat
        for i in range(lo, hi):
            scores[i] = f(payload, entries[i][1], gap_open, gap_extend, m).score

    if threads <= 1 or n_entries < 2 * threads:
        score_slice(0, n_entries)
    else:
        import threading  # noqa: PLC0415 (only needed for the parallel path)

        bounds = [(n_entries * k // threads, n_entries * (k + 1) // threads) for k in range(threads)]
        workers = [threading.Thread(target=score_slice, args=b, daemon=True) for b in bounds]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
    # The search space is the whole database that was scored, not the candidates that reached traceback,
    # so the E-value does not change with scan depth.
    db_residues = int(sum(len(sq) for _, sq in entries))
    ln_max_e = math.log(max_evalue) if max_evalue is not None else None

    # With `max_evalue` set there is no cap and no ranking: the gate IS a score floor, so the candidates
    # are exactly those at or above it.
    #
    # For a fixed query and database, `E = K m N exp(-lambda S)` depends on nothing but the raw score, so
    # `E <= max_evalue` inverts to `S >= score_floor(...)`. Query length enters only logarithmically, so
    # that floor is 105.4 for a 196 aa payload and 115.7 for a 3,054 aa one against this database: nearly
    # flat across the whole range. Selecting on it makes truncation structurally impossible rather than
    # something to check for, and the count above the floor becomes a measurement to report rather than a
    # limit to defend.
    #
    # It also removes a mismatch that no amount of depth would have fixed. Ranking by the pairwise
    # E-value adds `ln(subject_length)`, worth up to 17.2 score points across the 50-5,000 aa window, so a
    # long subject with a qualifying score was pushed below the cut-off while a short one with a worse
    # score survived. The ranking and the gate were ordering by different quantities.
    floor = None
    if max_evalue is not None:
        floor = score_floor(max_evalue, len(payload), db_residues)
        chosen = np.flatnonzero(scores >= np.float32(floor))
        order = chosen[np.argsort(-scores[chosen], kind="stable")]
        n_candidates = int(order.size)
    else:
        # Without a gate there is no principled floor, so fall back to ranking and a cap.
        if rank_by == "raw":
            order = np.argsort(-scores)
        elif rank_by == "evalue":
            sub_len = np.fromiter((len(sq) for _, sq in entries), dtype=np.float64, count=n_entries)
            order = np.argsort(np.log(sub_len) - LAMBDA * scores.astype(np.float64))
        else:
            raise ValueError(f"rank_by must be 'evalue' or 'raw', not {rank_by!r}")
        order = order[:top_n]
        n_candidates = int(min(top_n, n_entries))

    out: list[Homolog] = []
    n_fail_gates = n_fail_evalue = 0
    for i in order:
        acc, seq = entries[int(i)]
        r = parasail.sw_trace_striped_sat(payload, seq, gap_open, gap_extend, m)
        comp = r.traceback.comp
        if not comp:
            continue
        aligned = len(comp)
        identity = comp.count("|") / aligned
        coverage = coverage_of(aligned, len(payload))
        if not (identity >= min_identity and coverage >= min_coverage):
            n_fail_gates += 1
            continue
        if ln_max_e is not None and log_evalue_search(
                float(scores[int(i)]), len(payload), db_residues) > ln_max_e:
            n_fail_evalue += 1
            continue
        out.append(Homolog(acc, float(scores[int(i)]), round(identity, 4), round(coverage, 4), len(seq)))
    if stats is not None:
        # With a score floor `truncated` is False by construction; without one it records whether the cap
        # bound. The marginal keep count is what detects truncation in that case; the reject fraction is
        # not, because candidates arrive best-first and the deeper ones are always likelier to fail.
        stats.update(n_considered=n_candidates, n_kept=len(out),
                     n_rejected=n_candidates - len(out),
                     reject_fraction=round((n_candidates - len(out)) / max(n_candidates, 1), 4),
                     scanned_whole_db=True, rank_by=("score_floor" if floor is not None else rank_by),
                     db_residues=db_residues, n_fail_identity_or_coverage=n_fail_gates,
                     n_fail_evalue=n_fail_evalue, max_evalue=max_evalue,
                     score_floor=None if floor is None else round(floor, 2),
                     n_above_floor=None if floor is None else n_candidates,
                     truncated=bool(floor is None and n_candidates >= top_n))
    return sorted(out, key=lambda h: -h.identity)


def coverage_of(aligned_columns: int, payload_length: int) -> float:
    """Fraction of the **payload** explained by the alignment, clamped to 1.

    The denominator is the payload, not the subject, and that is the whole point of this function
    existing. The question a homolog record answers is "how much of the protein being searched for does
    this hit account for", so a 240 aa payload matching 200 columns of a 1,400 aa subject has covered
    0.83 of what was asked about, not 0.14.

    `scripts/16_build_twilight_refs.py` divided by the subject length instead, so the two routes to the
    same `homologs.csv` disagreed by up to 5x per payload and in both directions. Both now
    call this. The clamp matters because the aligned column count includes gaps and can exceed the
    payload length when the subject carries insertions.
    """
    return min(aligned_columns / payload_length, 1.0) if payload_length > 0 else 0.0


def holdout_mask(homologs: list[Homolog], threshold: float) -> set[str]:
    """Accessions to remove from the reference set at this holdout: anything *above* `threshold` identity."""
    return {h.accession for h in homologs if h.identity > threshold}


def targets_at(homologs: list[Homolog], threshold: float) -> set[str]:
    """The accessions that count as a correct answer at this holdout: true homologs still in the set.

    A set rather than a single answer, because below ~50% identity there is rarely one uniquely right
    relative: any genuine homolog retrieved is a success for a screen, which is the question being asked.
    """
    return {h.accession for h in homologs if h.identity <= threshold}
