"""Cut a called ORF into the parts that were fused together to make it.

The pilot's central measurement is this: a naive pipeline calls one ORF per *construct*, but the thing a
screen has to recognise is one *protein*. A 2A cassette, an affinity tag or a C-terminal reporter all
produce a single long ORF whose embedding is a blend of everything in it, and the payload's signal is
diluted in proportion to how much of the chain is something else. Decomposition is the step that undoes
that, and this module measures how much it is worth.

Four arms are compared in `scripts/06_decompose.py` and `scripts/10_twilight.py`:

* **whole**: the ORF exactly as the caller produced it. The current baseline.
* **auto**: `decompose()`. Motif-driven and *answer-blind*: it sees only the protein sequence, finds
                known ribosome-skipping and protease sites, cuts there, and trims recognised terminal tags
                and linkers. This is a deployable method, so its result is the honest one.
* **bhs**: `best_hit_segments()`. Iterative best-hit segmentation, also answer-blind: align the ORF
                against the catalogue, claim the span that aligns to the best-scoring reference, mask it,
                repeat until nothing clears the significance floor. Deployable, and unlike `auto` it fires
                on every construct rather than only where a motif happens to sit.
* **oracle**: `oracle_span()`. Locally aligns the ORF against the *known correct answer* and returns the
                matching sub-span. It cannot be deployed, it uses the label, so it is not a method but
                a ceiling: the best any decomposition step could do on this construct.

The gap between `whole` and `oracle` is how much the problem costs. The gap between `auto`/`bhs` and
`oracle` is how much of it a deployable rule can already recover, and therefore how much is left for a
learned method.

Coordinates throughout are 1-based inclusive residue positions in the ORF, matching the convention in
`fusion.py` and the audit sheet.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Motif catalogue
# ---------------------------------------------------------------------------
# Cleavage motifs mark places where one chain becomes two products: either co-translationally (2A
# ribosome skipping) or by a protease the construct is designed to be cut with. `cut_offset` is the
# number of residues from the start of the match after which the break falls.
#
# 2A: skipping happens between the penultimate G and the final P of the conserved D-x-E-x-NPG|P core, so
# the downstream product begins with that proline. The pattern is deliberately 9 residues long; the
# commonly quoted 4-residue core (NPGP) hits random sequence often enough to matter.
#
# Enterokinase (DDDDK) and Factor Xa (IEGR) are intentionally *absent*: at 4-5 residues with no
# constrained positions they fire on unrelated sequence, and DDDDK is a substring of the FLAG tag.
CLEAVAGE: tuple[tuple[str, str, int], ...] = (
    ("2A_skip", r"GD[VI]E[A-Z][NS]PG[PS]", 8),
    ("TEV_site", r"ENLYFQ[GS]", 6),
    ("3C_site", r"LEVLFQ[GP]", 6),
    ("thrombin_site", r"LVPR[GS]", 4),
)

# Terminal purification/localisation tags. Order matters only for readability; overlapping matches are
# resolved by keeping the longest (so 3xFLAG wins over the FLAG it contains).
TAGS: tuple[tuple[str, str], ...] = (
    ("3xFLAG", r"(?:DYKDHD[GI]){1,3}DYKDDDDK"),
    ("FLAG", r"DYKDDDDK"),
    ("His", r"H{6,12}"),
    ("HA", r"YPYDVPDYA"),
    ("Myc", r"EQKLISEEDL"),
    ("V5", r"GKPIPNPLLGLDST"),
    ("StrepII", r"WSHPQFEK"),
    ("SV40_NLS", r"P[KR]KKRKV"),
    ("Arg", r"R{6,}"),
    ("Avi", r"GLNDIFEAQKIEWHE"),
    ("S_tag", r"KETAAAKFERQHMDS"),
)

# Flexible linkers. These are the joins themselves rather than functional parts, so they are trimmed off
# the ends of a segment like tags.
LINKERS: tuple[tuple[str, str], ...] = (
    ("GS_linker", r"(?:GGGGS)+"),
    ("G_rich", r"[GS]{6,}"),
)

# Which motifs may be read as a *junction* when they turn up in the interior of a chain (rule 2 in
# `decompose`). Measured against the 5,016-protein reference set: allowing every linker to imply a seam
# splits 85/5,016 real single proteins (1.69%); restricting it to the tags plus the tight `GGGGS` repeat
# splits 18/5,016 (0.36%) and still separates Addgene #171782's MBP from its payload at the internal His6.
# `G_rich` ([GS]{6,}) is the culprit, it fires in 2.07% of real proteins, so it trims ends but never
# cuts. Anything added here must be re-measured the same way; see the working record.
INTERIOR_JUNCTION: frozenset[str] = frozenset({n for n, _ in TAGS} | {"GS_linker"})


@dataclass(frozen=True)
class MotifHit:
    name: str
    kind: str  # "cleave" | "tag" | "linker"
    start: int  # 1-based inclusive
    end: int  # 1-based inclusive
    cut_after: int | None = None  # residue index after which the chain breaks; cleavage motifs only

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class Segment:
    """One decomposition product: a contiguous sub-span of the ORF."""

    start: int
    end: int
    seq: str
    removed: tuple[str, ...] = ()  # e.g. ("N:3xFLAG", "C:SV40_NLS"), what was trimmed off each end

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class OracleSpan:
    """The sub-span of an ORF that locally aligns to a known payload. Uses the label, so: a ceiling."""

    start: int
    end: int
    seq: str
    identity: float  # fraction of aligned columns that are identical
    payload_coverage: float  # fraction of the payload covered by the alignment
    score: float

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def find_motifs(protein: str) -> list[MotifHit]:
    """Every catalogued motif in `protein`, longest-match-wins where two overlap.

    Returned in position order. Overlap resolution drops any hit wholly inside another so that, for
    example, `MDYKDHDGDYKDHDIDYKDDDDK...` yields one 3xFLAG rather than a 3xFLAG plus the FLAG inside it.
    """
    hits: list[MotifHit] = []
    for name, pat, off in CLEAVAGE:
        for m in re.finditer(pat, protein):
            hits.append(MotifHit(name, "cleave", m.start() + 1, m.end(), m.start() + off))
    for kind, table in (("tag", TAGS), ("linker", LINKERS)):
        for name, pat in table:
            for m in re.finditer(pat, protein):
                hits.append(MotifHit(name, kind, m.start() + 1, m.end()))
    keep = [
        h
        for h in hits
        if not any(
            o is not h and o.start <= h.start and h.end <= o.end and o.length > h.length for o in hits
        )
    ]
    return sorted(keep, key=lambda h: (h.start, -h.length))


def _trim(protein: str, start: int, end: int, hits: list[MotifHit], window: int) -> Segment:
    """Strip tag and linker motifs that sit within `window` residues of either end, repeatedly.

    Repetition matters because tags come in stacks: `3xFLAG-GGGGS-payload` needs two passes, and stopping
    after one would leave a linker on the front of the segment that gets embedded.
    """
    removed: list[str] = []
    inner = [h for h in hits if h.kind != "cleave"]
    changed = True
    while changed and start <= end:
        changed = False
        for h in inner:
            if h.start < start or h.end > end:
                continue
            if h.start - start <= window:  # near the N-terminus
                start, changed = h.end + 1, True
                removed.append(f"N:{h.name}")
                break
            if end - h.end <= window:  # near the C-terminus
                end, changed = h.start - 1, True
                removed.append(f"C:{h.name}")
                break
    return Segment(start, end, protein[start - 1 : end] if start <= end else "", tuple(removed))


def decompose(
    protein: str,
    min_len: int = 50,
    trim_window: int = 30,
    cut_at_internal_tags: bool = True,
) -> list[Segment]:
    """Answer-blind decomposition of one ORF into candidate single-protein segments.

    Three rules, in order:

    1. **Cut at cleavage motifs.** A 2A peptide or an engineered protease site means the ribosome or the
       purification protocol produces two separate proteins, so one ORF is two queries.
    2. **Cut around interior tags and tight linkers**: the motifs in `INTERIOR_JUNCTION`. A His6 or a
       `GGGGS` repeat in the middle of a chain is not part of either neighbour; it is the seam between
       them. `cut_at_internal_tags=False` disables this, which is the ablation reported in the results:
       rule 1 alone cannot split a tagged fusion that carries no cleavage site, such as Addgene #171782's
       MBP-His6-TEV protease.
    3. **Trim terminal tags and linkers** from each resulting piece, repeatedly, then drop pieces shorter
       than `min_len`: the same floor the ORF caller uses, so a segment is never scored on evidence the
       caller would itself have discarded.

    Returns segments in N-to-C order. A protein with no motifs comes back as a single segment covering the
    whole chain, which makes `whole` the degenerate case of `auto` rather than a separate code path.

    **Never returns an empty list.** If trimming leaves nothing above `min_len`, which happens for short
    tag-like proteins such as the arginine-rich protamines, where the `R{6,}` pattern consumes most of the
    chain, the whole chain is returned unchanged, flagged `fallback:nothing_above_min_len`. A screening
    pipeline that silently drops a query is worse than one that over-reports, so this failure mode is made
    loud rather than quiet.
    """
    n = len(protein)
    hits = find_motifs(protein)
    cuts = {h.cut_after for h in hits if h.kind == "cleave" and h.cut_after is not None}
    if cut_at_internal_tags:
        for h in hits:
            if h.name in INTERIOR_JUNCTION and h.start - 1 >= trim_window and n - h.end >= trim_window:
                cuts.update({h.start - 1, h.end})
    bounds = [0, *sorted(c for c in cuts if 0 < c < n), n]
    out: list[Segment] = []
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        seg = _trim(protein, a + 1, b, hits, trim_window)
        if seg.length >= min_len:
            out.append(seg)
    if not out:
        return [Segment(1, n, protein, ("fallback:nothing_above_min_len",))]
    return out


def _aligner():
    """BLOSUM62 local aligner with BLASTP-like gap costs; built lazily so importing this module is cheap."""
    from Bio import Align

    a = Align.PairwiseAligner(scoring="blastp")
    a.mode = "local"
    return a


def oracle_span(orf_protein: str, payload_protein: str, aligner=None) -> OracleSpan | None:
    """The sub-span of `orf_protein` that locally aligns to `payload_protein`, or None if nothing aligns.

    The span runs from the first to the last aligned residue, so an internal gap (a payload interrupted by
    an inserted tag, for instance) stays inside one span rather than being split: a decomposition step
    that returned the pieces separately would be doing something different and harder to compare.

    **This function is the ceiling arm because of how it is used, not because of what it computes.** What
    it computes is "the sub-span of a chain that locally aligns to some other protein", which uses a label
    only when the caller passes the *known answer* as `payload_protein`. `best_hit_segments` calls it with
    the best-scoring **catalogue** entry instead, which is available at screening time, and is therefore a
    deployable method built on the same primitive. Read the second argument, not the function name, to
    know whether a given call is label-free.
    """
    if not orf_protein or not payload_protein:
        return None
    al = aligner or _aligner()
    try:
        best = al.align(orf_protein, payload_protein)[0]
    except (IndexError, ValueError):
        return None
    tgt, qry = best.aligned
    if len(tgt) == 0:
        return None
    start, end = int(tgt[0][0]) + 1, int(tgt[-1][1])
    same = total = covered = 0
    for (ts, te), (qs, qe) in zip(tgt, qry):
        total += te - ts
        covered += qe - qs
        same += sum(1 for i, j in zip(range(ts, te), range(qs, qe)) if orf_protein[i] == payload_protein[j])
    return OracleSpan(
        start,
        end,
        orf_protein[start - 1 : end],
        round(same / total, 4) if total else 0.0,
        round(covered / len(payload_protein), 4),
        float(best.score),
    )


MASK_CHAR = "X"


@dataclass(frozen=True)
class BestHitIteration:
    """One round of `best_hit_segments`: what won, what it claimed, and whether that cleared the floor."""

    iteration: int
    ref_index: int
    ref_accession: str
    score: float
    floor: float
    start: int  # 1-based inclusive, in ORF coordinates
    end: int
    identity: float
    ref_coverage: float  # fraction of the winning reference covered by the alignment
    emitted: bool  # False when the span cleared the floor but no unmasked run reached `min_len`
    n_masked_now: int  # residues masked by this iteration that were not already masked
    emitted_spans: tuple[tuple[int, int], ...] = ()  # the span clipped to its unmasked runs


@dataclass(frozen=True)
class BestHitDecomposition:
    """`best_hit_segments`' full result: the arm itself, plus the trace that makes it auditable."""

    segments: tuple[Segment, ...]
    trace: tuple[BestHitIteration, ...]
    floor: float
    stop_reason: str  # "below_floor" | "no_alignment" | "no_progress" | "exhausted"

    @property
    def n_iterations(self) -> int:
        """Rounds that claimed a span. The round that stops the loop is not counted as an iteration."""
        return len(self.trace)


def _unmasked_runs(start: int, end: int, masked: list[bool]) -> list[tuple[int, int]]:
    """The contiguous unmasked runs inside a claimed span, 1-based inclusive.

    Factored out so that `best_hit_segments` and `replay_best_hit_segments` cannot drift: a replay that
    clipped spans even slightly differently from the run it replays would be the worst possible failure
    here, because it would be silent. There is one implementation and both call it.
    """
    runs: list[tuple[int, int]] = []
    i = start - 1
    while i < end:
        if masked[i]:
            i += 1
            continue
        j = i
        while j < end and not masked[j]:
            j += 1
        runs.append((i + 1, j))
        i = j
    return runs


class ReplayMismatch(ValueError):
    """A span file does not reproduce the run it claims to describe. Never recoverable, never warned."""


def span_rows_at(rows: list[dict], threshold: float | None = None) -> list[dict]:
    """The span rows describing ONE segmentation, out of a `bhs_spans_*.csv` that may hold several.

    Since 20 Oct a span file carries one segmentation per *(construct, threshold)* and a
    `threshold` column to say which. A reader that ignores that column treats six thresholds' spans as
    one segmentation, so `k_bhs` counted from it comes out up to six times too large: a **wrong
    number** rather than a crash, which is the failure mode this repository's register exists to catch.
    Every consumer of a span file therefore goes through here.

    A file with no `threshold` column predates the change and holds exactly one segmentation, built at
    the strictest threshold in the sweep; `threshold` is then ignored and the rows come back as they are.
    A file that HAS the column and is asked for no particular threshold raises, naming the thresholds it
    holds, because there is no sensible default: picking one silently is how a number from the 1.00
    column gets reported as the 0.25 column's.
    """
    if not rows or "threshold" not in rows[0]:
        return list(rows)
    present = sorted({float(r["threshold"]) for r in rows}, reverse=True)
    if threshold is None:
        raise ValueError(
            "this span file carries one segmentation per (construct, threshold) and holds thresholds "
            f"{present}; pass the one to read. Reading them all as one segmentation would multiply "
            "k_bhs by the number of thresholds.")
    out = [r for r in rows if abs(float(r["threshold"]) - float(threshold)) < 1e-9]
    if not out:
        raise ValueError(f"no span rows at threshold {threshold}; the file holds {present}")
    return out


def replay_best_hit_segments(
    orf_protein: str,
    rows: list[dict],
    *,
    min_len: int = 50,
    floor: float | None = None,
) -> BestHitDecomposition:
    """Rebuild a `best_hit_segments` result from the span rows a previous run wrote, with no scoring.

    **Why this exists.** Segmentation costs one full-catalogue alignment pass per iteration per
    construct, and it depends on the *segmenting* scorer, the catalogue and the holdout mask, but not
    on the retrieval scorer. So a hybrid run (`--bhs-scorer align --scorer <embedder>`) recomputes, on a
    GPU window's clock, a segmentation the repository may already own from an alignment run against the
    same catalogue at the same holdout.

    **The holdout is in that list as of 20 Oct ** and used not to be: `10_twilight.py` built
    the segmenter's mask once at the strictest threshold in its sweep, which made segmentation
    holdout-invariant by construction and wrong at every looser threshold. A span file now
    carries one segmentation per (construct, threshold), and `rows` must be the rows for ONE construct at
    ONE threshold. A file written before the change has no `threshold` column; the caller is responsible
    for refusing to replay it as if it were per-threshold.

    **Why it is safe to replay rather than recompute.** Everything after the catalogue pass is pure
    bookkeeping over the claimed spans: mask the span, clip the next claim to the unmasked runs, keep
    the runs that reach `min_len`. The span rows record every claim in order, so the bookkeeping can be
    re-executed exactly, by the same `_unmasked_runs` the original used.

    **And it is checked rather than trusted.** The rows carry three quantities this function recomputes
    independently, `newly_masked`, `emitted` and `orf_len`, and any disagreement raises
    `ReplayMismatch`. There is deliberately no repair path and no warning: a span file that does not
    reproduce its own invariants is describing a different run, and the only correct response is to
    stop. `n_catalogue_entries_masked` is checked by the caller, which is where the holdout mask lives.

    `rows` must be the span rows for ONE construct, in iteration order. An empty list is legitimate and
    common: a construct whose first claim fell below the floor emits no rows at all (measured: 11 of 336
    on synth_real, 72 of 336 on synthetic, 102 of 410 on the corpus). Such a construct has no segments,
    so `arms.build_arms` falls back to the whole ORF, exactly as it would after a real run. Its
    `stop_reason` is **not recoverable** from an empty row set, so it is reported as
    `replay:absent_from_span_file` rather than guessed at; nothing downstream consumes it, because a
    construct with no trace emits no span rows on the way back out either.

    One field is genuinely absent and is **not** faked: `BestHitIteration.ref_index` comes back as -1,
    because the span file records the winning accession and not its row in the catalogue, and the two
    catalogues need not be ordered alike. Nothing outside this module reads `ref_index`; a grep over
    `src/`, `scripts/` and `tests/` is the check, and `tests/test_bhs_replay.py` pins it.
    """
    n = len(orf_protein)
    if not rows:
        return BestHitDecomposition((), (), float(floor or 0.0), "replay:absent_from_span_file")

    rows = sorted(rows, key=lambda r: int(r["iteration"]))
    declared_len = int(rows[0]["orf_len"])
    if declared_len != n:
        raise ReplayMismatch(
            f"span file says orf_len={declared_len}, this run's ORF is {n} residues long")

    masked = [False] * n
    segments: list[Segment] = []
    trace: list[BestHitIteration] = []
    for k, r in enumerate(rows, start=1):
        if int(r["iteration"]) != k:
            raise ReplayMismatch(f"iterations are not 1..n in order: saw {r['iteration']} at position {k}")
        start, end = int(r["seg_start"]), int(r["seg_end"])
        runs = _unmasked_runs(start, end, masked)
        fresh = sum(b - a + 1 for a, b in runs)
        if fresh != int(r["newly_masked"]):
            raise ReplayMismatch(
                f"iteration {k}: replay masks {fresh} new residues, the file recorded "
                f"{r['newly_masked']}")
        kept = [(a, b) for a, b in runs if b - a + 1 >= min_len]
        if bool(kept) != bool(int(r["emitted"])):
            raise ReplayMismatch(
                f"iteration {k}: replay emits {bool(kept)}, the file recorded {bool(int(r['emitted']))}")
        acc = r["hit_accession"]
        for a, b in kept:
            segments.append(Segment(a, b, orf_protein[a - 1 : b],
                                    (f"bhs:iter{k}", f"hit:{acc}" if acc else "hit:?")))
        trace.append(BestHitIteration(
            k, -1, acc, float(r["hit_score"]), float(r["floor"]), start, end,
            float(r["identity"]), float(r["hit_coverage"]), bool(kept), fresh, tuple(kept)))
        for i in range(start - 1, end):
            masked[i] = True

    return BestHitDecomposition(tuple(segments), tuple(trace), float(rows[0]["floor"]),
                                rows[-1]["stop_reason"])


def best_hit_segments(
    orf_protein: str,
    ref_seqs: list[str],
    scorer,
    db_residues: int,
    ref_accessions: list[str] | None = None,
    max_evalue: float = 1e-3,
    min_len: int = 50,
    exclude_idx=None,
    floor_fn=None,
) -> BestHitDecomposition:
    """Label-free iterative best-hit segmentation of one ORF. **The payload is not a parameter.**

    Align the ORF against the catalogue, take the best-scoring reference, claim the sub-span of the ORF
    that aligns to it, mask that span, and repeat on what is left until no remaining hit clears the
    significance floor. The claimed spans are the arm.

    **Why this is label-free, structurally rather than by promise.** The signature takes the ORF, the
    catalogue and a scorer; there is no `payload_seq` parameter and no way to pass one. The method aligns
    against whatever wins in the catalogue, not against the answer. `tests/test_best_hit_segments.py`
    asserts the output is identical when the payload is deleted from the catalogue entirely, which is the
    property that makes this a method rather than a ceiling. (`arms.pick_payload_orf` does legitimately
    use the label, to choose *which* ORF carries the payload: that isolates decomposition from ORF
    selection. This function runs on the ORF that returns.)

    **`exclude_idx` is not an optimisation; it is what keeps the arm honest.** The benchmark's catalogue
    contains the payload itself, and `homologs.csv` records it as its own homolog at identity 1.000, so
    the holdout mask at any threshold below 1.0 removes it. A segmenter searching the *unmasked*
    catalogue would therefore find the answer sitting in the database at 100% identity, claim exactly its
    span in iteration 1, and reproduce `oracle` while appearing to be label-free: the leak would be
    laundered through the catalogue rather than passed as an argument, and the structural guarantee above
    would not catch it. Passing the holdout mask here is what makes "it finds spans using only remote
    relatives" true rather than aspirational. Excluded entries are dropped from the `argmax` only; the
    score row is still computed over everything, since filtering afterwards is free.

    **Why it should work**: the mechanism behind the whole-ORF penalty is rank competition
    from the co-encoded gene's own family, and the co-encoded genes in real constructs are reporters,
    markers and tags: the best-represented families in any catalogue. So iteration 1 usually claims the
    distractor, and iteration 2 exposes the payload.

    **The stopping rule is derived, not chosen.** `E <= max_evalue` inverts to a raw-score floor,
    `S_min = (ln(K m N) + ln(1/E)) / lambda` (`homology.score_floor`). `N` is **the database actually
    searched**, which here is the catalogue, not Swiss-Prot: at 20,806,731 catalogue residues against
    Swiss-Prot's 207,099,657 the search space is 9.95x smaller and the floor is 8.61 points **lower**, so
    using the Swiss-Prot floor would stop the iteration early and silently. The floor is computed per ORF
    from `db_residues` and is returned for printing. There is **no iteration cap**: the loop is bounded by
    the floor, and the iteration count is a measurement to report (this project's working rules, "derive the limit instead of
    choosing it": the same move that retired `--top-n`).

    **Masking is by X-substitution**, in place, preserving length. Two properties are required and both
    are tested:

    1. *A claimed span must not win again.* In BLOSUM62 `X` scores **at most 0** against every standard
       residue (0 against A/S/T, -1 or -2 otherwise), so a masked span's maximum possible contribution to
       any local alignment is zero. Smith-Waterman's local optimum never includes a stretch that
       contributes nothing, so the span cannot be re-claimed.
    2. *Masking must not manufacture a junction.* Deleting the span and concatenating the flanks would
       join two residues that were never adjacent, creating a novel seam that can score as a new hit.
       X-substitution leaves the mask in place as a spacer of the original length, so the flanks are never
       brought into adjacency and no new seam exists to be found.

    Property 1 holds at the level of *score*, and that is not sufficient on its own, which a test caught
    rather than inspection: aligning through a run of X costs only about one point per residue, so a
    long-enough pair of flanks can still align end to end **across** a mask: a 198 aa protein split by a
    198 aa mask aligns 1-396 at score 817, against 1,026 for its halves concatenated. Masking is
    therefore enforced twice: in the score, where a masked span contributes nothing, and at emission,
    where a claimed span is clipped to its contiguous unmasked runs. Without the second, iteration 2 could
    emit a segment that re-contains iteration 1's span, which is the chimera this arm exists to avoid.
    Masking stays strictly more conservative than deletion either way (817 < 1,026).

    X-substitution also keeps every coordinate in ORF space with no bookkeeping, which is where this
    repo's off-by-one bugs have lived.

    Termination: each iteration must newly mask at least one residue, or the loop stops with
    `no_progress`. That bounds the iteration count by the ORF length independently of the floor.

    **`floor_fn` exists because the floor is in the scorer's units, and the scorer is replaceable.**
    `homology.score_floor` inverts Karlin-Altschul on *raw Smith-Waterman* scores, which is meaningless
    for a cosine-substitution scorer, so the arm could not run on an embedding at all (the working record,
    open item 25). Pass `floor_fn(max_evalue, query_len, db_residues) -> float` to supply a floor in the
    units the scorer actually returns. Nothing else about the loop changes, and it should not: `bhs_stop`
    is `below_floor` for all 777 constructs on all four tiers, so this one call *is* the stopping rule.
    """
    import numpy as np  # noqa: PLC0415 (keeps the motif-only import path light)

    from .homology import score_floor  # noqa: PLC0415 (avoids a circular import at module load)

    n = len(orf_protein)
    accs = ref_accessions or [""] * len(ref_seqs)
    floor = (floor_fn or score_floor)(max_evalue, n, db_residues)
    if not orf_protein or not ref_seqs:
        return BestHitDecomposition((), (), floor, "no_alignment")

    aligner = _aligner()
    working = list(orf_protein)
    masked = [False] * n
    segments: list[Segment] = []
    trace: list[BestHitIteration] = []
    row = None
    stop = "exhausted"
    excl = None if exclude_idx is None else np.asarray(exclude_idx, dtype=int)
    if excl is not None and excl.size and excl.size >= len(ref_seqs):
        return BestHitDecomposition((), (), floor, "no_alignment")

    while True:
        if row is None:
            row = np.asarray(scorer.score_all("".join(working)), dtype=np.float64).copy()
            if excl is not None and excl.size:
                row[excl] = -np.inf
        best = int(np.argmax(row))
        best_score = float(row[best])
        if not np.isfinite(best_score):
            stop = "no_alignment"
            break
        if best_score < floor:
            stop = "below_floor"
            break
        span = oracle_span("".join(working), ref_seqs[best], aligner=aligner)
        if span is None:
            stop = "no_alignment"
            break
        # An alignment can BRIDGE an earlier mask: aligning through a run of X costs only about one point
        # per residue, so a long pair of flanks either side of a claimed span can still align end to end
        # (measured: a 198 aa protein split by a 198 aa mask aligns 1-396 at score 817 against 1,026 for
        # the same halves concatenated). Masking therefore has to be enforced at *emission* as well as in
        # the scoring: the claimed span is clipped to its contiguous unmasked runs, so a segment can never
        # re-contain an already-claimed region and no segment is a chimera welded across a mask. The whole
        # span is still masked, which is what stops the same bridge being found again.
        runs = _unmasked_runs(span.start, span.end, masked)
        fresh = sum(b - a + 1 for a, b in runs)
        if fresh == 0:  # the winner claimed only already-masked residues; masking more would be a no-op
            stop = "no_progress"
            break
        kept = [(a, b) for a, b in runs if b - a + 1 >= min_len]
        for a, b in kept:
            # the segment's sequence comes from the ORIGINAL ORF, so an earlier iteration's mask can never
            # leak an X into a query that gets scored
            segments.append(
                Segment(a, b, orf_protein[a - 1 : b],
                        (f"bhs:iter{len(trace) + 1}", f"hit:{accs[best]}" if accs[best] else "hit:?"))
            )
        trace.append(BestHitIteration(len(trace) + 1, best, accs[best], best_score, floor,
                                      span.start, span.end, span.identity, span.payload_coverage,
                                      bool(kept), fresh, tuple(kept)))
        for i in range(span.start - 1, span.end):
            working[i], masked[i] = MASK_CHAR, True
        if all(masked):
            stop = "exhausted"
            break
        row = None  # force a rescore of the newly masked chain

    return BestHitDecomposition(tuple(segments), tuple(trace), floor, stop)


def tile_segments(
    protein: str,
    window: int = 200,
    step: int = 100,
    min_len: int = 50,
) -> list[Segment]:
    """Fixed-width tiling of one ORF: the segmenter with no external dependency.

    `decompose` needs the motif catalogue and `best_hit_segments` needs the retrieval catalogue, a
    scorer and the holdout mask. This one needs none of them, the windows follow from the ORF's
    length alone, so it is the arm that can run where the other two cannot, and the control for what
    segmentation *as such* is worth as against a segmentation that knows something.

    **Why it exists.** the working record traced the whole of `auto_per_segment`'s residual filler effect to
    the 144 constructs the motif rules never cut: `motifless_fusion`, the architecture built with no
    cleavage site, no tag and no linker, where the motif arm's only query *is* the whole ORF and it is
    bit-identical to `whole` on every cell. A fixed window has nothing to look for and so cannot fail
    to find it.

    **The falsifying case, named before any run, and it is structural rather than hypothetical.**
    Tiling buys purity by fragmenting the payload: a 200 aa window over a 705 aa payload queries 28% of
    it, and `scripts/50_fragment_bound.py` with the working record fix a significance floor: on the
    40,750-protein catalogue no fragment under roughly 18 aa reaches `E <= 1e-3` even against a perfect
    copy of itself. So this arm trades dilution for truncation, and what refutes it is a long payload
    whose every window is too short to clear significance, where purity rises and rank falls.

    **Coordinates are 1-based inclusive**, as for every other `Segment` in this module.

    Three rules, and each one is a way to produce a meaningless arm:

    1. **Gapless.** `step > window` leaves residues in no window at all, which is a silent truncation of
       the query and precisely the class this project's binding-limit register collects. Refused, not clamped.
    2. **The tail is anchored, never dropped and never short.** A 250 aa ORF at 200/100 leaves a 50 aa
       remainder, and a short final window is the one most likely to sit under the significance floor.
       Dropping it would leave the C-terminus unqueried; emitting it short would put a window below the
       floor and let the policy look worse than it is. The final window is moved back to end at the
       ORF's end instead, so every window is full width and the ORF is wholly covered. It overlaps its
       predecessor by more than `step`, which is the price and is visible in the coordinates.
    3. **An ORF no longer than one window yields one segment spanning it**, which makes this arm
       identical to `whole` for that construct. That is not a failure and it is not hidden: the segment
       carries `tile:whole_orf`, so the undecomposed subset is a column in the output rather than an
       inference, which is what 68.1 had to recover after the fact by joining geometry back onto
       finished runs.
    """
    if window <= 0 or step <= 0:
        raise ValueError(f"window and step must be positive, got window={window} step={step}")
    if step > window:
        raise ValueError(
            f"step {step} exceeds window {window}, which would leave residues in no window at all")
    if window < min_len:
        raise ValueError(
            f"window {window} is below min_len {min_len}: every window would be refused and the arm "
            f"would come back silently empty")
    n = len(protein)
    if n == 0:
        return []
    tag = f"tile:{window}/{step}"
    if n <= window:
        return [Segment(1, n, protein, (tag, "tile:whole_orf"))]

    bounds: list[tuple[int, int]] = []
    start = 1
    while start + window - 1 <= n:
        bounds.append((start, start + window - 1))
        start += step
    if bounds[-1][1] < n:
        bounds.append((n - window + 1, n))

    out: list[Segment] = []
    seen: set[tuple[int, int]] = set()
    for a, b in bounds:
        if (a, b) in seen:
            continue
        seen.add((a, b))
        anchored = ("tile:anchored_tail",) if (a - 1) % step else ()
        out.append(Segment(a, b, protein[a - 1 : b], (tag, *anchored)))
    return out


def tile_binding_counts(segments: list[Segment]) -> dict[str, int]:
    """What a tiling emitted, as counts, because a policy that silently does nothing looks like a result.

    this project's working rules: *for every limit in catalogue construction, print how many items hit it.* The limits here
    are the window and the step, and the two ways they bind are an ORF shorter than one window (the arm
    degenerates to `whole`, which is 68.1's undecomposed subset by another name) and a tail that had to
    be anchored. Both are free to compute and neither is visible in a rank.
    """
    return {
        "segments": len(segments),
        "whole_orf": sum(1 for s in segments if "tile:whole_orf" in s.removed),
        "anchored_tail": sum(1 for s in segments if "tile:anchored_tail" in s.removed),
    }


def dilution(orf_protein: str, span: OracleSpan | Segment) -> float:
    """Fraction of the ORF that is *not* the payload: how much foreign sequence the embedding carries."""
    if not orf_protein:
        return 0.0
    return round(1.0 - span.length / len(orf_protein), 4)
