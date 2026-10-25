"""Build the comparison arms for one plasmid, once, so every experiment scores the same sequences.

`scripts/06_decompose.py` and `scripts/10_twilight.py` both need "the payload-bearing ORF, its motif
decomposition, and its oracle span". Deriving that twice invites the two scripts to drift apart and the
results to stop being comparable, which would be the quiet kind of error: nothing crashes, the numbers
just stop meaning the same thing.

Four arms: `whole`, `auto` (motif rules), `bhs` (iterative best-hit segmentation) and `oracle` (the
ceiling). `bhs` needs the catalogue and a scorer, which the motif arms do not, so it is built only when
those are supplied: `18_eba_sweep.py` and `22_segment_swissprot.py` call `build_arms` without them and
get an empty `bhs`, unchanged in behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .decompose import (
    BestHitDecomposition,
    OracleSpan,
    Segment,
    best_hit_segments,
    decompose,
    oracle_span,
    replay_best_hit_segments,
    tile_segments,
)

MIN_AA = 50
PREFILTER_TOP = 3

#: Arm names in report order. The single source of truth: scripts iterate this rather than repeating a
#: literal tuple, which is how `10_twilight.py` came to hardcode the same three names in five places.
#:
#: `tile` is appended rather than inserted, and that position is deliberate: `REPORT_ARM_NAMES` is
#: built from this tuple, so appending leaves every existing arm at its existing index and a run that
#: does not ask for a tiling writes a file byte-identical to one from before the arm existed. The arm
#: is opt-in for the same reason `bhs` is, see `build_arms`.
ARM_NAMES: tuple[str, ...] = ("whole", "auto", "bhs", "oracle", "tile")

#: A segment arm, and the name under which the SAME segments are reported ranked one query at a time
#: instead of max-pooled. These are report-only arms: they add no query and no catalogue pass, they
#: re-aggregate score rows the pooled arm already computed.
#:
#: They are deliberately NEW names rather than a redefinition of `auto` and `bhs`. Every comparison in
#: the working record reads `auto` as the pooled quantity, and the twilight CSVs are keyed on
#: (threshold, plasmid_id, arm), so changing what `auto` means would silently reinterpret stored results
#:, the failure mode this project's register exists to prevent.
PER_SEGMENT_ARMS: dict[str, str] = {"auto": "auto_per_segment", "bhs": "bhs_per_segment",
                                    "tile": "tile_per_segment"}

#: Every arm a report may contain: the four query arms first, in `ARM_NAMES` order, then the per-segment
#: re-aggregations. The order is load-bearing for one specific check: because the new names come last
#: within each threshold block, dropping the rows whose arm is in `PER_SEGMENT_ARMS.values()` from a new
#: twilight CSV reproduces the old file exactly, so "the existing arms did not move" is a diff and not an
#: argument.
REPORT_ARM_NAMES: tuple[str, ...] = ARM_NAMES + tuple(
    PER_SEGMENT_ARMS[a] for a in ARM_NAMES if a in PER_SEGMENT_ARMS)


@dataclass
class Arms:
    plasmid_id: str
    orf_id: str
    orf_seq: str
    span: OracleSpan | None
    auto: list[Segment] = field(default_factory=list)
    auto_noint: list[Segment] = field(default_factory=list)
    bhs: list[Segment] = field(default_factory=list)
    bhs_result: BestHitDecomposition | None = None
    #: Fixed-width tiling of `orf_seq`. Empty unless `build_arms` was given a window, so every caller
    #: that predates the arm is unaffected, `queries()` drops empty arms.
    tile: list[Segment] = field(default_factory=list)

    @property
    def whole(self) -> str:
        return self.orf_seq

    @property
    def oracle(self) -> str:
        return self.span.seq if self.span else ""

    def queries(self) -> dict[str, list[str]]:
        """Each arm's query set, in `ARM_NAMES` order, with empty arms dropped.

        A single-sequence arm yields a one-element list and a segment arm yields one entry per segment, so
        a caller can treat all four uniformly: score every query in the list and take the best score each
        reference gets from any of them, which is what a pipeline submitting every segment as its own
        query would see. Returning this mapping rather than exposing the arms individually is what keeps
        a new arm from having to be special-cased at each call site.
        """
        out = {
            "whole": [self.whole],
            "auto": [s.seq for s in self.auto if s.seq],
            "bhs": [s.seq for s in self.bhs if s.seq],
            "oracle": [self.oracle],
            "tile": [s.seq for s in self.tile if s.seq],
        }
        return {k: v for k in ARM_NAMES if (v := [q for q in out[k] if q])}


def pick_payload_orf(orfs: dict[str, str], payload_seq: str, min_len: int = MIN_AA):
    """The ORF that best locally aligns to the payload, with its oracle span.

    A cheap 4-mer containment prefilter narrows the field before alignment: on a 15 kb plasmid a naive
    caller emits 30-odd ORFs, and aligning the payload against every one of them dominates the runtime
    without changing the answer.

    Choosing the ORF this way uses the label, which is deliberate: it isolates decomposition from ORF
    *selection*, so a selection failure cannot be mistaken for a decomposition failure. The deployable
    insert-size heuristic is scored separately.
    """
    if not payload_seq:
        return "", None
    km = {payload_seq[i : i + 4] for i in range(len(payload_seq) - 3)}
    scored = sorted(
        ((sum(1 for i in range(len(s) - 3) if s[i : i + 4] in km), oid)
         for oid, s in orfs.items() if len(s) >= min_len),
        reverse=True,
    )
    best_oid, best_span = "", None
    for _, oid in scored[:PREFILTER_TOP]:
        sp = oracle_span(orfs[oid], payload_seq)
        if sp and (best_span is None or sp.score > best_span.score):
            best_oid, best_span = oid, sp
    return best_oid, best_span


def build_arms(
    plasmid_id: str,
    orfs: dict[str, str],
    payload_seq: str,
    min_len: int = MIN_AA,
    *,
    ref_seqs: list[str] | None = None,
    scorer=None,
    db_residues: int | None = None,
    ref_accessions: list[str] | None = None,
    max_evalue: float = 1e-3,
    exclude_idx=None,
    bhs_replay: list[dict] | None = None,
    tile_window: int | None = None,
    tile_step: int | None = None,
) -> Arms | None:
    """All arms for one plasmid, or None when no ORF carries the payload.

    The `bhs` arm is built only when `ref_seqs`, `scorer` and `db_residues` are all supplied; otherwise it
    comes back empty and the other three arms are exactly as before. Note the asymmetry in what uses the
    label: `payload_seq` picks the ORF and builds `oracle`, and is **not** passed to `best_hit_segments`,
    which receives only the chosen ORF, the catalogue and the scorer.

    `exclude_idx` is the holdout mask, forwarded to the segmenter so it cannot win on the payload's own
    catalogue entry. See `best_hit_segments` for why that is a correctness requirement and not a tuning
    knob, and `segment_arms` for why a holdout sweep must call that function once per threshold instead
    of reusing one segmentation.

    `bhs_replay` hands back the span rows an earlier run wrote for THIS construct, and when it is given
    the segmenter does not run at all: no catalogue, no scorer, no alignment pass. It takes precedence
    over `ref_seqs`/`scorer`/`db_residues`, which are then used only for the floor. This is sound only
    because segmentation depends on the segmenting scorer, the catalogue **and the holdout mask**, not
    on the retrieval scorer, and `replay_best_hit_segments` checks what it rebuilds against the
    invariants the file carries rather than trusting it. See that function. **The mask is part of that
    list as of 20 Oct:** it used to be left out because `10_twilight.py` built one mask for
    a whole holdout sweep, so a span file carried one segmentation per construct. It now carries one per
    (construct, threshold), and a caller replaying a file must hand back the rows for the threshold it
    is scoring at.

    `tile_window` builds the `tile` arm and nothing else does, so every caller that predates it gets an
    empty list and `queries()` drops the arm: a run that does not ask for a tiling is unchanged, which
    is the same opt-in `bhs` has and for the same reason: adding an arm must never silently reinterpret
    a stored result. `tile_step` defaults to **half the window**, which is not a free choice: all
    three policies this project priced (300/150, 200/100, 100/50) are 50% overlap, so the default is
    the one already costed rather than a new one.

    **The tiling is mask-independent and must stay that way.** The windows come from the ORF's length,
    so no holdout can reach them, and `segment_arms` therefore leaves `tile` alone when it rebuilds
    `bhs` per threshold: `dataclasses.replace` carries it through untouched. That is what lets a
    tiling arm cost **one pass for a whole six-holdout sweep** rather than six (the working record, and the
    same property `whole`, `auto` and `oracle` already exploit at 58.6).
    """
    oid, span = pick_payload_orf(orfs, payload_seq, min_len)
    if not oid:
        return None
    seq = orfs[oid]
    arms = Arms(plasmid_id, oid, seq, span,
                auto=decompose(seq, min_len=min_len),
                auto_noint=decompose(seq, min_len=min_len, cut_at_internal_tags=False),
                tile=(tile_segments(seq, tile_window, tile_step if tile_step is not None
                                    else tile_window // 2, min_len=min_len)
                      if tile_window else []))
    if bhs_replay is None and not (ref_seqs is not None and scorer is not None
                                   and db_residues is not None):
        return arms
    return segment_arms(arms, ref_seqs=ref_seqs, scorer=scorer, db_residues=db_residues,
                        ref_accessions=ref_accessions, max_evalue=max_evalue, min_len=min_len,
                        exclude_idx=exclude_idx, bhs_replay=bhs_replay)


def segment_arms(
    arms: Arms,
    *,
    ref_seqs: list[str] | None = None,
    scorer=None,
    db_residues: int | None = None,
    ref_accessions: list[str] | None = None,
    max_evalue: float = 1e-3,
    min_len: int = MIN_AA,
    exclude_idx=None,
    bhs_replay: list[dict] | None = None,
) -> Arms:
    """`arms` with its `bhs` arm (re)built under `exclude_idx`, returned as a NEW object.

    **Why this is separate from `build_arms`.** `exclude_idx` is the holdout mask, and the mask is an
    argument to the *segmenter*: which catalogue entries it may claim a span against. So a sweep over
    holdouts has to segment once per threshold. Reusing one segmentation across a sweep is what this project measured, and it is not conservative in either direction: on the corpus tier at the 0.70
    holdout it manufactures 14,717 rank points of loss and 6 of gain against per-threshold segmentation.

    The other three arms do **not** depend on the mask: `whole` and `oracle` are sequences, and `auto`
    cuts on motifs in the ORF and never consults the catalogue. So a caller builds them once with
    `build_arms` and calls this per threshold, which is why a new object comes back rather than a
    mutation in place: a caller holding one threshold's arms must not have them change underfoot when
    the next threshold is segmented.
    """
    out = replace(arms, bhs=[], bhs_result=None)
    seq = out.orf_seq
    if bhs_replay is not None:
        # The segmentation this construct got in an earlier run, handed back in rather than recomputed.
        # The floor is a closed-form function of the ORF length and the searched residue count, so a
        # construct with no span rows still gets its real floor without a catalogue pass.
        from .homology import score_floor  # noqa: PLC0415 (avoids a circular import at module load)

        res = replay_best_hit_segments(
            seq, bhs_replay, min_len=min_len,
            floor=None if db_residues is None else score_floor(max_evalue, len(seq), db_residues))
    elif ref_seqs is not None and scorer is not None and db_residues is not None:
        res = best_hit_segments(seq, ref_seqs, scorer, db_residues,
                                ref_accessions=ref_accessions, max_evalue=max_evalue,
                                min_len=min_len, exclude_idx=exclude_idx)
    else:
        return out
    out.bhs_result = res
    # Same convention as `decompose`, and for the same reason: a screening pipeline that silently
    # drops a query is worse than one that over-reports, so when nothing clears the floor the arm
    # falls back to the whole ORF rather than becoming unscorable. That keeps the comparison paired
    # on the same constructs instead of quietly shrinking n, and it is label-free: the trigger is
    # "no segments at all", which the method knows without being told the answer. The flag makes the
    # fallback countable, which is what this project's working rules asks of any limit that binds.
    out.bhs = (list(res.segments) if res.segments
               else [Segment(1, len(seq), seq, ("fallback:bhs_found_nothing",))])
    return out
