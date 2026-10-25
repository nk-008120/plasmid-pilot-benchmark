"""Tests for handing a previous run's segmentation back in (`decompose.replay_best_hit_segments`).

**What this flag is for.** Segmentation costs one full-catalogue alignment pass per iteration per
construct, and it depends on the segmenting scorer and the catalogue and on nothing else in the run:
not on `--scorer`, and not on the holdout, since the segmenter's mask is built once at the strictest
threshold. So a hybrid run under an embedding scorer can reuse an alignment run's segmentation over the
same catalogue, and spend a scarce GPU window on retrieval instead.

**What makes that safe is not the argument above; it is that the replay is checked.** Everything after
the catalogue pass is bookkeeping over the claimed spans, and the span file records three quantities
the replay recomputes independently: `orf_len`, `newly_masked` and `emitted`. The tests below tamper
with each one and require a refusal. A replay that quietly differed from the run it replays would be
the worst available failure here, because nothing downstream would show it.

The round-trip test is the important one: run the real segmenter on a real catalogue, serialise its
trace exactly as `10_twilight.py` does, replay that, and demand the same segments and the same trace.
"""
from __future__ import annotations

import numpy as np
import pytest

from pilot.align import AlignmentScorer
from pilot.arms import build_arms
from pilot.decompose import ReplayMismatch, best_hit_segments, replay_best_hit_segments

parasail = pytest.importorskip("parasail", reason="the alignment arm is an optional dependency")

PAYLOAD = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIF"
    "FKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDN"
)
DISTRACTOR = (
    "MVSKGEALFKEIMVRFDGDVNGHEFTIEGEGTGHPYEGTQTAKLTVTKGGPLPFAWDILSPQFQYGSKVYVKHPADIPDYKKLSFPEGFKWERVMNFED"
    "GGIIHVNQDSSLQDGVFIYKVKLRGTNFPPDGPVMQKKTMGWEASTEQLYPEDGVLKGDIKMALRLKDGGRYLADFKTTYKAKKPVQMPGAYNVDRKLD"
)
FILLER = (
    "MRTEYCGQLRLSHVGQQVTLCGWVNRRRDLGSLIFIDMRDTGFSKLVTGEDIEQCFQIKSEVCRRFCMQVTGTVRARDEKNINRDMATGEIEVLASSLT"
    "ILNRSDTPPFTIQDQNDVSEELRLKYRYLDLRRPEMAQRLKTRAKITSLVRRFMDDNGFLDIETPILTKATPEGARDYLVPSRVHKGKFYALPQSPQLF"
)

DB_RESIDUES = 20_000_000


def _segmented():
    """A real two-gene ORF segmented by the real segmenter, with its catalogue."""
    orf = DISTRACTOR + PAYLOAD
    refs = [PAYLOAD, DISTRACTOR, FILLER]
    accs = ["PAY", "DIS", "FIL"]
    scorer = AlignmentScorer(refs)
    res = best_hit_segments(orf, refs, scorer, DB_RESIDUES, ref_accessions=accs, min_len=50)
    return orf, refs, accs, scorer, res


def _as_span_rows(orf: str, res, tier: str = "t", pid: str = "p", acc: str = "A", n_masked: int = 7):
    """Serialise a decomposition exactly as `10_twilight.py` writes its bhs_spans CSV.

    Deliberately a copy of the writer's field set rather than an import of it: if the writer drops a
    field the replay needs, this test is where that shows up, and a shared helper would hide it. Values
    are strings, because the replay reads them back out of a CSV.
    """
    return [{
        "tier": tier, "plasmid_id": pid, "payload_accession": acc, "orf_len": str(len(orf)),
        "iteration": str(t.iteration), "hit_accession": t.ref_accession,
        "hit_score": str(round(t.score, 1)), "floor": str(round(t.floor, 2)),
        "seg_start": str(t.start), "seg_end": str(t.end), "seg_len": str(t.end - t.start + 1),
        "identity": str(t.identity), "hit_coverage": str(t.ref_coverage),
        "emitted": str(int(t.emitted)), "newly_masked": str(t.n_masked_now),
        "n_iterations": str(res.n_iterations), "stop_reason": res.stop_reason,
        "n_catalogue_entries_masked": str(n_masked),
    } for t in res.trace]


def test_replay_round_trips_the_real_segmenter():
    """The whole claim: replaying a trace gives back the segments that produced it."""
    orf, _refs, _accs, _scorer, res = _segmented()
    assert len(res.trace) >= 2, "the fixture must exercise masking, or this proves nothing"

    back = replay_best_hit_segments(orf, _as_span_rows(orf, res), min_len=50)

    assert [(s.start, s.end, s.seq) for s in back.segments] == \
           [(s.start, s.end, s.seq) for s in res.segments]
    assert [(t.iteration, t.ref_accession, t.start, t.end, t.emitted, t.n_masked_now)
            for t in back.trace] == \
           [(t.iteration, t.ref_accession, t.start, t.end, t.emitted, t.n_masked_now)
            for t in res.trace]
    assert back.stop_reason == res.stop_reason
    assert back.floor == pytest.approx(res.floor, abs=0.01)


def test_replayed_segment_labels_match_too():
    """The segment labels carry the iteration and the winning accession, and a reader uses them."""
    orf, _refs, _accs, _scorer, res = _segmented()
    back = replay_best_hit_segments(orf, _as_span_rows(orf, res), min_len=50)
    assert [s.removed for s in back.segments] == [s.removed for s in res.segments]


def test_ref_index_is_absent_rather_than_invented():
    """The span file records the winning accession, not its catalogue row, so the row comes back -1.

    Pinned because the honest answer here is a sentinel, and a later change that quietly filled it with
    a plausible-looking index would be inventing provenance. Nothing outside `decompose` reads it.
    """
    orf, _refs, _accs, _scorer, res = _segmented()
    back = replay_best_hit_segments(orf, _as_span_rows(orf, res), min_len=50)
    assert all(t.ref_index == -1 for t in back.trace)
    assert any(t.ref_index >= 0 for t in res.trace)


def test_absent_construct_replays_as_no_segments():
    """A construct whose first claim fell below the floor emits no rows, and that is not an error.

    Measured on the tiers this flag exists for: 11 of 336 on synth_real, 72 of 336 on synthetic, 102 of
    410 on the corpus. Such a construct has no segments, so `build_arms` falls back to the whole ORF
    exactly as it would after a real run. Its stop reason is not recoverable and is labelled, not
    guessed.
    """
    back = replay_best_hit_segments(PAYLOAD, [], min_len=50, floor=101.5)
    assert back.segments == ()
    assert back.trace == ()
    assert back.floor == pytest.approx(101.5)
    assert back.stop_reason == "replay:absent_from_span_file"


def test_wrong_orf_length_is_refused():
    """The cheapest way to point a span file at the wrong construct set, and the first thing checked."""
    orf, _refs, _accs, _scorer, res = _segmented()
    rows = _as_span_rows(orf, res)
    with pytest.raises(ReplayMismatch, match="orf_len"):
        replay_best_hit_segments(orf + "AAAA", rows, min_len=50)


def test_tampered_newly_masked_is_refused():
    """`newly_masked` is the replay's checksum on its own mask bookkeeping."""
    orf, _refs, _accs, _scorer, res = _segmented()
    rows = _as_span_rows(orf, res)
    rows[0]["newly_masked"] = str(int(rows[0]["newly_masked"]) + 1)
    with pytest.raises(ReplayMismatch, match="new residues"):
        replay_best_hit_segments(orf, rows, min_len=50)


def test_tampered_emitted_is_refused():
    """`emitted` is the checksum on the min_len rule, which is the other half of the bookkeeping."""
    orf, _refs, _accs, _scorer, res = _segmented()
    rows = _as_span_rows(orf, res)
    rows[0]["emitted"] = "0" if rows[0]["emitted"] == "1" else "1"
    with pytest.raises(ReplayMismatch, match="emits"):
        replay_best_hit_segments(orf, rows, min_len=50)


def test_out_of_order_iterations_are_refused():
    """Masking is order-dependent, so a shuffled or gapped trace cannot be replayed at all."""
    orf, _refs, _accs, _scorer, res = _segmented()
    rows = _as_span_rows(orf, res)
    rows[0]["iteration"] = "3"
    with pytest.raises(ReplayMismatch, match="order"):
        replay_best_hit_segments(orf, rows, min_len=50)


def test_a_different_min_len_changes_the_answer_and_is_caught():
    """`min_len` is not in the span file, so a run that changed it must not silently replay.

    It is caught by the `emitted` checksum rather than by a declared field, which is weaker than a
    recorded value would be but is not nothing: a min_len large enough to drop a segment flips
    `emitted` on the iteration that produced it. This test pins that the protection exists; it is also
    the reason `min_len` should be added to the span file if that file is ever revised.
    """
    orf, _refs, _accs, _scorer, res = _segmented()
    rows = _as_span_rows(orf, res)
    with pytest.raises(ReplayMismatch, match="emits"):
        replay_best_hit_segments(orf, rows, min_len=10_000)


def test_build_arms_replay_matches_build_arms_live():
    """End to end through the caller the script actually uses, including the whole-ORF fallback."""
    orf = DISTRACTOR + PAYLOAD
    refs = [PAYLOAD, DISTRACTOR, FILLER]
    accs = ["PAY", "DIS", "FIL"]
    scorer = AlignmentScorer(refs)
    orfs = {"orf1": orf}

    live = build_arms("p", orfs, PAYLOAD, ref_seqs=refs, scorer=scorer, db_residues=DB_RESIDUES,
                      ref_accessions=accs, exclude_idx=np.array([], dtype=int))
    rows = _as_span_rows(orf, live.bhs_result)
    replayed = build_arms("p", orfs, PAYLOAD, ref_seqs=refs, scorer=scorer, db_residues=DB_RESIDUES,
                          ref_accessions=accs, exclude_idx=np.array([], dtype=int), bhs_replay=rows)

    assert [(s.start, s.end, s.seq, s.removed) for s in replayed.bhs] == \
           [(s.start, s.end, s.seq, s.removed) for s in live.bhs]


def test_build_arms_replay_falls_back_to_the_whole_orf():
    """No segments means the whole ORF, flagged, so the comparison stays paired on the same constructs."""
    orfs = {"orf1": PAYLOAD}
    arms = build_arms("p", orfs, PAYLOAD, ref_seqs=[PAYLOAD], scorer=AlignmentScorer([PAYLOAD]),
                      db_residues=DB_RESIDUES, ref_accessions=["PAY"], bhs_replay=[])
    assert len(arms.bhs) == 1
    assert arms.bhs[0].start == 1 and arms.bhs[0].end == len(PAYLOAD)
    assert "fallback:bhs_found_nothing" in arms.bhs[0].removed


def test_replay_never_touches_the_scorer():
    """Structural, not behavioural: the saving is real only if no catalogue pass happens.

    A scorer that raises on use proves the replay path never scores anything. Without this the flag
    could "work" while still paying the cost it exists to avoid.
    """
    class Exploding:
        name = "must-not-be-called"

        def score_all(self, _q):  # pragma: no cover - the test fails if this runs
            raise AssertionError("the replay path scored against the catalogue")

    orf, _refs, _accs, _scorer, res = _segmented()
    arms = build_arms("p", {"orf1": orf}, PAYLOAD, ref_seqs=[PAYLOAD], scorer=Exploding(),
                      db_residues=DB_RESIDUES, ref_accessions=["PAY"],
                      bhs_replay=_as_span_rows(orf, res))
    assert len(arms.bhs) == len(res.segments)
