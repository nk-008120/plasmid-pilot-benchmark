"""Tests for the label-free iterative best-hit segmenter (`decompose.best_hit_segments`).

Three groups, matching the three things the arm has to be trusted on:

1. **The contamination guarantee.** The arm's whole value is that it is deployable, which is worth
   nothing if it can see the answer. Two tests: a structural one over the signature, and a behavioural
   one that deletes the payload from the catalogue and demands identical output.
2. **The two masking properties.** A claimed span must not win again, and masking must not manufacture a
   junction that scores as a new hit.
3. **The floor derivation.** The stopping rule is the significance gate inverted, and the database in it
   is the catalogue rather than Swiss-Prot. Getting that wrong stops the iteration early and silently,
   which is the failure mode that leaves no trace in the output.

The scorer here is the real `AlignmentScorer`, not a stub, so these exercise the same Smith-Waterman the
benchmark runs on. Catalogues are tiny, so they are fast.
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from pilot.align import AlignmentScorer
from pilot.decompose import MASK_CHAR, best_hit_segments, oracle_span
from pilot.homology import LAMBDA, K_PARAM, score_floor

parasail = pytest.importorskip("parasail", reason="the alignment arm is an optional dependency")

# Three unrelated, realistic-composition proteins. PAYLOAD is the thing we want found; DISTRACTOR stands
# for the co-encoded reporter or marker that the working record identifies as the real competitor; FILLER is
# catalogue background.
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


def _mutate(seq: str, every: int, seed: int = 0) -> str:
    """A diverged relative of `seq`: substitute one residue in `every`, deterministically."""
    rng = np.random.default_rng(seed)
    alpha = "ACDEFGHIKLMNPQRSTVWY"
    out = list(seq)
    for i in range(0, len(seq), every):
        out[i] = alpha[int(rng.integers(len(alpha)))]
    return "".join(out)


@pytest.fixture
def world():
    """A two-gene construct and a catalogue holding remote relatives of both, plus the payload itself.

    The ORF is DISTRACTOR then PAYLOAD, which is the real architecture: a reporter co-encoded with the
    gene of interest. The catalogue carries a 1-in-4-substituted relative of each, which is remote enough
    to be a twilight-zone hit but still the best thing available, plus the exact payload so the
    contamination tests have something to delete.
    """
    orf = DISTRACTOR + PAYLOAD
    accs = ["REL_DIST", "REL_PAY", "FILLER1", "FILLER2", "EXACT_PAYLOAD"]
    seqs = [_mutate(DISTRACTOR, 4, 1), _mutate(PAYLOAD, 4, 2), FILLER, _mutate(FILLER, 6, 3), PAYLOAD]
    return orf, accs, seqs


def _run(orf, accs, seqs, *, exclude=None, db_residues=None, max_evalue=1e-3, min_len=50):
    scorer = AlignmentScorer(seqs)
    return best_hit_segments(orf, seqs, scorer, db_residues or sum(len(s) for s in seqs),
                             ref_accessions=accs, max_evalue=max_evalue, min_len=min_len,
                             exclude_idx=exclude)


# ---------------------------------------------------------------------------
# 1. the contamination guarantee
# ---------------------------------------------------------------------------


def test_signature_cannot_accept_a_payload():
    """The structural guarantee: there is no parameter the label could be passed through.

    This is the test the brief asks for in the form it asks for: a guarantee rather than a promise.
    It fails loudly if someone later adds a convenience argument that reintroduces the label.
    """
    params = set(inspect.signature(best_hit_segments).parameters)
    for forbidden in ("payload_seq", "payload", "payload_protein", "answer", "target", "truth",
                      "payload_accession", "target_idx"):
        assert forbidden not in params, f"best_hit_segments must not take {forbidden!r}"
    assert params == {"orf_protein", "ref_seqs", "scorer", "db_residues", "ref_accessions",
                      "max_evalue", "min_len", "exclude_idx", "floor_fn"}


def test_floor_fn_is_called_with_no_label_bearing_argument():
    """`floor_fn` was added so the stopping rule can be in a cosine scorer's units (open item 25).

    It is a callable, so the guarantee above, no parameter the label could be passed through, has
    to be restated for what the callable receives. It gets the significance gate, the query length and
    the database residue count, and nothing else: no sequence, no accession, no index.
    """
    seen = []

    def spy(max_evalue, query_len, db_residues):
        seen.append((max_evalue, query_len, db_residues))
        return 0.0

    orf = "M" + "ACDEFGHIKLMNPQRSTVWY" * 12
    refs = ["ACDEFGHIKLMNPQRSTVWY" * 6, "WWWWYYYYFFFFLLLLIIII" * 5]
    best_hit_segments(orf, refs, AlignmentScorer(refs), 1_000_000, max_evalue=1e-3,
                      floor_fn=spy)
    assert seen, "floor_fn was never called, so it is not the stopping rule"
    assert all(args == (1e-3, len(orf), 1_000_000) for args in seen), seen


def test_identical_output_when_the_payload_is_deleted_from_the_catalogue(world):
    """Behavioural: with the holdout mask applied, the payload's own entry changes nothing.

    This is the arm's central claim. The mask excludes the payload (in the benchmark it is its own
    homolog at identity 1.000), so deleting the entry outright must be a no-op: the segmenter was never
    using it. Compared on spans and on the winning accessions, not just on segment count.
    """
    orf, accs, seqs = world
    exact = accs.index("EXACT_PAYLOAD")

    with_payload = _run(orf, accs, seqs, exclude=[exact])
    # delete it entirely and renumber; nothing else about the catalogue changes
    kept = [i for i in range(len(seqs)) if i != exact]
    without = _run(orf, [accs[i] for i in kept], [seqs[i] for i in kept], exclude=[])

    assert [(s.start, s.end, s.seq) for s in with_payload.segments] == \
           [(s.start, s.end, s.seq) for s in without.segments]
    assert [t.ref_accession for t in with_payload.trace] == [t.ref_accession for t in without.trace]
    assert [(t.start, t.end) for t in with_payload.trace] == [(t.start, t.end) for t in without.trace]
    assert with_payload.stop_reason == without.stop_reason


def test_an_unmasked_catalogue_would_leak_the_answer(world):
    """The negative control that justifies `exclude_idx` existing at all.

    Without the mask the exact payload wins and the segmenter reproduces the oracle span, which is the
    contamination the mask prevents. Asserting it here means the leak is a documented, tested property of
    the unmasked configuration rather than something a future reader has to take on trust.
    """
    orf, accs, seqs = world
    unmasked = _run(orf, accs, seqs)  # no exclude_idx
    assert unmasked.trace, "expected at least one span"
    assert unmasked.trace[0].ref_accession == "EXACT_PAYLOAD"
    oracle = oracle_span(orf, PAYLOAD)
    assert oracle is not None
    first = unmasked.trace[0]
    # it lands on the oracle span, which is exactly the leak
    assert abs(first.start - oracle.start) <= 2 and abs(first.end - oracle.end) <= 2


# ---------------------------------------------------------------------------
# 2. the two masking properties
# ---------------------------------------------------------------------------


def test_a_claimed_span_cannot_win_again(world):
    """Property 1: no span is claimed twice, and no two claimed spans overlap.

    The mechanism is that `X` scores at most 0 against every standard residue in BLOSUM62, so a masked
    stretch cannot contribute positively to a local alignment and Smith-Waterman's optimum excludes it.
    """
    orf, accs, seqs = world
    res = _run(orf, accs, seqs, exclude=[accs.index("EXACT_PAYLOAD")])
    spans = [(t.start, t.end) for t in res.trace]
    assert len(spans) == len(set(spans)), "a span was claimed twice"
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        assert not (a1 <= b2 and a2 <= b1), f"spans {(a1, b1)} and {(a2, b2)} overlap"
    # and every iteration made real progress, which is what bounds the loop
    assert all(t.n_masked_now > 0 for t in res.trace)


def test_the_mask_character_cannot_score_above_zero():
    """Property 1's mechanism, asserted directly on BLOSUM62 rather than inferred from behaviour.

    If a future scoring matrix gave `X` a positive score against anything, masking would stop working and
    every other test here could still pass by luck.
    """
    alpha = "ARNDCQEGHILKMFPSTWYVBZX*"
    mat = np.array(parasail.blosum62.matrix)
    xrow = mat[alpha.index(MASK_CHAR)]
    for aa in "ARNDCQEGHILKMFPSTWYV":
        assert xrow[alpha.index(aa)] <= 0, f"{MASK_CHAR} vs {aa} scores above zero"
    masked_block = MASK_CHAR * 120
    assert AlignmentScorer([PAYLOAD, DISTRACTOR, FILLER]).score_all(masked_block).max() <= 0


def test_masking_does_not_manufacture_a_junction():
    """Property 2: masking never scores higher than deletion would, because it creates no new seam.

    The construct is HALF_A + CLAIMED + HALF_B where the two halves come from one real protein that has
    been split apart. Deleting the claimed middle rejoins them into a chimera aligning to that protein
    full-length at full strength: a hit manufactured by the masking step itself. X-substitution leaves
    the spacer in place, so the same alignment has to pay for crossing it.
    """
    half = len(FILLER) // 2
    a, b = FILLER[:half], FILLER[half:]
    claimed = DISTRACTOR
    accs, seqs = ["WHOLE_FILLER", "REL_DIST"], [FILLER, _mutate(DISTRACTOR, 4, 1)]
    i = accs.index("WHOLE_FILLER")

    deleted = a + b  # what naive deletion would produce
    masked = a + MASK_CHAR * len(claimed) + b
    sc = AlignmentScorer(seqs)
    assert sc.score_all(deleted)[i] > sc.score_all(masked)[i], (
        "masking must be strictly more conservative than deletion"
    )


def test_a_bridging_alignment_cannot_emit_a_chimera():
    """Property 2's teeth: a span that bridges a mask is clipped, never emitted whole.

    Score alone does not stop a bridge: aligning through X costs about a point per residue, so a long
    pair of flanks still aligns end to end across a 198 aa mask. The emission clip is what prevents
    iteration 2 from handing back a segment that re-contains iteration 1's span.
    """
    half = len(FILLER) // 2
    a, b = FILLER[:half], FILLER[half:]
    orf = a + DISTRACTOR + b
    # the distractor's relative wins first and claims the middle; the whole filler then bridges the mask
    accs = ["REL_DIST", "WHOLE_FILLER"]
    seqs = [_mutate(DISTRACTOR, 4, 1), FILLER]

    # confirm the bridge really happens, or this test proves nothing
    masked = a + MASK_CHAR * len(DISTRACTOR) + b
    bridged = oracle_span(masked, FILLER)
    assert bridged is not None and bridged.start <= len(a) and bridged.end > len(a) + len(DISTRACTOR), (
        "expected a bridging alignment; this test's construction no longer exercises the property"
    )

    res = _run(orf, accs, seqs, min_len=40)
    claimed = [(t.start, t.end) for t in res.trace]
    assert claimed, "no span claimed"
    # no emitted segment may overlap any earlier iteration's claimed span
    for k, t in enumerate(res.trace):
        for a2, b2 in t.emitted_spans:
            for pa, pb in claimed[:k]:
                assert not (pa <= b2 and a2 <= pb), (
                    f"segment {(a2, b2)} from iteration {t.iteration} re-contains claimed span {(pa, pb)}"
                )
    # and no two emitted segments overlap each other
    allseg = [(s.start, s.end) for s in res.segments]
    for x in range(len(allseg)):
        for y in range(x + 1, len(allseg)):
            (a1, b1), (a2, b2) = allseg[x], allseg[y]
            assert not (a1 <= b2 and a2 <= b1), f"segments {allseg[x]} and {allseg[y]} overlap"


def test_masking_preserves_coordinates_and_segment_sequences(world):
    """Segment sequences come from the original ORF, so no mask character can reach a scored query."""
    orf, accs, seqs = world
    res = _run(orf, accs, seqs, exclude=[accs.index("EXACT_PAYLOAD")])
    for s in res.segments:
        assert MASK_CHAR not in s.seq, "a mask character leaked into a query"
        assert s.seq == orf[s.start - 1 : s.end]
        assert 1 <= s.start <= s.end <= len(orf)


def test_segments_below_min_len_are_masked_but_not_emitted(world):
    """A significant but short hit must still be masked, or the loop would reclaim it forever."""
    orf, accs, seqs = world
    res = _run(orf, accs, seqs, exclude=[accs.index("EXACT_PAYLOAD")], min_len=10_000)
    assert res.segments == ()  # nothing clears an impossible length floor
    assert res.trace, "spans were still claimed"
    assert all(not t.emitted for t in res.trace)
    assert all(t.n_masked_now > 0 for t in res.trace)


# ---------------------------------------------------------------------------
# 3. the floor derivation
# ---------------------------------------------------------------------------


def test_floor_is_the_significance_gate_inverted():
    """`score_floor` must be the exact inverse of the E-value, at the catalogue's residue count."""
    db = 20_806_731
    for m in (100, 196, 647, 3054):
        s_min = score_floor(1e-3, m, db)
        recovered = K_PARAM * m * db * math.exp(-LAMBDA * s_min)
        assert recovered == pytest.approx(1e-3, rel=1e-9)


def test_catalogue_floor_sits_8_61_points_below_the_swissprot_floor():
    """The trap the brief names: borrowing Swiss-Prot's floor stops the iteration early and silently.

    The offset is `ln(N_sprot / N_cat) / lambda` and is independent of query length, so one number covers
    the whole range.
    """
    cat, sprot = 20_806_731, 207_099_657
    expected = math.log(sprot / cat) / LAMBDA
    assert expected == pytest.approx(8.61, abs=0.01)
    for m in (100, 500, 3000):
        diff = score_floor(1e-3, m, sprot) - score_floor(1e-3, m, cat)
        assert diff == pytest.approx(expected, abs=1e-9)
        assert diff > 0, "the catalogue floor must be LOWER, or iteration stops early"


def test_the_floor_actually_stops_the_iteration(world):
    """An impossible floor stops immediately; a permissive one claims more. The gate is load-bearing."""
    orf, accs, seqs = world
    excl = [accs.index("EXACT_PAYLOAD")]
    strict = _run(orf, accs, seqs, exclude=excl, max_evalue=1e-300)
    assert strict.segments == () and strict.stop_reason == "below_floor"
    loose = _run(orf, accs, seqs, exclude=excl, max_evalue=10.0)
    assert len(loose.trace) >= len(_run(orf, accs, seqs, exclude=excl).trace)


def test_a_larger_database_raises_the_floor(world):
    """`N` is in the floor, so the database's identity is part of the result."""
    orf, accs, seqs = world
    small = _run(orf, accs, seqs, exclude=[accs.index("EXACT_PAYLOAD")], db_residues=10_000)
    big = _run(orf, accs, seqs, exclude=[accs.index("EXACT_PAYLOAD")], db_residues=200_000_000)
    assert big.floor > small.floor


# ---------------------------------------------------------------------------
# behaviour: does it do the thing it exists to do?
# ---------------------------------------------------------------------------


def test_it_separates_a_payload_from_a_co_encoded_distractor(world):
    """The mechanism from iteration 1 claims one gene, iteration 2 exposes the other.

    This is the arm's reason to exist, so it is asserted rather than left to the benchmark. Both genes
    must come back as separate segments, each matching its own half of the ORF.
    """
    orf, accs, seqs = world
    res = _run(orf, accs, seqs, exclude=[accs.index("EXACT_PAYLOAD")])
    assert len(res.segments) >= 2, f"expected both genes; got {len(res.segments)}"
    assert {t.ref_accession for t in res.trace} >= {"REL_DIST", "REL_PAY"}
    # one segment should cover the payload half, which starts after the distractor
    boundary = len(DISTRACTOR)
    assert any(s.start > boundary * 0.75 for s in res.segments), "no segment landed on the payload half"
    assert any(s.end < boundary * 1.25 for s in res.segments), "no segment landed on the distractor half"


def test_empty_and_degenerate_inputs():
    """An empty ORF, an empty catalogue and a fully excluded catalogue all return empty, not a crash."""
    assert best_hit_segments("", [PAYLOAD], AlignmentScorer([PAYLOAD]), 1000).segments == ()
    assert best_hit_segments(PAYLOAD, [], None, 1000).segments == ()
    res = best_hit_segments(PAYLOAD, [PAYLOAD], AlignmentScorer([PAYLOAD]), 1000, exclude_idx=[0])
    assert res.segments == () and res.stop_reason == "no_alignment"


def test_no_iteration_cap_but_termination_is_guaranteed():
    """Repeating one gene many times: the loop keeps going until the floor stops it, and does terminate.

    There is deliberately no iteration cap (the floor is the limit), so termination has to
    come from the masking making progress every round.
    """
    orf = (DISTRACTOR + FILLER) * 3
    accs = ["REL_DIST", "REL_FILL"]
    seqs = [_mutate(DISTRACTOR, 5, 1), _mutate(FILLER, 5, 2)]
    res = _run(orf, accs, seqs)
    assert res.n_iterations >= 3, "should claim several copies"
    assert res.n_iterations <= len(orf)  # the structural bound
    assert sum(t.n_masked_now for t in res.trace) <= len(orf)


# ---------------------------------------------------------------------------
# the arm as `build_arms` assembles it
# ---------------------------------------------------------------------------


def test_build_arms_falls_back_to_the_whole_orf_when_nothing_clears_the_floor(world):
    """A construct the segmenter finds nothing on must stay scorable, flagged, not silently dropped.

    Same convention as `decompose`, and it matters for the comparison as well as for deployability: an
    arm that goes unscorable on some constructs shrinks n, and a table paired on a different n is the
    error the working record exists to prevent.
    """
    from pilot.arms import build_arms

    orf, accs, seqs = world
    arms = build_arms("t", {"t|orf1": orf}, PAYLOAD, ref_seqs=seqs, scorer=AlignmentScorer(seqs),
                      db_residues=sum(len(s) for s in seqs), ref_accessions=accs,
                      max_evalue=1e-300)  # an impossible floor: nothing can clear it
    assert arms is not None
    assert arms.bhs_result.segments == ()
    assert len(arms.bhs) == 1
    assert arms.bhs[0].seq == arms.orf_seq
    assert arms.bhs[0].removed == ("fallback:bhs_found_nothing",)
    assert arms.queries()["bhs"] == [arms.orf_seq]


def test_build_arms_leaves_bhs_empty_without_a_catalogue(world):
    """The three existing callers pass no catalogue and must be unaffected."""
    from pilot.arms import build_arms

    orf, _, _ = world
    arms = build_arms("t", {"t|orf1": orf}, PAYLOAD)
    assert arms is not None and arms.bhs == [] and arms.bhs_result is None
    assert "bhs" not in arms.queries()


def test_queries_is_ordered_and_drops_empty_arms(world):
    """`queries()` is what every call site iterates, so its contract is worth pinning down."""
    from pilot.arms import ARM_NAMES, build_arms

    orf, accs, seqs = world
    arms = build_arms("t", {"t|orf1": orf}, PAYLOAD, ref_seqs=seqs, scorer=AlignmentScorer(seqs),
                      db_residues=sum(len(s) for s in seqs), ref_accessions=accs,
                      exclude_idx=[accs.index("EXACT_PAYLOAD")])
    q = arms.queries()
    assert list(q) == [a for a in ARM_NAMES if a in q]
    assert all(qs and all(s for s in qs) for qs in q.values())
    assert q["whole"] == [orf]
