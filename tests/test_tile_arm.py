"""Tests for the fixed-window tiling arm (`decompose.tile_segments`, `arms.Arms.tile`).

The arm exists because the working record traced the whole of `auto_per_segment`'s residual filler effect to
the 144 constructs the motif rules never cut, where the motif arm's only query *is* the whole ORF. A
fixed window has nothing to look for and so cannot fail to find it, and it is the only segmenter in the
project that consults neither the motif catalogue nor the retrieval catalogue.

Three groups of tests, and the last two are the ones that would catch a real mistake:

* the segmenter's own invariants, including the two refusals;
* **the cross-check against the working record**, which priced three tiling policies over the 336
  `synth_real` ORFs and recorded exact window and residue counts. Reproducing those is a check against
  a measurement already in the record rather than against this file's own expectations;
* **the hand-tier validation this project's working rules requires** before any rule reaches the corpus or the synthetic
  tiers, and it does not merely pass, it bounds where the arm can work.
"""
import csv
from pathlib import Path
from statistics import median

import pytest

from pilot.arms import ARM_NAMES, PER_SEGMENT_ARMS, REPORT_ARM_NAMES, build_arms, segment_arms
from pilot.decompose import oracle_span, tile_binding_counts, tile_segments
from pilot.reference import iter_fasta

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmark" / "v1"
CATALOGUE = REPO / "results" / "gate_all173" / "reference_twilight.fasta"


# --------------------------------------------------------------------------- the segmenter

def test_coordinates_are_one_based_inclusive_and_the_sequence_matches():
    """Every other `Segment` in `decompose` is 1-based inclusive; an off-by-one here is silent."""
    p = "".join("ACDEFGHIKLMNPQRSTVWY"[i % 20] for i in range(500))
    for s in tile_segments(p, 200, 100):
        assert s.seq == p[s.start - 1 : s.end]
        assert s.length == s.end - s.start + 1 == len(s.seq)


def test_the_tiling_is_gapless():
    """`step <= window` is enforced so that no residue sits in no window. Checked, not assumed."""
    for n in (201, 250, 399, 400, 401, 1000, 3525):
        p = "M" * n
        for w, st in ((300, 150), (200, 100), (100, 50)):
            covered = set()
            for s in tile_segments(p, w, st):
                covered.update(range(s.start, s.end + 1))
            assert covered == set(range(1, n + 1)), f"n={n}  {w}/{st} left residues unqueried"


def test_the_tail_is_anchored_so_no_window_is_short():
    """A short final window is the one most likely to sit under the significance floor.

    It is moved back to end at the ORF's end instead, so it is full width and the C-terminus is still
    queried. The overlap that buys is visible in the coordinates and counted by `tile_binding_counts`.
    """
    segs = tile_segments("M" * 250, 200, 100)
    assert [(s.start, s.end) for s in segs] == [(1, 200), (51, 250)]
    assert all(s.length == 200 for s in segs)
    assert segs[-1].end == 250
    assert tile_binding_counts(segs)["anchored_tail"] == 1


def test_an_orf_no_longer_than_one_window_is_tagged_rather_than_hidden():
    """This arm then equals `whole`, which is 68.1's undecomposed subset: a column, not an inference."""
    segs = tile_segments("M" * 120, 200, 100)
    assert len(segs) == 1
    assert (segs[0].start, segs[0].end) == (1, 120)
    assert "tile:whole_orf" in segs[0].removed
    assert tile_binding_counts(segs) == {"segments": 1, "whole_orf": 1, "anchored_tail": 0}


def test_an_exact_multiple_needs_no_anchored_tail():
    segs = tile_segments("M" * 200, 200, 100)
    assert [(s.start, s.end) for s in segs] == [(1, 200)]
    assert tile_binding_counts(segs)["anchored_tail"] == 0


def test_a_gapped_policy_is_refused_rather_than_clamped():
    """`step > window` leaves residues in no window: a silent truncation of the query, so it raises."""
    with pytest.raises(ValueError, match="leave residues in no window"):
        tile_segments("M" * 500, 100, 200)


def test_a_window_below_min_len_is_refused_rather_than_returning_nothing():
    """Otherwise the arm comes back empty and reads as "the tiling found nothing" rather than as a bug."""
    with pytest.raises(ValueError, match="below min_len"):
        tile_segments("M" * 500, 40, 20, min_len=50)


@pytest.mark.parametrize("window,step", [(0, 100), (200, 0), (-1, 1)])
def test_non_positive_geometry_is_refused(window, step):
    with pytest.raises(ValueError, match="must be positive"):
        tile_segments("M" * 500, window, step)


def test_an_empty_protein_yields_no_segments():
    assert tile_segments("", 200, 100) == []


# ------------------------------------------------- the cross-check against a recorded measurement

@pytest.mark.skipif(not (BENCH / "items.csv").exists(), reason="benchmark/v1 not present")
def test_the_window_counts_reproduce_context_68_4():
    """this project priced three policies over the 336 `synth_real` ORFs and recorded exact counts.

    This is the check worth having: the numbers come from a section already in the record, measured
    before this function existed, so agreeing with them means the tiling implemented here is the tiling
    that was priced: including the anchored-tail rule, which changes the counts.
    """
    with open(BENCH / "items.csv", newline="", encoding="utf-8-sig") as fh:
        lens = [int(r["orf_len"]) for r in csv.DictReader(fh) if r["tier"] == "synth_real"]
    assert (len(lens), sum(lens)) == (336, 295_315), "the tier moved; 68.4's counts no longer apply"

    for window, step, n_windows, residues in ((300, 150, 1816, 543_015),
                                              (200, 100, 2788, 557_520),
                                              (100, 50, 5749, 574_900)):
        segs = [s for n in lens for s in tile_segments("M" * n, window, step)]
        assert len(segs) == n_windows, f"{window}/{step}: 68.4 recorded {n_windows}"
        assert sum(s.length for s in segs) == residues


# --------------------------------------------------------------------------- the arms wiring

def _orfs(n=701):
    return {"o1": "".join("ACDEFGHIKLMNPQRSTVWY"[i % 20] for i in range(n))}


def test_the_arm_is_opt_in_so_every_existing_caller_is_unchanged():
    """Adding an arm must never silently reinterpret a stored result (the reason `PER_SEGMENT_ARMS`
    took new names rather than redefining `auto`). Without a window there is no tile arm at all."""
    orfs = _orfs()
    payload = orfs["o1"][200:400]
    a = build_arms("p1", orfs, payload)
    assert a.tile == []
    assert "tile" not in a.queries()


def test_a_window_builds_the_arm_and_queries_it_one_window_at_a_time():
    orfs = _orfs()
    payload = orfs["o1"][200:400]
    a = build_arms("p1", orfs, payload, tile_window=200)
    assert len(a.tile) == 7
    assert a.queries()["tile"] == [s.seq for s in a.tile]


def test_the_step_defaults_to_half_the_window():
    """All three policies this project priced are 50% overlap, so the default is one already costed."""
    orfs = _orfs()
    payload = orfs["o1"][200:400]
    assert ([(s.start, s.end) for s in build_arms("p1", orfs, payload, tile_window=200).tile]
            == [(s.start, s.end) for s in
                build_arms("p1", orfs, payload, tile_window=200, tile_step=100).tile])


def test_the_tiling_survives_resegmentation_because_no_holdout_can_reach_it():
    """The windows come from the ORF's length, so `segment_arms` must leave them alone when it rebuilds
    `bhs` per threshold. That is what lets a tiling arm cost one pass for a whole six-holdout sweep."""
    orfs = _orfs()
    payload = orfs["o1"][200:400]
    a = build_arms("p1", orfs, payload, tile_window=200)
    before = [(s.start, s.end, s.seq) for s in a.tile]
    after = segment_arms(a)
    assert [(s.start, s.end, s.seq) for s in after.tile] == before


def test_dropping_the_tile_names_reproduces_the_arm_order_that_predates_them():
    """`tile` is appended rather than inserted, so a run that does not ask for it writes a file
    byte-identical to one from before the arm existed: a diff, not an argument (the `auto_per_segment`
    rule applied to a new query arm)."""
    assert ARM_NAMES[:4] == ("whole", "auto", "bhs", "oracle")
    assert tuple(a for a in REPORT_ARM_NAMES if "tile" not in a) == (
        "whole", "auto", "bhs", "oracle", "auto_per_segment", "bhs_per_segment")
    assert PER_SEGMENT_ARMS["tile"] == "tile_per_segment"


# ------------------------------------------- the hand-tier validation this project's working rules requires (rule 4)

@pytest.mark.skipif(not (CATALOGUE.exists() and (BENCH / "queries.faa").exists()),
                    reason="the 40,750-protein catalogue or benchmark/v1 is not present")
def test_the_hand_tier_bounds_where_the_tiling_arm_can_help():
    """Scored against the 23 hand-audited constructs before the arm touches any other tier.

    A tiling consults nothing, so there is no segmentation rule here to be right or wrong about. What
    the hand tier *can* say is whether the arm does the thing it exists to do, and the answer is a
    bound rather than a pass:

    * **purity is already 1.000 at the median**: the best window is pure payload, which is what the
      arm is for;
    * **and it buys almost nothing here**, because the whole ORF is already 0.911 payload at the
      median on this tier, while coverage of the payload by that best window falls to 0.685.

    That is the falsifying case `tile_segments` names, produced by the tier the rule sends it to: on
    long payloads every window is a fragment. `addgene_85400_Lenti_iCas9_neo` is 1,367 aa of Cas9 in a
    1,909 aa ORF and its best 200 aa window holds 15% of the payload. So the arm is scoped to low `f`
    with a payload short enough to sit inside a window, which is 68.1's `motifless_fusion` subset and
    not this tier, and that scoping was free.
    """
    orfs = {e.accession: e.seq for e in iter_fasta(BENCH / "queries.faa")}
    payloads = {e.accession: e.seq for e in iter_fasta(CATALOGUE)}
    with open(BENCH / "items.csv", newline="", encoding="utf-8-sig") as fh:
        items = [r for r in csv.DictReader(fh) if r["tier"] == "hand"]
    assert len(items) == 20, "the hand tier's scorable count moved"

    covs, purs, whole_purs = [], [], []
    for it in items:
        orf, pay = orfs[it["construct_id"]], payloads[it["payload_accession"]]
        span = oracle_span(orf, pay)
        assert span is not None, f"no oracle span for {it['construct_id']}"
        best_ov, best_len = -1, 0
        for s in tile_segments(orf, 200, 100):
            ov = max(0, min(s.end, span.end) - max(s.start, span.start) + 1)
            if ov > best_ov:
                best_ov, best_len = ov, s.length
        assert best_ov > 0, f"no window touches the payload of {it['construct_id']}"
        covs.append(best_ov / span.length)
        purs.append(best_ov / best_len)
        whole_purs.append(span.length / len(orf))

    assert median(purs) == pytest.approx(1.000, abs=0.001)
    assert median(whole_purs) == pytest.approx(0.911, abs=0.002)
    # the trade, recorded so that a regression in either direction shows up as a failure
    assert median(covs) == pytest.approx(0.685, abs=0.002)
    assert min(covs) == pytest.approx(0.146, abs=0.002)


@pytest.mark.skipif(not (CATALOGUE.exists() and (BENCH / "queries.faa").exists()),
                    reason="the 40,750-protein catalogue or benchmark/v1 is not present")
def test_a_narrower_window_trades_coverage_for_nothing_on_this_tier():
    """Three policies, one direction: coverage falls and purity does not rise, because it is already 1.

    This is the dose-response of the falsifying case, and it is why the arm's window is a parameter to
    be swept on a tier that has dilution rather than a constant to be chosen here.
    """
    orfs = {e.accession: e.seq for e in iter_fasta(BENCH / "queries.faa")}
    payloads = {e.accession: e.seq for e in iter_fasta(CATALOGUE)}
    with open(BENCH / "items.csv", newline="", encoding="utf-8-sig") as fh:
        items = [r for r in csv.DictReader(fh) if r["tier"] == "hand"]

    med_cov = []
    for window, step in ((300, 150), (200, 100), (100, 50)):
        covs = []
        for it in items:
            orf, pay = orfs[it["construct_id"]], payloads[it["payload_accession"]]
            span = oracle_span(orf, pay)
            best_ov = max(max(0, min(s.end, span.end) - max(s.start, span.start) + 1)
                          for s in tile_segments(orf, window, step))
            covs.append(best_ov / span.length)
        med_cov.append(median(covs))
    assert med_cov == sorted(med_cov, reverse=True), f"coverage should fall with the window: {med_cov}"
    assert med_cov[0] == pytest.approx(0.871, abs=0.002)
    assert med_cov[2] == pytest.approx(0.343, abs=0.002)
