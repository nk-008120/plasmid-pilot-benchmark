"""the corpus tier's one catastrophic `bhs_per_segment` failure, and its cause.

`benchmark/v1/baseline_alignment.json`, corpus tier, `f < 0.50`, 0.70 holdout, n = 6: `bhs_per_segment`
mean rank **2205.0** against `whole` **1.33**, from one construct of the six at rank 13,225. The cause is
not the segmenter and not the label. `scripts/10_twilight.py` builds the segmenter's holdout mask **once,
at the strictest threshold in the sweep**, and reuses that one segmentation at every looser threshold. A
construct whose payload half has *every* relative masked at the strictest threshold cannot be claimed at
all, so the loop stops at k = 1 holding only the co-encoded gene, and is then scored as a decomposition
failure at the looser thresholds, where the segmenter run under that holdout's own mask claims the payload
and returns rank 1.

Two groups:

1. **The mechanism, on a tiny catalogue.** Three tests over the same two-gene construct the segmenter's
   own tests use. They pin that the mask *choice* and not the floor is what produces the k = 1 miss, which
   is the distinction the 20 Oct investigation had to measure rather than assume.
2. **The named cases, as recorded.** Gated on the corpus run being in the checkout. These pin the two
   constructs and their recorded ranks so the case stays named rather than being rediscovered: this project's working rules
   asks for the falsifying case to be named, and `bhs_spans_*.csv` exists precisely so a miss can be told
   apart from a chimera.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pilot.align import AlignmentScorer
from pilot.decompose import best_hit_segments

# the same three proteins and the same construct architecture as tests/test_best_hit_segments.py:
# DISTRACTOR (the co-encoded reporter) then PAYLOAD (the gene of interest)
from tests.test_best_hit_segments import DISTRACTOR, FILLER, PAYLOAD, _mutate

parasail = pytest.importorskip("parasail", reason="the alignment arm is an optional dependency")

ROOT = Path(__file__).resolve().parents[1]
SPANS = ROOT / "results/persegment_corpus/bhs_spans_sw-blosum62-11-1_corpus.csv"
TWILIGHT = ROOT / "results/persegment_corpus/twilight_sw-blosum62-11-1_corpus.csv"
needs_corpus = pytest.mark.skipif(
    not (SPANS.is_file() and TWILIGHT.is_file()),
    reason="the per-segment corpus run is not in this checkout")

# ---------------------------------------------------------------------------
# 1. the mechanism, on a tiny catalogue
# ---------------------------------------------------------------------------

# ORF = DISTRACTOR (residues 1-198) then the payload half, FILLER (200-396). FILLER rather than
# `test_best_hit_segments`'s PAYLOAD on purpose: PAYLOAD and DISTRACTOR are both GFP-family, so a masked
# payload half still finds the distractor's relative and the fixture would not isolate the mask. FILLER is
# unrelated to both, which is the architecture `ncbi_M88535_1` has: a chemokine fused to a thrombin
# inhibitor. The payload half's only catalogue support is REL_PAY and EXACT_PAYLOAD; masking both is what
# the strictest threshold does to a payload whose entire homolog family sits above it.
ORF = DISTRACTOR + FILLER
PAY_START = len(DISTRACTOR) + 1
ACCS = ["REL_DIST", "REL_PAY", "BG1", "BG2", "EXACT_PAYLOAD"]
SEQS = [_mutate(DISTRACTOR, 4, 1), _mutate(FILLER, 4, 2), PAYLOAD, _mutate(PAYLOAD, 6, 3), FILLER]
# the real catalogue's in-window residue count, so the floor these tests run against is the project's own
DB_RESIDUES = 20_806_731


def _run(exclude):
    scorer = AlignmentScorer(SEQS)
    return best_hit_segments(ORF, SEQS, scorer, DB_RESIDUES, ref_accessions=ACCS,
                             max_evalue=1e-3, min_len=50,
                             exclude_idx=[ACCS.index(a) for a in exclude])


def _overlap(seg):
    return max(0, min(seg.end, len(ORF)) - max(seg.start, PAY_START) + 1)


def test_a_loose_mask_claims_both_halves():
    """The looser holdout, where only the payload's own entry is masked: k = 2, payload claimed."""
    res = _run(["EXACT_PAYLOAD"])
    assert len(res.segments) == 2, [(s.start, s.end) for s in res.segments]
    assert max(_overlap(s) for s in res.segments) >= len(FILLER) - 20
    assert {t.ref_accession for t in res.trace if t.emitted} == {"REL_DIST", "REL_PAY"}


def test_the_strictest_mask_leaves_k_equals_1_holding_only_the_co_encoded_gene():
    """Mask every payload relative and the payload half is unclaimable: k = 1, zero overlap.

    This is `ncbi_M88535_1` in miniature. The segmenter is not wrong, there is nothing in the catalogue
    left for the payload half to hit, but the one segment it emits contains none of the payload, so a
    per-segment arm built from it can only miss.
    """
    res = _run(["EXACT_PAYLOAD", "REL_PAY"])
    assert len(res.segments) == 1, [(s.start, s.end) for s in res.segments]
    assert res.segments[0].start < PAY_START
    assert _overlap(res.segments[0]) == 0
    assert [t.ref_accession for t in res.trace if t.emitted] == ["REL_DIST"]
    assert res.stop_reason == "below_floor"


def test_the_mask_choice_and_not_the_floor_is_what_loses_the_payload():
    """The same floor, the same ORF, the same catalogue: only the mask differs.

    The point the 20 Oct investigation had to settle. Under the loose mask the payload half's best hit
    clears the floor comfortably; under the strict mask the best thing left is below it. So the k = 1 miss
    is a property of which entries the segmenter was allowed to see, not of the stopping rule, and a
    per-threshold mask would not need a different floor to fix it.
    """
    loose, strict = _run(["EXACT_PAYLOAD"]), _run(["EXACT_PAYLOAD", "REL_PAY"])
    assert loose.floor == strict.floor
    payload_iter = next(t for t in loose.trace if t.ref_accession == "REL_PAY")
    assert payload_iter.score >= loose.floor
    assert strict.n_iterations == 1


# ---------------------------------------------------------------------------
# 2. the named cases, as recorded
# ---------------------------------------------------------------------------

# Quoted from results/persegment_corpus/bhs_spans_sw-blosum62-11-1_corpus.csv. `ncbi_M88535_1` is the
# CTAP-III(Leu21)--hirudin fusion of GenBank M88535.1, whose depositor annotates hirudin at 3964..4158 =
# ORF residues 98..162, exactly the oracle span. `ncbi_AF087958_1` is pMA632, a "translational fusion
# between truncated alkaline phosphatase and beta-galactosidase alpha fragment" in the deposit's own
# words. Both labels are the depositor's; neither failure is a label error.
NAMED = {
    "ncbi_M88535_1": dict(payload="P01050", orf_len=162, oracle=(98, 162), n_iterations=1,
                          claimed=(23, 88), hit="P50228", score=193.0, floor=96.07, masked=17),
    "ncbi_AF087958_1": dict(payload="P00634", orf_len=508, oracle=(1, 397), n_iterations=1,
                            claimed=(400, 469), hit="A1A831", score=303.0, floor=100.36, masked=27),
}


def _spans():
    with open(SPANS, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


@needs_corpus
def test_the_two_named_constructs_claimed_only_the_co_encoded_gene():
    rows = {r["plasmid_id"]: r for r in _spans() if r["plasmid_id"] in NAMED}
    assert set(rows) == set(NAMED)
    for pid, want in NAMED.items():
        r = rows[pid]
        assert r["payload_accession"] == want["payload"]
        assert int(r["orf_len"]) == want["orf_len"]
        assert (int(r["oracle_start"]), int(r["oracle_end"])) == want["oracle"]
        assert (int(r["seg_start"]), int(r["seg_end"])) == want["claimed"]
        assert r["hit_accession"] == want["hit"] and float(r["hit_score"]) == want["score"]
        assert round(float(r["floor"]), 2) == want["floor"]
        assert int(r["n_catalogue_entries_masked"]) == want["masked"]
        assert int(r["n_iterations"]) == want["n_iterations"] and r["stop_reason"] == "below_floor"
        assert r["emitted"] == "1"
        # the whole of the finding: one emitted segment, and not one residue of the payload in it
        assert int(r["overlap_residues"]) == 0 and float(r["frac_of_oracle_covered"]) == 0.0


@needs_corpus
def test_these_two_are_the_only_corpus_constructs_that_claim_nothing_of_the_payload():
    """The binding count, the class is 2 of the 305 constructs with an emitted segment.

    A limit nothing reaches is free; this one reaches two constructs, and reporting that is what makes it
    a measurement rather than a worry.
    """
    by_pid: dict[str, list[dict]] = {}
    for r in _spans():
        by_pid.setdefault(r["plasmid_id"], []).append(r)
    emitted = {p: [r for r in rs if r["emitted"] == "1"] for p, rs in by_pid.items()}
    emitted = {p: rs for p, rs in emitted.items() if rs}
    assert len(emitted) == 305
    zero = {p for p, rs in emitted.items() if all(int(r["overlap_residues"]) == 0 for r in rs)}
    assert zero == set(NAMED)


@needs_corpus
def test_the_failure_is_absent_at_the_headline_holdout_and_present_at_every_looser_one():
    """0.25 is where the segmentation mask IS the run's own, so the artefact cannot reach the headline.

    Both constructs are unscorable at 0.25 (no homolog survives the holdout), and both carry the blow-up
    at every threshold where they are scorable. That is the scoping claim the working record rests on, and it is
    read from the run rather than argued.
    """
    ranks: dict[tuple[str, str], dict[str, str]] = {}
    with open(TWILIGHT, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["plasmid_id"] in NAMED:
                ranks.setdefault((r["plasmid_id"], r["threshold"]), {})[r["arm"]] = r["best_target_rank"]
    assert ranks[("ncbi_M88535_1", "0.25")]["bhs_per_segment"] == ""
    assert ranks[("ncbi_AF087958_1", "0.25")]["bhs_per_segment"] == ""
    for pid, thr, whole, bhs in (("ncbi_M88535_1", "0.7", 1, 13225),
                                 ("ncbi_M88535_1", "0.9", 1, 13228),
                                 ("ncbi_M88535_1", "1.0", 1, 26362),
                                 ("ncbi_AF087958_1", "0.35", 1, 1470),
                                 ("ncbi_AF087958_1", "0.7", 1, 1470)):
        d = ranks[(pid, thr)]
        assert int(d["whole"]) == whole and int(d["bhs_per_segment"]) == bhs
        # pooled and per-segment agree, which is what k = 1 means: one query, one hit list
        assert d["bhs"] == d["bhs_per_segment"]
        # and the oracle span is retrievable throughout, so nothing about the catalogue is at fault
        assert int(d["oracle"]) <= 10
