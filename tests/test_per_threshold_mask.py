"""the segmenter's holdout mask is built per threshold, and the 1.00 column is guarded.

`tests/test_strictest_mask_segmentation.py` pins the **defect** of `scripts/10_twilight.py`
built the segmenter's mask once at `min(THRESHOLDS)` and reused it at every looser holdout. This file
pins the **fix**, and it has to cover three separate claims, because each one is a place the fix could be
wrong in a way no output would show:

1. **The predicate.** `segmentation_mask` is now the same predicate the scoring mask uses, plus the
   payload's own accession. The union has to be a provable no-op below 1.00 or the fix would have changed
   the headline holdout, where the working record established the artefact is zero by construction. That is
   checked against **every payload of both of this project's homolog tables**, not argued from
   "self-identity is 1.0".

2. **The `thr >= 1.0` branch, which is a different question and must stay one.** There the scoring mask
   is empty and the target is the payload accession alone. An empty *segmentation* mask
   there is the leak `best_hit_segments` documents as a correctness requirement: the segmenter finds the
   payload in the catalogue at 100% identity, claims exactly its span, and reproduces `oracle` while
   looking label-free. `26_bhs_protocol.py` has no guard and leaks on 28 of 43 constructs at
   `--threshold 1.0`. The leak is reproduced here on a tiny catalogue so that "the union
   is what prevents it" is a measurement rather than a claim about intent.

3. **`segment_arms` returns a new object.** The whole point of splitting it out of `build_arms` is that
   one base `Arms` is segmented once per threshold, so a caller holding threshold A's arms must not see
   threshold B's segments appear in them.

The old protocol stays reachable behind `--bhs-mask strictest` precisely so this project's figures stay
reproducible rather than becoming folklore, and the two settings have to agree at `min(THRESHOLDS)` by
construction. Both are checked.
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pilot.align import AlignmentScorer
from pilot.arms import Arms, segment_arms
from pilot.decompose import best_hit_segments
from pilot.homology import Homolog

# the same three proteins and the same construct architecture as the the working record tests
from tests.test_best_hit_segments import DISTRACTOR, FILLER, PAYLOAD, _mutate

parasail = pytest.importorskip("parasail", reason="the alignment arm is an optional dependency")

ROOT = Path(__file__).resolve().parents[1]
DB_RESIDUES = 20_806_731


@pytest.fixture(scope="module")
def tw():
    """`scripts/10_twilight.py` loaded by path: `scripts/` is not an importable package."""
    spec = importlib.util.spec_from_file_location("twilight", ROOT / "scripts" / "10_twilight.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# 1. the predicate
# ---------------------------------------------------------------------------

# One payload, `PAY`, with a homolog family spread across the sweep so that every threshold cuts it in a
# different place. `OTHER` is a second payload's row set, present only to check the mask is per payload.
# One identity per gap in the sweep, so that every pair of adjacent thresholds really does cut the
# family in a different place: otherwise "strictest differs from per-threshold" would pass or fail on
# the fixture rather than on the code.
IDX_OF = {a: i for i, a in enumerate(
    ["PAY", "H99", "H80", "H60", "H40", "H30", "H20", "ABSENT_FROM_CAT"])}
IDX_OF.pop("ABSENT_FROM_CAT")
HOM = {"PAY": [Homolog("PAY", 999.0, 1.00, 1.0, 100),
               Homolog("H99", 900.0, 0.99, 1.0, 100),
               Homolog("H80", 700.0, 0.80, 1.0, 100),
               Homolog("H60", 500.0, 0.60, 1.0, 100),
               Homolog("H40", 300.0, 0.40, 1.0, 100),
               Homolog("H30", 200.0, 0.30, 1.0, 100),
               Homolog("H20", 100.0, 0.20, 1.0, 100),
               # in the homolog table and NOT in the catalogue: the mask must skip it rather than raise
               Homolog("ABSENT_FROM_CAT", 90.0, 0.95, 1.0, 100)]}


def _mask(tw, thr, protocol="per-threshold"):
    return set(tw.segmentation_mask(HOM, IDX_OF, "PAY", thr, protocol=protocol).tolist())


def test_the_mask_tracks_the_threshold(tw):
    """`identity > thr`, the scoring mask's own predicate, one mask per threshold."""
    assert _mask(tw, 0.25) == {IDX_OF[a] for a in ("PAY", "H99", "H80", "H60", "H40", "H30")}
    assert _mask(tw, 0.35) == {IDX_OF[a] for a in ("PAY", "H99", "H80", "H60", "H40")}
    assert _mask(tw, 0.50) == {IDX_OF[a] for a in ("PAY", "H99", "H80", "H60")}
    assert _mask(tw, 0.70) == {IDX_OF[a] for a in ("PAY", "H99", "H80")}
    assert _mask(tw, 0.90) == {IDX_OF[a] for a in ("PAY", "H99")}


def test_the_mask_matches_the_scoring_mask_exactly_below_one(tw):
    """Below 1.00 the segmenter and the scorer see the same catalogue. That is the fix, stated as an
    identity rather than as an intention: the union with the payload accession adds nothing here."""
    for thr in (0.25, 0.35, 0.50, 0.70, 0.90):
        scoring = {IDX_OF[h.accession] for h in HOM["PAY"]
                   if h.identity > thr and h.accession in IDX_OF}
        assert _mask(tw, thr) == scoring, thr


def test_strictest_reproduces_the_old_protocol_and_agrees_at_the_headline(tw):
    """The the working record behaviour, kept reachable so its figures stay reproducible.

    One mask for the sweep, built at `min(THRESHOLDS)`; and the two protocols are *identical* at that
    threshold, which is why the project's 25%-holdout headline was never affected by the defect.
    """
    strict = {frozenset(_mask(tw, t, "strictest")) for t in tw.THRESHOLDS}
    assert len(strict) == 1
    assert _mask(tw, min(tw.THRESHOLDS), "strictest") == _mask(tw, min(tw.THRESHOLDS))
    # and it differs from the correct mask at every looser threshold, which is the defect's reach
    for thr in (0.35, 0.50, 0.70, 0.90, 1.00):
        assert _mask(tw, thr, "strictest") != _mask(tw, thr), thr


def test_a_homolog_outside_the_catalogue_is_skipped_not_raised(tw):
    """`ABSENT_FROM_CAT` sits at identity 0.95 and is not in the catalogue. this project's Q9TT33 is the
    real instance: in the file, excluded by the file's own 50 aa reader."""
    assert IDX_OF.get("ABSENT_FROM_CAT") is None
    assert all(i in set(IDX_OF.values()) for i in _mask(tw, 0.25))


def test_a_payload_with_no_homolog_rows_still_masks_itself(tw):
    """The degenerate case, and it must not come back empty: an empty segmentation mask is the leak."""
    assert _mask(tw, 0.25) and tw.segmentation_mask({}, IDX_OF, "PAY", 0.25).tolist() == [IDX_OF["PAY"]]


# ---------------------------------------------------------------------------
# 2. the thr >= 1.0 branch
# ---------------------------------------------------------------------------


def test_at_the_1_00_holdout_the_predicate_is_empty_and_the_union_is_the_whole_mask(tw):
    """`identity > 1.0` selects nothing, because an identity cannot exceed 1. So at 1.00 the mask is the
    payload's own entry and nothing else, which is the minimum that keeps the segmenter honest."""
    assert not [h for h in HOM["PAY"] if h.identity > 1.0]
    assert _mask(tw, 1.00) == {IDX_OF["PAY"]}


def test_the_union_is_a_no_op_below_one_for_every_real_payload():
    """The claim the fix rests on, checked against the project's own homolog tables rather than argued.

    If a payload's self-identity were ever below 1.0, the union would add an entry at some `thr < 1.0`
    and the fix would have moved the headline holdout. It does not: every payload in both tables carries
    a row at identity 1.0 for itself. Q9TT33 is the one payload without a self row, and it is excluded
    from the catalogue by the catalogue's own 50 aa reader (41 aa), so it never reaches the
    segmenter: named here rather than silently tolerated.
    """
    tables = [p for p in (ROOT / "results/gate_all173/homologs.csv",
                          ROOT / "results/route_b_homologs/homologs.csv") if p.is_file()]
    if not tables:
        pytest.skip("neither homolog table is in this checkout")
    for path in tables:
        best: dict[str, float] = {}
        with open(path, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r["homolog_accession"] == r["payload_accession"]:
                    best[r["payload_accession"]] = max(
                        best.get(r["payload_accession"], 0.0), float(r["identity"]))
        assert best, path
        assert {a for a, i in best.items() if i < 1.0} == set(), path


def test_an_empty_segmentation_mask_lets_the_segmenter_claim_the_payload_span():
    """The leak, reproduced. This is why the 1.00 column needs the union and not an empty mask.

    `EXACT_PAYLOAD` is the payload sitting in the catalogue at 100% identity. Unmasked, the segmenter's
    first claim is that entry and its span is the payload half: `oracle` reproduced by a method that
    never saw the label, which is the laundering `best_hit_segments`'s docstring names. Masked, the best
    thing left is `REL_PAY`, a genuine relative, and the claim is a real retrieval.
    """
    orf = DISTRACTOR + FILLER
    pay_start = len(DISTRACTOR) + 1
    accs = ["REL_DIST", "REL_PAY", "BG1", "BG2", "EXACT_PAYLOAD"]
    seqs = [_mutate(DISTRACTOR, 4, 1), _mutate(FILLER, 4, 2), PAYLOAD, _mutate(PAYLOAD, 6, 3), FILLER]
    scorer = AlignmentScorer(seqs)

    def run(exclude):
        return best_hit_segments(orf, seqs, scorer, DB_RESIDUES, ref_accessions=accs,
                                 max_evalue=1e-3, min_len=50,
                                 exclude_idx=[accs.index(a) for a in exclude])

    leaked, guarded = run([]), run(["EXACT_PAYLOAD"])
    payload_claim = next(t for t in leaked.trace if t.start >= pay_start - 1)
    assert payload_claim.ref_accession == "EXACT_PAYLOAD"
    assert payload_claim.identity == pytest.approx(1.0, abs=1e-6)
    # the guarded run claims the same region from a relative instead, so the span survives and the
    # 100%-identity shortcut does not
    assert "EXACT_PAYLOAD" not in {t.ref_accession for t in guarded.trace}
    assert "REL_PAY" in {t.ref_accession for t in guarded.trace if t.emitted}


# ---------------------------------------------------------------------------
# 3. segment_arms
# ---------------------------------------------------------------------------


def _base() -> Arms:
    orf = DISTRACTOR + FILLER
    return Arms("pid", "orf1", orf, None)


def test_segment_arms_returns_a_new_object_and_leaves_the_base_untouched():
    accs = ["REL_DIST", "REL_PAY", "BG1", "BG2", "EXACT_PAYLOAD"]
    seqs = [_mutate(DISTRACTOR, 4, 1), _mutate(FILLER, 4, 2), PAYLOAD, _mutate(PAYLOAD, 6, 3), FILLER]
    scorer = AlignmentScorer(seqs)
    base = _base()
    kw = dict(ref_seqs=seqs, scorer=scorer, db_residues=DB_RESIDUES, ref_accessions=accs,
              max_evalue=1e-3, min_len=50)
    loose = segment_arms(base, exclude_idx=np.array([4]), **kw)
    strict = segment_arms(base, exclude_idx=np.array([1, 4]), **kw)
    # the base never acquires a bhs arm, however many thresholds are segmented from it
    assert base.bhs == [] and base.bhs_result is None
    assert loose is not base and strict is not base and loose is not strict
    # and the two masks really do give two segmentations, which is the thing the sweep now measures
    assert len(loose.bhs) == 2 and len(strict.bhs) == 1
    # the shared, mask-independent fields are the same object's values, not recomputed
    assert strict.orf_seq == base.orf_seq and strict.auto == base.auto


def test_segment_arms_without_a_catalogue_is_a_no_op():
    """`18_eba_sweep.py` and `22_segment_swissprot.py` call `build_arms` with no scorer and get an empty
    `bhs`; the split must not change that."""
    base = _base()
    out = segment_arms(base)
    assert out is not base and out.bhs == [] and out.bhs_result is None


# ---------------------------------------------------------------------------
# 4. span_rows_at: every reader of a span file goes through it
# ---------------------------------------------------------------------------

OLD_ROWS = [{"plasmid_id": "p", "iteration": "1", "emitted": "1"},
            {"plasmid_id": "p", "iteration": "2", "emitted": "1"}]
NEW_ROWS = [{"plasmid_id": "p", "threshold": t, "iteration": "1", "emitted": "1"}
            for t in ("1.0", "0.9", "0.7", "0.5", "0.35", "0.25")]


def test_a_pre_20_oct_span_file_comes_back_unchanged():
    """No `threshold` column means one segmentation for the whole sweep, so there is nothing to pick."""
    from pilot.decompose import span_rows_at
    assert span_rows_at(OLD_ROWS) == OLD_ROWS
    assert span_rows_at(OLD_ROWS, 0.25) == OLD_ROWS
    assert span_rows_at([]) == []


def test_a_per_threshold_span_file_must_be_asked_for_a_threshold():
    """The failure this guard exists for is a WRONG NUMBER, not a crash.

    `scripts/49_route_b_kmer_arm.py` counts emitted spans to get `k_bhs`. Handed a six-threshold file
    and no threshold it would count six segmentations as one and report `k_bhs` up to six times too
    large: a number that looks like a measurement. So the reader refuses and names what it holds.
    """
    from pilot.decompose import span_rows_at
    with pytest.raises(ValueError, match="one segmentation per"):
        span_rows_at(NEW_ROWS)
    assert len(span_rows_at(NEW_ROWS, 0.25)) == 1
    assert span_rows_at(NEW_ROWS, 0.25)[0]["threshold"] == "0.25"
    # and float comparison, not string equality: 0.5 must find the row written as "0.5"
    assert len(span_rows_at(NEW_ROWS, 0.50)) == 1
    with pytest.raises(ValueError, match="no span rows at threshold"):
        span_rows_at(NEW_ROWS, 0.42)


def test_every_span_file_reader_in_the_repo_goes_through_it():
    """A grep, as the check. A new reader that opens a `bhs_spans_*.csv` and iterates it directly is
    the way this defect comes back, and it would come back as a six-fold `k_bhs` with no error."""
    import re
    hits = []
    for d in ("scripts", "src", "tools"):
        for f in sorted((ROOT / d).rglob("*.py")):
            txt = f.read_text(encoding="utf-8")
            if re.search(r"bhs_spans[^\"'\s]*\.csv|--spans", txt) and "csv.DictReader" in txt:
                hits.append((f.name, "span_rows_at" in txt or f.name == "10_twilight.py"))
    assert hits, "no span-file readers found: the grep is wrong, not the code"
    assert all(ok for _, ok in hits), [n for n, ok in hits if not ok]
