"""Tests for corpus screening and deduplication."""
from __future__ import annotations

import random

import pytest

from pilot.corpus import (
    Record,
    canonical_kmers,
    jaccard,
    screen,
    sequence_hash,
    sketch,
    stratify,
)

RC = str.maketrans("ACGT", "TGCA")


def dna(n, seed=0):
    r = random.Random(seed)
    return "".join(r.choice("ACGT") for _ in range(n))


def rc(s):
    return s.translate(RC)[::-1]


def rec(acc, seq, n_cds=3):
    return Record(acc, f"{acc}.gbk", seq, n_cds, "circular", "test construct")


class TestSketch:
    def test_a_plasmid_and_its_reverse_complement_sketch_identically(self):
        """GenBank records the same plasmid in either orientation; a sketch that missed this would let
        every construct through twice."""
        s = dna(4000)
        assert jaccard(sketch(s), sketch(rc(s))) == 1.0
        assert sequence_hash(s) == sequence_hash(rc(s))

    def test_unrelated_sequences_share_nothing(self):
        assert jaccard(sketch(dna(4000, 1)), sketch(dna(4000, 2))) == 0.0

    def test_similarity_rises_with_shared_sequence(self):
        base = dna(5000, 3)
        near = base[:4750] + dna(250, 4)
        far = base[:2500] + dna(2500, 5)
        assert jaccard(sketch(base), sketch(near)) > jaccard(sketch(base), sketch(far))

    def test_kmers_containing_n_are_skipped(self):
        assert not any("N" in k for k in canonical_kmers("ACGT" * 10 + "N" * 20 + "ACGT" * 10))

    def test_sketch_of_a_sequence_shorter_than_k_is_empty(self):
        assert sketch("ACGT") == frozenset()
        assert jaccard(frozenset(), sketch(dna(1000))) == 0.0


class TestScreen:
    def test_size_window_is_enforced_at_both_ends(self):
        res = screen([rec("SHORT", dna(500)), rec("OK", dna(5000)), rec("HUGE", dna(40000))])
        assert [r.accession for r in res.kept] == ["OK"]
        assert res.rejected["too_short"] == ["SHORT"]
        assert res.rejected["too_long"] == ["HUGE"]

    def test_records_without_cds_are_dropped_by_default_and_keepable_on_request(self):
        rs = [rec("A", dna(5000), n_cds=0)]
        assert screen(rs).rejected["no_cds"] == ["A"]
        assert len(screen(rs, require_cds=False).kept) == 1

    def test_ambiguous_sequence_is_dropped(self):
        s = dna(5000)
        assert screen([rec("N", "N" * 500 + s[500:])]).rejected["ambiguous_bases"] == ["N"]

    def test_exact_duplicates_collapse_to_one(self):
        s = dna(5000, 7)
        res = screen([rec("A", s), rec("B", s), rec("C", dna(5000, 8))])
        assert len(res.kept) == 2
        assert len(res.rejected["exact_duplicate"]) == 1

    def test_a_reverse_complement_deposit_is_an_exact_duplicate(self):
        s = dna(5000, 9)
        res = screen([rec("FWD", s), rec("REV", rc(s))])
        assert len(res.kept) == 1
        assert len(res.rejected["exact_duplicate"]) == 1

    def test_near_duplicates_are_removed(self):
        """pEGFP-N1 vs pEGFP-C1: the same backbone with a small rearrangement."""
        base = dna(6000, 11)
        twin = base[:5900] + dna(100, 12)
        res = screen([rec("A", base), rec("B", twin)])
        assert len(res.kept) == 1
        assert res.rejected["near_duplicate"] == ["B"] or res.rejected["near_duplicate"] == ["A"]

    def test_genuine_homologs_survive_the_near_duplicate_filter(self):
        """The threshold must drop reruns of one construct without thinning real diversity: 90% shared
        sequence sketches at roughly 0.80 Jaccard, well under the 0.90 cut."""
        base = dna(6000, 13)
        cousin = base[:5400] + dna(600, 14)
        res = screen([rec("A", base), rec("B", cousin)])
        assert len(res.kept) == 2

    def test_the_better_annotated_copy_survives_a_duplicate_pair(self):
        s = dna(5000, 15)
        res = screen([rec("SPARSE", s, n_cds=1), rec("RICH", s, n_cds=9)])
        assert [r.accession for r in res.kept] == ["RICH"]

    def test_every_input_is_either_kept_or_given_a_reason(self):
        rs = [rec("A", dna(5000, 16)), rec("B", dna(500)), rec("C", dna(5000, 16)),
              rec("D", dna(5000, 17), n_cds=0)]
        res = screen(rs)
        accounted = len(res.kept) + sum(len(v) for v in res.rejected.values())
        assert accounted == len(rs)

    def test_counts_summarises_the_outcome(self):
        res = screen([rec("A", dna(5000, 18)), rec("B", dna(100))])
        assert res.counts["kept"] == 1
        assert res.counts["too_short"] == 1


class TestStratify:
    def test_each_archetype_is_capped(self):
        rs = [rec(f"A{i}", dna(5000, i)) for i in range(30)]
        got = stratify(rs, lambda r: "x" if int(r.accession[1:]) % 2 else "y", per_archetype=5)
        assert len(got) == 10

    def test_a_small_archetype_is_taken_whole_not_padded(self):
        rs = [rec(f"A{i}", dna(5000, i)) for i in range(8)]
        got = stratify(rs, lambda r: "rare" if r.accession == "A0" else "common", per_archetype=5)
        assert sum(1 for r in got if r.accession == "A0") == 1
        assert len(got) == 6

    def test_selection_is_deterministic_for_a_seed(self):
        rs = [rec(f"A{i}", dna(5000, i)) for i in range(30)]
        f = lambda r: "x"  # noqa: E731
        assert [r.accession for r in stratify(rs, f, 7, seed=1)] == \
               [r.accession for r in stratify(rs, f, 7, seed=1)]

    def test_different_seeds_can_select_differently(self):
        rs = [rec(f"A{i}", dna(5000, i)) for i in range(40)]
        f = lambda r: "x"  # noqa: E731
        assert [r.accession for r in stratify(rs, f, 5, seed=1)] != \
               [r.accession for r in stratify(rs, f, 5, seed=99)]
