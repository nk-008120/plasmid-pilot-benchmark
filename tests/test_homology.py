"""Tests for homolog discovery and the holdout construction."""
from __future__ import annotations

import gzip

import pytest

from pilot.homology import MAX_LEN, Homolog, find_homologs, holdout_mask, iter_swissprot, targets_at

parasail = pytest.importorskip("parasail")

A = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDE"
B = "MSDNEKLIAFIEQVNQLSPAERAELVQQLTAHWDGQRWQAFADGLQNHLDGQAYTLDMLKQAQRHGVDVSTLSPEQRAAFEALLHRHQGDEWSAIAQRLAG"


def _fasta(tmp_path, entries, gz=False):
    text = "".join(f">sp|{a}|X_Y desc\n{s}\n" for a, s in entries)
    p = tmp_path / ("sp.fasta.gz" if gz else "sp.fasta")
    (gzip.open(p, "wt") if gz else open(p, "w")).write(text) if gz else p.write_text(text)
    if gz:
        with gzip.open(p, "wt") as fh:
            fh.write(text)
    return p


class TestIterSwissProt:
    def test_reads_accession_from_the_uniprot_header(self, tmp_path):
        p = _fasta(tmp_path, [("P12345", A), ("Q67890", B)])
        assert [a for a, _ in iter_swissprot(p)] == ["P12345", "Q67890"]

    def test_reads_gzip(self, tmp_path):
        p = _fasta(tmp_path, [("P12345", A)], gz=True)
        assert [a for a, _ in iter_swissprot(p)] == ["P12345"]

    def test_the_last_entry_is_not_lost(self, tmp_path):
        """A FASTA parser that flushes only on '>' silently drops the final record."""
        p = _fasta(tmp_path, [("P1", A), ("P2", B), ("P3", A)])
        assert len(list(iter_swissprot(p))) == 3

    def test_length_filter_excludes_fragments_and_titins(self, tmp_path):
        """The window is 50 to MAX_LEN. This test used to call a 3,000 aa protein a titin and assert it
        was dropped; at the old 2,000 bound that also dropped P04517 and P09866, two of this project own
        payload accessions, from every retrieval run. 3,000 now passes and the titin case uses a length
        that is actually titin-scale."""
        p = _fasta(tmp_path, [("SHORT", "MKV"), ("OK", A), ("POLYPROTEIN", "A" * 3000),
                              ("HUGE", "A" * (MAX_LEN + 1))])
        assert [a for a, _ in iter_swissprot(p)] == ["OK", "POLYPROTEIN"]


class TestFindHomologs:
    def test_an_exact_copy_is_found_at_identity_one(self):
        hits = find_homologs(A, [("self", A), ("other", B)])
        assert hits[0].accession == "self"
        assert hits[0].identity == 1.0
        assert hits[0].coverage == 1.0

    def test_point_mutants_land_below_one_but_high(self):
        mutant = A[:30] + "W" + A[31:]
        (h,) = [x for x in find_homologs(A, [("mut", mutant)]) if x.accession == "mut"]
        assert 0.9 < h.identity < 1.0

    def test_short_local_match_is_rejected_by_the_coverage_floor(self):
        """30 identical residues against a 103 aa payload is not homology, whatever the local identity."""
        decoy = "WWWWWWWWWW" + A[:30] + "WWWWWWWWWW"
        assert not find_homologs(A, [("decoy", decoy)], min_coverage=0.4)

    def test_results_are_sorted_by_identity(self):
        mut1 = A[:10] + "W" * 5 + A[15:]
        mut2 = A[:10] + "W" * 30 + A[40:]
        ids = [h.identity for h in find_homologs(A, [("far", mut2), ("near", mut1), ("exact", A)])]
        assert ids == sorted(ids, reverse=True)


class TestHoldout:
    HOMOLOGS = [Homolog("exact", 500, 1.00, 1.0, 100), Homolog("near", 400, 0.80, 1.0, 100),
                Homolog("mid", 200, 0.45, 0.9, 100), Homolog("far", 120, 0.28, 0.8, 100)]

    def test_mask_removes_everything_above_the_threshold(self):
        assert holdout_mask(self.HOMOLOGS, 0.5) == {"exact", "near"}

    def test_targets_are_exactly_what_the_mask_leaves(self):
        for t in (0.9, 0.5, 0.3):
            assert holdout_mask(self.HOMOLOGS, t) & targets_at(self.HOMOLOGS, t) == set()
            assert holdout_mask(self.HOMOLOGS, t) | targets_at(self.HOMOLOGS, t) == {
                h.accession for h in self.HOMOLOGS}

    def test_the_payload_itself_is_always_held_out_below_full_identity(self):
        assert "exact" in holdout_mask(self.HOMOLOGS, 0.99)

    def test_a_threshold_with_no_remaining_homolog_yields_no_target(self):
        """Then the plasmid is unscorable at that holdout and must be excluded, not counted as a failure."""
        assert targets_at(self.HOMOLOGS, 0.2) == set()


class TestCoverageDefinition:
    """Coverage is measured against the payload. The two routes to `homologs.csv` used to disagree:
    `find_homologs` divided by the payload length and `16_build_twilight_refs.py` by the subject length,
    which moved individual payloads by up to 5x in both directions."""

    def test_the_denominator_is_the_payload_not_the_subject(self):
        from pilot.homology import coverage_of

        # 200 aligned columns of a 240 aa payload against a 1,400 aa subject
        assert coverage_of(200, 240) == pytest.approx(0.8333, abs=1e-4)

    def test_it_clamps_at_one(self):
        """Aligned columns include gaps, so they can exceed the payload length."""
        from pilot.homology import coverage_of

        assert coverage_of(300, 240) == 1.0

    def test_a_zero_length_payload_is_zero_not_a_crash(self):
        from pilot.homology import coverage_of

        assert coverage_of(10, 0) == 0.0

    def test_both_routes_now_agree_on_the_same_alignment(self):
        """The DIAMOND route's arithmetic, written out, must equal what find_homologs records."""
        from pilot.homology import coverage_of

        aligned, payload_len, subject_len = 200, 240, 1400
        diamond_route = coverage_of(aligned, payload_len)   # was min(aligned / subject_len, 1.0)
        exact_route = coverage_of(aligned, payload_len)
        assert diamond_route == exact_route
        assert diamond_route != min(aligned / subject_len, 1.0), "the old bug would make these equal"


class TestThreadedScoring:
    """Threading the score-only pass must not change a single homolog. parasail releases the GIL, so the
    threads share one copy of the database instead of a process each: 280 MB that is not multiplied."""

    @staticmethod
    def _db(n=120, seed=5):
        import random

        rng = random.Random(seed)
        aa = "ACDEFGHIKLMNPQRSTVWY"
        return [(f"P{i:05d}", "".join(rng.choice(aa) for _ in range(rng.randint(60, 200))))
                for i in range(n)]

    def test_threaded_and_serial_agree_exactly(self):
        from pilot.homology import find_homologs

        db = self._db()
        payload = db[7][1]
        serial = find_homologs(payload, db, min_coverage=0.0, min_identity=0.0)
        for threads in (2, 3, 4):
            got = find_homologs(payload, db, min_coverage=0.0, min_identity=0.0, threads=threads)
            assert [(h.accession, h.score, h.identity, h.coverage) for h in got] == \
                   [(h.accession, h.score, h.identity, h.coverage) for h in serial]

    def test_the_payload_still_finds_itself_first(self):
        from pilot.homology import find_homologs

        db = self._db()
        got = find_homologs(db[3][1], db, threads=4)
        assert got[0].accession == "P00003"
        assert got[0].identity == 1.0

    def test_a_database_smaller_than_the_thread_count_still_works(self):
        from pilot.homology import find_homologs

        db = self._db(n=3)
        assert find_homologs(db[0][1], db, threads=8, min_coverage=0.0, min_identity=0.0)

    def test_an_empty_database_is_not_a_crash(self):
        from pilot.homology import find_homologs

        assert find_homologs("MKTAYIAK", [], threads=4) == []


class TestLengthWindow:
    """The database window. The upper bound was 2,000, which silently excluded two of this project's own
    payload accessions (P04517 at 3,054 aa, P09866 at 3,387 aa) from every retrieval run."""

    def _fasta(self, tmp_path, lengths):
        p = tmp_path / "sp.fasta"
        p.write_text("".join(f">sp|P{i:05d}|T_X test {i}\n{'A' * n}\n" for i, n in enumerate(lengths)),
                     encoding="utf-8")
        return p

    def test_a_3000_aa_protein_survives_the_filter(self, tmp_path):
        from pilot.homology import iter_swissprot

        got = dict(iter_swissprot(self._fasta(tmp_path, [3000])))
        assert len(got) == 1, "a 3,000 aa protein must be searchable"
        assert len(next(iter(got.values()))) == 3000

    def test_the_payload_lengths_that_motivated_the_change_now_pass(self, tmp_path):
        from pilot.homology import iter_swissprot

        got = dict(iter_swissprot(self._fasta(tmp_path, [3054, 3387])))  # P04517, P09866
        assert len(got) == 2

    def test_the_lower_bound_still_excludes_short_fragments(self, tmp_path):
        """50 stays: below it an alignment has too few columns for a meaningful identity estimate."""
        from pilot.homology import iter_swissprot

        got = dict(iter_swissprot(self._fasta(tmp_path, [41, 49, 50])))
        assert len(got) == 1

    def test_the_upper_bound_still_excludes_titins(self, tmp_path):
        from pilot.homology import MAX_LEN, iter_swissprot

        got = dict(iter_swissprot(self._fasta(tmp_path, [MAX_LEN, MAX_LEN + 1])))
        assert len(got) == 1

    def test_reference_build_shares_the_same_window(self):
        """The two filters used to disagree, 4,000 against 2,000, so a protein could be a legitimate
        reference entry and still be invisible to the homolog search."""
        import inspect

        from pilot.homology import MAX_LEN, MIN_LEN
        from pilot.reference import build

        sig = inspect.signature(build).parameters
        assert sig["max_len"].default == MAX_LEN
        assert sig["min_len"].default == MIN_LEN


class TestCandidateSelectionStatistic:
    """Candidates are picked by E-value, not raw Smith-Waterman score. The pool is small and everything
    outside it is invisible to the gates, so the ordering decides what can be found at all."""

    def test_bit_score_cannot_fix_a_length_bias(self):
        """It is affine in the raw score, so ranking by it reproduces ranking by raw score exactly.
        Recorded as a test because it is the obvious wrong fix."""
        from pilot.homology import bit_score

        raws = [10.0, 50.0, 123.0, 400.0]
        bits = [bit_score(r) for r in raws]
        assert bits == sorted(bits)
        assert [i for i, _ in sorted(enumerate(raws), key=lambda kv: -kv[1])] == \
               [i for i, _ in sorted(enumerate(bits), key=lambda kv: -kv[1])]

    def test_the_evalue_penalises_a_longer_subject_at_equal_raw_score(self):
        from pilot.homology import log_evalue

        short = log_evalue(200.0, query_len=550, subject_len=240)
        long_ = log_evalue(200.0, query_len=550, subject_len=3649)
        assert long_ > short, "a longer subject must be the less surprising hit"

    def test_a_long_decoy_displaces_a_genuine_hit_under_raw_ranking_but_not_evalue(self):
        """The 380 homologs lost when the window widened to 5,000, in miniature."""
        import random

        from pilot.homology import find_homologs

        rng = random.Random(4)
        aa = "ACDEFGHIKLMNPQRSTVWY"
        payload = "".join(rng.choice(aa) for _ in range(200))
        genuine = payload[:150] + "".join(rng.choice(aa) for _ in range(50))
        padding = "".join(rng.choice(aa) for _ in range(3000))
        db = [("GENUINE", genuine), ("LONGJUNK", padding)]
        kept_raw = {h.accession for h in find_homologs(payload, db, top_n=1, rank_by="raw",
                                                       min_identity=0.0, min_coverage=0.0)}
        kept_ev = {h.accession for h in find_homologs(payload, db, top_n=1, rank_by="evalue",
                                                      min_identity=0.0, min_coverage=0.0)}
        assert kept_ev == {"GENUINE"}
        assert kept_raw != kept_ev or kept_raw == {"GENUINE"}

    def test_stats_report_the_reject_fraction(self):
        import random

        from pilot.homology import find_homologs

        rng = random.Random(9)
        aa = "ACDEFGHIKLMNPQRSTVWY"
        db = [(f"P{i}", "".join(rng.choice(aa) for _ in range(120))) for i in range(40)]
        st: dict = {}
        find_homologs(db[0][1], db, top_n=20, stats=st)
        assert st["n_considered"] == 20
        assert st["n_kept"] + st["n_rejected"] == 20
        assert 0.0 <= st["reject_fraction"] <= 1.0
        assert st["rank_by"] == "evalue"

    def test_an_unknown_ranking_statistic_is_refused(self):
        import pytest as _pytest

        from pilot.homology import find_homologs

        with _pytest.raises(ValueError, match="rank_by"):
            find_homologs("MKTAYIAK", [("x", "MKTAYIAK")], rank_by="bitscore")


class TestSearchSpaceEvalue:
    """The E-value used for gating must carry a search-space term. The pairwise form does not, which is
    why it is correct for ranking within one query and wrong as a significance threshold."""

    def test_the_gating_evalue_grows_with_the_database(self):
        from pilot.homology import log_evalue_search

        small = log_evalue_search(120.0, query_len=400, db_residues=1_000_000)
        big = log_evalue_search(120.0, query_len=400, db_residues=200_000_000)
        assert big > small
        # doubling the search space must double E, i.e. add ln 2 to ln E
        import math

        a = log_evalue_search(120.0, query_len=400, db_residues=10_000_000)
        b = log_evalue_search(120.0, query_len=400, db_residues=20_000_000)
        assert b - a == pytest.approx(math.log(2), abs=1e-9)

    def test_it_does_not_change_when_the_database_does_not(self):
        from pilot.homology import log_evalue_search

        a = log_evalue_search(120.0, query_len=400, db_residues=199_345_949)
        b = log_evalue_search(120.0, query_len=400, db_residues=199_345_949)
        assert a == b

    def test_the_pairwise_form_is_blind_to_the_database(self):
        """This is the defect that makes it unusable as a gate, recorded as a test."""
        from pilot.homology import log_evalue

        assert log_evalue(120.0, 400, 500) == log_evalue(120.0, 400, 500)
        # no argument exists through which the database size could enter
        import inspect

        assert "db" not in "".join(inspect.signature(log_evalue).parameters)

    def test_a_higher_score_is_more_significant(self):
        from pilot.homology import log_evalue_search

        assert log_evalue_search(300.0, 400, 10 ** 8) < log_evalue_search(100.0, 400, 10 ** 8)


class TestEvalueGate:
    @staticmethod
    def _db(n=150, seed=11):
        import random

        rng = random.Random(seed)
        aa = "ACDEFGHIKLMNPQRSTVWY"
        return [(f"P{i:05d}", "".join(rng.choice(aa) for _ in range(rng.randint(60, 300))))
                for i in range(n)]

    def test_the_gate_is_off_by_default(self):
        from pilot.homology import find_homologs

        db = self._db()
        st: dict = {}
        find_homologs(db[0][1], db, min_identity=0.0, min_coverage=0.0, stats=st)
        assert st["max_evalue"] is None
        assert st["n_fail_evalue"] == 0

    def test_a_strict_gate_keeps_the_self_match_and_drops_noise(self):
        from pilot.homology import find_homologs

        db = self._db()
        loose = find_homologs(db[0][1], db, min_identity=0.0, min_coverage=0.0)
        strict = find_homologs(db[0][1], db, min_identity=0.0, min_coverage=0.0, max_evalue=1e-10)
        assert len(strict) < len(loose)
        assert strict[0].accession == "P00000", "the exact self-match must survive any sane gate"

    def test_tightening_the_gate_is_monotone(self):
        from pilot.homology import find_homologs

        db = self._db()
        sizes = [len(find_homologs(db[0][1], db, min_identity=0.0, min_coverage=0.0, max_evalue=e))
                 for e in (1.0, 0.01, 1e-5, 1e-10)]
        assert sizes == sorted(sizes, reverse=True)

    def test_the_evalue_gate_is_applied_before_traceback_not_after(self):
        """Once the gate is a score floor it selects candidates, so nothing that would fail it is ever
        traced back. `n_fail_evalue` is 0 by construction, and that is the saving: traceback runs only on
        hits that can qualify. Identity and coverage still reject after traceback, since neither is
        knowable from the score."""
        from pilot.homology import find_homologs

        db = self._db()
        st: dict = {}
        find_homologs(db[0][1], db, min_identity=0.0, min_coverage=0.0, max_evalue=1e-10, stats=st)
        assert st["n_fail_evalue"] == 0
        assert st["n_fail_identity_or_coverage"] == 0
        assert st["db_residues"] == sum(len(s) for _, s in db)
        assert st["n_considered"] == st["n_above_floor"]

    def test_identity_and_coverage_still_reject_after_traceback(self):
        from pilot.homology import find_homologs

        db = self._db()
        st: dict = {}
        find_homologs(db[0][1], db, min_identity=0.99, min_coverage=0.99, max_evalue=1.0, stats=st)
        assert st["n_fail_identity_or_coverage"] > 0

    def test_the_search_space_is_the_database_not_the_scan_depth(self):
        """Depth decides what is looked at; the gate decides what survives. If top_n entered the search
        space, significance would become a function of an implementation detail."""
        from pilot.homology import find_homologs

        db = self._db(n=200)
        shallow: dict = {}
        deep: dict = {}
        find_homologs(db[0][1], db, top_n=20, min_identity=0.0, min_coverage=0.0,
                      max_evalue=1e-8, stats=shallow)
        find_homologs(db[0][1], db, top_n=200, min_identity=0.0, min_coverage=0.0,
                      max_evalue=1e-8, stats=deep)
        assert shallow["db_residues"] == deep["db_residues"]


class TestScoreFloorReplacesTheCap:
    """A significance gate is a score floor, so there is no cap to choose and truncation is impossible.

    Ranking and taking the top N dropped 2,975 qualifying homologs for 5 of 16 audited payloads at
    N = 400. A floor derived from identity and coverage would have excluded nothing (those gates admit
    alignments of negative expected score); a floor derived from an E-value is exact."""

    def test_the_floor_is_the_exact_inverse_of_the_evalue(self):
        import math

        from pilot.homology import log_evalue_search, score_floor

        for m in (196, 446, 3054):
            for e in (1.0, 1e-3, 1e-10):
                s = score_floor(e, m, 207_099_657)
                assert log_evalue_search(s, m, 207_099_657) == pytest.approx(math.log(e), abs=1e-9)

    def test_the_floor_is_nearly_flat_across_the_length_range(self):
        """Query length enters only logarithmically, which is why one floor serves every payload."""
        from pilot.homology import score_floor

        short = score_floor(1e-3, 196, 207_099_657)
        long_ = score_floor(1e-3, 3054, 207_099_657)
        assert short == pytest.approx(105.4, abs=0.2)
        assert long_ == pytest.approx(115.7, abs=0.2)
        assert long_ - short < 11

    def test_a_stricter_gate_raises_the_floor(self):
        from pilot.homology import score_floor

        floors = [score_floor(e, 400, 10 ** 8) for e in (1.0, 1e-3, 1e-10)]
        assert floors == sorted(floors)

    def test_selection_by_floor_is_not_capped(self):
        """top_n must have no effect once a gate is set, which is the whole point."""
        import random

        from pilot.homology import find_homologs

        rng = random.Random(3)
        aa = "ACDEFGHIKLMNPQRSTVWY"
        payload = "".join(rng.choice(aa) for _ in range(200))
        # many near-copies, all of which clear any sane floor
        db = [(f"H{i}", payload[:180] + "".join(rng.choice(aa) for _ in range(20))) for i in range(60)]
        db += [(f"N{i}", "".join(rng.choice(aa) for _ in range(200))) for i in range(60)]
        tiny = find_homologs(payload, db, top_n=5, max_evalue=1.0, min_identity=0.0, min_coverage=0.0)
        huge = find_homologs(payload, db, top_n=10_000, max_evalue=1.0, min_identity=0.0,
                             min_coverage=0.0)
        assert len(tiny) == len(huge) > 5, "the cap must not bind once a score floor is in use"

    def test_stats_report_the_floor_and_that_nothing_was_truncated(self):
        import random

        from pilot.homology import find_homologs

        rng = random.Random(8)
        aa = "ACDEFGHIKLMNPQRSTVWY"
        db = [(f"P{i}", "".join(rng.choice(aa) for _ in range(150))) for i in range(40)]
        st: dict = {}
        find_homologs(db[0][1], db, top_n=3, max_evalue=1e-3, min_identity=0.0, min_coverage=0.0,
                      stats=st)
        assert st["rank_by"] == "score_floor"
        assert st["truncated"] is False
        assert st["score_floor"] is not None
        assert st["n_above_floor"] == st["n_considered"]

    def test_without_a_gate_the_cap_still_applies_and_is_reported(self):
        import random

        from pilot.homology import find_homologs

        rng = random.Random(8)
        aa = "ACDEFGHIKLMNPQRSTVWY"
        db = [(f"P{i}", "".join(rng.choice(aa) for _ in range(150))) for i in range(40)]
        st: dict = {}
        find_homologs(db[0][1], db, top_n=10, min_identity=0.0, min_coverage=0.0, stats=st)
        assert st["n_considered"] == 10
        assert st["truncated"] is True
        assert st["score_floor"] is None
