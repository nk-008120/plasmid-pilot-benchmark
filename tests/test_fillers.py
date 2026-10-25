"""`pilot.fillers`: the filler selection rules, and the one that keeps the benchmark from inverting.

The correctness requirement is that a filler is never a relative of its own construct's payload. If it
were, the filler's family would be a *true* homolog of the answer sitting in the catalogue, retrieving it
would score as success, and a construct would look easier the more diluted it got: the benchmark would
measure the reverse of its claim. That is tested here directly rather than trusted, because nothing in
the output of a scoring run would reveal it: the numbers would simply be good.

The remaining tests pin the choices that have to stay fixed for conditions A, B and C to be paired:
a payload's fillers must not change with archetype or dilution, and the filler must never alter the
padding budget, since an ORF length that differed between conditions would change the `m` term of the
significance floor and confound the comparison the three conditions exist to make.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest

from pilot.fillers import (
    FillerDraw,
    RealProteinFiller,
    eligible_pool,
    fit_to_length,
    is_eligible_filler,
    load_homolog_sets,
)
from pilot.synthconstruct import ARCHETYPES, design, random_protein

PAYLOAD = "M" + "ACDEFGHIKLMNPQRSTVWY" * 15  # 301 aa


def pool(n=12, length=200):
    return [(f"Q{i:05d}", "".join(random.Random(i).choice("ACDEFGHIKLMNPQRSTVWY")
                                  for _ in range(length))) for i in range(n)]


def test_fit_to_length_tiles_truncates_and_reports_which(fil=None):
    assert fit_to_length("ABCDE", 5) == ("ABCDE", "exact")
    assert fit_to_length("ABCDEFGH", 3) == ("ABC", "truncated")
    out, mode = fit_to_length("ABC", 8)
    assert (out, mode) == ("ABCABCAB", "tiled")
    assert len(out) == 8


def test_fit_to_length_of_zero_is_empty_so_an_unused_slot_places_nothing():
    assert fit_to_length("ABC", 0) == ("", "exact")
    assert fit_to_length("ABC", -5) == ("", "exact")


def test_is_eligible_filler_rejects_a_vector_part_by_its_description():
    """`is_vector_part` classifies on the description. A beta-lactamase filler would put the construct's
    own resistance marker's family into the competition, so the arm would measure the backbone."""
    assert not is_eligible_filler("P62593", "Beta-lactamase TEM OS=Escherichia coli", forbidden=set())
    assert not is_eligible_filler("X1", "Uncharacterized protein OS=E. coli", forbidden=set())
    assert is_eligible_filler("X2", "Carbonic anhydrase 2 OS=Homo sapiens", forbidden=set())


def test_is_eligible_filler_rejects_an_explicitly_forbidden_accession():
    assert not is_eligible_filler("X2", "Carbonic anhydrase 2", forbidden={"X2"})


def test_load_homolog_sets_ignores_identity_entirely(tmp_path):
    """Presence, not degree: a filler must not be a payload relative at ANY holdout threshold, so a
    0.21-identity row disqualifies exactly as a 1.00 row does."""
    p = tmp_path / "h.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["payload_accession", "homolog_accession", "score",
                                           "identity", "coverage", "length"])
        w.writeheader()
        w.writerows([
            {"payload_accession": "P1", "homolog_accession": "P1", "score": 1, "identity": 1.0,
             "coverage": 1.0, "length": 100},
            {"payload_accession": "P1", "homolog_accession": "H9", "score": 1, "identity": 0.21,
             "coverage": 0.4, "length": 100},
            {"payload_accession": "P2", "homolog_accession": "H8", "score": 1, "identity": 0.9,
             "coverage": 0.9, "length": 100},
        ])
    rel, payloads = load_homolog_sets(p)
    assert rel["P1"] == {"P1", "H9"}
    assert payloads == {"P1", "P2"}


def test_eligible_pool_excludes_payloads_vector_parts_and_relatives():
    entries = [("A1", "AAAA", "Carbonic anhydrase"),
               ("A2", "AAAA", "Beta-lactamase TEM"),
               ("P1", "AAAA", "Some payload"),
               ("H9", "AAAA", "A relative of P1"),
               ("A3", "AAAA", "Alcohol dehydrogenase")]
    pool_, forbidden = eligible_pool(entries, payload_relatives={"P1": {"H9"}},
                                     tier_payloads={"P1"}, catalogue_payloads={"P1"})
    accs = [a for a, _ in pool_]
    assert "A2" not in accs          # vector part
    assert "P1" not in accs          # a payload
    assert "A1" in accs and "A3" in accs
    assert "H9" in accs              # in the pool globally...
    assert "H9" in forbidden["P1"]   #... but forbidden for P1 specifically


def test_the_inversion_rule_a_payload_relative_is_never_drawn_for_that_payload():
    """THE correctness test. Every eligible filler is checked against the payload's relative set."""
    entries = [(f"A{i}", "AA", "Carbonic anhydrase") for i in range(6)]
    relatives = {"P1": {"A0", "A1", "A2"}}
    pool_, forbidden = eligible_pool(entries, payload_relatives=relatives,
                                     tier_payloads={"P1"}, catalogue_payloads=set())
    f = RealProteinFiller(pool_, forbidden, seed=0)
    for _ in range(50):
        f.begin("P1")
        for acc, _seq in f._assignment:
            assert acc not in relatives["P1"]


def test_a_payload_whose_eligible_pool_is_too_small_raises_rather_than_reusing_a_relative():
    pool_ = [("A0", "AA"), ("A1", "AA")]
    f = RealProteinFiller(pool_, {"P1": {"A0"}}, seed=0)
    with pytest.raises(ValueError, match="only 1 eligible fillers"):
        f.begin("P1")


def test_an_empty_pool_is_an_error_at_construction(tmp_path):
    with pytest.raises(ValueError, match="empty filler pool"):
        RealProteinFiller([], {})


def test_calling_the_filler_without_begin_is_an_error_not_a_silent_random_draw():
    f = RealProteinFiller(pool(), {})
    with pytest.raises(RuntimeError, match="begin"):
        f(random.Random(0), 50)


def test_a_payloads_fillers_are_identical_across_archetypes_and_dilutions():
    """The dilution axis must vary the AMOUNT of filler and never its identity, or the experiment
    confounds the two. The caller's rng has been advanced a different number of times for each
    construct, so this only holds because the assignment is keyed on the payload, not on that rng."""
    p, forb = pool(), {}
    seen = set()
    for archetype in ("tagged_fusion", "motifless_fusion", "multi_gene_cassette"):
        for dil in (0.2, 0.4, 0.6, 0.8):
            f = RealProteinFiller(p, forb, seed=0)
            f.begin("P1")
            design(PAYLOAD, "P1", archetype, dil, random.Random(hash((archetype, dil)) % 10**6),
                   filler=f)
            seen.add(tuple(sorted({d.accession for d in f.last_draws})))
    # every construct drew from the same two-accession assignment for this payload
    assert len({a for t in seen for a in t}) <= 2


def test_different_payloads_get_different_assignments():
    p = pool(n=40)
    f = RealProteinFiller(p, {}, seed=0)
    f.begin("P1")
    a = {d for d, _ in f._assignment}
    f.begin("P2")
    b = {d for d, _ in f._assignment}
    assert a != b


def test_the_filler_does_not_change_the_orf_length_so_conditions_stay_paired():
    """A, B and C must have identical ORF lengths construct-for-construct: the padding budget is computed
    before the filler is consulted, and the filler is fitted to it exactly. An ORF length that differed
    between conditions would move the `m` term of the significance floor and confound the comparison."""
    p, forb = pool(), {}
    for archetype in ARCHETYPES:
        for dil in (0.0, 0.2, 0.4, 0.6, 0.8):
            rand_protein, _ = design(PAYLOAD, "P1", archetype, dil, random.Random(7))
            f = RealProteinFiller(p, forb, seed=0)
            f.begin("P1")
            real, _ = design(PAYLOAD, "P1", archetype, dil, random.Random(7), filler=f)
            assert len(real) == len(rand_protein), (archetype, dil)


def test_the_default_filler_is_still_random_protein_so_condition_a_is_unchanged():
    a, _ = design(PAYLOAD, "P1", "motifless_fusion", 0.6, random.Random(3))
    b, _ = design(PAYLOAD, "P1", "motifless_fusion", 0.6, random.Random(3), filler=random_protein)
    assert a == b


def test_draws_are_logged_with_the_mode_so_truncation_counts_can_be_reported():
    f = RealProteinFiller([("A0", "ACDEFGHIKL" * 5)], {}, seed=0, slots=1)  # 50 aa source
    f.begin("P1")
    f(random.Random(0), 20)
    f(random.Random(0), 500)
    assert [d.mode for d in f.last_draws] == ["truncated", "tiled"]
    assert [d.requested_aa for d in f.last_draws] == [20, 500]
    assert all(d.source_aa == 50 for d in f.last_draws)
    assert all(isinstance(d, FillerDraw) for d in f.draws)


def test_last_draws_resets_per_construct_but_draws_accumulates():
    f = RealProteinFiller(pool(), {}, seed=0)
    f.begin("P1")
    f(random.Random(0), 60)
    f.begin("P2")
    f(random.Random(0), 60)
    assert len(f.last_draws) == 1
    assert len(f.draws) == 2


def test_a_zero_length_slot_places_nothing_and_logs_nothing():
    """`design`'s tagged_fusion can compute a carrier of exactly 0. That is not a filler block and must
    not appear in the manifest as one."""
    f = RealProteinFiller(pool(), {}, seed=0)
    f.begin("P1")
    assert f(random.Random(0), 0) == ""
    assert f.last_draws == []


def test_restrict_to_keeps_only_accessions_present_in_the_search_database():
    """The correction that makes condition C mean what it says. The catalogue is not Swiss-Prot, it is
    the gated homologs of this project's 173 payloads, so a filler drawn uniformly from Swiss-Prot
    usually has no relative in the search database and competes for rank as little as random sequence
    does. Measured on the first build: 2 of 48 fillers were in the catalogue."""
    entries = [("A1", "AA", "Carbonic anhydrase"),
               ("A2", "AA", "Alcohol dehydrogenase"),
               ("A3", "AA", "Serum albumin")]
    pool_, _ = eligible_pool(entries, payload_relatives={}, tier_payloads=set(),
                             catalogue_payloads=set(), restrict_to={"A1", "A3", "NOT_IN_ENTRIES"})
    assert [a for a, _ in pool_] == ["A1", "A3"]


def test_restrict_to_none_is_the_unrestricted_pool():
    entries = [("A1", "AA", "Carbonic anhydrase"), ("A2", "AA", "Alcohol dehydrogenase")]
    pool_, _ = eligible_pool(entries, payload_relatives={}, tier_payloads=set(),
                             catalogue_payloads=set(), restrict_to=None)
    assert len(pool_) == 2


def test_restrict_to_is_applied_before_the_payload_relative_exclusion_not_instead_of_it():
    """Both filters must apply. A catalogue entry that is the payload's relative is still the inversion
    case and must still be forbidden."""
    entries = [("H9", "AA", "A relative"), ("A1", "AA", "Carbonic anhydrase")]
    pool_, forbidden = eligible_pool(entries, payload_relatives={"P1": {"H9"}},
                                     tier_payloads={"P1"}, catalogue_payloads=set(),
                                     restrict_to={"H9", "A1"})
    assert {a for a, _ in pool_} == {"H9", "A1"}
    assert "H9" in forbidden["P1"]
