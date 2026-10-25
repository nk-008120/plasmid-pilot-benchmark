"""Tests for sequence-based archetype classification.

Each class below locks in a rule that was wrong in a first version and corrected against the 23
hand-audited constructs. The corrections are the point; the rules look obvious only in hindsight.
"""
from __future__ import annotations

import pytest

from pilot.archetype_seq import FUSION_DILUTION, classify
from pilot.autolabel import Hit

GENE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQ" * 3
OTHER = "MSDNEKLIAFIEQVNQLSPAERAELVQQLTAHWDGQRWQAFADGLQNHLDGQAYTLDMLKQAQRHGVDVSTLSPEQRAAF" * 3
P2A = "ATNFSLLKQAGDVEENPGP"
TEV = "ENLYFQG"
FLAG3 = "DYKDHDGDYKDHDIDYKDDDDK"
PLASMID = 6000


def hit(oid, acc, desc="Adapter molecule crk", *, ident=0.99, cov=0.95, bits=500.0, orf_aa=240, ref_aa=240):
    return Hit(oid, acc, desc, ident, cov, bits, orf_aa, ref_aa)


O1, O2 = "p|orf001|+|1..720", "p|orf002|+|2000..2720"


class TestBasicArchetypes:
    def test_no_confident_payload_is_no_payload(self):
        c = classify({O1: GENE}, [hit(O1, "P1", "Beta-lactamase TEM")], PLASMID)
        assert c.archetype == "no_payload"
        assert c.n_payload_orfs == 0

    def test_a_bare_gene_is_single_cds(self):
        """A 240 aa gene is 720 bp, so the plasmid has to be small enough that the payload clears the
        15% backbone threshold: otherwise `backbone_heavy` fires correctly and this test is about
        something else."""
        c = classify({O1: GENE}, [hit(O1, "P1", orf_aa=len(GENE), ref_aa=len(GENE))], 3000)
        assert c.archetype == "single_cds"
        assert c.flags == frozenset()

    def test_a_2a_peptide_makes_a_cassette(self):
        c = classify({O1: GENE + P2A + OTHER}, [hit(O1, "P1")], PLASMID)
        assert c.archetype == "multi_gene_cassette"
        assert "has_2a" in c.flags

    def test_a_terminal_tag_makes_a_tagged_fusion(self):
        seq = "M" + FLAG3 + GENE
        c = classify({O1: seq}, [hit(O1, "P1", orf_aa=len(seq), ref_aa=len(GENE))], PLASMID)
        assert c.archetype == "tagged_fusion"
        assert "has_tag" in c.flags


class TestCorrections:
    def test_homolog_families_are_not_counted_as_separate_payloads(self):
        """#134898 carries one untagged PCNA; DIAMOND returns twelve PCNA orthologues. Counting distinct
        accessions read that as a twelve-gene cassette."""
        hits = [hit(O1, f"P{i}", "Proliferating cell nuclear antigen", bits=500 - i,
                    orf_aa=len(GENE), ref_aa=len(GENE)) for i in range(12)]
        c = classify({O1: GENE}, hits, PLASMID)
        assert c.archetype == "single_cds"
        assert "multi_payload" not in c.flags

    def test_two_genuinely_different_payloads_on_separate_orfs_is_a_cassette(self):
        hits = [hit(O1, "P1", "Adapter molecule crk", orf_aa=len(GENE), ref_aa=len(GENE)),
                hit(O2, "P2", "Angiotensin-converting enzyme 2", orf_aa=len(OTHER), ref_aa=len(OTHER))]
        c = classify({O1: GENE, O2: OTHER}, hits, PLASMID)
        assert c.archetype == "multi_gene_cassette"
        assert "multi_payload" in c.flags

    def test_a_protease_site_is_a_tag_marker_not_a_cassette(self):
        """#187828 is GST-TEV-NDP52. A TEV site marks a removable tag; a 2A peptide makes two proteins.
        Treating both as cassette evidence mislabelled it."""
        seq = GENE + TEV + OTHER
        c = classify({O1: seq}, [hit(O1, "P1", orf_aa=len(seq), ref_aa=len(OTHER))], PLASMID)
        assert c.archetype == "tagged_fusion"
        assert "has_protease_site" in c.flags
        assert "has_2a" not in c.flags

    def test_a_motifless_fusion_is_caught_by_length(self):
        """#17551's FOXO1-EGFP has no linker, tag or cleavage site: 899 aa of ORF against a 652 aa
        reference is the only evidence there is."""
        seq = GENE + OTHER
        c = classify({O1: seq}, [hit(O1, "P1", orf_aa=len(seq), ref_aa=len(GENE))], PLASMID)
        assert c.archetype == "tagged_fusion"
        assert "diluted_payload" in c.flags
        assert c.n_tag_motifs == 0

    def test_an_undiluted_payload_raises_no_fusion_flag(self):
        c = classify({O1: GENE}, [hit(O1, "P1", orf_aa=len(GENE), ref_aa=len(GENE) - 2)], PLASMID)
        assert "diluted_payload" not in c.flags
        assert c.archetype == "single_cds"

    def test_dilution_threshold_is_the_documented_one(self):
        assert FUSION_DILUTION == 0.20


class TestFlagsAreOrthogonal:
    def test_backbone_heavy_is_a_flag_not_a_competing_label(self):
        """#134898 is a single-CDS construct whose payload is 12% of the plasmid. The old exclusive
        scheme forced a choice between those two true statements."""
        c = classify({O1: GENE}, [hit(O1, "P1", orf_aa=len(GENE), ref_aa=len(GENE))], 40000)
        assert c.archetype == "single_cds"
        assert "backbone_heavy" in c.flags

    def test_a_cassette_can_also_be_backbone_heavy(self):
        c = classify({O1: GENE + P2A + OTHER}, [hit(O1, "P1")], 60000)
        assert c.archetype == "multi_gene_cassette"
        assert {"has_2a", "backbone_heavy"} <= c.flags

    def test_archetype_is_always_one_of_the_declared_set(self):
        from pilot.archetype_seq import ARCHETYPES

        for seq, hits in ((GENE, [hit(O1, "P1")]), (GENE + P2A + OTHER, [hit(O1, "P1")]),
                          (GENE, [hit(O1, "P1", "Beta-lactamase TEM")])):
            assert classify({O1: seq}, hits, PLASMID).archetype in ARCHETYPES

    def test_evidence_is_recorded_for_every_flag(self):
        c = classify({O1: "M" + FLAG3 + GENE + P2A + OTHER}, [hit(O1, "P1")], 60000)
        assert len(c.evidence) >= len(c.flags) - 1
        assert all(isinstance(e, str) and e for e in c.evidence)


class TestDilutionOnHit:
    def test_dilution_is_zero_when_the_orf_matches_the_reference(self):
        assert hit(O1, "P1", orf_aa=300, ref_aa=300).dilution == 0.0

    def test_dilution_rises_with_extra_sequence(self):
        assert hit(O1, "P1", orf_aa=600, ref_aa=300).dilution == pytest.approx(0.5)

    def test_an_orf_shorter_than_its_reference_is_not_negatively_diluted(self):
        assert hit(O1, "P1", orf_aa=200, ref_aa=300).dilution == 0.0

    def test_missing_lengths_do_not_raise(self):
        assert hit(O1, "P1", orf_aa=0, ref_aa=0).dilution == 0.0
