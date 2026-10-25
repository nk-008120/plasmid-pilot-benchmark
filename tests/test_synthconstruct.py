"""Tests for the synthetic construct tier.

This tier's only claim is that its ground truth is exact by construction. Two bugs falsified that claim
during the build and each has a named test below: a chain that did not begin with Met got one prepended,
shifting every coordinate by one; and the ORF caller walked back to an upstream in-frame ATG in the
spacer, returning a longer chain than was designed. Both were invisible until the constructs were run
through the real pipeline, which is what `TestEndToEnd` does.
"""
from __future__ import annotations

import random

import pytest
from Bio.Seq import Seq

from pilot.orfs import find_orfs
from pilot.synthconstruct import (
    ARCHETYPES,
    FLAG3,
    P2A,
    Construct,
    build_construct,
    design,
    random_protein,
    random_spacer,
    reverse_translate,
    write_genbank,
)

PAYLOAD = "M" + "ACDEFGHIKLMNPQRSTVWY" * 15  # 301 aa
MARKERS = [("AmpR", "M" + random_protein(random.Random(99), 250))]


def rng():
    return random.Random(7)


class TestReverseTranslation:
    def test_round_trips_through_the_genetic_code(self):
        dna = reverse_translate(PAYLOAD, rng())
        assert str(Seq(dna).translate(table=11)) == PAYLOAD

    def test_never_emits_a_stop_codon(self):
        dna = reverse_translate(random_protein(rng(), 2000), rng())
        assert "*" not in str(Seq(dna).translate(table=11))

    def test_unknown_residues_do_not_raise(self):
        assert len(reverse_translate("MXXXBZJ", rng())) == 21


class TestDesign:
    @pytest.mark.parametrize("archetype", ARCHETYPES)
    def test_every_archetype_records_exactly_one_payload_part(self, archetype):
        _, parts = design(PAYLOAD, "P1", archetype, 0.5, rng())
        assert sum(1 for p in parts if p.kind == "payload") == 1

    @pytest.mark.parametrize("archetype", ("tagged_fusion", "motifless_fusion", "multi_gene_cassette"))
    @pytest.mark.parametrize("target", (0.2, 0.4, 0.6, 0.8))
    def test_dilution_hits_its_target(self, archetype, target):
        protein, _ = design(PAYLOAD, "P1", archetype, target, rng())
        assert abs((1 - len(PAYLOAD) / len(protein)) - target) < 0.02

    def test_parts_tile_the_chain_without_gaps_or_overlaps(self):
        protein, parts = design(PAYLOAD, "P1", "tagged_fusion", 0.6, rng())
        assert parts[0].start_aa == 1
        assert parts[-1].end_aa == len(protein)
        for a, b in zip(parts, parts[1:]):
            assert b.start_aa == a.end_aa + 1

    def test_the_payload_part_slices_back_to_the_payload(self):
        for archetype in ARCHETYPES:
            protein, parts = design(PAYLOAD, "P1", archetype, 0.5, rng())
            p = next(x for x in parts if x.kind == "payload")
            assert protein[p.start_aa - 1 : p.end_aa] == PAYLOAD

    def test_a_cassette_carries_a_real_2a_peptide(self):
        protein, _ = design(PAYLOAD, "P1", "multi_gene_cassette", 0.5, rng())
        assert P2A in protein

    def test_a_motifless_fusion_carries_no_motif_at_all(self):
        """The case motif decomposition is blind to. If this ever starts finding motifs, the tier has
        stopped testing what it exists to test."""
        from pilot.decompose import find_motifs

        protein, _ = design(PAYLOAD, "P1", "motifless_fusion", 0.5, random.Random(3))
        assert not [h for h in find_motifs(protein) if h.kind == "cleave"]

    def test_unknown_archetype_is_rejected(self):
        with pytest.raises(ValueError, match="unknown archetype"):
            design(PAYLOAD, "P1", "nonsense", 0.5, rng())

    def test_zero_dilution_is_the_bare_payload(self):
        protein, parts = design(PAYLOAD, "P1", "tagged_fusion", 0.0, rng())
        assert protein == PAYLOAD
        assert len(parts) == 1


class TestRegressions:
    def test_a_tagged_chain_begins_with_met(self):
        """A chain starting with the FLAG tag got an extra Met from the start codon, shifting every
        recorded coordinate by one and making the payload ORF uncallable at its designed length."""
        assert FLAG3.startswith("M")
        protein, _ = design(PAYLOAD, "P1", "tagged_fusion", 0.5, rng())
        assert protein.startswith("M")

    def test_an_in_frame_stop_guards_the_start_codon(self):
        """Without it the caller extends the ORF back to any upstream in-frame ATG in the spacer."""
        c = build_construct("c", PAYLOAD, "P1", "single_cds", 0.0, MARKERS, rng())
        guard = c.dna[c.orf_start_nt - 3 : c.orf_start_nt]
        assert guard in ("TAA", "TGA", "TAG")
        assert c.dna[c.orf_start_nt : c.orf_start_nt + 3] == "ATG"

    def test_spacers_contain_stop_codons(self):
        """Stop-free spacers let reading frames run through them, which turned 545 of 1,008 annotated
        markers from `exact` into `contained`."""
        s = random_spacer(random.Random(5), 3000)
        assert any(stop in s for stop in ("TAA", "TGA", "TAG"))


class TestBuildConstruct:
    def test_the_recorded_orf_translates_to_the_designed_chain(self):
        c = build_construct("c", PAYLOAD, "P1", "multi_gene_cassette", 0.5, MARKERS, rng())
        orf = c.dna[c.orf_start_nt : c.orf_start_nt + (c.orf_aa + 1) * 3]
        tr = str(Seq(orf).translate(table=11))
        assert tr.endswith("*")
        assert len(tr) - 1 == c.orf_aa
        assert tr[c.payload_start_aa - 1 : c.payload_end_aa] == PAYLOAD

    def test_dilution_is_recorded_not_assumed(self):
        c = build_construct("c", PAYLOAD, "P1", "tagged_fusion", 0.6, MARKERS, rng())
        assert abs(c.dilution - 0.6) < 0.02
        assert c.dilution == round(1 - c.payload_aa / c.orf_aa, 4)

    def test_only_markers_are_annotated(self, tmp_path):
        """Real deposits annotate recognised vector parts and leave the insert unlabelled.
        A construct whose payload is helpfully labelled would make the benchmark easier than reality."""
        from Bio import SeqIO

        c = build_construct("c", PAYLOAD, "P1", "tagged_fusion", 0.5, MARKERS, rng())
        p = tmp_path / "c.gbk"
        write_genbank(c, p)
        rec = next(SeqIO.parse(p, "genbank"))
        labels = [f.qualifiers["label"][0] for f in rec.features if f.type == "CDS"]
        assert labels == ["AmpR"]
        assert rec.annotations["topology"] == "circular"

    def test_backbone_padding_lowers_the_payload_fraction(self):
        small = build_construct("a", PAYLOAD, "P1", "single_cds", 0.0, MARKERS, rng())
        big = build_construct("b", PAYLOAD, "P1", "single_cds", 0.0, MARKERS, rng(), backbone_pad_bp=12000)
        assert big.payload_fraction < small.payload_fraction


class TestEndToEnd:
    """The claim this tier makes, checked through the real ORF caller rather than by inspection."""

    @pytest.mark.parametrize("archetype", ARCHETYPES)
    @pytest.mark.parametrize("dilution", (0.0, 0.4, 0.8))
    def test_the_payload_orf_is_called_at_its_designed_length(self, archetype, dilution):
        c = build_construct("c", PAYLOAD, "P1", archetype, dilution, MARKERS, random.Random(11))
        called = find_orfs(c.dna, circular=True, min_aa=50)
        match = [o for o in called if len(o.protein) == c.orf_aa]
        assert match, f"{archetype} at {dilution}: no ORF of {c.orf_aa} aa among {sorted({len(o.protein) for o in called}, reverse=True)[:5]}"
        assert match[0].protein[c.payload_start_aa - 1 : c.payload_end_aa] == PAYLOAD
