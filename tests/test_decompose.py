"""Tests for motif-driven decomposition and the oracle alignment arm.

Several tests are named for the real Addgene construct whose behaviour they lock in; those are the
regressions that matter, because each one was a surprise the first time the pipeline met it.
"""
from __future__ import annotations

import pytest

from pilot.decompose import (
    CLEAVAGE,
    INTERIOR_JUNCTION,
    Segment,
    decompose,
    dilution,
    find_motifs,
    oracle_span,
)

P2A = "ATNFSLLKQAGDVEENPGP"
T2A = "EGRGSLLTCGDVEENPGP"
E2A = "QCTNYALLKLAGDVESNPGP"
F2A = "VKQTLNFDLLKLAGDVELNPGP"
FLAG3 = "DYKDHDGDYKDHDIDYKDDDDK"
NLS = "PKKKRKV"
# two unrelated filler proteins, long enough to survive the 50 aa floor
A = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDE"
B = "MSDNEKLIAFIEQVNQLSPAERAELVQQLTAHWDGQRWQAFADGLQNHLDGQAYTLDMLKQAQRHGVDVSTLSPEQRAAFEALLHRHQGDEWSAIAQRLAG"


class TestMotifs:
    @pytest.mark.parametrize("peptide", [P2A, T2A, E2A, F2A])
    def test_every_common_2a_is_found(self, peptide):
        hits = [h for h in find_motifs(A + peptide + B) if h.name == "2A_skip"]
        assert len(hits) == 1

    def test_2a_cut_falls_before_the_final_proline(self):
        """Ribosome skipping happens between the penultimate G and the final P, so the downstream product
        begins with that proline: an off-by-one here silently shifts every segment in a cassette."""
        seq = A + P2A + B
        (hit,) = [h for h in find_motifs(seq) if h.name == "2A_skip"]
        assert seq[hit.cut_after] == "P"
        assert seq[hit.cut_after - 1] == "G"

    def test_naive_npgp_core_is_not_used(self):
        """The 4-residue core fires on unrelated sequence; the catalogue must use the longer motif."""
        assert not any("NPGP" == pat for _, pat, _ in CLEAVAGE)
        assert not find_motifs(A + "NPGP" + B)

    def test_3xflag_wins_over_the_flag_inside_it(self):
        hits = [h for h in find_motifs("M" + FLAG3 + A) if h.kind == "tag"]
        assert [h.name for h in hits] == ["3xFLAG"]
        assert (hits[0].start, hits[0].end) == (2, 23)

    def test_tev_site_cuts_after_the_glutamine(self):
        seq = A + "ENLYFQG" + B
        (hit,) = [h for h in find_motifs(seq) if h.name == "TEV_site"]
        assert seq[hit.cut_after - 1] == "Q"


class TestDecompose:
    def test_motifless_protein_is_one_whole_segment(self):
        """`whole` has to be the degenerate case of `auto`, not a separate code path."""
        segs = decompose(A + B)
        assert len(segs) == 1
        assert (segs[0].start, segs[0].end) == (1, len(A + B))

    def test_two_gene_2a_cassette_splits_in_two(self):
        segs = decompose(A + P2A + B)
        assert len(segs) == 2
        assert segs[0].seq.startswith(A[:20])
        assert segs[1].seq.endswith(B[-20:])

    def test_three_gene_cassette_splits_in_three(self):
        segs = decompose(A + P2A + B + T2A + A)
        assert len(segs) == 3

    def test_terminal_tags_are_stripped_from_both_ends(self):
        segs = decompose("M" + FLAG3 + A + NLS)
        assert len(segs) == 1
        assert segs[0].seq == A
        assert set(segs[0].removed) == {"N:3xFLAG", "C:SV40_NLS"}

    def test_stacked_tags_need_more_than_one_pass(self):
        """`3xFLAG-GGGGS-payload`: a single trimming pass leaves the linker on the query."""
        segs = decompose("M" + FLAG3 + "GGGGSGGGGS" + A)
        assert segs[0].seq == A

    def test_segments_below_the_floor_are_dropped(self):
        segs = decompose(A + P2A + "MKV", min_len=50)
        assert len(segs) == 1

    def test_interior_tag_is_treated_as_a_junction(self):
        segs = decompose(A + "HHHHHH" + B)
        assert len(segs) == 2
        assert segs[0].seq == A and segs[1].seq == B

    def test_loose_g_rich_linker_trims_but_never_cuts(self):
        """`[GS]{6,}` fires in 2.07% of real proteins. Letting it imply a junction split 85/5,016 real
        single proteins; restricting interior cuts to `INTERIOR_JUNCTION` brings that to 18/5,016. This
        test is the guard on that measurement: see scripts/08_motif_specificity.py."""
        assert "G_rich" not in INTERIOR_JUNCTION
        assert "GS_linker" in INTERIOR_JUNCTION
        assert len(decompose(A + "GSGSGSGS" + B)) == 1

    def test_a_query_is_never_silently_dropped(self):
        """An arginine-rich protamine is almost entirely `R{6,}`; trimming leaves nothing above the floor.
        Returning [] there would make a screening pipeline lose the query without saying so."""
        protamine = "MARYRHSRSRSRSRYRRRRRRRRSRYRSRRRRYRRSRRRRRRGRRRGYSRRRYSRRRRRRY"
        segs = decompose(protamine)
        assert len(segs) == 1
        assert segs[0].seq == protamine
        assert segs[0].removed == ("fallback:nothing_above_min_len",)

    def test_interior_tag_rule_can_be_switched_off(self):
        """The ablation reported in the results: rule 1 alone cannot split a cleavage-free fusion."""
        segs = decompose(A + "HHHHHH" + B, cut_at_internal_tags=False)
        assert len(segs) == 1

    def test_terminal_tag_is_trimmed_not_treated_as_a_junction(self):
        """A His6 within `trim_window` of an end is a tag; only an interior one implies a seam."""
        segs = decompose("HHHHHH" + A + B)
        assert len(segs) == 1
        assert segs[0].removed == ("N:His",)

    def test_segments_are_returned_in_n_to_c_order(self):
        segs = decompose(A + P2A + B + T2A + A)
        assert [s.start for s in segs] == sorted(s.start for s in segs)

    def test_coordinates_are_one_based_and_slice_back_to_the_segment(self):
        seq = "M" + FLAG3 + A + P2A + B
        for s in decompose(seq):
            assert seq[s.start - 1 : s.end] == s.seq


class TestOracleSpan:
    def test_recovers_an_embedded_payload_exactly(self):
        span = oracle_span("M" + FLAG3 + A + NLS, A)
        assert (span.start, span.end) == (24, 23 + len(A))
        assert span.identity == 1.0
        assert span.payload_coverage == 1.0

    def test_identity_falls_with_substitutions(self):
        mutated = A[:40] + "W" * 10 + A[50:]
        span = oracle_span("MM" + mutated + "MM", A)
        assert 0.5 < span.identity < 1.0

    def test_returns_none_when_nothing_aligns(self):
        assert oracle_span("", A) is None
        assert oracle_span(A, "") is None

    def test_dilution_is_the_foreign_fraction_of_the_orf(self):
        orf = "M" + FLAG3 + A + NLS
        span = oracle_span(orf, A)
        assert dilution(orf, span) == pytest.approx(1 - len(A) / len(orf), abs=1e-4)
        assert dilution(A, oracle_span(A, A)) == 0.0


class TestRealConstructRegressions:
    """Boundaries taken from the A2/A3 audit of the real Addgene files; see the working record."""

    def test_73037_three_gene_cassette_yields_three_segments(self):
        """Nluc-P2A-copGFP-T2A-PuroR: one 669 aa ORF whose payload is 171 aa, 74% dilution. Cutting at the
        two 2A peptides is what moves this plasmid's payload from rank 50 to rank 1."""
        orf = A * 2 + P2A + B * 2 + T2A + A
        segs = decompose(orf)
        assert len(segs) == 3

    def test_171782_internal_his_tag_splits_mbp_from_the_payload(self):
        """MBP-His6-TEV protease carries no cleavage site, so rule 1 alone leaves it whole; the interior-tag
        rule is the only thing that separates the 392 aa carrier from the 236 aa payload."""
        orf = A * 4 + "HHHHHH" + B * 2
        assert len(decompose(orf)) == 2
        assert len(decompose(orf, cut_at_internal_tags=False)) == 1

    def test_17551_direct_fusion_defeats_motif_rules(self):
        """FOXO1-EGFP is joined with no linker, tag or cleavage site. Motif decomposition cannot see the
        boundary and returns the whole chain: the documented failure mode that motivates a learned
        method. This test exists to fail loudly if someone 'fixes' it with a heuristic that guesses."""
        segs = decompose(A * 3 + B * 2)
        assert len(segs) == 1
