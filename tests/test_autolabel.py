"""Tests for automatic payload assignment.

The description strings used here are real Swiss-Prot `DE` lines, not invented ones. That matters: the
first version of `CARRIER_DESC` matched "maltose-binding protein" and missed the actual Swiss-Prot name,
"Maltose/maltodextrin-binding periplasmic protein", which would have let MBP be chosen as the payload of
every pET-MBP construct in the corpus.
"""
from __future__ import annotations

import pytest

from pilot.autolabel import (
    MIN_COVERAGE,
    MIN_IDENTITY,
    RULES,
    AutoLabel,
    Hit,
    accession_of,
    assign_payload,
    is_vector_part,
    read_tabular,
)

P = "addgene_00001_demo"


def hit(acc, desc, *, ident=0.99, cov=0.95, bits=500.0, orf=1, start=100, aa=300):
    return Hit(f"{P}|orf{orf:03d}|+|{start}..{start + aa * 3}", acc, desc, ident, cov, bits, aa)


MARKERS = [
    ("P62593", "Beta-lactamase TEM"),
    ("P00552", "Aminoglycoside 3'-phosphotransferase"),
    ("P62577", "Chloramphenicol acetyltransferase"),
    ("P03023", "Lactose operon repressor"),
    ("P03051", "Regulatory protein rop"),
    ("P04483", "Tetracycline repressor protein class B"),
]
CARRIERS = [
    ("P08515", "Glutathione S-transferase class-mu 26 kDa isozyme"),
    ("P0AEX9", "Maltose/maltodextrin-binding periplasmic protein"),
    ("P0AA25", "Thioredoxin 1"),
    ("Q12306", "Ubiquitin-like protein SMT3"),
]
PAYLOADS = [
    ("Q99ZW2", "CRISPR-associated endonuclease Cas9"),
    ("P46108", "Adapter molecule crk"),
    ("Q13137", "Calcium-binding and coiled-coil domain-containing protein 2"),
    ("Q9BYF1", "Angiotensin-converting enzyme 2"),
]


class TestClassification:
    @pytest.mark.parametrize("acc,desc", MARKERS)
    def test_markers_are_vector_parts(self, acc, desc):
        assert is_vector_part(hit(acc, desc))

    @pytest.mark.parametrize("acc,desc", CARRIERS)
    def test_carriers_are_vector_parts(self, acc, desc):
        assert is_vector_part(hit(acc, desc))

    @pytest.mark.parametrize("acc,desc", PAYLOADS)
    def test_real_payloads_are_not_excluded(self, acc, desc):
        assert not is_vector_part(hit(acc, desc))

    def test_reporters_are_payloads_by_default(self):
        """pLenti-CMV-GFP's payload IS the GFP; excluding reporters wholesale would mislabel it."""
        g = hit("P42212", "Green fluorescent protein")
        assert not is_vector_part(g)
        assert is_vector_part(g, exclude_reporters=True)

    def test_accession_is_pulled_out_of_a_uniprot_id(self):
        assert accession_of("sp|P42212|GFP_AEQVI") == "P42212"
        assert accession_of("tr|A0A123|A0A123_ECOLI") == "A0A123"
        assert accession_of("P42212") == "P42212"


class TestAssignPayload:
    def test_the_marker_is_not_chosen_even_when_it_scores_highest(self):
        """The failure this whole module exists to prevent: AmpR is the most confident hit on most plasmids."""
        hits = [hit("P62593", "Beta-lactamase TEM", bits=900.0),
                hit("Q99ZW2", "CRISPR-associated endonuclease Cas9", bits=600.0, orf=2)]
        got = assign_payload(hits)
        assert got.payload_accession == "Q99ZW2"
        assert got.n_excluded_marker == 1

    def test_the_carrier_is_not_chosen_over_its_cargo(self):
        """Addgene #187828: GST-TEV-NDP52, where GST is the handle and NDP52 the cargo."""
        hits = [hit("P08515", "Glutathione S-transferase class-mu 26 kDa isozyme", bits=800.0),
                hit("Q13137", "Calcium-binding and coiled-coil domain-containing protein 2", bits=700.0, orf=2)]
        assert assign_payload(hits).payload_accession == "Q13137"

    def test_nothing_but_vector_parts_fails_the_gate_rather_than_guessing(self):
        got = assign_payload([hit("P62593", "Beta-lactamase TEM")])
        assert got.passed_gate is False
        assert got.payload_accession == ""
        assert "gate" in got.reason

    def test_low_identity_fails_the_gate(self):
        got = assign_payload([hit("Q99ZW2", "CRISPR-associated endonuclease Cas9", ident=0.55)])
        assert got.passed_gate is False

    def test_an_engineered_variant_clears_the_gate_against_its_parent(self):
        """mCherry is 0.86 identical to its DsRed parent. A 0.90 floor abstained on all three mCherry
        constructs in the pilot set; 0.85 accepts them, which is why the default moved."""
        got = assign_payload([hit("Q9U6Y8", "Red fluorescent protein drFP583", ident=0.86, cov=0.96)])
        assert got.passed_gate
        assert got.payload_accession == "Q9U6Y8"

    def test_a_payload_matched_over_a_sliver_of_a_polyprotein_still_fails(self):
        """Addgene #171782: 0.97 identity but 0.08 coverage of the TEV genome polyprotein. Lowering the
        identity floor must not let this through: coverage is the control that catches it."""
        got = assign_payload([hit("P04517", "Genome polyprotein", ident=0.97, cov=0.08)])
        assert got.passed_gate is False

    def test_low_coverage_fails_the_gate(self):
        """A 30-residue match to a 1,368 aa protein is not an identification."""
        got = assign_payload([hit("Q99ZW2", "CRISPR-associated endonuclease Cas9", cov=0.02)])
        assert got.passed_gate is False

    def test_empty_input_returns_none_not_a_label(self):
        assert assign_payload([]) is None

    @pytest.mark.parametrize("rule", RULES)
    def test_every_rule_runs_and_avoids_vector_parts(self, rule):
        hits = [hit("P62593", "Beta-lactamase TEM", bits=900.0, start=50),
                hit("Q99ZW2", "CRISPR-associated endonuclease Cas9", bits=600.0, orf=2, start=900, aa=1368),
                hit("P46108", "Adapter molecule crk", bits=650.0, orf=3, start=200, aa=304)]
        got = assign_payload(hits, rule=rule)
        assert got.passed_gate
        assert got.payload_accession != "P62593"
        assert got.rule == rule

    def test_the_rules_genuinely_disagree(self):
        """If all three rules always agreed there would be nothing to choose between them, and the
        validation step against the hand-audited set would be pointless."""
        hits = [hit("P46108", "Adapter molecule crk", bits=650.0, orf=1, start=200, aa=304),
                hit("Q99ZW2", "CRISPR-associated endonuclease Cas9", bits=600.0, orf=2, start=900, aa=1368)]
        chosen = {r: assign_payload(hits, rule=r).payload_accession for r in RULES}
        assert chosen["best_bitscore"] == "P46108"
        assert chosen["longest_orf"] == "Q99ZW2"
        assert chosen["first_in_construct"] == "P46108"
        assert len(set(chosen.values())) > 1

    def test_unknown_rule_is_rejected_loudly(self):
        with pytest.raises(ValueError, match="unknown rule"):
            assign_payload([hit("Q99ZW2", "Cas9")], rule="whatever")


class TestBackboneExclusions:
    """Exclusions added after the 23-construct validation run; each fixed a real mislabelling."""

    @pytest.mark.parametrize("acc,desc", [
        ("P03853", "Uncharacterized 9.2 kDa protein"),
        ("P03852", "Uncharacterized 15.3 kDa protein"),
        ("X00001", "Hypothetical 12 kDa protein"),
    ])
    def test_cryptic_backbone_orfs_are_excluded(self, acc, desc):
        """P03853 was the chosen payload on four unrelated constructs before this rule existed."""
        assert is_vector_part(hit(acc, desc))

    @pytest.mark.parametrize("acc,desc", [
        ("Q9QPN3", "Protein Nef (Fragment)"),
        ("P12497", "Gag-Pol polyprotein"),
    ])
    def test_retroviral_backbone_remnants_are_excluded(self, acc, desc):
        """In a 3rd-generation transfer vector these are vestigial delivery machinery, not cargo."""
        assert is_vector_part(hit(acc, desc))

    def test_the_gate_defaults_are_the_validated_ones(self):
        assert (MIN_IDENTITY, MIN_COVERAGE) == (0.85, 0.80)


class TestReadTabular:
    ROW = "{q}\tsp|{a}|X_Y\t{p}\t{l}\t1000\t{s}\t{b}\t1e-50\t{d}\n"

    def _write(self, tmp_path, rows):
        p = tmp_path / "hits.tsv"
        p.write_text("".join(rows), encoding="utf-8")
        return p

    def test_coverage_is_measured_against_the_subject_not_the_query(self, tmp_path):
        """A 1,400 aa fusion ORF matching a 240 aa reference over 240 columns has covered it fully."""
        p = self._write(tmp_path, [self.ROW.format(q=f"{P}|orf001|+|1..4200", a="P42212", p=99.0,
                                                   l=240, s=240, b=500, d="Green fluorescent protein")])
        (h,) = read_tabular(p)[P]
        assert h.coverage == 1.0
        assert h.identity == 0.99

    def test_weak_hits_are_dropped(self, tmp_path):
        p = self._write(tmp_path, [self.ROW.format(q=f"{P}|orf001|+|1..900", a="P42212", p=25.0,
                                                   l=30, s=240, b=20, d="Green fluorescent protein")])
        assert read_tabular(p) == {}

    def test_rows_are_grouped_by_plasmid(self, tmp_path):
        p = self._write(tmp_path, [
            self.ROW.format(q="plasmid_a|orf001|+|1..900", a="P1", p=99.0, l=240, s=240, b=500, d="X"),
            self.ROW.format(q="plasmid_b|orf002|-|1..900", a="P2", p=99.0, l=240, s=240, b=500, d="Y"),
        ])
        assert set(read_tabular(p)) == {"plasmid_a", "plasmid_b"}

    def test_malformed_and_comment_lines_are_skipped_not_fatal(self, tmp_path):
        p = self._write(tmp_path, ["# comment\n", "too\tfew\tfields\n",
                                   self.ROW.format(q=f"{P}|orf001|+|1..900", a="P42212", p=99.0,
                                                   l=240, s=240, b=500, d="GFP")])
        assert len(read_tabular(p)[P]) == 1
