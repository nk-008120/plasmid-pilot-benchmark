"""Guards for the route B payload-fraction measurement.

The one that matters most is `test_reader_handles_both_catalogue_header_styles`. The first version of
`read_payload_seqs` took the header's first token, which works on the route B catalogue and resolved
**zero of 16** payloads on `gate_all173`, where entries are written Swiss-Prot style as
`>sp|P42212|GFP_AEQVI...`. Nothing raised: the payload set came back empty and every construct looked
unscorable for a reason unrelated to the construct. Two rows of this project's register, two formats.

The rest assert the properties the reported numbers rest on, so that a re-run on new deposits fails
loudly rather than quietly changing what was claimed.
"""
import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/43_route_b_f.py"
OUT = ROOT / "results/route_b_f/f_by_construct.csv"

spec = importlib.util.spec_from_file_location("route_b_f", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_reader_handles_both_catalogue_header_styles(tmp_path):
    fa = tmp_path / "ref.fasta"
    fa.write_text(
        ">sp|P42212|GFP_AEQVI Green fluorescent protein OS=Aequorea victoria\n"
        "MSKGEELFTG\n"
        ">O43474 homolog_of:O43474\n"
        "MAVSDALLPS\n"
        ">P83690 homolog_of:P42212\n"
        "MNOTAPAYLOAD\n",
        encoding="utf-8",
    )
    got = mod.read_payload_seqs(fa, {"P42212", "O43474"})
    assert got == {"P42212": "MSKGEELFTG", "O43474": "MAVSDALLPS"}, (
        "both the Swiss-Prot style payload entry and the self-tagged one must resolve")


def test_reader_does_not_return_homologs_as_payloads(tmp_path):
    fa = tmp_path / "ref.fasta"
    fa.write_text(">P83690 homolog_of:P42212\nMNOTAPAYLOAD\n", encoding="utf-8")
    assert mod.read_payload_seqs(fa, {"P42212"}) == {}


def test_stratum_boundaries_are_the_project_strata():
    assert mod.F_STRATA == (0.50, 0.70, 0.95)
    assert mod.stratum(0.49) == "<0.50"
    assert mod.stratum(0.50) == "0.50-0.70"
    assert mod.stratum(0.70) == "0.70-0.95"
    assert mod.stratum(0.95) == ">=0.95"
    assert mod.stratum(1.00) == ">=0.95"


def test_the_known_true_floor_is_below_every_canonical_call():
    """The floor is read off true pairs, so it cannot sit above the class it admits."""
    assert mod.ABSENT_BELOW_IDENTITY < mod.KNOWN_TRUE_MIN_IDENTITY
    assert mod.KNOWN_TRUE_N == 20


pytestmark_out = pytest.mark.skipif(
    not OUT.is_file(), reason="route B f output is not present in this checkout")


@pytest.fixture(scope="module")
def rows():
    with open(OUT, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


@pytestmark_out
def test_one_sequence_record_per_deposit_and_payload(rows):
    """Three deposits were downloaded twice; counting both would double them in every denominator."""
    seen = [(r["addgene_id"], r["payload_accession"]) for r in rows]
    dupes = {k for k in seen if seen.count(k) > 1}
    assert not dupes, f"(deposit, payload) counted more than once: {dupes}"


@pytestmark_out
def test_f_is_span_over_orf_as_context_26_3_defines_it(rows):
    for r in rows:
        if not r["f"]:
            continue
        assert abs(float(r["f"]) - int(r["span_aa"]) / int(r["orf_aa"])) < 5e-4, r["addgene_id"]


@pytestmark_out
def test_classification_follows_the_identity_bands(rows):
    for r in rows:
        if not r["span_identity"]:
            continue
        i = float(r["span_identity"])
        want = ("absent" if i < mod.ABSENT_BELOW_IDENTITY
                else "engineered_variant" if i < mod.KNOWN_TRUE_MIN_IDENTITY else "canonical")
        assert r["payload_present"] == want, (r["addgene_id"], r["payload_accession"], i)


@pytestmark_out
def test_the_identity_gap_the_absent_class_rests_on_is_still_empty(rows):
    """The absent/variant boundary is an empty region in the data, not a chosen number.

    If a future harvest puts a pair inside it, the classification is no longer reading a gap and the
    boundary has to be re-derived rather than kept. That is the whole difference between this and a
    tuned threshold, so it is asserted.
    """
    ids = sorted(float(r["span_identity"]) for r in rows if r["span_identity"])
    inside = [i for i in ids if 0.40 < i < 0.77]
    assert not inside, f"identities now fall inside the gap: {inside}"


@pytestmark_out
def test_the_guide_plasmid_is_called_absent(rows):
    """238174 is a CRISPRa guide against KLF4 and carries no KLF4 ORF: the harvest's own false positive."""
    r = [x for x in rows if x["addgene_id"] == "238174"]
    assert r and r[0]["payload_present"] == "absent", r


def test_the_longest_record_wins_among_equally_verified_ones():
    """this project's erratum, pinned.

    236780 carries two Addgene-Verified Full records, 19,523 bp and 20,956 bp. They are not two
    readings of one molecule, one is a partial record, and taking whichever sorted first called
    T7 RNAP absent from a construct that carries it at identity 0.998. A dedupe rule is a limit like
    any other, and this is the limit that bound.
    """
    kind_of = {("236780", "494140"): "addgene_verified_full",
               ("236780", "505276"): "addgene_verified_full"}
    bp_of = {("236780", "494140"): 19523, ("236780", "505276"): 20956}
    assert mod.choose_records(kind_of, bp_of) == {"236780": "505276"}


def test_verified_provenance_still_outranks_a_longer_depositor_record():
    """Length is the tie-break, not the rule. Provenance is still first."""
    kind_of = {("50466", "75420"): "depositor_full",
               ("50466", "305068"): "addgene_verified_full"}
    bp_of = {("50466", "75420"): 99999, ("50466", "305068"): 8476}
    assert mod.choose_records(kind_of, bp_of) == {"50466": "305068"}


def test_a_deposit_with_no_downloaded_record_still_resolves():
    """A missing download must degrade to a choice, not drop the deposit silently."""
    kind_of = {("999999", "111"): "addgene_verified_full"}
    assert mod.choose_records(kind_of, {}) == {"999999": "111"}


@pytestmark_out
def test_t7_rnap_is_present_in_all_three_deposits(rows):
    """The three pTB10x deposits are one construct series and must agree; two of them read as
    absent only while the partial records were being measured."""
    t7 = [r for r in rows if r["payload_accession"] == "P00573"]
    assert len(t7) == 3, t7
    assert {r["payload_present"] for r in t7} == {"canonical"}, t7
    assert len({r["f"] for r in t7}) == 1, f"one construct series, one f: {[r['f'] for r in t7]}"
