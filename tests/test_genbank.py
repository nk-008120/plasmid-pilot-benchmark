from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from pilot.genbank import annotated_cds, load_plasmids


def test_wrapping_cds_on_both_strands_roundtrips_through_genbank(make_gb):
    specs = [
        {"label": "fwd_wrap", "strand": 1, "n_codons": 100},
        {"label": "rev", "strand": -1, "n_codons": 90},
    ]
    path, seq, planted = make_gb("wrapfwd", 11, specs, rotate_to_wrap=True)
    [pl] = load_plasmids(path)
    assert pl.seq == seq and pl.topology == "circular" and not pl.topology_assumed
    cds = {c.label: c for c in annotated_cds(pl)}
    assert cds["fwd_wrap"].protein == planted[0].protein
    assert len(cds["fwd_wrap"].segments) == 2  # crosses the origin
    assert cds["rev"].protein == planted[1].protein and cds["rev"].strand == -1
    assert all(c.invalid_reason == "" for c in cds.values())

    path2, seq2, planted2 = make_gb("wraprev", 12, [{"label": "rev_wrap", "strand": -1, "n_codons": 100}], rotate_to_wrap=True)
    [pl2] = load_plasmids(path2)
    [c] = annotated_cds(pl2)
    assert planted2[0].wraps and len(c.segments) == 2
    assert c.protein == planted2[0].protein and c.strand == -1


def test_topology_is_read_from_file_and_assumed_only_when_absent(make_gb):
    specs = [{"label": "g", "strand": 1, "n_codons": 60}]
    path, _, _ = make_gb("notopo", 5, specs, topology="")
    assert "circular" not in Path(path).read_text().splitlines()[0]  # the file really has no topology
    [pl] = load_plasmids(path, assume_topology="circular")
    assert (pl.topology, pl.topology_assumed) == ("circular", True)
    [pl] = load_plasmids(path, assume_topology="linear")
    assert (pl.topology, pl.topology_assumed) == ("linear", True)

    path, _, _ = make_gb("lin", 5, specs, topology="linear")
    [pl] = load_plasmids(path, assume_topology="circular")  # the file wins over the assumption
    assert (pl.topology, pl.topology_assumed, pl.circular) == ("linear", False, False)


def _write_tiny(path, seq, start, end):
    rec = SeqRecord(Seq(seq), id="tiny", name="tiny", features=[SeqFeature(FeatureLocation(start, end, 1), type="CDS", qualifiers={"label": ["x"]})])
    rec.annotations["molecule_type"] = "DNA"
    rec.annotations["topology"] = "circular"
    SeqIO.write(rec, str(path), "genbank")


def test_invalid_annotations_are_flagged_not_scored(tmp_path):
    seq = "ATGAAATAAAAATAA" + "C" * 30
    _write_tiny(tmp_path / "a.gb", seq, 0, 15)
    [c] = annotated_cds(load_plasmids(tmp_path / "a.gb")[0])
    assert c.invalid_reason == "internal_stop"
    _write_tiny(tmp_path / "b.gb", seq, 0, 14)
    [c] = annotated_cds(load_plasmids(tmp_path / "b.gb")[0])
    assert c.invalid_reason == "length_not_multiple_of_3"


def test_cds_without_terminal_stop_is_usable(tmp_path):
    seq = "ATGAAACCC" + "C" * 30
    _write_tiny(tmp_path / "c.gb", seq, 0, 9)
    [c] = annotated_cds(load_plasmids(tmp_path / "c.gb")[0])
    assert c.invalid_reason == "" and not c.has_terminal_stop and c.protein == "MKP"
