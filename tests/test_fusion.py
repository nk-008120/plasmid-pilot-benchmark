import csv

from pilot.fusion import features_inside, payload_orf
from pilot.genbank import Feature, Plasmid, load_plasmids
from pilot.orfs import ORF, find_orfs

L = 6000


def pl(features=()):
    return Plasmid("p", "p.gb", "p", "A" * L, "circular", False, list(features))


def feat(label, start, end, strand=1, kind="CDS"):
    return Feature(kind, label, strand, [(start, end)], f"{start+1}..{end}")


def test_offsets_are_measured_from_the_orf_start_forward_strand():
    orf = ORF(1, 1000, 903, False, "M" * 300, "M" * 300)   # 300 aa at nt 1000..1903
    [tag, gene] = features_inside(pl([feat("6xHis", 1000, 1018), feat("GeneA", 1018, 1900)]), orf)
    assert (tag.label, tag.start_aa, tag.end_aa) == ("6xHis", 0, 6)
    assert (gene.label, gene.start_aa) == ("GeneA", 6)
    assert tag.same_strand and gene.same_strand


def test_offsets_run_from_the_other_end_on_the_reverse_strand():
    orf = ORF(-1, 1000, 903, False, "M" * 300, "M" * 300)  # translated high -> low
    [h] = features_inside(pl([feat("tag", 1885, 1903, strand=-1)]), orf)
    assert h.start_aa == 0 and h.same_strand          # at the ORF's start, i.e. the high end


def test_opposite_strand_features_are_reported_but_flagged():
    orf = ORF(1, 1000, 903, False, "M" * 300, "M" * 300)
    [h] = features_inside(pl([feat("antisense", 1200, 1500, strand=-1)]), orf)
    assert not h.same_strand


def test_non_overlapping_and_unlabelled_and_source_features_are_skipped():
    orf = ORF(1, 1000, 903, False, "M" * 300, "M" * 300)
    feats = [feat("far away", 4000, 4500), feat("", 1100, 1200), feat("src", 0, L, kind="source"),
             feat("whole", 0, L, kind="gene")]
    assert features_inside(pl(feats), orf) == []


def test_partial_overlap_reports_covered_fraction():
    orf = ORF(1, 1000, 903, False, "M" * 300, "M" * 300)
    [h] = features_inside(pl([feat("half_in", 1800, 2200)]), orf)   # 103 of 400 bp inside
    assert 0.2 < h.covered_fraction < 0.3


def test_payload_orf_returns_none_without_a_stated_size():
    orf = ORF(1, 0, 903, False, "M" * 300, "M" * 300)
    got, chk = payload_orf(pl(), [orf], None)
    assert got is None and chk.verdict == ""



def test_worksheet_script_end_to_end(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, "scripts")
    from importlib import import_module

    from pilot.synthetic import build_fusion_plasmid, write_genbank

    raw = tmp_path / "raw"
    raw.mkdir()
    seq, planted = build_fusion_plasmid(9, tag_codons=60, gene_codons=150)
    write_genbank(raw / "fusion.gb", seq, planted, name="fusion")
    insert_nt = planted[0].length_nt + planted[1].length_nt
    man = tmp_path / "m.csv"
    man.write_text(f"filename,insert_size_bp\nfusion.gb,{insert_nt}\n", encoding="utf-8-sig")
    out = tmp_path / "out"

    mod = import_module("03_fusion_worksheet")
    assert mod.main(["--input-dir", str(raw), "--manifest", str(man), "--out-dir", str(out)]) == 0
    with open(out / "fusion_worksheet.csv", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["feature_label"] for r in rows} == {"tag_part", "gene_part"}
    tag = next(r for r in rows if r["feature_label"] == "tag_part")
    assert tag["start_aa"] == "0"                      # the tag sits at the ORF's start
    assert tag["piece_is"] == "" and tag["notes"] == ""  # left blank for the human
    assert mod.main(["--input-dir", str(raw), "--manifest", str(man), "--out-dir", str(out)]) == 2  # no clobber


def test_origin_crossing_feature_does_not_collapse_to_one_residue():
    """Regression: min(start)/max(end) mapped both ends of a wrapping feature to the same offset.

    Seen on real data: #171782's 367 aa MBP annotation crosses the origin and was reported as 311..311.
    """
    orf = ORF(1, 5000, 1503, True, "M" * 500, "M" * 500)  # wraps: 5000..6000 then 0..503
    wrap = Feature("CDS", "MBP-like", 1, [(5600, L), (0, 200)], "join(5601..6000,1..200)")
    [h] = features_inside(pl([wrap]), orf)
    assert h.n_segments == 2
    assert h.end_aa - h.start_aa > 100  # ~200 aa of span, not 0
    assert h.start_aa == 200  # 600 nt past the ORF start


def test_feature_starting_upstream_of_the_orf_gets_a_negative_offset():
    """Regression: the circular modulo wrapped an upstream position to a huge positive offset.

    Seen on #63155: a Kozak site just before a 1,656 aa ORF was reported at residue 2,559.
    """
    orf = ORF(1, 1000, 903, False, "M" * 300, "M" * 300)
    [h] = features_inside(pl([feat("Kozak sequence", 988, 1006, kind="regulatory")]), orf)
    assert h.start_aa == -4 and h.end_aa == 2      # four residues upstream, not ~1996
    assert -orf.length_nt // 3 < h.start_aa < orf.length_aa


def test_merged_multi_site_feature_reports_only_the_overlapping_site():
    """Regression: #63155's 10 bp Kozak feature spanned 2,559 residues to reach a second copy."""
    orf = ORF(1, 1000, 903, False, "M" * 300, "M" * 300)
    merged = Feature("regulatory", "Kozak sequence", 1, [(1030, 1040), (4000, 4010)], "order(...)")
    [h] = features_inside(pl([merged]), orf)
    assert (h.start_aa, h.end_aa) == (10, 13)  # only the copy inside the ORF
    assert h.n_segments == 2 and h.covered_fraction == 0.5
