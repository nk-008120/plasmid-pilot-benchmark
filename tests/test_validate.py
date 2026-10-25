import csv
import json
import shutil

from pilot.validate import main


def read(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_end_to_end_on_synthetic_plasmids(tmp_path, make_gb):
    raw = tmp_path / "raw"
    raw.mkdir()
    _, _, p1 = make_gb("fwd", 1, [{"label": "gA", "strand": 1, "n_codons": 120}], directory=raw)
    _, _, p2 = make_gb("revwrap", 2, [{"label": "gB", "strand": -1, "n_codons": 110}], rotate_to_wrap=True, directory=raw)
    _, _, p3 = make_gb(
        "two", 3, [{"label": "gC", "strand": 1, "n_codons": 100}, {"label": "gD", "strand": -1, "n_codons": 90}], directory=raw
    )
    shutil.copy(raw / "fwd.gb", raw / "fwd_again.gb")  # same sequence downloaded twice
    (raw / "broken.gb").write_text("this is not genbank\n")
    manifest = tmp_path / "manifest.csv"
    # written with a BOM, exactly as Excel's "CSV UTF-8" does; the reader must cope with it
    manifest.write_text("filename,source_id,source_url,archetype_manual,downloaded_on,annotation_audited,notes\n"
                        "fwd.gb,Addgene #1,https://example.org/1,single_cds,2026-10-14,yes,hello\n"
                        "ghost.gb,Addgene #2,https://example.org/2,tagged_fusion,2026-10-14,no,no such file\n",
                        encoding="utf-8-sig")
    out = tmp_path / "out"

    code = main(["--input-dir", str(raw), "--out-dir", str(out), "--min-aa", "60", "--manifest", str(manifest)])
    assert code == 0

    cds = read(out / "cds_recovery.csv")
    assert len(cds) == 5  # fwd, revwrap, two (2 CDSs), fwd_again
    assert {r["status"] for r in cds} == {"exact"}
    assert [r["wraps_origin"] for r in cds if r["plasmid_id"] == "revwrap"] == ["True"]

    summary = {r["plasmid_id"]: r for r in read(out / "pilot_summary.csv")}
    assert set(summary) == {"fwd", "revwrap", "two", "fwd_again"}  # broken.gb reported, not fatal
    assert summary["fwd_again"]["duplicate_of"] == "fwd" and summary["fwd"]["duplicate_of"] == ""
    assert summary["fwd"]["archetype_manual"] == "single_cds" and summary["fwd"]["source_id"] == "Addgene #1"
    assert summary["two"]["n_cds_annotated"] == "2" and summary["two"]["cds_exact"] == "2"

    orfs = read(out / "orf_calls.csv")
    faa = (out / "orfs.faa").read_text().split(">")[1:]
    assert len(faa) == len(orfs)
    proteins = {line.split("\n")[1] for line in faa}
    for planted in (p1, p2, p3):
        for p in planted:
            assert p.protein in proteins

    info = json.loads((out / "run_info.json").read_text())
    assert {i["file"] for i in info["inputs"]} == {"fwd.gb", "revwrap.gb", "two.gb", "fwd_again.gb", "broken.gb"}
    assert all(len(i["sha256"]) == 64 for i in info["inputs"])
    assert info["load_errors"][0]["file"] == "broken.gb"
    assert info["parameters"]["min_aa"] == 60 and info["biopython"]
    assert info["aggregate"]["manifest_rows_without_file"] == ["ghost.gb"]
    assert info["aggregate"]["files_without_manifest_row"] == ["broken.gb", "fwd_again.gb", "revwrap.gb", "two.gb"]


def test_fusion_parts_are_contained_in_one_long_orf(tmp_path):
    from pilot.synthetic import build_fusion_plasmid, write_genbank

    raw = tmp_path / "raw"
    raw.mkdir()
    seq, planted = build_fusion_plasmid(9, tag_codons=60, gene_codons=150)
    write_genbank(raw / "fusion.gb", seq, planted, name="fusion")
    out = tmp_path / "out"
    assert main(["--input-dir", str(raw), "--out-dir", str(out)]) == 0

    rows = {r["cds_label"]: r for r in read(out / "cds_recovery.csv")}
    tag, gene = rows["tag_part"], rows["gene_part"]
    assert (tag["status"], gene["status"]) == ("contained", "contained")
    assert tag["orf_id"] == gene["orf_id"]  # both parts live in the same called ORF
    assert (gene["n_extra_aa"], gene["c_extra_aa"]) == (str(len(planted[0].protein)), "0")
    assert (tag["n_extra_aa"], tag["c_extra_aa"]) == ("0", str(len(planted[1].protein)))
    orfs = [r for r in read(out / "orf_calls.csv") if r["class"] == "cds_match"]
    assert len(orfs) == 1 and int(orfs[0]["length_aa"]) == len(planted[0].protein) + len(planted[1].protein)


def test_empty_input_directory_exits_nonzero(tmp_path):
    (tmp_path / "raw").mkdir()
    assert main(["--input-dir", str(tmp_path / "raw"), "--out-dir", str(tmp_path / "out")]) == 1
    assert main(["--input-dir", str(tmp_path / "nope"), "--out-dir", str(tmp_path / "out")]) == 2


def test_manifest_without_filename_column_fails_loudly(tmp_path):
    import pytest

    raw = tmp_path / "raw"
    raw.mkdir()
    bad = tmp_path / "bad.csv"
    bad.write_text("file,notes\nx.gb,hi\n")
    with pytest.raises(SystemExit, match="filename"):
        main(["--input-dir", str(raw), "--out-dir", str(tmp_path / "out"), "--manifest", str(bad)])


def test_gtg_start_gene_inside_longer_orf_scores_end_to_end(tmp_path):
    """End-to-end regression for the Met-substitution bug, through real GenBank parsing."""
    import random

    from pilot.synthetic import Planted, random_codons, random_dna, write_genbank

    rng = random.Random(21)
    bg = random_dna(rng, 300)
    upstream = "ATG" + random_codons(rng, 20)          # in-frame extension, no stop
    gene = "GTG" + random_codons(rng, 100) + "TAA"     # the annotated CDS, non-ATG start
    seq = bg + "TAA" + upstream + gene + random_dna(rng, 300)
    start = len(bg) + 3 + len(upstream)

    raw = tmp_path / "raw"
    raw.mkdir()
    planted = [Planted("gtg_gene", 1, "", start, len(gene), False)]
    write_genbank(raw / "gtg.gb", seq, planted, name="gtg")

    out = tmp_path / "out"
    assert main(["--input-dir", str(raw), "--out-dir", str(out)]) == 0
    [row] = [r for r in read(out / "cds_recovery.csv") if r["cds_label"] == "gtg_gene"]
    assert row["status"] == "contained"          # was "missed" before the fix
    assert row["n_extra_aa"] == "21" and row["c_extra_aa"] == "0"
