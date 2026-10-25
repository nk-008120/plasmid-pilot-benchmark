import csv

from pilot.audit import COLUMNS, main as audit_main
from pilot.synthetic import build_fusion_plasmid, build_plasmid, write_genbank
from pilot.validate import main as validate_main


def rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def make_results(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    seq, planted = build_fusion_plasmid(9)
    write_genbank(raw / "fusion.gb", seq, planted, name="fusion")
    seq, planted = build_plasmid(4, [{"label": "ok_gene", "strand": 1, "n_codons": 120}])
    write_genbank(raw / "clean.gb", seq, planted, name="clean")
    out = tmp_path / "results"
    assert validate_main(["--input-dir", str(raw), "--out-dir", str(out)]) == 0
    return out


def test_audit_sheet_lists_only_non_exact_rows_with_blank_verdicts(tmp_path):
    out = make_results(tmp_path)
    assert audit_main(["--results-dir", str(out)]) == 0
    sheet = rows(out / "audit.csv")
    assert list(sheet[0].keys()) == COLUMNS
    assert {r["cds_label"] for r in sheet} == {"tag_part", "gene_part"}  # the exact gene is not listed
    assert all(r["reason_for_review"] == "contained" for r in sheet)
    assert all(r["verdict"] == "" and r["extra_residues_are"] == "" for r in sheet)


def test_include_exact_adds_seeded_samples(tmp_path):
    out = make_results(tmp_path)
    assert audit_main(["--results-dir", str(out), "--include-exact", "5"]) == 0
    sheet = rows(out / "audit.csv")
    assert [r["cds_label"] for r in sheet if r["reason_for_review"] == "exact_sample"] == ["ok_gene"]


def test_existing_worksheet_is_never_overwritten_without_force(tmp_path):
    out = make_results(tmp_path)
    assert audit_main(["--results-dir", str(out)]) == 0
    (out / "audit.csv").write_text("my hand-written notes", encoding="utf-8")
    assert audit_main(["--results-dir", str(out)]) == 2
    assert (out / "audit.csv").read_text(encoding="utf-8") == "my hand-written notes"
    assert audit_main(["--results-dir", str(out), "--force"]) == 0
    assert rows(out / "audit.csv")


def test_missing_results_dir_is_reported(tmp_path):
    assert audit_main(["--results-dir", str(tmp_path / "nope")]) == 2
