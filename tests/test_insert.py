import csv

import pytest

from pilot.insert import DEFAULT_TOLERANCE, check_insert, parse_insert_size
from pilot.orfs import ORF
from pilot.validate import main


def orf(nt):
    return ORF(1, 0, nt, False, "M" * (nt // 3 - 1), "M" * (nt // 3 - 1))


@pytest.mark.parametrize(
    "text,want",
    [("2025", 2025), (" 2,025 ", 2025), ("2025 bp", 2025), ("", None), ("  ", None), ("NA", None),
     ("n/a", None), ("-", None), ("?", None), ("abc", None), ("0", None), ("-5", None), (None, None)],
)
def test_parse_insert_size(text, want):
    assert parse_insert_size(text) == want


def test_picks_closest_not_longest():
    """In a small vector the resistance marker is often the longest ORF, so 'longest' would pick backbone."""
    orfs = [orf(861), orf(750), orf(3000)]  # AmpR-sized, insert-sized, a long spurious one
    r = check_insert(orfs, 747)
    assert (r.best_orf_index, r.best_orf_nt) == (1, 750)
    assert r.recovered and r.verdict == "recovered" and r.ratio == pytest.approx(750 / 747, abs=1e-4)


def test_tolerance_band_edges():
    assert check_insert([orf(1100)], 1000, tolerance=0.10).recovered       # exactly +10%
    assert check_insert([orf(900)], 1000, tolerance=0.10).recovered        # exactly -10%
    assert not check_insert([orf(1104)], 1000, tolerance=0.10).recovered
    assert check_insert([orf(1104)], 1000, tolerance=0.15).recovered       # tolerance is configurable


def test_verdicts():
    assert check_insert([orf(600)], 1000).verdict == "short"
    assert check_insert([orf(2000)], 1000).verdict == "long"
    assert check_insert([], 1000).verdict == "no_orfs"
    assert check_insert([orf(600)], None).verdict == ""        # no stated size -> no claim
    assert check_insert([orf(600)], None).recovered is None
    assert DEFAULT_TOLERANCE == 0.10


def test_end_to_end_columns(tmp_path, make_gb):
    raw = tmp_path / "raw"
    raw.mkdir()
    # a 120-codon CDS -> 363 nt ORF including its stop
    _, _, planted = make_gb("sized", 31, [{"label": "g", "strand": 1, "n_codons": 120}], directory=raw)
    assert planted[0].length_nt == 366
    manifest = tmp_path / "m.csv"
    manifest.write_text(
        "filename,source_id,archetype_manual,insert_size_bp,notes\n"
        "sized.gb,X,single_cds,366,exact\n", encoding="utf-8-sig")
    out = tmp_path / "out"
    assert main(["--input-dir", str(raw), "--out-dir", str(out), "--manifest", str(manifest)]) == 0
    with open(out / "pilot_summary.csv", newline="", encoding="utf-8-sig") as fh:
        [row] = list(csv.DictReader(fh))
    assert row["insert_size_bp"] == "366"
    assert row["insert_best_orf_nt"] == "366"
    assert row["insert_verdict"] == "recovered"
    assert row["insert_best_orf_class"] == "cds_match"
    assert row["insert_best_orf_id"].startswith("sized|orf")


def test_missing_insert_size_leaves_columns_blank(tmp_path, make_gb):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_gb("nosize", 32, [{"label": "g", "strand": 1, "n_codons": 90}], directory=raw)
    out = tmp_path / "out"
    assert main(["--input-dir", str(raw), "--out-dir", str(out)]) == 0
    with open(out / "pilot_summary.csv", newline="", encoding="utf-8-sig") as fh:
        [row] = list(csv.DictReader(fh))
    assert row["insert_size_bp"] == "" and row["insert_verdict"] == "" and row["insert_ratio"] == ""
