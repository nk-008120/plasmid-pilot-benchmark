"""Condition B's per-construct exclusion: the union rule, and the confound it must not hide.

Loaded by path because `scripts/` is not an importable package, as in `tests/test_compare_tiers.py`.

Condition B holds each filler's own relatives out of the search database for that construct. Two things
there are easy to get wrong in a way no output would reveal:

* **The union.** A construct with two fillers must lose entries close to *either* of them. Storing the
  maximum identity per accession is what makes one `identity > threshold` comparison implement that
  union; storing the last-seen or the minimum would silently keep entries that should be gone.
* **The confound.** Removing entries shrinks the database, and `N` in
  `S_min = (ln(K m N) + ln(1/E))/lambda` is the residue count actually searched, so condition B's score
  floor differs from condition C's for a reason that has nothing to do with the fillers. The direction is
  fixed by the algebra, fewer residues means a LOWER floor, so the segmenter iterates MORE, and a
  9.95x database ratio was worth 8.61 points earlier in this project. The floor must therefore be
  computed per construct, which is what makes the shift reportable at all.
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

from pilot.homology import score_floor

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("bhs_protocol", ROOT / "scripts" / "26_bhs_protocol.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def write_filler_homologs(path: Path, rows: list[tuple[str, str, float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["filler_accession", "homolog_accession", "score",
                                           "identity", "coverage", "length"])
        w.writeheader()
        for facc, hacc, ident in rows:
            w.writerow({"filler_accession": facc, "homolog_accession": hacc, "score": 100,
                        "identity": ident, "coverage": 0.9, "length": 200})


def test_exclusions_are_the_union_over_a_constructs_fillers(mod, tmp_path):
    p = tmp_path / "fh.csv"
    write_filler_homologs(p, [("F1", "H1", 0.90), ("F2", "H2", 0.80)])
    rows = [{"construct_id": "c1", "filler_accessions": "F1|F2"}]
    got, stats = mod.load_filler_exclusions(p, rows)
    assert got["c1"] == {"H1": 0.90, "H2": 0.80}
    assert stats["n_constructs_with_exclusions"] == 1


def test_the_identity_kept_per_accession_is_the_maximum_over_the_fillers(mod, tmp_path):
    """The entry is close to F2 even if distant from F1, so the union must see 0.95 and not 0.21.
    Keeping the minimum or the last-seen value would leave it in the database above a 0.25 threshold."""
    p = tmp_path / "fh.csv"
    write_filler_homologs(p, [("F1", "H1", 0.21), ("F2", "H1", 0.95)])
    got, _ = mod.load_filler_exclusions(p, [{"construct_id": "c1", "filler_accessions": "F1|F2"}])
    assert got["c1"]["H1"] == pytest.approx(0.95)


def test_the_maximum_is_taken_regardless_of_row_order(mod, tmp_path):
    p = tmp_path / "fh.csv"
    write_filler_homologs(p, [("F1", "H1", 0.95), ("F1", "H1", 0.21)])
    got, _ = mod.load_filler_exclusions(p, [{"construct_id": "c1", "filler_accessions": "F1"}])
    assert got["c1"]["H1"] == pytest.approx(0.95)


def test_a_construct_with_no_filler_gets_an_empty_set_and_is_counted(mod, tmp_path):
    """54 of the 336 synthetic constructs carry no filler (dilution 0, or a carrier of length 0). They
    must come back empty rather than missing, or condition B would silently fall back to condition C."""
    p = tmp_path / "fh.csv"
    write_filler_homologs(p, [("F1", "H1", 0.9)])
    rows = [{"construct_id": "c1", "filler_accessions": ""},
            {"construct_id": "c2", "filler_accessions": "F1"}]
    got, stats = mod.load_filler_exclusions(p, rows)
    assert got["c1"] == {}
    assert stats["n_constructs_without_filler"] == 1
    assert stats["n_constructs_with_exclusions"] == 1


def test_a_filler_with_no_recorded_relatives_contributes_nothing_rather_than_raising(mod, tmp_path):
    p = tmp_path / "fh.csv"
    write_filler_homologs(p, [("F1", "H1", 0.9)])
    got, _ = mod.load_filler_exclusions(p, [{"construct_id": "c1", "filler_accessions": "F1|F_UNKNOWN"}])
    assert got["c1"] == {"H1": 0.9}


def test_the_id_field_can_be_the_filename_when_a_manifest_has_no_construct_id(mod, tmp_path):
    p = tmp_path / "fh.csv"
    write_filler_homologs(p, [("F1", "H1", 0.9)])
    got, _ = mod.load_filler_exclusions(p, [{"filename": "x.gbk", "filler_accessions": "F1"}],
                                        id_field="filename")
    assert got["x.gbk"] == {"H1": 0.9}


def test_shrinking_the_database_lowers_the_floor_which_is_the_confound_to_report():
    """The direction is fixed by the algebra and is the opposite of the intuition that masking makes the
    task harder: fewer residues searched means a lower significance floor, so condition B's segmenter
    iterates MORE than condition C's for a reason unrelated to the fillers."""
    full, shrunk = 20_806_731, 20_806_731 - 500_000
    assert score_floor(1e-3, 600, shrunk) < score_floor(1e-3, 600, full)


def test_the_floor_shift_is_the_log_ratio_and_is_independent_of_query_length():
    """`score_floor` differences cancel the `ln(m)` term, so the shift a database change causes is the
    same at every query length. That is what makes one reported number honest for the whole run."""
    full, shrunk = 20_806_731, 10_403_365
    shifts = {round(score_floor(1e-3, m, shrunk) - score_floor(1e-3, m, full), 9)
              for m in (100, 600, 3000)}
    assert len(shifts) == 1


def test_the_known_swissprot_offset_is_reproduced(mod):
    """this project records 8.61 points between this catalogue and Swiss-Prot. If this drifts, the floor
    arithmetic has changed and condition B's reported shift cannot be compared to it."""
    got = score_floor(1e-3, 100, 207_099_657) - score_floor(1e-3, 100, 20_806_731)
    assert got == pytest.approx(8.61, abs=0.01)
