"""`scripts/34_crosscheck_per_segment.py`: the comparison has to be able to fail.

Loaded by path because `scripts/` is not an importable package, as in `tests/test_bhs_protocol_sample.py`.

A cross-check that cannot report a disagreement is worse than none, because it is quoted as evidence.
Three ways this one could be silently vacuous, all of them tested here:

1. **The arm-to-column map is wrong**, so it compares `bhs_per_segment` against the pooled column and
   reports agreement between two different quantities. The map is asserted directly.
2. **It intersects the frames and says nothing**, so a tier where one file holds 286 constructs and the
   other 200 reports "0 mismatches" over the 200 that happen to overlap. Frame differences are named.
3. **It compares aggregates**, which two implementations can match while disagreeing per construct:
   a swapped pair leaves every mean untouched. The exit code must come from the cells.
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "crosscheck", ROOT / "scripts" / "34_crosscheck_per_segment.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ARMS = ("whole", "auto", "auto_per_segment", "bhs", "bhs_per_segment", "oracle")


def write_twilight(path: Path, rows: dict[str, dict[str, int]], threshold: float = 0.25) -> None:
    out = []
    for pid, by_arm in rows.items():
        for arm, rank in by_arm.items():
            out.append(dict(tier="t", threshold=threshold, plasmid_id=pid, payload_accession="P1",
                            arm=arm, n_homologs_available=5, n_held_out=1,
                            best_target_rank="" if rank is None else rank))
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


def write_detail(path: Path, rows: dict[str, dict[str, int]], mod) -> None:
    out = []
    for pid, by_arm in rows.items():
        r = {"plasmid_id": pid, "payload_accession": "P1", "f": "0.5"}
        for arm, rank in by_arm.items():
            r[mod.ARM_TO_DETAIL[arm]] = rank
        out.append(r)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


def pair(tmp_path: Path, mod, tw_rows, det_rows):
    tw, det = tmp_path / "tw.csv", tmp_path / "detail.csv"
    write_twilight(tw, tw_rows)
    write_detail(det, det_rows, mod)
    return ["--twilight", str(tw), "--detail", str(det)]


def agreeing(n=3) -> dict[str, dict[str, int]]:
    return {f"c{i}": {a: i + j for j, a in enumerate(ARMS)} for i in range(1, n + 1)}


def test_the_arm_to_column_map_is_the_one_the_two_scripts_actually_use(mod):
    """A wrong entry here compares two different quantities and calls them equal."""
    assert mod.ARM_TO_DETAIL["bhs_per_segment"] == "rank_per_segment"
    assert mod.ARM_TO_DETAIL["bhs"] == "rank_pooled"
    assert mod.ARM_TO_DETAIL["auto"] == "rank_auto_pooled"
    assert mod.ARM_TO_DETAIL["auto_per_segment"] == "rank_auto_per_segment"
    # the label-using diagnostic has no arm in script 10 and must not be compared
    assert "oracle_selected" not in mod.ARM_TO_DETAIL
    assert "rank_oracle_selected" not in mod.ARM_TO_DETAIL.values()


def test_identical_files_agree(mod, tmp_path, capsys):
    rows = agreeing()
    assert mod.main(pair(tmp_path, mod, rows, rows)) == 0
    out = capsys.readouterr().out
    assert "0 mismatches" in out and "AGREE: 18 of 18 cells" in out


def test_one_changed_cell_fails_and_is_named(mod, tmp_path, capsys):
    tw = agreeing()
    det = {p: dict(v) for p, v in tw.items()}
    det["c2"]["bhs_per_segment"] = 999
    assert mod.main(pair(tmp_path, mod, tw, det)) == 1
    out = capsys.readouterr().out
    assert "1 mismatches" in out and "DISAGREE" in out
    assert "c2" in out and "bhs_per_segment" in out and "999" in out


def test_a_swapped_pair_fails_although_every_aggregate_matches(mod, tmp_path, capsys):
    """The reason this compares cells: a swap leaves top-1, MRR, mean rank and recall@10 identical."""
    tw = {"c1": {a: 1 for a in ARMS}, "c2": {a: 7 for a in ARMS}}
    det = {"c1": {a: 7 for a in ARMS}, "c2": {a: 1 for a in ARMS}}
    assert mod.main(pair(tmp_path, mod, tw, det)) == 1
    out = capsys.readouterr().out
    assert "12 mismatches" in out
    # Every arm's two aggregate blocks are identical, same top-1, same MRR, same mean rank, which
    # is exactly the trap: an aggregate comparison would have passed this file pair.
    for line in out.splitlines():
        if line.startswith(ARMS):
            halves = " ".join(line.split()).split()[1:]
            assert halves[:4] == halves[4:], line
    assert out.count("0.571") == 2 * len(ARMS)


def test_a_frame_difference_is_reported_not_silently_intersected(mod, tmp_path, capsys):
    tw = agreeing(4)
    det = {p: v for p, v in tw.items() if p != "c4"}
    det["extra"] = {a: 1 for a in ARMS}
    assert mod.main(pair(tmp_path, mod, tw, det)) == 0
    out = capsys.readouterr().out
    assert "3 shared" in out
    assert "only in the twilight CSV" in out and "c4" in out
    assert "only in the detail file" in out and "extra" in out


def test_an_arm_missing_from_one_file_is_skipped_and_said(mod, tmp_path, capsys):
    """A `--no-bhs` run has no bhs arms; comparing the rest is still worth doing, silently is not."""
    tw = {p: {a: v for a, v in by.items() if not a.startswith("bhs")}
          for p, by in agreeing().items()}
    det = agreeing()
    assert mod.main(pair(tmp_path, mod, tw, det)) == 0
    out = capsys.readouterr().out
    assert "skipped: bhs, bhs_per_segment" in out
    assert "x 4 arms" in out


def test_unscorable_cells_are_dropped_rather_than_read_as_zero(mod, tmp_path, capsys):
    tw = agreeing(2)
    tw["c3"] = {a: None for a in ARMS}  # written as empty strings by 10_twilight.py
    det = agreeing(2)
    assert mod.main(pair(tmp_path, mod, tw, det)) == 0
    assert "2 shared" in capsys.readouterr().out


def test_no_shared_constructs_is_a_failure_not_a_pass(mod, tmp_path, capsys):
    assert mod.main(pair(tmp_path, mod, agreeing(2), {"z1": {a: 1 for a in ARMS}})) == 1
    assert "nothing to compare" in capsys.readouterr().out
