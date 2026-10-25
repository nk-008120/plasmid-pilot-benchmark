"""Guards for `scripts/44_route_b_bench.py`, which turns the f table into a scorable tier.

The property that matters is that the retrieval tier and the f table describe the same molecules.
`data/route_b_raw/` holds five superseded records, and if the two steps chose differently, every
route B rank would be reported against an f measured on another sequence: the coverage-denominator
bug in a new costume, and invisible in every output.
"""
import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rb_bench", ROOT / "scripts/44_route_b_bench.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

F_TABLE = ROOT / "results/route_b_f/f_by_construct.csv"
MANIFEST = ROOT / "data/manifest_route_b.csv"
SECOND = ROOT / "data/manifest_route_b_second.csv"
needs_outputs = pytest.mark.skipif(not (F_TABLE.is_file() and MANIFEST.is_file()),
                                   reason="route B tier is not built in this checkout")


def _row(file="a.gbk", acc="P00001", cls="canonical", f="0.5"):
    return {"file": file, "addgene_id": "1", "deposit_title": "t", "payload_accession": acc,
            "payload_name": "n", "group": "A", "f": f, "stratum": "0.50-0.70",
            "payload_present": cls}


def test_absent_and_variant_pairs_are_excluded_by_default():
    """The default is a decision, not a convenience.

    All four engineered-variant pairs sit at identity 0.779-0.822, below the 0.856 known-true floor
    measured over 20 hand pairs, so including them would decide by judgement what the floor exists to
    decide by measurement. `--include-variants` reverses it deliberately; the default must not drift.
    """
    rows = [_row(acc="P1"), _row(file="b.gbk", acc="P2", cls="absent"),
            _row(file="c.gbk", acc="P3", cls="engineered_variant")]
    primary, second, excluded = mod.partition(rows)
    assert [r["payload_accession"] for r in primary] == ["P1"]
    assert not second
    assert {r["payload_accession"] for r in excluded} == {"P2", "P3"}


def test_include_variants_scores_them_because_that_is_my_call():
    rows = [_row(acc="P1"), _row(file="c.gbk", acc="P3", cls="engineered_variant")]
    primary, _, excluded = mod.partition(rows, include_variants=True)
    assert {r["payload_accession"] for r in primary} == {"P1", "P3"}
    assert not excluded


def test_a_second_payload_on_one_file_goes_to_the_second_pass_not_the_bin():
    """The manifest holds one payload per file, so the second must be scored separately rather
    than dropped, and the split is by accession, which no measurement can move."""
    rows = [_row(acc="P18887"), _row(acc="P06746")]
    primary, second, excluded = mod.partition(rows)
    assert [r["payload_accession"] for r in primary] == ["P06746"]
    assert [r["payload_accession"] for r in second] == ["P18887"]
    assert not excluded


def test_a_pair_with_no_alignment_is_excluded_rather_than_scored():
    primary, second, excluded = mod.partition([_row(f="")])
    assert not primary and not second and len(excluded) == 1


@needs_outputs
def test_every_manifest_row_names_a_record_the_f_table_measured():
    with open(F_TABLE, newline="", encoding="utf-8-sig") as fh:
        measured = {(r["file"], r["payload_accession"]) for r in csv.DictReader(fh) if r["f"]}
    for path in (MANIFEST, SECOND):
        with open(path, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                assert (r["filename"], r["payload_accession"]) in measured, (path.name, r)


@needs_outputs
def test_the_kept_directory_holds_exactly_the_manifests_records():
    kept = {p.name for p in (ROOT / "data/route_b_kept").glob("*.gbk")}
    named = set()
    for path in (MANIFEST, SECOND):
        with open(path, newline="", encoding="utf-8-sig") as fh:
            named |= {r["filename"] for r in csv.DictReader(fh)}
    assert kept == named, f"only in kept: {kept - named}; only in manifests: {named - kept}"


@needs_outputs
def test_no_deposit_appears_twice_in_the_primary_manifest():
    with open(MANIFEST, newline="", encoding="utf-8-sig") as fh:
        ids = [r["source_id"] for r in csv.DictReader(fh)]
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]


def test_the_variant_default_is_exclusion_and_is_my_decision():
    """Pins the flag's default as well as the function's, since the script is the entry point."""
    import inspect
    sig = inspect.signature(mod.partition)
    assert sig.parameters["include_variants"].default is False
    rows = [_row(acc="P48431", cls="engineered_variant"),
            _row(file="b.gbk", acc="P20309", cls="engineered_variant")]
    primary, second, excluded = mod.partition(rows)
    assert not primary and not second
    assert len(excluded) == 2, "the SOX2-17 chimera and rM3D stay out unless asked for"
