"""`scripts/19_find_homologs_all.py`: the payload selection that lets one parameter be probed cheaply.

Loaded by path because `scripts/` is not an importable package, as in `tests/test_compare_tiers.py`.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "find_homologs_all", ROOT / "scripts" / "19_find_homologs_all.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["find_homologs_all"] = mod
    spec.loader.exec_module(mod)
    return mod


fha = _load()
AVAILABLE = ["P01012", "P04517", "P42212", "Q9GV45"]


def test_selects_the_named_accessions_in_order():
    assert fha.select_only("Q9GV45,P04517", AVAILABLE) == ["Q9GV45", "P04517"]


def test_a_single_accession_works():
    assert fha.select_only("P01012", AVAILABLE) == ["P01012"]


def test_whitespace_is_tolerated():
    assert fha.select_only(" Q9GV45, P04517 ", AVAILABLE) == ["Q9GV45", "P04517"]


def test_unknown_accessions_are_dropped_not_guessed():
    """A typo must not silently become an empty run that looks like a finished one."""
    assert fha.select_only("Q9GV45,NOTREAL", AVAILABLE) == ["Q9GV45"]
    assert fha.select_only("NOTREAL", AVAILABLE) == []


def test_duplicates_are_collapsed():
    assert fha.select_only("P01012,P01012", AVAILABLE) == ["P01012"]


def test_an_empty_spec_selects_nothing():
    assert fha.select_only("", AVAILABLE) == []
    assert fha.select_only(", ", AVAILABLE) == []
