"""The route B selection must stay consistent with the measurements it was derived from.

`data/route_b_selected.csv` is a *decision*, not an output, so nothing re-derives it on
each run and nothing would notice if the candidate menu or the homolog gate moved underneath it. These
tests are that noticing. They assert the properties this project states as criteria, so if a future
harvest changes a gate count the failure names which criterion broke rather than leaving the tier to be
built on a stale number.
"""
import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "data/route_b_selected.csv"
MENU = ROOT / "data/route_b_candidates.csv"
GATE = ROOT / "results/route_b_homologs/payload_lengths.csv"

pytestmark = pytest.mark.skipif(
    not (SELECTED.is_file() and MENU.is_file() and GATE.is_file()),
    reason="route B selection or its inputs are not present in this checkout",
)


def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def selected():
    return _read(SELECTED)


@pytest.fixture(scope="module")
def gate():
    return {r["payload_accession"]: r for r in _read(GATE)}


def test_selection_is_the_size_context_48_records(selected):
    assert len(selected) == 21
    assert len({r["uniprot_accession"] for r in selected}) == 21


def test_every_pick_was_actually_searched(selected, gate):
    missing = [r["uniprot_accession"] for r in selected if r["uniprot_accession"] not in gate]
    assert not missing, f"selected but never put through the homolog search: {missing}"


def test_every_pick_clears_the_25_percent_gate(selected, gate):
    """48.2 criterion 1. A payload with no remote homolog left is unscorable, not merely hard."""
    unscorable = [r["uniprot_accession"] for r in selected
                  if int(gate[r["uniprot_accession"]]["available_at_25"]) < 1]
    assert not unscorable, f"selected but unscorable at the 25% holdout: {unscorable}"


def test_gate_counts_in_the_selection_match_the_measurement(selected, gate):
    """The selection file carries copies of measured counts; copies drift, so check them."""
    for r in selected:
        g = gate[r["uniprot_accession"]]
        assert r["available_at_25"] == g["available_at_25"], r["uniprot_accession"]
        assert r["available_at_35"] == g["available_at_35"], r["uniprot_accession"]
        assert r["length_aa"] == g["payload_aa"], r["uniprot_accession"]


def test_only_oct4_sits_below_the_thin_family_bar(selected):
    """48.2 names Q01860 as the one pick below 4 remote homologs, and says to read it at 35%."""
    thin = {r["uniprot_accession"] for r in selected if int(r["available_at_25"]) < 4}
    assert thin == {"Q01860"}, f"the set of thin-family picks has changed: {thin}"


def test_no_pick_has_a_family_so_large_it_has_no_headroom(selected):
    """48.2 criterion 2: large remote families are the warning sign, not the merit."""
    too_easy = [(r["uniprot_accession"], r["available_at_25"]) for r in selected
                if int(r["available_at_25"]) > 200]
    assert not too_easy, f"picks with no retrieval headroom: {too_easy}"


def test_only_sacas9_is_truncated_by_the_esm2_window(selected):
    """this project's truncation row, counted before the run rather than after it."""
    over = {r["uniprot_accession"] for r in selected if int(r["length_aa"]) > 1022}
    assert over == {"J7RUA5"}, f"the set of truncated picks has changed: {over}"


def test_picks_are_candidates_and_not_excluded_rows(selected):
    menu = {r["uniprot_accession"].strip(): r for r in _read(MENU) if r["uniprot_accession"].strip()}
    for r in selected:
        row = menu.get(r["uniprot_accession"])
        assert row is not None, f"{r['uniprot_accession']} is not in the candidate menu"
        assert row["status"] == "candidate", (
            f"{r['uniprot_accession']} is marked {row['status']} in the menu: {row['notes'][:80]}")


def test_both_architectures_are_represented(selected):
    """48.2 criterion 5. Deposit-level flags, so this is coverage of the menu's claim, not of f."""
    for token in ("2A", "IRES"):
        n = sum(1 for r in selected if token in r["architectures"])
        assert n >= 10, f"only {n} picks carry a {token} deposit flag"
