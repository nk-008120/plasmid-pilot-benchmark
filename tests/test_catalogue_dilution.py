"""`pilot.catalogue_dilution` and `scripts/40_catalogue_dilution.py`: reference-side dilution.

The arithmetic here is small, which is exactly why it needs tests: every quantity in this module is a
ratio, and this repo's recurring bug is a ratio whose two sides are computed differently (the coverage
denominator). `reference_payload_fraction` recovers aligned columns from a *capped*
coverage, so the cap's effect on the ratio is pinned here rather than discovered in a report; and
`stratify`'s boundary convention is pinned because an off-by-one-stratum error moves counts between
published cells without changing any total.

The script is loaded by path, as in `tests/test_bhs_strata.py`, and run end to end on a hand-built
catalogue small enough that every number in the output can be checked by hand.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import random
from pathlib import Path

import pytest

from pilot.catalogue_dilution import (F_STRATA, family_length_ratio, family_lengths_from_rows,
                                      length_census, macro_micro, reference_payload_fraction,
                                      repeat_fraction, stratify, summarise_repeats)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "catalogue_dilution_script", ROOT / "scripts" / "40_catalogue_dilution.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --- repeat_fraction ---------------------------------------------------------------------------

def test_repeat_fraction_is_zero_on_a_sequence_with_no_duplicate_kmer():
    assert repeat_fraction("ACDEFGHIKLMNPQRSTVWY", k=6) == 0.0


def test_repeat_fraction_approaches_one_on_a_tandem_repeat():
    assert repeat_fraction("ACDEFG" * 50, k=6) > 0.95


def test_repeat_fraction_is_zero_when_the_sequence_is_shorter_than_k():
    # the honest answer is "no evidence of repetition", not a division by a non-positive window
    assert repeat_fraction("ACD", k=6) == 0.0
    assert repeat_fraction("", k=6) == 0.0


def test_repeat_fraction_counts_positions_not_distinct_kmers():
    # "AAAAAAA" has 2 windows, both "AAAAAA": 1 distinct of 2 -> 0.5, not 0 and not 1
    assert repeat_fraction("AAAAAAA", k=6) == pytest.approx(0.5)


# --- family_length_ratio -----------------------------------------------------------------------

def test_family_length_ratio_is_none_for_a_family_too_small_to_compare_against():
    # None rather than 1.0: a singleton's ratio is 1.0 by construction and would be counted as
    # evidence that the entry is clean
    assert family_length_ratio(300, [300]) is None
    assert family_length_ratio(300, []) is None


def test_family_length_ratio_uses_the_median_not_the_mean():
    # one 10,000 aa outlier must not drag the reference point; the median is 300
    assert family_length_ratio(600, [300, 300, 300, 10_000]) == pytest.approx(2.0)


def test_family_length_ratio_calls_a_uniformly_long_family_clean():
    """The proxy's blind spot, pinned so it cannot be quietly forgotten.

    P09866 is 3,387 aa in a family whose median is 3,411 aa, so the ratio is 0.993 and the proxy says
    "ordinary for its family" about one of the two most diluted entries in the catalogue. This is a
    failure mode of the proxy, not a bug, and the report must state it.
    """
    assert family_length_ratio(3387, [3411] * 10) == pytest.approx(0.993, abs=1e-3)


# --- reference_payload_fraction ----------------------------------------------------------------

def test_reference_payload_fraction_recovers_aligned_columns_from_coverage():
    # coverage 0.50 of a 400 aa payload = 200 aligned columns against a 1,000 aa subject
    f_ref, is_bound = reference_payload_fraction(0.50, 400, 1000)
    assert f_ref == pytest.approx(0.20)
    assert is_bound is False


def test_reference_payload_fraction_flags_the_coverage_cap_as_a_lower_bound():
    """`homology.coverage_of` caps at 1.0, so a saturated row's aligned count is only known to be
    at least the payload length: the ratio is then a bound and the flag has to say so."""
    f_ref, is_bound = reference_payload_fraction(1.0, 240, 3054)
    assert is_bound is True
    assert f_ref == pytest.approx(240 / 3054, abs=1e-6)


def test_reference_payload_fraction_never_exceeds_one():
    # a payload longer than its subject would otherwise give f_ref > 1, which is not a fraction
    f_ref, _ = reference_payload_fraction(1.0, 3000, 500)
    assert f_ref == pytest.approx(1.0)


def test_reference_payload_fraction_rejects_a_non_positive_subject():
    with pytest.raises(ValueError):
        reference_payload_fraction(0.5, 100, 0)


def test_reference_payload_fraction_reproduces_the_motivating_construct():
    """a ~240 aa protease domain against the 3,054 aa TEV polyprotein is ~8% of the
    reference, which is the number the whole measurement exists to generalise."""
    f_ref, _ = reference_payload_fraction(1.0, 240, 3054)
    assert 0.07 < f_ref < 0.09


# --- stratify ----------------------------------------------------------------------------------

def test_stratify_puts_a_value_exactly_on_a_bound_in_the_upper_stratum():
    out = stratify([0.50, 0.70, 0.95], F_STRATA)
    assert out == {"<0.50": 0, "0.50-0.70": 1, "0.70-0.95": 1, ">=0.95": 1}


def test_stratify_totals_equal_the_input_length():
    vals = [0.0, 0.1, 0.49, 0.5, 0.69, 0.7, 0.94, 0.95, 1.0]
    assert sum(stratify(vals, F_STRATA).values()) == len(vals)


def test_stratify_rejects_unsorted_bounds_instead_of_sorting_them():
    with pytest.raises(ValueError):
        stratify([0.5], (0.70, 0.50))


# --- length_census -----------------------------------------------------------------------------

def test_length_census_thresholds_are_strictly_greater():
    cen = length_census([1022, 1023], thresholds=(1022,))
    assert cen.above[1022] == (1, 1023)


def test_length_census_reports_entry_and_residue_shares_separately():
    # the gap between the two is the finding: few long entries, many long residues
    cen = length_census([100] * 9 + [1000], thresholds=(500,))
    assert cen.fraction_entries(500) == pytest.approx(0.10)
    assert cen.fraction_residues(500) == pytest.approx(1000 / 1900)


def test_length_census_window_binding_counts_both_ends():
    cen = length_census([41, 100, 6000], thresholds=(1022,), window=(50, 5000))
    assert (cen.n_below_window, cen.n_above_window) == (1, 1)
    assert cen.residues_in_window == 100


def test_length_census_rejects_an_empty_catalogue():
    with pytest.raises(ValueError):
        length_census([])


# --- macro_micro -------------------------------------------------------------------------------

def test_macro_and_micro_disagree_when_one_group_dominates_the_rows():
    """The reason both are reported. 9 rows of payload A all pass, 1 row of payload B fails: micro
    says 10% fail, macro says 50% of payloads have a failing target."""
    rows = [("A", 1.0)] * 9 + [("B", 0.0)]
    micro, macro, n = macro_micro(rows, lambda v: v == 0.0)
    assert micro == pytest.approx(0.10)
    assert macro == pytest.approx(0.50)
    assert n == 2


def test_macro_micro_on_no_rows_is_zero_rather_than_an_error():
    assert macro_micro([], lambda v: True) == (0.0, 0.0, 0)


# --- family_lengths_from_rows / summarise_repeats -----------------------------------------------

def test_family_lengths_from_rows_groups_by_payload():
    rows = [{"payload_accession": "P1", "length": "100"},
            {"payload_accession": "P1", "length": "300"},
            {"payload_accession": "P2", "length": "50"}]
    out = family_lengths_from_rows(rows, lambda r: int(r["length"]))
    assert out == {"P1": [100, 300], "P2": [50]}


def test_summarise_repeats_counts_the_disqualified():
    out = summarise_repeats([0.0, 0.1, 0.4, 0.9], disqualify=0.30)
    assert out["n"] == 4 and out["n_disqualified"] == 2
    assert out["max"] == pytest.approx(0.9)


def test_summarise_repeats_on_nothing_has_every_key_a_caller_reads():
    out = summarise_repeats([])
    assert set(out) >= {"n", "n_disqualified", "median", "p90", "max"}


# --- the script, end to end ----------------------------------------------------------------------

def _tiny_inputs(tmp_path: Path) -> dict[str, Path]:
    """A six-entry catalogue whose every reported number is checkable by hand.

    P1 is a 45 aa payload, deliberately below `homology.MIN_LEN` so the run exercises the window's
    binding count: the path this project's Q9TT33 takes in the real catalogue. P2 is a 3,000 aa
    payload standing in for P04517. S1 is an ordinary 300 aa subject that P1 matches in full
    (f_ref = 45/300 = 0.15); S2 is a 2,500 aa subject that P2 matches over 300 columns (f_ref = 0.12);
    S3 is a 400 aa subject P1 matches over 22.5 columns (f_ref 0.056). D1 is a decoy in no family.

    Sequences are seeded-random rather than periodic: a periodic filler would sit at
    `repeat_fraction` ~ 1 and the run would disqualify every entry, which is the test fooling itself.
    `random.Random` and not `hash`, because `hash` of a str is salted per interpreter run.
    """
    def seq(n: int, salt: str) -> str:
        rng = random.Random(f"catalogue-dilution-{salt}")
        return "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(n))

    ref = tmp_path / "reference_twilight.fasta"
    with open(ref, "w", encoding="utf-8") as fh:
        for acc, n in (("P1", 45), ("P2", 3000), ("S1", 300), ("S2", 2500), ("S3", 400),
                       ("D1", 500)):
            fh.write(f">sp|{acc}|X_TEST test entry {acc}\n{seq(n, acc)}\n")

    hom = tmp_path / "homologs.csv"
    with open(hom, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["payload_accession", "homolog_accession", "score", "identity", "coverage",
                    "length"])
        w.writerow(["P1", "P1", 300.0, 1.0, 1.0, 45])
        w.writerow(["P1", "S1", 200.0, 0.4, 1.0, 300])      # 45 of 300 -> 0.15, lower bound
        w.writerow(["P1", "S3", 120.0, 0.3, 0.5, 400])      # 22.5 of 400 -> 0.056
        w.writerow(["P2", "P2", 900.0, 1.0, 1.0, 3000])
        w.writerow(["P2", "S2", 400.0, 0.3, 0.1, 2500])     # 300 of 2500 -> 0.12

    pls = tmp_path / "payload_lengths.csv"
    with open(pls, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["payload_accession", "payload_aa"])
        w.writerow(["P1", 45])
        w.writerow(["P2", 3000])

    man = tmp_path / "manifest.csv"
    with open(man, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "payload_accession"])
        w.writerow(["a.gb", "P1"])
    man_auto = tmp_path / "manifest_auto.csv"
    with open(man_auto, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["source_id", "payload_accession"])
        w.writerow(["x", "P2"])

    return {"ref": ref, "hom": hom, "pls": pls, "man": man, "man_auto": man_auto}


def test_script_runs_end_to_end_and_the_numbers_are_the_hand_computed_ones(mod, tmp_path):
    p = _tiny_inputs(tmp_path)
    out_dir = tmp_path / "out"
    assert mod.main(["--reference", str(p["ref"]), "--homologs", str(p["hom"]),
                     "--payload-lengths", str(p["pls"]), "--manifest", str(p["man"]),
                     "--manifest-auto", str(p["man_auto"]), "--out-dir", str(out_dir)]) == 0

    data = json.loads((out_dir / "catalogue_dilution.json").read_text(encoding="utf-8"))
    lc = data["length_census"]
    assert lc["n_entries"] == 6
    assert lc["n_residues"] == 45 + 3000 + 300 + 2500 + 400 + 500
    # P2 (3,000) and S2 (2,500) are the only entries over either threshold
    assert lc["above"]["1022"]["entries"] == 2
    assert lc["above"]["2048"]["entries"] == 2
    assert lc["above"]["5000"]["entries"] == 0
    # the 45 aa payload is below the project's reference window and the reader would drop it
    assert lc["n_below_window"] == 1 and lc["n_above_window"] == 0

    assert data["payloads"]["above"]["1022"] == ["P2"]
    assert data["payloads"]["missing_from_catalogue"] == []

    fr = data["f_ref"]
    assert fr["n_rows"] == 5
    # P1/P1, P1/S1 and P2/P2 all have coverage 1.0
    assert fr["n_lower_bound_rows"] == 3
    # four of five rows are below 0.70: only the two self-hits are not... P1/P1 is 1.0, P2/P2 is 1.0
    assert fr["strata"]["<0.50"] == 3
    assert fr["strata"][">=0.95"] == 2

    mech = data["mechanism"]
    # only P2 -> S2 has a subject over 1,022 aa with f_ref < 0.70 (P2 -> P2 is f_ref 1.0)
    assert mech["n_rows_long_subject_and_f_ref_below_070"] == 1
    assert mech["n_target_subjects"] == 5          # P1, S1, S3, P2, S2; D1 is in no family
    assert mech["n_target_subjects_above_1022"] == 2

    for name in ("catalogue_entries.csv", "payload_dilution.csv", "f_ref_by_payload.csv",
                 "f_ref_rows.csv", "catalogue_dilution.log"):
        assert (out_dir / name).is_file(), name


def test_script_outputs_are_readable_as_utf_8_sig(mod, tmp_path):
    """Every CSV this project writes is `utf-8-sig` for Excel, and every reader opens it that way.
    A CSV written without the BOM is read with a leading byte glued to the first column name, which
    fails as a silently missing column rather than as an error."""
    p = _tiny_inputs(tmp_path)
    out_dir = tmp_path / "out"
    mod.main(["--reference", str(p["ref"]), "--homologs", str(p["hom"]),
              "--payload-lengths", str(p["pls"]), "--manifest", str(p["man"]),
              "--manifest-auto", str(p["man_auto"]), "--out-dir", str(out_dir)])
    with open(out_dir / "f_ref_rows.csv", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and "payload_accession" in rows[0]
    assert {r["homolog_accession"] for r in rows} == {"P1", "S1", "S3", "P2", "S2"}


def test_script_refuses_a_missing_input_rather_than_writing_a_partial_run(mod, tmp_path):
    p = _tiny_inputs(tmp_path)
    assert mod.main(["--reference", str(tmp_path / "nope.fasta"), "--homologs", str(p["hom"]),
                     "--payload-lengths", str(p["pls"]), "--manifest", str(p["man"]),
                     "--manifest-auto", str(p["man_auto"]),
                     "--out-dir", str(tmp_path / "out2")]) == 2
    assert not (tmp_path / "out2").exists()
