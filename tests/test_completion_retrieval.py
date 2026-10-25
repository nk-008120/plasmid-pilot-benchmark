"""Tests for R15's wiring: `pilot.complete`'s extension arms and `scripts/58_completion_retrieval.py`.

R15 asks whether extending a 33-66 aa fragment with a 650M completion lifts the payload's remote
family. The ways that question can be answered wrongly are all in the *construction of the query*, not
in the ranking: the ranking is `pilot.align.best_target_rank`, shared with `10_twilight.py` and
`51_fragment_probe.py`, and a test here that re-checked it would be testing the wrong module.

So what is pinned is this:

* the four arms exist and isolate one change each, because a two-arm design would credit the model for
  a query simply getting longer: a raw local-alignment score rises with length for free;
* the shuffle preserves composition exactly and is **stable across processes**, since a shuffle seeded
  from Python's salted `hash()` would differ between the ThinkPad dry run and the Omen run and the two
  would not be comparable;
* `truth` and `extended` are the same length, which is what makes `truth` a ceiling rather than a
  different experiment;
* the prefix-cut convention is *checked* rather than assumed, and the checker returns a count so it
  cannot pass having compared nothing (this project's lesson applied to a check);
* the paired counter excludes an unranked cell from both sides and counts it, because dropping it on
  one side only would give the two arms different denominators.
"""
from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

from pilot.complete import (  # noqa: E402
    GENERIC_ARMS,
    donor_completions,
    fixed_tails,
    most_frequent_residue,
)
from pilot.complete import (
    EXTENSION_ARMS,
    Completion,
    check_prefix_cut,
    deterministic_shuffle,
    extension_arms,
    read_completion_sweep,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.split(".")[0], ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FRAG = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
COMP = "ACDEFGHIKLMN"
TRUE = "APNLQIYNVLGG"
#: A catalogue stand-in for the fixed tails. Small on purpose: the tails must be a function of
#: whatever they are derived from, and a two-sequence "catalogue" makes that visible.
CAT = ["ACDEFGHIKLMNPQRSTVWYLLLL", "LLLLAAAA"]
TAILS = fixed_tails(CAT, 12)
#: A stand-in for another payload's completion: same length, model-shaped, not this query's.
DONOR = "WYFMKTGAVLIS"


def test_the_six_arms_isolate_one_change_each():
    """Two arms would credit the model for length. Six separate length, composition, order, and
    whether the extension has anything to do with this query at all."""
    q = extension_arms(FRAG, COMP, TRUE, "k", TAILS, DONOR)
    assert tuple(q) == EXTENSION_ARMS
    assert q["fragment"] == FRAG
    assert q["extended"] == FRAG + COMP
    assert q["truth"] == FRAG + TRUE
    # Every extended arm is the same length, so a rank difference between any two of them cannot be
    # a length effect, which is the whole reason the five exist rather than one.
    assert len({len(q[a]) for a in EXTENSION_ARMS if a != "fragment"}) == 1
    assert len(q["extended"]) == len(FRAG) + len(COMP)


def test_the_shuffle_preserves_composition_and_only_moves_the_completion():
    q = extension_arms(FRAG, COMP, TRUE, "k", TAILS, DONOR)
    tail = q["shuffled"][len(FRAG):]
    assert Counter(tail) == Counter(COMP), "a composition control that changes composition controls nothing"
    assert q["shuffled"].startswith(FRAG), "the fragment must be untouched; only the extension is permuted"
    assert tail != COMP


def test_the_shuffle_is_stable_across_processes_and_keyed():
    """`hash()` is salted per process, so a shuffle seeded from it would differ between the ThinkPad
    dry run and the Omen run with nothing in either output saying so. These values are pinned."""
    assert deterministic_shuffle("ACDEFGHIKLMN", "P12345:33:single") == "DMCFKINLHEAG"
    assert deterministic_shuffle("ACDEFGHIKLMN", "P12345:66:single") == "NMKCIDEFLAGH"


def test_a_completion_and_a_truth_of_different_lengths_is_refused():
    """A ragged pair would make `truth` a longer query than `extended`, so the ceiling would be a
    length effect. both sides of a comparison are computed the same way or neither is."""
    with pytest.raises(ValueError):
        extension_arms(FRAG, COMP, TRUE[:-1], "k", TAILS, DONOR)


def test_a_degenerate_completion_shuffles_to_itself_and_that_is_correct():
    """The 650M degenerates to one repeated residue on some payloads. There the arm and
    its composition control are the same string, and the right behaviour is to produce it, not to
    perturb it: a shuffle that forced a difference would make the control a different query."""
    q = extension_arms(FRAG, "L" * 12, TRUE, "k", TAILS, DONOR)
    assert q["shuffled"] == q["extended"]


def test_the_prefix_cut_is_checked_against_the_catalogue_and_counts_what_it_checked():
    seq = "M" * 33 + TRUE + "YYYY"
    rows = [Completion("P1", 33, 12, "single", COMP, TRUE)]
    ok, bad = check_prefix_cut(rows, {"P1": seq}.get)
    assert (ok, bad) == (1, [])
    # A payload absent from the catalogue and one whose truth is not the prefix continuation are
    # different failures and both have to be named rather than counted as a pass.
    ok, bad = check_prefix_cut(rows, {}.get)
    assert ok == 0 and "not in the catalogue" in bad[0]
    ok, bad = check_prefix_cut([Completion("P1", 33, 12, "single", COMP, "A" * 12)], {"P1": seq}.get)
    assert ok == 0 and "prefix continuation" in bad[0]


def test_the_sweep_reader_filters_by_length_and_by_mode(tmp_path):
    p = tmp_path / "sweep.csv"
    p.write_text(
        "model,payload_accession,payload_len,fragment_aa,n_completed,mode,completed,truth,identity,"
        "degenerate\n"
        f"m,P1,200,33,12,single,{COMP},{TRUE},0.1,0\n"
        f"m,P1,200,33,12,iterative,{COMP},{TRUE},0.1,0\n"
        f"m,P1,200,8,12,single,{COMP},{TRUE},0.1,0\n", encoding="utf-8")
    assert len(read_completion_sweep(p)) == 3
    assert len(read_completion_sweep(p, lengths=[33])) == 2
    got = read_completion_sweep(p, lengths=[33], mode="single")
    assert len(got) == 1 and got[0].fragment_len == 33 and got[0].mode == "single"


def test_the_paired_counter_excludes_an_unranked_cell_from_both_sides():
    """One arm ranking where the other does not is not a win; it is a cell neither arm can be scored
    on. Counting it as a win for whichever arm returned a number is how a denominator drifts."""
    mod = _load("58_completion_retrieval.py")
    rows = [
        dict(rank_fragment=10, rank_extended=5),    # better
        dict(rank_fragment=5, rank_extended=10),    # worse
        dict(rank_fragment=7, rank_extended=7),     # equal
        dict(rank_fragment="", rank_extended=1),    # neither arm is scorable here
        dict(rank_fragment=3, rank_extended=""),
    ]
    p = mod.paired(rows, "extended")
    assert (p["better"], p["equal"], p["worse"], p["unpaired"]) == (1, 1, 1, 2)
    assert p["net_rank_points"] == (10 - 5) + (5 - 10) + 0


def test_lengths_below_33_are_refused_because_both_checkpoints_sit_on_the_floor_there():
    """at 8 and 16 aa the 650M is 1.3 SE *below* the matched baseline and degenerates to
    one residue on 24 of 24 payloads. A run there cannot inform anything, so it is not offered."""
    mod = _load("58_completion_retrieval.py")
    assert mod.main(["--lengths", "8", "33"]) == 2
    assert mod.DEFAULT_LENGTHS == (33, 66)


def test_the_summary_reports_every_arm_and_names_the_stratum():
    """the working record forbids a pooled scalar across strata; the cell here is (threshold, length) and
    each payload contributes one row to each cell, so micro and macro coincide at the cell."""
    mod = _load("58_completion_retrieval.py")
    rows = []
    for thr in (1.0, 0.25):
        for i, acc in enumerate(("P1", "P2")):
            rows.append(dict(threshold=thr, frag_len_aa=33, payload_accession=acc,
                             identity_to_truth=0.2, rank_fragment=10 + i, rank_extended=5 + i,
                             rank_shuffled=20 + i, rank_truth=1 + i,
                             rank_homopolymer=6 + i, rank_generic=7 + i,
                             rank_misconditioned=8 + i))
    out = mod.summarise(rows)
    assert [r["threshold"] for r in out] == [1.0, 0.25]
    assert all(r["n_payloads"] == 2 for r in out)
    for arm in EXTENSION_ARMS:
        assert f"recall_at_10_{arm}" in out[0] and f"mrr_{arm}" in out[0]
    assert out[0]["extended_better"] == 2 and out[0]["truth_better"] == 2
    assert out[0]["shuffled_worse"] == 2


# ---- the fixed-tail control (wired 23 Oct) -------------------------------------

def test_a_fixed_tail_carries_nothing_about_the_query_it_is_appended_to():
    """That is the entire property under test. `shuffled` still carries this query's composition;
    these two carry nothing of it, so if they match `extended` the model is not in the claim."""
    a = extension_arms(FRAG, COMP, TRUE, "k", TAILS, DONOR)
    b = extension_arms("WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW", "QRSTVWYACDEF", TRUE, "other", TAILS, DONOR)
    for arm in GENERIC_ARMS:
        assert a[arm][len(FRAG):] == b[arm][33:], f"{arm} differs between two unrelated queries"


def test_the_tails_are_derived_from_the_catalogue_and_not_chosen():
    """Change the catalogue and the tails must change, or they are literals wearing a function."""
    other = fixed_tails(["WWWWWWWWWWWWWWWWWWWW", "WWYY"], 12)
    assert other["homopolymer"] == "W" * 12
    assert TAILS["homopolymer"] == "L" * 12, TAILS
    assert other["generic"] != TAILS["generic"]


def test_the_generic_tail_is_stable_across_processes():
    """The `deterministic_shuffle` lesson again: a Counter iterates in insertion order, so a draw
    over an unsorted residue list would depend on the order the catalogue was read in. Pinned."""
    assert TAILS["generic"] == "QHLLTALLSLLL"
    assert fixed_tails(CAT[::-1], 12)["generic"] == "QHLLTALLSLLL", "catalogue order must not matter"


def test_the_homopolymer_tail_is_what_the_composition_baseline_predicts():
    """It is the composition-only predictor's actual output, which is why it is this arm and not an
    arbitrary residue: this project found the completions that collapse to it are the best subgroup,
    and this makes that deliberate."""
    assert TAILS["homopolymer"] == most_frequent_residue(CAT) * 12


@pytest.mark.parametrize("bad", [{}, {"homopolymer": "L" * 12}, {"generic": "A" * 12}])
def test_a_missing_tail_is_refused_rather_than_dropping_an_arm(bad):
    """An absent arm reads as a measured blank in the summary, which is the defect class this
    project's register collects. So it raises instead."""
    with pytest.raises(ValueError, match="fixed tail"):
        extension_arms(FRAG, COMP, TRUE, "k", bad, DONOR)


def test_a_tail_of_the_wrong_length_is_refused():
    """A short tail would make its arm a shorter query than `extended`, so any difference between
    them would be a length effect: the same reason a ragged completion and truth is refused."""
    with pytest.raises(ValueError, match="residues"):
        extension_arms(FRAG, COMP, TRUE, "k",
                       {"homopolymer": "L" * 11, "generic": "A" * 12}, DONOR)


def test_a_zero_length_tail_is_refused():
    with pytest.raises(ValueError):
        fixed_tails(CAT, 0)


def test_tails_need_standard_residues_to_be_derived_from():
    with pytest.raises(ValueError):
        fixed_tails(["XXXX", "ZZZZ"], 12)


def test_the_summary_pairs_the_fixed_tails_against_the_model_and_not_only_the_fragment():
    """Against `fragment` both tails would read as wins, which is not the question. this project predicts they MATCH `extended`, so `extended` is what they have to be paired against."""
    mod = _load("58_completion_retrieval.py")
    rows = []
    for i, acc in enumerate(("P1", "P2")):
        rows.append(dict(threshold=0.25, frag_len_aa=33, payload_accession=acc,
                         identity_to_truth=0.2, rank_fragment=100, rank_extended=50,
                         rank_shuffled=80, rank_truth=10,
                         rank_homopolymer=50, rank_generic=40 + i,
                         rank_misconditioned=60 + i))
    out = mod.summarise(rows)[0]
    # Identical to the model's arm on both payloads: the control's predicted outcome.
    assert out["homopolymer_vs_extended_equal"] == 2
    assert out["homopolymer_vs_extended_better"] == 0 == out["homopolymer_vs_extended_worse"]
    # And strictly better on both, which would be the stronger version of the same finding.
    assert out["generic_vs_extended_better"] == 2
    assert out["generic_vs_extended_net"] == (50 - 40) + (50 - 41)
    # Every extended arm still gets its own pairing against the fragment baseline.
    for arm in EXTENSION_ARMS[1:]:
        assert f"{arm}_better" in out and f"{arm}_net" in out
    assert out["homopolymer_net"] == 2 * (100 - 50)


# ---- 77.5's control: the same model, conditioned on someone else ----------------------------

def test_the_donor_arm_is_the_model_output_of_a_different_query():
    q = extension_arms(FRAG, COMP, TRUE, "k", TAILS, DONOR)
    assert q["misconditioned"] == FRAG + DONOR
    assert q["misconditioned"] != q["extended"], "a donor equal to the completion is not a control"
    assert len(q["misconditioned"]) == len(q["extended"])


def test_no_row_receives_its_own_completion():
    """It is a derangement by construction, and that is the whole validity of the arm: a row that
    got its own completion back would make `misconditioned` a copy of `extended` for that cell."""
    rows = [Completion(f"P{i}", 33, 12, "single", f"{'A' * 11}{chr(65 + i)}", "Q" * 12)
            for i in range(12)]
    d = donor_completions(rows)
    assert len(d) == 12
    for r in rows:
        assert d[(r.payload_accession, r.fragment_len, r.mode)] != r.completed


def test_donors_are_drawn_within_a_length_and_mode_group():
    """Length and decoding mode are held fixed so that only the conditioning moves; a donor from
    the 66 aa group would change the query length as well and confound the two."""
    rows = ([Completion(f"P{i}", 33, 12, "single", f"{'A' * 11}{chr(65 + i)}", "Q" * 12)
             for i in range(3)]
            + [Completion(f"P{i}", 66, 12, "single", f"{'C' * 11}{chr(65 + i)}", "Q" * 12)
               for i in range(3)])
    d = donor_completions(rows)
    for (acc, m, mode), got in d.items():
        assert got.startswith("A" if m == 33 else "C"), (acc, m, got)


def test_a_group_of_one_is_refused_rather_than_handed_itself():
    with pytest.raises(ValueError, match="borrow"):
        donor_completions([Completion("P1", 33, 12, "single", "A" * 12, "Q" * 12)])


def test_the_donor_assignment_is_stable_across_processes_and_ignores_input_order():
    """`deterministic_shuffle`'s lesson for a third time in this module."""
    rows = [Completion(f"P{i}", 33, 12, "single", f"{'A' * 11}{chr(65 + i)}", "Q" * 12)
            for i in range(8)]
    assert donor_completions(rows) == donor_completions(rows[::-1])


def test_a_donor_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="residues"):
        extension_arms(FRAG, COMP, TRUE, "k", TAILS, "AAA")


def test_a_donor_may_equal_its_recipient_by_value_and_that_is_correct():
    """Two payloads can complete identically, 13 of the 650M's collapse to `L`, so a perfect
    derangement by slot can still hand a row a string equal to its own. The first run of this arm
    refused on that and the refusal was wrong, for the reason the shuffle test above gives: forcing
    a difference would make the control a different query. It is a count to report, not an error."""
    rows = [Completion("P1", 33, 12, "single", "L" * 12, "Q" * 12),
            Completion("P2", 33, 12, "single", "L" * 12, "Q" * 12)]
    d = donor_completions(rows)
    # Deranged by slot: each took the other's row.
    assert d[("P1", 33, "single")] == d[("P2", 33, "single")] == "L" * 12
    # And equal by value, which the caller counts rather than rejects.
    assert sum(1 for r in rows
               if d[(r.payload_accession, r.fragment_len, r.mode)] == r.completed) == 2
