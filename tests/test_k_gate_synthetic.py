"""Guards for `scripts/48_k_gate_synthetic.py`, which fires this project's named falsifier.

52.2 argued the k = 1 branch of the decomposition gate from a premise rather than a threshold: at
k = 1 the segmenter found one catalogue-supported thing, so decomposing can only discard residues.
52.3 named the refuting case, k = 1 and `per_segment` strictly better than `whole`, counted 0 of 22
across route B and the 23, and said the hand tier cannot produce it at all.

The synthetic tier produces it, in both of this project's paired conditions. The tests below pin the
refutation so it cannot quietly disappear, and pin the two things that make it a refutation rather
than an artefact: it fires in the **control** condition as well as the upper bound, and it hurts five
distinct payloads where the branch helps one.

They also pin the honest limits. `test_the_synthetic_tier_contains_no_catastrophic_decomposition_cost`
is the counterweight: this tier cannot test the k >= 2 branch's *benefit*, because its worst
decomposition loss is 21 rank points against route B's 9,446. Neither tier holds both cases.
"""
import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("k_gate_synthetic", ROOT / "scripts/48_k_gate_synthetic.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

COND_A = ROOT / "results/synthetic_real/A_detail_238.csv"
COND_C = ROOT / "results/synthetic_real/C_detail_238.csv"
needs_a = pytest.mark.skipif(not COND_A.is_file(),
                             reason="condition A's detail CSV is not in this checkout")
needs_c = pytest.mark.skipif(not COND_C.is_file(),
                             reason="condition C's detail CSV is not in this checkout")

# A falsifying construct as measured, condition C at the 25% holdout (`syn_Q6F1K1_motifless_fusion_20`).
FALSIFIER = {"plasmid_id": "syn_Q6F1K1_motifless_fusion_20", "payload": "Q6F1K1", "f": 0.800948,
             "cov": 0.199052, "best_hit": "C5BET0", "rank_whole": 66, "rank_per_segment": 1,
             "rank_oracle": 1, "k_bhs": 1, "bhs_iterations": 2, "bhs_stop": "below_floor",
             "bhs_fellback": 0, "best_hit_emitted": 0, "orf_len": 211, "n_targets": 20,
             "db_residues_searched": 20806731, "condition": "real_competing"}


def test_verdict_names_the_arm_that_actually_won():
    assert mod.verdict(FALSIFIER) == "per_segment"
    assert mod.verdict(dict(FALSIFIER, rank_whole=1, rank_per_segment=44)) == "whole"
    assert mod.verdict(dict(FALSIFIER, rank_whole=1, rank_per_segment=1)) == "tie"


def test_the_gate_sends_the_falsifying_construct_the_wrong_way():
    """The refutation in one assertion: k = 1, so the gate queries the whole ORF at rank 66, while
    the arm it declined to use returns rank 1. 52.2's premise, decomposing can only discard
    residues, is true and its conclusion does not follow, because the residues discarded are the
    ones competing for the rank."""
    assert FALSIFIER["k_bhs"] == 1
    assert mod.k_gated_rank(FALSIFIER, 2) == 66
    assert FALSIFIER["rank_per_segment"] == 1


def test_falsifiers_finds_only_the_named_case():
    rows = [FALSIFIER,
            dict(FALSIFIER, plasmid_id="tie", rank_whole=1, rank_per_segment=1),
            dict(FALSIFIER, plasmid_id="whole_wins", rank_whole=23, rank_per_segment=44),
            dict(FALSIFIER, plasmid_id="k2", k_bhs=3)]
    assert [r["plasmid_id"] for r in mod.falsifiers(rows, 2)] == ["syn_Q6F1K1_motifless_fusion_20"]


def test_branch_rank_points_signs_the_branch_rather_than_scoring_it():
    """`net` is positive when leaving k = 1 constructs whole earns its place. Both directions are
    counted, because a branch that only saves is not being measured."""
    rows = [FALSIFIER, dict(FALSIFIER, plasmid_id="w", rank_whole=23, rank_per_segment=44)]
    rp = mod.branch_rank_points(rows, 2)
    assert rp == {"n": 2, "saved": 21, "lost": 65, "net": -44}


def test_census_counts_payloads_and_not_only_constructs():
    """27.4: constructs are not independent trials and the macro view governs. A branch that helps
    14 deposits of one payload and hurts 5 deposits of 5 is not a 14-to-5 win."""
    rows = [dict(FALSIFIER, plasmid_id=f"p{i}", payload=f"P{i}") for i in range(5)]
    rows += [dict(FALSIFIER, plasmid_id=f"w{i}", payload="Q7MBQ5",
                  rank_whole=23, rank_per_segment=44) for i in range(14)]
    c = mod.census(rows, 2)["k=1"]
    assert c["per_segment"] == 5 and c["whole"] == 14
    assert c["payloads_per_segment"] == ["P0", "P1", "P2", "P3", "P4"]
    assert c["payloads_whole"] == ["Q7MBQ5"]


def test_binding_reports_the_absent_constructs_rather_than_the_scored_ones():
    """A gate measured on a silently truncated subset is the failure 52.3 is about, so the tier
    total is a constant in the script and the shortfall is printed."""
    b = mod.binding([FALSIFIER])
    assert b["tier_total"] == mod.TIER_TOTAL == 336
    assert b["scored"] == 1 and b["absent_at_holdout"] == 335
    assert b["no_best_hit"] == 0 and b["segmenter_fell_back"] == 0
    assert b["stop_reasons"] == {"below_floor": 1}


def test_binding_counts_a_construct_with_no_best_hit():
    assert mod.binding([dict(FALSIFIER, cov=None)])["no_best_hit"] == 1


@needs_a
@needs_c
def test_both_conditions_read_the_same_catalogue_and_the_same_constructs():
    """A and C are paired construct-for-construct with only the filler differing (30.1). If the
    pairing breaks, a sample change and a condition change are being summed into one number, which
    is 21.3's error."""
    a = {r["plasmid_id"]: r for r in mod.read_detail(COND_A)}
    c = {r["plasmid_id"]: r for r in mod.read_detail(COND_C)}
    assert set(a) == set(c) and len(a) == 238
    assert {r["db_residues_searched"] for r in a.values()} == {20806731}
    assert {r["db_residues_searched"] for r in c.values()} == {20806731}
    for pid in a:
        assert a[pid]["rank_oracle"] == c[pid]["rank_oracle"], (
            f"{pid}: the oracle arm queries only the payload span, which the conditions share, "
            f"so it is 30.1's built-in pairing check and must be identical")


@needs_a
def test_the_falsifier_fires_in_the_control_condition():
    """Condition A is `random_protein()` fillers and 30.2's measured null: whole-to-oracle is
    -0.00 in every stratum. A falsifier that fires HERE cannot be dismissed as condition C's
    extremity, which is the first thing a reader will reach for."""
    rows = mod.read_detail(COND_A)
    fal = mod.falsifiers(rows, 2)
    assert len(fal) == 1
    assert fal[0]["plasmid_id"] == "syn_Q6F1K1_tagged_fusion_80"
    assert fal[0]["rank_whole"] == 2 and fal[0]["rank_per_segment"] == 1


@needs_c
def test_the_falsifier_fires_five_times_in_the_bound_over_five_payloads():
    rows = mod.read_detail(COND_C)
    fal = mod.falsifiers(rows, 2)
    assert len(fal) == 5
    assert len({r["payload"] for r in fal}) == 5, (
        "five distinct payloads, against the one payload the k = 1 branch helps")
    assert all(0.70 <= r["f"] < 0.95 for r in fal), (
        "all five sit in 0.70-0.95, the same stratum GAL4 broke the segmenter in: the gate's one "
        "measured win and its refutation live in the same cell")


@needs_c
def test_the_k_gate_is_beaten_by_per_segment_in_the_stratum_it_was_built_for():
    """52.2's claim was that the k gate equals the labelled oracle router in every stratum. On
    condition C at f 0.70-0.95 it does not: it is worse than the oracle router AND worse than
    decomposing unconditionally, on the mean, the MRR and the operational secondary alike."""
    rows = [r for r in mod.read_detail(COND_C) if mod.stratum(r["f"]) == "0.70-0.95"]
    gated = mod.metrics([mod.k_gated_rank(r, 2) for r in rows])
    always = mod.metrics([r["rank_per_segment"] for r in rows])
    router = mod.metrics([min(r["rank_whole"], r["rank_per_segment"]) for r in rows])
    assert gated["mean"] > always["mean"] > router["mean"]
    assert gated["mrr"] < always["mrr"]
    assert gated["r10"] < always["r10"], "recall@10 is the operational secondary (26.3, 42.1)"


@needs_a
@needs_c
def test_the_k_two_branch_is_not_refuted_by_either_condition():
    """The other direction holds, on 189 constructs across both conditions: not one where the
    segmenter claimed two or more segments and `whole` still won."""
    total = 0
    for path in (COND_A, COND_C):
        c = mod.census(mod.read_detail(path), 2)["k>=2"]
        assert c["whole"] == 0, f"{path.name}: {c['payloads_whole']}"
        total += c["n"]
    assert total == 189


@needs_a
@needs_c
def test_the_branch_changes_sign_across_the_bracket():
    """this project's working rules: an overshooting constructed condition is a bound and the real case is bracketed
    between conditions. The branch is worth +293 rank points on the control and -10 on the bound, so
    the synthetic tier refutes the branch's premise without settling its value."""
    net_a = mod.branch_rank_points(mod.read_detail(COND_A), 2)["net"]
    net_c = mod.branch_rank_points(mod.read_detail(COND_C), 2)["net"]
    assert net_a == 293 and net_c == -10
    assert net_c < 0 < net_a


@needs_a
@needs_c
def test_the_synthetic_tier_contains_no_catastrophic_decomposition_cost():
    """The limit that keeps this from being read as the whole answer. The k >= 2 branch exists to
    prevent route B's 9,446-rank GAL4 loss; the worst decomposition cost anywhere in this tier is 21
    ranks, so the tier can price the branch's cost and cannot price its benefit."""
    for path in (COND_A, COND_C):
        rows = mod.read_detail(path)
        worst = max(r["rank_per_segment"] - r["rank_whole"] for r in rows)
        assert worst == 21, f"{path.name}: worst per_segment-minus-whole is {worst}"


@needs_c
def test_the_emitted_probe_is_reported_as_in_sample_and_incomplete():
    """`best_hit_emitted == 0` separates all 5 of condition C's falsifiers from its other 47 k = 1
    constructs, and misses condition A's. It is pinned as a hypothesis with a known miss so it
    cannot be promoted to a gate on the strength of the condition it was found in."""
    ep = mod.emitted_probe(mod.read_detail(COND_C), 2)
    assert ep == {"k1": 52, "fired": 5, "fired_and_falsifying": 5, "falsifying": 5, "missed": []}
    ep_a = mod.emitted_probe(mod.read_detail(COND_A), 2)
    assert ep_a["fired"] == 0 and ep_a["missed"] == ["syn_Q6F1K1_tagged_fusion_80"], (
        "it misses the control condition's falsifier, so it is not a complete separator")


@needs_a
def test_read_detail_agrees_with_47_on_every_shared_field():
    """48 extends 47's reader rather than replacing it, and the gate itself is imported. This pins
    that the two scripts are reading the same rows, so a change to 47 cannot silently re-define the
    rule 48 reports on."""
    spec47 = importlib.util.spec_from_file_location("cov_gate_direct", ROOT / "scripts/47_cov_gate.py")
    m47 = importlib.util.module_from_spec(spec47)
    spec47.loader.exec_module(m47)
    base = {r["plasmid_id"]: r for r in m47.read_detail(COND_A)}
    mine = {r["plasmid_id"]: r for r in mod.read_detail(COND_A)}
    assert set(base) == set(mine)
    for pid, r in base.items():
        for key, value in r.items():
            assert mine[pid][key] == value, f"{pid}.{key}"


@needs_a
def test_the_scored_subset_is_the_holdout_and_nothing_else():
    """98 of 336 are absent at the 25% holdout. They are 7 whole payloads at 14 constructs each,
    the holdout removes payloads, not dilution cells, so every architecture and every dilution
    level survives and the absence cannot hide the falsifying shape."""
    rows = mod.read_detail(COND_A)
    man = list(csv.DictReader(open(ROOT / "data/synthetic/synthetic_manifest.csv",
                                   newline="", encoding="utf-8-sig")))
    assert len(man) == mod.TIER_TOTAL
    scored = {r["plasmid_id"] for r in rows}
    absent = [r for r in man if r["construct_id"] not in scored]
    assert len(absent) == 98
    lost = {r["payload_accession"] for r in absent}
    assert len(lost) == 7
    assert all(sum(1 for r in absent if r["payload_accession"] == p) == 14 for p in lost)
    kept_cells = {r["dilution_target"] for r in man if r["construct_id"] in scored}
    assert kept_cells == {r["dilution_target"] for r in man}


@needs_a
@needs_c
def test_write_csvs_carries_the_stratum_and_the_condition_on_every_row(tmp_path):
    """this project's working rules: state the catalogue and never pool across strata. The census CSV therefore keys
    every row by condition and stratum, with the pooled `ALL` row labelled as such rather than
    unlabelled."""
    loaded = [("A", "random (control / floor)", mod.read_detail(COND_A)),
              ("C", "real_competing (upper bound)", mod.read_detail(COND_C))]
    mod.write_csvs(tmp_path, loaded, 2)
    rows = list(csv.DictReader(open(tmp_path / "census.csv", newline="", encoding="utf-8-sig")))
    assert {r["f_stratum"] for r in rows} == {"ALL", "<0.50", "0.50-0.70", "0.70-0.95", ">=0.95"}
    assert all(r["condition"] in ("A", "C") for r in rows)
    pooled_c = next(r for r in rows if r["condition"] == "C" and r["f_stratum"] == "ALL")
    assert int(pooled_c["k1_per_segment_wins"]) == 5 and int(pooled_c["kge_whole_wins"]) == 0
    fal = list(csv.DictReader(open(tmp_path / "falsifiers.csv", newline="", encoding="utf-8-sig")))
    assert len(fal) == 6, "one from the control condition and five from the bound"
    assert all(f["f_stratum"] == "0.70-0.95" for f in fal if f["condition"] == "C")
