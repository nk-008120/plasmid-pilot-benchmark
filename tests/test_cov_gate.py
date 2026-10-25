"""Guards for `scripts/47_cov_gate.py`, which tests the gate the working record pre-registered.

The pre-registered falsifier was: the `best_hit_cov` gate works if it leaves the GAL4 constructs on
`whole` and costs nothing in the f < 0.50 stratum. It fired. GAL4's `best_hit_cov` is 0.175, near the
bottom of a distribution whose median is 0.354, so the gate decomposes exactly the constructs
decomposition destroys. `test_the_gate_sends_the_gal4_constructs_the_wrong_way` pins that refutation
so it cannot quietly disappear if the gate is ever re-tuned, and
`test_no_cov_threshold_can_route_gal4_to_whole_without_taking_everything` pins why the failure is not
a threshold-tuning problem: the signal's ordering is wrong, not its cut point.

`k_bhs >= 2` is the replacement. The tests below hold it to the claim that *now* stands for it, which is
weaker than the one originally made: 2 is the interior winner of a three-point sweep whose neighbours
break different strata, not a definition. The original argument, that at k = 1 decomposition and the
whole-ORF query are the same claim about the sequence, was **refuted on the synthetic tier** (the working record, `tests/test_k_gate_synthetic.py`), so `test_two_is_the_boundary_and_three_is_not_the_same_rule`
below should be read as pinning the sweep's shape and not as pinning a derivation.
"""
import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cov_gate", ROOT / "scripts/47_cov_gate.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

ROUTE_B = ROOT / "results/route_b_gate/detail_h25.csv"
needs_route_b = pytest.mark.skipif(not ROUTE_B.is_file(),
                                   reason="route B gate detail is not in this checkout")

# The three GAL4 constructs, as measured at the 25% holdout.
GAL4 = {"plasmid_id": "addgene-plasmid-257515-sequence-519066", "payload": "P04386",
        "f": 0.8961, "cov": 0.175325, "best_hit": "Q9QUR7", "rank_whole": 4,
        "rank_per_segment": 636, "rank_oracle": 4, "k_bhs": 1}


def test_the_cov_gate_keeps_a_confidently_covered_orf_whole():
    r = dict(GAL4, cov=0.90)
    assert mod.gated_rank(r, 0.65) == r["rank_whole"]


def test_the_cov_gate_decomposes_when_coverage_is_low():
    assert mod.gated_rank(dict(GAL4, cov=0.30), 0.65) == GAL4["rank_per_segment"]


def test_the_gate_sends_the_gal4_constructs_the_wrong_way():
    """The pre-registered falsifier, fired. 51.3 proposed this gate to rescue exactly this case."""
    assert GAL4["cov"] < 0.65
    assert mod.gated_rank(GAL4, 0.65) == 636, (
        "the cov gate decomposes GAL4, which is the 9,446-place loss it was proposed to prevent")


def test_no_cov_threshold_can_route_gal4_to_whole_without_taking_everything():
    """Why the refutation is structural. GAL4 sits at 0.175 against a median of 0.354, so any
    threshold low enough to keep it whole keeps nearly the whole tier whole and gives up the
    f < 0.50 rescues. The ordering is wrong, not the cut point."""
    diluted = [dict(GAL4, cov=c) for c in (0.102, 0.15, 0.20, 0.25, 0.354)]
    for thr in (0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 0.90):
        keeps_gal4 = mod.gated_rank(GAL4, thr) == GAL4["rank_whole"]
        also_keeps_diluted = any(mod.gated_rank(d, thr) == d["rank_whole"] for d in diluted)
        assert not keeps_gal4 or also_keeps_diluted, (
            f"threshold {thr} would separate GAL4 from the diluted constructs, which the "
            f"measured distribution says is impossible")


def test_the_k_gate_sends_a_single_segment_construct_whole():
    """k = 1 means the segmenter found one thing, so decomposing can only discard residues."""
    assert mod.k_gated_rank(GAL4, 2) == GAL4["rank_whole"] == 4


def test_the_k_gate_decomposes_a_multi_segment_construct():
    assert mod.k_gated_rank(dict(GAL4, k_bhs=3), 2) == GAL4["rank_per_segment"]


def test_two_is_the_boundary_and_three_is_not_the_same_rule():
    """If k >= 3 behaved identically, 2 would be an arbitrary point on a plateau. It does not:
    a two-segment construct is decomposed at 2 and left whole at 3."""
    two = dict(GAL4, k_bhs=2)
    assert mod.k_gated_rank(two, 2) == two["rank_per_segment"]
    assert mod.k_gated_rank(two, 3) == two["rank_whole"]


def test_metrics_reports_median_beside_the_mean():
    m = mod.metrics([1, 1, 1, 636])
    assert m["median"] == 1 and m["mean"] == 159.75, (
        "one catastrophic rank moves the mean 160-fold and the median not at all; both are reported")
    assert m["r10"] == 0.75


def test_macro_counts_one_payload_once():
    rows = [dict(GAL4, payload="P1", rank_whole=1), dict(GAL4, payload="P1", rank_whole=1),
            dict(GAL4, payload="P2", rank_whole=100)]
    assert mod.macro_mean(rows, lambda r: r["rank_whole"]) == 50.5


def test_stratum_boundaries_are_the_project_strata():
    assert mod.F_STRATA == (0.50, 0.70, 0.95)
    assert mod.stratum(0.8961) == "0.70-0.95", "GAL4 is a high-f construct, which is the whole point"


@needs_route_b
def test_the_k_gate_matches_the_labelled_oracle_router_on_route_b():
    """In-sample on route B and recorded as such. It is asserted because if a future
    change breaks the match, the claim in 52 is no longer true and should fail loudly."""
    rows = mod.read_detail(ROUTE_B)
    assert len(rows) == 43
    for s in mod.STRATUM_NAMES:
        rows_s = [r for r in rows if mod.stratum(r["f"]) == s]
        if not rows_s:
            continue
        gated = mod.metrics([mod.k_gated_rank(r, 2) for r in rows_s])["mean"]
        oracle = mod.metrics([min(r["rank_whole"], r["rank_per_segment"]) for r in rows_s])["mean"]
        assert gated == oracle, f"stratum {s}: k gate {gated} vs oracle router {oracle}"


@needs_route_b
def test_no_f_below_0_50_construct_has_a_single_segment():
    """The mechanism the k gate rests on. If a diluted construct ever yields k = 1, the gate starts
    costing rescues and its justification has to be re-derived rather than kept."""
    rows = mod.read_detail(ROUTE_B)
    singles = [r["plasmid_id"] for r in rows if r["f"] < 0.50 and r["k_bhs"] < 2]
    assert not singles, f"diluted constructs the k gate would leave whole: {singles}"
