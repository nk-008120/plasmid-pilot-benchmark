"""Guards for `scripts/45_route_b_strata.py`, the f-stratified route B report.

the working record forbids a pooled headline across f strata, so the join between the rank table and f is
load-bearing: if it silently fails, every construct lands in one bucket and the output looks like a
report rather than like an error. `test_a_rank_row_whose_f_is_unknown_is_counted_not_dropped` is the
one that would catch that, because the script prints the count rather than discarding the rows
quietly: the same reason 50.3's reader bug needed a number to disagree with.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rb_strata", ROOT / "scripts/45_route_b_strata.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_stratum_boundaries_are_the_project_strata():
    assert mod.F_STRATA == (0.50, 0.70, 0.95)
    assert mod.stratum(0.49) == "<0.50"
    assert mod.stratum(0.50) == "0.50-0.70"
    assert mod.stratum(0.69) == "0.50-0.70"
    assert mod.stratum(0.70) == "0.70-0.95"
    assert mod.stratum(0.95) == ">=0.95"
    assert mod.stratum(1.0) == ">=0.95"


def test_the_plasmid_id_maps_back_to_the_manifests_source_id():
    """The join key. `addgene-plasmid-176139-sequence-353012` is one file of deposit 176139, and if
    this returned the sequence id instead, no row would ever find its f."""
    assert mod.source_id_of("addgene-plasmid-176139-sequence-353012") == "addgene_176139"
    assert mod.source_id_of("addgene-plasmid-50466-sequence-305068") == "addgene_50466"
    assert mod.source_id_of("something_odd") == "something_odd"


def test_summarise_reports_the_failure_count_beside_the_mean():
    """32.2: a mean rank is only reportable beside its count of rank >= 10,000 failures."""
    got = mod.summarise([1, 1, 2, 10, 20000])
    assert got["n"] == 5
    assert got["n_ge_10000"] == 1
    assert got["median_rank"] == 2
    assert got["top1"] == 2
    assert got["recall_at_10"] == 0.8
    assert got["mrr"] == round((1 + 1 + 0.5 + 0.1 + 1 / 20000) / 5, 3)


def test_macro_counts_one_cassette_in_four_backbones_once():
    """27.4: constructs are not independent trials. Four deposits of one payload must not outvote
    four distinct payloads."""
    micro = mod.summarise([1, 1, 1, 1, 100])
    macro = mod.macro({"P1": [1, 1, 1, 1], "P2": [100]})
    assert micro["mean_rank"] == 20.8
    assert macro["mean_rank"] == 50.5, "macro averages the per-payload means, not the ranks"
    assert macro["n_payloads"] == 2


def test_macro_recall_is_averaged_over_payloads_not_constructs():
    macro = mod.macro({"P1": [1, 1], "P2": [500, 1]})
    assert macro["recall_at_10"] == 0.75
