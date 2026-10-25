"""Guards for `scripts/57_purity_doseresponse.py`, the the working record purity dose-response figure.

Three of these are the figure's argument rather than housekeeping:

* `test_the_floor_band_holds_the_segments_that_round_up_into_it` pins the one trap this script has.
  68.1's table reports **240** constructs at "purity >= 0.904", and the four
  `syn_Q6F1K1_multi_gene_cassette_*` segments sit at **0.9037**. A floor written at the rounded value
  drops them and reports 236, which is a band losing members to a printed figure's own precision:
  this project's "state the catalogue a number came from" one level down, at the band boundary.
* `test_no_construct_falls_outside_every_reported_band` is *report binding, not settings*. The bands
  are a reported partition of all 336 constructs, so a construct in none of them is a silent omission
  from the figure, and the script prints the count whether or not it is zero.
* `test_a_negative_net_means_random_filler_costs_the_arm_rank` fixes the sign. The whole figure is a
  paired contrast whose direction is the finding (60.2, 66.2), and a reversed subtraction would draw
  a correct magnitude upside down with nothing to catch it.

The rest guard the joins and the units: the architecture comes from the construct id rather than from
a manifest, `recall@10` is drawn in `pilot.figstyle`'s percent, and the two decomposed architectures
really do span the same four f values as the undecomposed one, which is what makes the right panel a
contrast and not two different samples.
"""
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEOM = ROOT / "results" / "auto_geometry_postr10" / "auto_geom.csv"


def _load(stem: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / stem)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pd = pytest.importorskip("matplotlib") and _load("57_purity_doseresponse.py", "purity_doseresponse")

#: The geometry is a results artefact, so a worktree without it skips rather than fails.
needs_geom = pytest.mark.skipif(not GEOM.is_file(), reason=f"{GEOM.name} not staged in this checkout")


def _rows():
    return pd.read_geometry(GEOM)


def test_architecture_comes_from_the_construct_id():
    assert pd.architecture("syn_A1ABP5_motifless_fusion_20") == "motifless_fusion"
    assert pd.architecture("syn_Q6F1K1_multi_gene_cassette_80") == "multi_gene_cassette"
    assert pd.architecture("syn_B9KFT3_single_cds_00") == "single_cds"


def test_a_negative_net_means_random_filler_costs_the_arm_rank():
    """Random worse than SwissProt must read negative: that is 60.2's and 66.2's reported sign."""
    rows = [{"esm_whole_real": "5", "esm_whole_random": "900"}]
    out = pd.paired(rows, "esm_whole")
    assert out["net"] == -895
    assert (out["better"], out["equal"], out["worse"]) == (0, 0, 1)
    flipped = pd.paired([{"esm_whole_real": "900", "esm_whole_random": "5"}], "esm_whole")
    assert flipped["net"] == 895 and flipped["better"] == 1


def test_recall_and_failure_counts_use_the_project_thresholds():
    assert pd.RECALL_DEPTH == 10, "recall@10 is the operational secondary "
    assert pd.FAIL_RANK == 10_000, "an embedding mean is admissible only beside this count (32.2)"
    rows = [{"esm_whole_real": "10", "esm_whole_random": "11"},
            {"esm_whole_real": "9999", "esm_whole_random": "10000"}]
    out = pd.paired(rows, "esm_whole")
    assert out["recall_swissprot"] == 0.5 and out["recall_random"] == 0.0
    assert out["fail_swissprot"] == 0 and out["fail_random"] == 1


def test_the_floor_band_holds_the_segments_that_round_up_into_it():
    """0.9037 prints as 0.904 and must not be excluded by a floor written at the printed value."""
    assert pd.BAND_FLOOR < 0.9037, "a floor at 0.904 drops the four Q6F1K1 segments and reports 236"
    assert pd.band_of(0.9037) == pd.BAND_FLOOR
    assert pd.band_of(0.8421) == 0.80, "the one trimmed motifless segment stays in its own band"


@needs_geom
def test_no_construct_falls_outside_every_reported_band():
    rows = _rows()
    assert len(rows) == 336
    assert [r for r in rows if pd.band_of(r["purity"]) is None] == []


@needs_geom
def test_the_band_counts_are_the_ones_context_68_1_reports():
    bands = {b["band"]: b["n"] for b in pd.by_band(_rows(), "esm_auto_per_segment")}
    assert bands == {0.20: 24, 0.40: 24, 0.60: 24, 0.80: 24, pd.BAND_FLOOR: 240}


@needs_geom
def test_the_purity_dose_response_reproduces_context_68_1():
    """The four measured points and the floor, recomputed rather than quoted."""
    net = {b["band"]: b["net"] for b in pd.by_band(_rows(), "esm_auto_per_segment")}
    assert net[0.20] == -305_780
    assert net[0.40] == -352_053
    assert net[0.60] == -159_487
    assert net[0.80] == -231
    assert net[pd.BAND_FLOOR] == 0, "at purity >= 0.90 the filler effect is exactly zero"


@needs_geom
def test_the_right_panel_is_f_matched_rather_than_two_samples():
    """`multi_gene_cassette` spans the same four f values at a purity that barely moves (68.1)."""
    series = pd.by_series(_rows(), "esm_auto_per_segment")
    cassette = series["multi_gene_cassette"]
    assert [r["f"] for r in cassette] == list(pd.F_LEVELS)
    assert all(r["n"] == 24 for r in cassette)
    assert all(r["net"] == 0 for r in cassette)
    assert all(round(r["recall_random"], 3) == 0.917 for r in cassette)
    assert max(r["purity_max"] for r in cassette) - min(r["purity_min"] for r in cassette) < 0.08
    # the undecomposed architecture moves across the same f values, which is the contrast
    motifless = series["motifless_fusion"]
    assert [r["f"] for r in motifless] == list(pd.F_LEVELS)
    assert round(motifless[0]["recall_random"], 3) == 0.0
    assert round(motifless[-1]["recall_random"], 3) == 0.708


@needs_geom
def test_the_whole_and_best_hit_cells_reproduce_context_66():
    """The two cells this figure's caption quotes, at f < 0.50, holdout 1.00 (66.1 and 66.2)."""
    sub = [r for r in _rows() if r["f"] < 0.50]
    assert len(sub) == 144
    whole = pd.paired(sub, "esm_whole")
    assert whole["net"] == -1_831_284
    assert (whole["better"], whole["equal"], whole["worse"]) == (16, 0, 128)
    assert (whole["fail_random"], whole["fail_swissprot"]) == (109, 35)
    assert whole["recall_random"] == 0.0
    bhs = pd.paired(sub, "esm_bhs_per_segment")
    assert (bhs["better"], bhs["equal"], bhs["worse"]) == (0, 144, 0)
    assert bhs["net"] == 0 and bhs["recall_random"] == 1.0 and bhs["recall_swissprot"] == 1.0
    auto = pd.paired(sub, "esm_auto_per_segment")
    assert auto["net"] == -657_833
    assert (auto["better"], auto["equal"], auto["worse"]) == (6, 96, 42)
    assert (auto["fail_random"], auto["fail_swissprot"]) == (36, 10)


@needs_geom
def test_the_two_tiers_differ_in_the_motif_decomposition_on_exactly_two_constructs():
    """"Identical but for the filler" is a claim about the construct, not about every derived quantity.

    The motif catalogue scans the filler too, so 68.1 reports 2 of 336 where it differs. Counted here
    rather than assumed, which is also why the undecomposed count is 142 on one tier and 144 on the
    other.
    """
    rows = _rows()
    moved = [r for r in rows if abs(r["purity"] - r["purity_random"]) > 1e-4]
    assert sorted(r["plasmid_id"] for r in moved) == [
        "syn_Q8R4A1_motifless_fusion_20", "syn_Q8R4A1_motifless_fusion_60"]
    assert sum(1 for r in rows if r["undecomposed"]) == 142
    assert sum(1 for r in rows
               if int(r["payload_seg_len_random"]) == int(r["orf_len"])) == 144


@needs_geom
def test_every_drawn_series_exists_in_the_data():
    rows = _rows()
    present = {r["architecture"] for r in rows}
    for arch, style_arm, _off_l, _off_r in pd.SERIES:
        assert arch in present
        assert style_arm in ("whole", "auto_per_segment")
    # 68.1's table calls these two f = 1.00; measured, they are 0.9967 and 1.0000, so the label is a
    # rounding. Pinned at what the data supports rather than at what the section prints.
    for arch in pd.F1_ARCHITECTURES:
        assert arch in present
        assert all(r["f"] >= pd.F1_MIN for r in rows if r["architecture"] == arch)
    assert pd.F1_MIN < 1.0


@needs_geom
def test_the_written_table_is_never_a_pooled_scalar(tmp_path):
    """Every row names the band or the architecture it belongs to."""
    rows = _rows()
    out = tmp_path / "purity.csv"
    pd.write_table(pd.by_band(rows, "esm_auto_per_segment"),
                   pd.by_series(rows, "esm_auto_per_segment"), out)
    with open(out, newline="", encoding="utf-8-sig") as fh:
        table = list(csv.DictReader(fh))
    assert table and all(r["subset"] in ("purity_band", "architecture") for r in table)
    assert all(r["key"] for r in table)
    assert sum(int(r["n"]) for r in table if r["subset"] == "purity_band") == len(rows)
