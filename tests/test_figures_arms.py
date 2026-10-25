"""The two figure scripts draw every arm a summary carries, and survive one that predates the metrics.

Loaded by path because `scripts/` is not an importable package, as in `tests/test_compare_tiers.py`.

These are end-to-end smoke tests on purpose. `tests/test_figstyle.py` pins the table the scripts read;
what it cannot catch is a script that reads the table and then still plots a hardcoded subset, or one
that crashes on a `_summary.json` written before `*_mean_rank` existed, and most of this project's
finished results folders hold exactly that file. Both failures are silent in a figure: a missing line
looks like a missing series and a crash happens after the earlier figures are already on disk.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(stem: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{stem}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def figs():
    return load("07_figures", "figures07")


@pytest.fixture(scope="module")
def tiers():
    return load("17_compare_tiers", "compare17")


def cell(arm: str, n: int, top1: int, top10: int, *, modern: bool) -> dict:
    d = {f"{arm}_n": n, f"{arm}_top1": top1, f"{arm}_top10": top10,
         f"{arm}_n_payloads": n, f"{arm}_top1_macro": 100.0 * top1 / n}
    if modern:
        d.update({f"{arm}_mean_rank": 1.0 + n - top1, f"{arm}_mrr": top1 / n,
                  f"{arm}_mean_rank_macro": 1.0 + n - top1, f"{arm}_mrr_macro": top1 / n,
                  f"{arm}_top10_macro": 100.0 * top10 / n})
    return d


def write_summary(d: Path, *, arms: list[str], modern: bool, tier: str = "hand",
                  scorer: str = "sw-blosum62-11-1") -> Path:
    by = {}
    for i, thr in enumerate((1.00, 0.50, 0.25)):
        c: dict = {"n_scorable": 10, "n_payloads": 8}
        for j, a in enumerate(arms):
            c.update(cell(a, 10, 9 - i - j, 10 - i, modern=modern))
        by[f"{thr:.2f}"] = c
    agg: dict = {"scorer": scorer, "tier": tier, "by_threshold": by}
    if modern:
        agg["arms"] = arms
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"twilight_{scorer}_{tier}_summary.json"
    p.write_text(json.dumps(agg), encoding="utf-8")
    return p


SIX = ["whole", "auto", "bhs", "oracle", "auto_per_segment", "bhs_per_segment"]
OLD_THREE = ["whole", "auto", "oracle"]


# ------------------------------------------------------------------ 07_figures.py

def test_twilight_figures_draw_every_arm_including_the_per_segment_ones(figs, tmp_path, capsys):
    run = tmp_path / "run"
    write_summary(run, arms=SIX, modern=True)
    out = tmp_path / "figures"
    figs.fig_twilight(run, out)
    figs.fig_twilight_metrics(run, out)
    printed = capsys.readouterr().out
    assert "auto_per_segment" in printed and "bhs_per_segment" in printed
    for stem in ("fig_d4_twilight_curve", "fig_d9_twilight_metrics"):
        assert (out / f"{stem}.png").is_file() and (out / f"{stem}.pdf").is_file()


def test_an_old_summary_still_draws_and_says_what_it_could_not(figs, tmp_path, capsys):
    """Most finished results folders hold a pre-17-Oct summary: no `arms` key, no mean rank, no MRR."""
    run = tmp_path / "old"
    write_summary(run, arms=OLD_THREE, modern=False)
    out = tmp_path / "figures_old"
    figs.fig_twilight(run, out)
    figs.fig_twilight_metrics(run, out)
    printed = capsys.readouterr().out
    assert "arms drawn -> whole, auto, oracle" in printed
    assert "mean_rank" in printed and "panel left empty" in printed
    assert (out / "fig_d9_twilight_metrics.png").is_file()


def test_main_draws_the_twilight_figures_without_a_decompose_run(figs, tmp_path):
    """A `10_twilight.py` results folder is a valid input; it used to be rejected outright."""
    run = tmp_path / "tw_only"
    write_summary(run, arms=SIX, modern=True)
    out = tmp_path / "out"
    assert figs.main(["--results-dir", str(run), "--out", str(out)]) == 0
    assert (out / "fig_d9_twilight_metrics.png").is_file()
    assert not (out / "fig_d1_dilution_doseresponse.png").exists()


def test_main_still_refuses_a_directory_with_nothing_in_it(figs, tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert figs.main(["--results-dir", str(empty), "--out", str(tmp_path / "o")]) == 2
    assert "no twilight_*_summary.json" in capsys.readouterr().out


def test_the_caption_is_wrapped_so_the_canvas_is_not_stretched(figs):
    """`bbox_inches="tight"` grows the figure to fit the caption: one long line tripled its width."""
    assert "\n" in figs.wrapped_note()
    assert max(len(line) for line in figs.wrapped_note().splitlines()) <= 104


# ------------------------------------------------------------------ 17_compare_tiers.py

def test_tier_comparison_draws_both_figures_and_names_every_arm(tiers, tmp_path, capsys):
    hand, corpus = tmp_path / "hand", tmp_path / "corpus"
    write_summary(hand, arms=SIX, modern=True, tier="hand")
    write_summary(corpus, arms=OLD_THREE, modern=False, tier="corpus")
    out = tmp_path / "figures"
    rc = tiers.main(["--results", str(hand), "--results", str(corpus), "--out", str(out)])
    assert rc == 0
    printed = capsys.readouterr().out
    # the printed table carries the per-segment arms, which it used to omit along with everything
    # that was not `whole` or `oracle`
    assert "best-hit segments, per-segment" in printed
    assert "motif segments, per-segment" in printed
    # and the tier that cannot supply the primary metrics says so rather than drawing a flat line
    assert "corpus [micro]" in printed and "panel left empty" in printed
    for stem in ("fig_d5_tiers", "fig_d10_tiers_metrics"):
        assert (out / f"{stem}.png").is_file() and (out / f"{stem}.pdf").is_file()


def test_a_tier_without_the_per_segment_arm_simply_has_no_dashed_line(tiers, tmp_path):
    """D5 is coloured by tier, so an absent arm must be skipped rather than plotted as nan or zero."""
    corpus = tmp_path / "corpus"
    write_summary(corpus, arms=OLD_THREE, modern=False, tier="corpus")
    out = tmp_path / "figures"
    assert tiers.main(["--results", str(corpus), "--out", str(out)]) == 0
    assert (out / "fig_d5_tiers.png").is_file()


def test_an_untiered_summary_is_still_skipped(tiers, tmp_path, capsys):
    """The 15.13 run predates `--tier` and was scored against a different catalogue."""
    d = tmp_path / "mixed"
    write_summary(d, arms=SIX, modern=True, tier="hand")
    stray = d / "twilight_sw-blosum62-11-1_summary.json"
    stray.write_text(json.dumps({"scorer": "sw-blosum62-11-1", "by_threshold": {}}), encoding="utf-8")
    assert tiers.main(["--results", str(d), "--out", str(tmp_path / "f")]) == 0
    assert "skipped" in capsys.readouterr().out
