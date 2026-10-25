"""The k triage analysis: the 2x2 arithmetic, and the guard that caught a CONTEXT error."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "k_triage", Path(__file__).resolve().parents[1] / "scripts" / "35_k_triage.py")
kt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kt)

COLS = ["plasmid_id", "payload_accession", "orf_len", "oracle_len", "f", "k_bhs", "k_auto",
        "bhs_iterations", "bhs_stop", "best_hit_accession", "best_hit_score", "best_hit_start",
        "best_hit_end", "best_hit_cov", "best_hit_is_iter1", "best_hit_emitted", "bhs_fellback",
        "rank_whole", "rank_auto_pooled", "rank_auto_per_segment", "rank_pooled",
        "rank_per_segment", "rank_oracle_selected", "rank_oracle", "per_segment_ranks"]


def _row(pid, k, f, cov, whole, per_seg, iters=None):
    d = dict.fromkeys(COLS, "")
    d.update(plasmid_id=pid, payload_accession="P00001", orf_len=400, oracle_len=300, f=f,
             k_bhs=k, k_auto=1, bhs_iterations=iters if iters is not None else k,
             bhs_stop="below_floor", best_hit_cov=cov, best_hit_emitted=1, bhs_fellback=0,
             rank_whole=whole, rank_per_segment=per_seg, rank_auto_pooled=whole,
             rank_auto_per_segment=whole, rank_pooled=whole, rank_oracle_selected=per_seg,
             rank_oracle=per_seg)
    return d


def _write(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def test_rank_treats_blank_and_nonpositive_as_unscorable():
    assert kt.rank("3") == 3
    assert kt.rank("") is None
    assert kt.rank("0") is None
    assert kt.rank("-1") is None
    assert kt.rank(None) is None


def test_table_2x2_arithmetic():
    d = kt.table_2x2([True, True, False, False], [True, False, True, False])
    assert (d["tp"], d["fp"], d["fn"], d["tn"]) == (1, 1, 1, 1)
    assert d["sensitivity"] == 0.5 and d["specificity"] == 0.5
    assert d["ppv"] == 0.5 and d["npv"] == 0.5


def test_table_2x2_reports_none_rather_than_dividing_by_zero():
    d = kt.table_2x2([False, False], [False, False])
    assert d["sensitivity"] is None and d["ppv"] is None
    assert d["specificity"] == 1.0


def test_mcnemar_matches_the_known_floor_for_a_nested_split():
    assert kt.mcnemar_exact(0, 0) == 1.0
    assert kt.mcnemar_exact(5, 0) == pytest.approx(2 ** -4)
    assert kt.mcnemar_exact(3, 3) == pytest.approx(1.0)


def test_pearson_is_none_when_a_variable_is_constant():
    assert kt.pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert kt.pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_the_outcome_label_is_rescue_not_success(tmp_path, capsys):
    # a construct `whole` already retrieves is NOT a positive, however good per-segment is
    rows = [_row("already_fine", 3, 0.4, 0.3, 2, 1),
            _row("rescued", 3, 0.4, 0.3, 500, 2),
            _row("hopeless", 1, 1.0, 0.9, 900, 900)]
    p = tmp_path / "d.csv"
    _write(p, rows)
    with open(tmp_path / "log", "w", encoding="utf-8") as out:
        res = kt.analyse("t", kt.read_detail(p), 10, 0.65, out)
    assert res["n_label_positive"] == 1
    assert res["rule_k_ge_2"]["tp"] == 1
    assert res["rule_k_ge_2"]["fp"] == 1  # already_fine is flagged and is not a rescue


def test_a_lost_construct_is_named_by_the_falsifying_check(tmp_path):
    rows = [_row("lost_it", 4, 0.3, 0.2, 5, 400)]
    p = tmp_path / "d.csv"
    _write(p, rows)
    with open(tmp_path / "log", "w", encoding="utf-8") as out:
        res = kt.analyse("t", kt.read_detail(p), 10, 0.65, out)
    assert res["falsifying"]["lost_top_n_under_per_segment"] == ["lost_it"]


def test_k_and_iterations_are_checked_separately(tmp_path):
    # They are different quantities: an iteration that emits no segment raises iterations and not k.
    # Conflating them put a wrong acceptance target into open item 25 on 18 Oct.
    rows = [_row(f"c{i}", 1, 1.0, 0.9, 1, 1, iters=3) for i in range(5)]
    p = tmp_path / "d.csv"
    _write(p, rows)
    with open(tmp_path / "log", "w", encoding="utf-8") as out:
        res = kt.analyse("corpus", kt.read_detail(p), 10, 0.65, out)
    assert res["k"]["k"]["mean"] == 1.0
    assert res["k"]["iterations"]["mean"] == 3.0
    # 1.10 is CONTEXT's corpus figure for both, so k matches nothing here and iterations certainly not
    assert res["k"]["k"]["matches_context"] is False
    assert res["k"]["iterations"]["matches_context"] is False
    assert res["k"]["matches_context"] is False


def test_main_reproduces_the_committed_detail_files_and_exits_zero(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    inputs = {t: repo / p for t, p in kt.DEFAULT_INPUTS.items()}
    if not all(p.is_file() for p in inputs.values()):
        pytest.skip("the detail CSVs are not present in this checkout")
    args = ["--out-dir", str(tmp_path / "out")]
    for tier, p in inputs.items():
        args += [f"--{tier.replace('_', '-')}", str(p)]
    assert kt.main(args) == 0
    summary = json.loads((tmp_path / "out" / "k_triage_summary.json").read_text(encoding="utf-8"))
    assert summary["context_33_5_mean_k_reproduced"] is True
    # the headline the proposal would quote: k >= 2 misses at most one rescue on any tier
    for tier, r in summary["tiers"].items():
        assert r["rule_k_ge_2"]["fn"] <= 1, tier
