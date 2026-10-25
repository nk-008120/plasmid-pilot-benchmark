"""Tests for the public benchmark: the rules as code, the builder, and the scorer.

`pilot.benchmark` exists so that a reviewer's run cannot quietly use a different rule from this
project's, which means the rules have to be *tested* and the two that are refusals have to actually
refuse. The cases below are in that order: the boundaries, the refusals, the metrics, the submission
forms, and the builder.

The one that matters most is the first. There are now three places in this repository that name the
payload-fraction bands, and three spellings of one set of boundaries is how they drift apart:
quietly, with every table still printing. So the public spelling is pinned against the internal ones on
a grid rather than by inspection.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pilot import benchmark as bm
from pilot.catalogue_dilution import F_STRATA
from pilot.homology import Homolog

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.split(".")[0], ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- the boundaries ------------------------------------------------------------------------------

def test_the_public_strata_agree_with_the_internal_ones_on_a_grid():
    """Three spellings of one set of boundaries is how they drift. Pinned, not inspected."""
    internal = _load("29_stratify_by_payload_fraction.py")
    for i in range(0, 1001):
        f = i / 1000
        assert bm.stratum_of(f) == internal.stratum_of(f), f
    assert bm.stratum_of(1.0) == "f >= 0.95"


def test_the_public_strata_use_the_same_bounds_as_catalogue_dilution():
    assert tuple(lo for _, lo, _ in bm.STRATA[1:]) == F_STRATA


def test_every_f_lands_in_exactly_one_stratum():
    for i in range(0, 1001):
        f = i / 1000
        assert sum(1 for _, lo, hi in bm.STRATA if lo <= f < hi) == 1


def test_an_f_above_one_is_a_bug_not_a_category():
    """`oracle_len > orf_len` should be impossible, so it raises rather than landing somewhere."""
    with pytest.raises(ValueError):
        bm.stratum_of(1.5)


# ---- the refusals -------------------------------------------------------------------------------

def test_a_pooled_scalar_is_refused_rather_than_returned():
    with pytest.raises(bm.PooledScalarRefused) as e:
        bm.pooled_scalar([1, 2, 3])
    # The message has to say what is refused and what to do instead. It used to be pinned by
    # its citation, which does not survive into the published tree, so the assertion pins the
    # substance: a reader who hits this needs the rule and the alternative, not a section number.
    msg = str(e.value)
    assert "not reportable" in msg and "stratum" in msg


def test_report_never_emits_a_pooled_row():
    items = [bm.Item(f"c{i}", "t", "P1", "o", 100, f_len) for i, f_len in enumerate((10, 60, 80, 100))]
    rows = bm.report([(it, 0.25, 1) for it in items])
    assert rows
    assert all(r["stratum"] in {n for n, _, _ in bm.STRATA} for r in rows)
    assert len({r["stratum"] for r in rows}) == 4, "four distinct f values, four strata, no pooled row"


def test_an_embedding_arms_mean_rank_cannot_be_quoted_without_its_failure_count():
    """the working record. The caveat is enforced rather than left to prose."""
    row = {"mean_rank": 1088.0, "n": 286, "n_rank_ge_failure": 15}
    assert bm.mean_rank_phrase(row, arm_kind="alignment") == "1088.00"
    phrase = bm.mean_rank_phrase(row, arm_kind="embedding")
    assert "1088.00" in phrase and "15 of 286" in phrase and "10,000" in phrase
    with pytest.raises(ValueError):
        bm.mean_rank_phrase({"mean_rank": 1088.0, "n": 286}, arm_kind="embedding")
    with pytest.raises(ValueError):
        bm.mean_rank_phrase(row, arm_kind="whatever")


def test_report_rejects_an_unknown_arm_kind():
    with pytest.raises(ValueError):
        bm.report([], arm_kind="guess")


# ---- the metrics --------------------------------------------------------------------------------

def test_an_unscorable_item_stays_in_the_denominator():
    """Dropping it would shrink n exactly where the measurement gets hard, and flatter every curve."""
    m = bm.metrics([1, 2, None, None], depth=10)
    assert m["n"] == 4 and m["n_unscorable"] == 2
    assert m["recall_at_10"] == 0.5, "2 of 4, not 2 of 2"
    assert m["mrr"] == round((1 + 0.5) / 4, 4)
    assert m["mean_rank"] == 1.5, "the mean describes the items that were found"


def test_recall_counts_the_depth_inclusively():
    assert bm.metrics([10], depth=10)["recall_at_10"] == 1.0
    assert bm.metrics([11], depth=10)["recall_at_10"] == 0.0


def test_the_failure_count_is_at_or_above_ten_thousand():
    m = bm.metrics([9_999, 10_000, 10_001])
    assert m["n_rank_ge_failure"] == 2


def test_macro_counts_each_payload_once_however_many_constructs_carry_it():
    """410 corpus constructs carry 138 payloads and one appears in 37, so a micro average can be 37
    repetitions of one answer."""
    pairs = [("P1", 1)] * 37 + [("P2", 100)]
    micro = bm.metrics([r for _, r in pairs])["mean_rank"]
    macro = bm.macro_by_payload(pairs, lambda v: sum(v) / len(v), 2)
    assert micro == pytest.approx(3.6, abs=0.1)
    assert macro == 50.5, "two payloads, (1 + 100) / 2"


def test_empty_cells_report_none_rather_than_zero():
    m = bm.metrics([])
    assert m["n"] == 0 and m["mean_rank"] is None and m["mrr"] is None
    assert bm.macro_by_payload([], lambda v: 1.0) is None


# ---- the submission forms -----------------------------------------------------------------------

def test_a_masked_accession_occupies_no_rank():
    """A held-out accession does not exist at this threshold, so returning it is returning nothing: it
    must not push the true answer down a place."""
    ordered = ["HELD", "HELD2", "TRUE", "OTHER"]
    assert bm.rank_from_ranked_list(ordered, {"TRUE"}, {"HELD", "HELD2"}) == 1
    assert bm.rank_from_ranked_list(ordered, {"TRUE"}, set()) == 3


def test_a_list_with_no_target_is_unknown_rather_than_large():
    assert bm.rank_from_ranked_list(["A", "B"], {"TRUE"}, set()) is None


@pytest.fixture
def tiny():
    """Two items on one payload, a four-entry catalogue, and a homolog table with a self row."""
    items = [bm.Item("c1", "t", "P1", "o1", 100, 100), bm.Item("c2", "t", "P1", "o2", 100, 40)]
    cat = ["P1", "H1", "H2", "D1"]
    homologs = {"P1": [Homolog("P1", 500.0, 1.0, 1.0, 100),
                       Homolog("H1", 300.0, 0.60, 1.0, 100),
                       Homolog("H2", 200.0, 0.30, 1.0, 100)]}
    return items, cat, homologs


def test_the_two_submission_forms_agree_on_the_same_ranking(tiny):
    """Nothing about the metrics may change with the form a method submits in."""
    items, cat, homologs = tiny
    order = ["P1", "H1", "H2", "D1"]
    mat = np.array([[4.0, 3.0, 2.0, 1.0]] * len(items))
    rows_m, _ = bm.score_submission(bm.Submission("matrix", matrix=mat), items, homologs, cat)
    ranked = {it.construct_id: order for it in items}
    rows_r, _ = bm.score_submission(
        bm.Submission("ranked", ranked=ranked, depth_submitted=4), items, homologs, cat)
    for a, b in zip(rows_m, rows_r):
        for k in ("threshold", "stratum", "n", "recall_at_10", "mrr", "mean_rank", "top1"):
            assert a[k] == b[k], k


def test_a_transposed_score_matrix_is_refused(tiny):
    """The one error here that would still produce plausible-looking numbers."""
    items, cat, homologs = tiny
    bad = np.zeros((len(cat), len(items)))
    with pytest.raises(ValueError) as e:
        bm.score_submission(bm.Submission("matrix", matrix=bad), items, homologs, cat)
    assert "one row per item" in str(e.value)


def test_binding_counts_are_per_item_not_per_threshold(tiny):
    """A count that multiplies by the length of the holdout sweep is not a count of anything."""
    items, cat, homologs = tiny
    sub = bm.Submission("ranked", ranked={"c1": ["NOPE", "P1"]}, depth_submitted=2)
    _, bind = bm.score_submission(sub, items, homologs, cat)
    assert bind["missing_from_submission"] == 1, "c2 is absent once, not six times"
    assert bind["unknown_accessions"] == 1, "NOPE counted once, not six times"
    assert bind["items"] == 2


def test_an_item_absent_from_the_submission_is_a_miss_not_a_removal(tiny):
    items, cat, homologs = tiny
    sub = bm.Submission("ranked", ranked={"c1": ["P1"]}, depth_submitted=1)
    rows, _ = bm.score_submission(sub, items, homologs, cat)
    at100 = [r for r in rows if r["threshold"] == 1.0]
    assert sum(r["n"] for r in at100) == 2, "both items are still in the denominator"
    assert sum(r["n_unscorable"] for r in at100) == 1


def test_the_no_holdout_threshold_makes_the_payload_its_own_answer(tiny):
    items, cat, homologs = tiny
    mat = np.array([[4.0, 3.0, 2.0, 1.0]] * len(items))
    rows, _ = bm.score_submission(bm.Submission("matrix", matrix=mat), items, homologs, cat)
    at100 = [r for r in rows if r["threshold"] == 1.0]
    assert at100 and all(r["top1"] == r["n"] for r in at100)
    # At 0.50 the payload's own entry and H1 (0.60) are both held out, so H2 at 0.30 is the answer.
    at50 = [r for r in rows if r["threshold"] == 0.50]
    assert at50 and all(r["mean_rank"] == 1.0 for r in at50)


def test_a_csv_submission_needs_its_three_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("construct_id,accession\nc1,P1\n", encoding="utf-8")
    with pytest.raises(ValueError) as e:
        bm.load_submission(p)
    assert "accession" in str(e.value)


def test_a_csv_submission_is_ordered_by_its_rank_column_not_by_file_order(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("construct_id,rank,accession\nc1,2,SECOND\nc1,1,FIRST\n", encoding="utf-8")
    sub = bm.load_submission(p)
    assert sub.ranked == {"c1": ["FIRST", "SECOND"]}
    assert sub.depth_submitted == 2


# ---- the builder --------------------------------------------------------------------------------

@pytest.fixture
def built(tmp_path):
    """Run `52_build_benchmark.py` over a one-construct tier and return its output directory."""
    payload = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
               "QTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFG")
    filler = "WWWWWWWWWWCCCCCCCCCCHHHHHHHHHHYYYYYYYYYY" * 2
    orf = filler + payload + filler

    ref = tmp_path / "ref.fasta"
    with open(ref, "w", encoding="utf-8") as fh:
        fh.write(f">P1 payload\n{payload}\n")
        fh.write(f">H1 homolog_of:P1\n{payload[:120] + 'A' * 40}\n")
        fh.write(f">D1 decoy\n{'MG' + 'ARNDCQEGHILKMFPSTWYV' * 6}\n")
    hom = tmp_path / "homologs.csv"
    with open(hom, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["payload_accession", "homolog_accession", "score", "identity", "coverage", "length"])
        w.writerow(["P1", "P1", 800.0, 1.0, 1.0, len(payload)])
        w.writerow(["P1", "H1", 400.0, 0.40, 1.0, len(payload)])

    rd = tmp_path / "res"
    rd.mkdir()
    with open(rd / "orfs.faa", "w", encoding="utf-8") as fh:
        fh.write(f">plas1|orf1\n{orf}\n")
    with open(rd / "pilot_summary.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["plasmid_id", "source_file"])
        w.writerow(["plas1", "plas1.gbk"])
    man = tmp_path / "manifest.csv"
    with open(man, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "payload_accession"])
        w.writerow(["plas1.gbk", "P1"])

    build = _load("52_build_benchmark.py")
    build.TIERS = (("hand", str(rd), str(man), "a one-construct tier for the test"),)
    out = tmp_path / "bench"
    rc = build.main(["--out-dir", str(out), "--reference", str(ref), "--homologs", str(hom),
                     "--tiers", "hand"])
    assert rc == 0
    return out, len(orf), len(payload)


def test_the_builder_publishes_f_from_the_payload_bearing_orf(built):
    """f is `oracle_span / len(arms.orf_seq)`, derived by `build_arms` rather than read from a second
    lengths file. One definition in the repository: the 17 Oct error came from having two."""
    out, orf_len, payload_len = built
    rows = list(csv.DictReader(open(out / "items.csv", encoding="utf-8-sig")))
    assert len(rows) == 1
    r = rows[0]
    assert int(r["orf_len"]) == orf_len
    assert int(r["oracle_len"]) == payload_len
    assert float(r["f"]) == pytest.approx(payload_len / orf_len, abs=1e-4)
    assert r["stratum"] == bm.stratum_of(payload_len / orf_len)


def test_the_builder_ships_the_catalogue_as_a_checksum_not_a_copy(built):
    out, _, _ = built
    mani = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert len(mani["catalogue"]["sha256"]) == 64
    assert mani["catalogue"]["entries"] == 3
    assert not (out / "catalogue.fasta").exists()
    assert (out / "catalogue_accessions.txt").read_text(encoding="utf-8").split() == ["P1", "H1", "D1"]


def test_the_builder_records_every_rule_the_scorer_applies(built):
    """The published artifact has to carry its own rules, or a number read off it a month later is
    uninterpretable, which is the whole reason this directory exists."""
    out, _, _ = built
    mani = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    joined = " ".join(mani["metric_rules"])
    assert "no pooled scalar" in joined
    assert "10,000" in joined
    assert "denominator" in joined
    assert mani["depth"] == bm.DEPTH
    assert mani["thresholds"] == list(bm.THRESHOLDS)
    assert "longest ORF" in mani["f_definition"]
    assert mani["files"]["items.csv"]["sha256"]


def test_the_published_queries_are_the_orfs_not_the_payloads(built):
    """A benchmark that handed over the payload span would be handing over the answer."""
    out, orf_len, payload_len = built
    text = (out / "queries.faa").read_text(encoding="utf-8")
    assert text.startswith(">plas1\n")
    seq = "".join(text.splitlines()[1:])
    assert len(seq) == orf_len > payload_len


def test_a_built_benchmark_scores_end_to_end(built):
    """The builder's output is the scorer's input, with nothing in between."""
    out, _, _ = built
    items = bm.load_items(out / "items.csv")
    cat = (out / "catalogue_accessions.txt").read_text(encoding="utf-8").split()
    score = _load("53_score_submission.py")
    homologs = score.load_homologs(out / "homologs.csv")
    sub = bm.Submission("ranked", ranked={items[0].construct_id: ["H1", "P1", "D1"]},
                        depth_submitted=3)
    rows, bind = bm.score_submission(sub, items, homologs, cat)
    assert bind["missing_from_submission"] == 0 and bind["unknown_accessions"] == 0
    at25 = [r for r in rows if r["threshold"] == 0.25]
    # H1 sits at identity 0.40, so at the 0.25 holdout it is held out too and nothing survives.
    assert at25 and all(r["n_unscorable"] == r["n"] for r in at25)
    at50 = [r for r in rows if r["threshold"] == 0.50]
    assert at50 and all(r["top1"] == r["n"] for r in at50), "H1 survives at 0.50 and was ranked first"


def test_the_mean_rank_column_never_touches_the_mrr_column(built, tmp_path, capsys):
    """A format width is a minimum, so a cell too wide for its column does not wrap or raise: it
    eats the gap. `53_score_submission.py` held the mean rank in a constant 22, which fits an
    alignment arm's `1.00` and the empty cell's `no scorable item` and is overrun by an embedding
    arm's `1.00 (0 of 1 at rank >= 10,000)`, so the published table printed the MRR value joined to
    the mean rank as `1.00001.00`. Every number in that row was correct.

    Found 25 Oct by running the release acceptance test from a clean clone, which is also the only
    reason it was ever seen: the defect is invisible unless `--arm embedding` is passed, and the
    embedding baseline is one of the two that ship in `benchmark/v1`.

    This asserts the contract rather than the width, because the width is derived and must stay
    derived: the phrase grows with the stratum's `n` and with `FAILURE_RANK`, so any constant only
    moves the binding point. Two properties, read off the printed table: the mean-rank field opens
    with at least one space, and its content ends on the same right edge as its header label.
    """
    out, _, _ = built
    items = bm.load_items(out / "items.csv")
    sub_path = tmp_path / "sub.csv"
    with open(sub_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["construct_id", "rank", "accession"])
        for it in items:
            w.writerow([it.construct_id, 1, it.payload_accession])

    score = _load("53_score_submission.py")
    for arm in ("alignment", "embedding"):
        assert score.main(["--benchmark", str(out), "--submission", str(sub_path),
                           "--arm", arm]) in (0, None)
        printed = capsys.readouterr().out
        lines = printed.splitlines()
        header = next(ln for ln in lines if ln.lstrip().startswith("holdout"))
        # The mean-rank field begins where the MRR column ends and ends at its own label's right
        # edge. Both are read from the header, so neither is a number this test has to keep current.
        field_start = header.index("MRR") + len("MRR")
        right_edge = header.index("mean rank") + len("mean rank")
        assert right_edge > field_start

        data = [ln for ln in lines if ln[:8].strip().endswith("%")]
        assert len(data) >= 4, f"no data rows to check for the {arm} arm"
        for ln in data:
            cell = ln[field_start:right_edge]
            assert ln[field_start] == " ", (
                f"{arm}: the mean rank is touching the MRR value, {ln[field_start - 8:right_edge]!r}")
            assert cell.strip(), f"{arm}: empty mean-rank cell in {ln!r}"
            assert not cell.endswith(" "), (
                f"{arm}: the mean rank is not on its header's right edge, {cell!r}")

    # The embedding arm is the one that overran, so pin that its phrase is what is being fitted and
    # not some shorter fallback the fix might have reached for instead.
    assert "at rank >= 10,000)" in printed


# ---- the baseline and the cross-check ------------------------------------------------------------

def test_rank_cells_returns_one_record_per_item_and_threshold(tiny):
    """`rank_cells` is split out so a rank can be checked against another route's rank without going
    through the metric layer. An aggregate agreeing is much weaker evidence than every cell agreeing."""
    items, cat, homologs = tiny
    mat = np.array([[4.0, 3.0, 2.0, 1.0]] * len(items))
    records, bind = bm.rank_cells(bm.Submission("matrix", matrix=mat), items, homologs, cat)
    assert len(records) == len(items) * len(bm.THRESHOLDS)
    assert bind["items"] == len(items)
    # and the metric layer is exactly `report` over those records
    rows_direct = bm.report(records)
    rows_scored, _ = bm.score_submission(bm.Submission("matrix", matrix=mat), items, homologs, cat)
    assert rows_direct == rows_scored


def test_the_baseline_refuses_a_partial_source_set(tmp_path, built):
    """A baseline missing a tier is not a baseline, and quietly covering two of three would be worse
    than failing: the row would read as though the third tier agreed."""
    out, _, _ = built
    base = _load("54_baseline_submission.py")
    base.SOURCES = (("hand", str(tmp_path / "nope.csv"), 1.0),)
    assert base.main(["--benchmark", str(out)]) == 2


def _fake_twilight(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["threshold", "plasmid_id", "payload_accession", "arm",
                                           "best_target_rank"])
        w.writeheader()
        w.writerows(rows)


def test_the_baseline_quantifies_the_denominator_difference(tmp_path, built):
    """Same ranks, two rules: the benchmark counts an unscorable item as a miss, the twilight summaries
    drop it. A reader comparing the two needs the size of the gap, not a warning, so it is computed."""
    out, _, _ = built
    item_id = bm.load_items(out / "items.csv")[0].construct_id
    tw = tmp_path / "tw.csv"
    # one scorable cell at rank 1 and one unscorable, at each of the two compared thresholds
    rows = []
    for thr in ("0.25", "0.7"):
        rows.append({"threshold": thr, "plasmid_id": item_id, "payload_accession": "P1",
                     "arm": "whole", "best_target_rank": "1"})
        rows.append({"threshold": thr, "plasmid_id": item_id, "payload_accession": "P1",
                     "arm": "bhs_per_segment", "best_target_rank": ""})
    _fake_twilight(tw, rows)
    base = _load("54_baseline_submission.py")
    base.SOURCES = (("hand", str(tw), 1.0),)
    assert base.main(["--benchmark", str(out)]) == 0
    got = json.loads((out / "baseline_alignment.json").read_text(encoding="utf-8"))
    d = got["denominator_difference"]
    assert d["cells_compared"] >= 1
    assert d["mean_delta"] is not None and d["max_delta"] >= 0
    assert "benchmark" in d["which_is_stricter"]
    assert got["no_run_needed"]["regeneration_cost_seconds"] == 1.0
    assert got["no_run_needed"]["read_only_cost_seconds"] is not None


def test_the_benchmark_denominator_is_the_stricter_of_the_two_rules():
    """The arithmetic behind the gap the baseline reports, pinned directly rather than through the
    script: a cell with two found and two unscorable items reads 0.5 here and 1.0 in the published
    summaries. Both are defensible; only one of them counts an item nobody could retrieve."""
    ranks = [1, 2, None, None]
    bench = bm.metrics(ranks)["recall_at_10"]
    twilight = sum(1 for r in ranks if r is not None and r <= 10) / len(
        [r for r in ranks if r is not None])
    assert bench == 0.5 and twilight == 1.0
    assert twilight > bench, "the benchmark's rule can never be the looser of the two"


def test_the_crosscheck_fails_on_a_single_mismatched_cell(tmp_path, built):
    """There is no tolerance parameter and there should not be: the ranks are integers produced from the
    same scores by the same rule, so one disagreement means they are not the same quantity."""
    pytest.importorskip("parasail")
    out, _, _ = built
    items = bm.load_items(out / "items.csv")
    tw = tmp_path / "tw_bad.csv"
    _fake_twilight(tw, [{"threshold": f"{t}", "plasmid_id": items[0].construct_id,
                         "payload_accession": "P1", "arm": "whole", "best_target_rank": "999"}
                        for t in (1.0, 0.9, 0.7, 0.5, 0.35, 0.25)])
    cross = _load("55_crosscheck_benchmark.py")
    cross.TWILIGHT = {"hand": str(tw)}
    rc = cross.main(["--benchmark", str(out), "--reference", str(tmp_path / "ref.fasta"),
                     "--tier", "hand", "--out", str(tmp_path / "x.json")])
    assert rc == 1, "a mismatch must be a non-zero exit, not a printed note"
    res = json.loads((tmp_path / "x.json").read_text(encoding="utf-8"))
    assert res["verdict"] == "MISMATCHED" and res["mismatched"] >= 1


def test_the_crosscheck_refuses_a_catalogue_whose_order_differs(tmp_path, built):
    """A matrix scored against one accession order cannot be read against another, and the failure would
    be silent: every rank would still be an integer."""
    out, _, _ = built
    shuffled = tmp_path / "shuffled.fasta"
    shuffled.write_text(">D1 decoy\nMGAR\n>P1 payload\nMGAR\n", encoding="utf-8")
    cross = _load("55_crosscheck_benchmark.py")
    cross.TWILIGHT = {"hand": str(tmp_path / "ref.fasta")}  # exists; the order check fires first
    rc = cross.main(["--benchmark", str(out), "--reference", str(shuffled), "--tier", "hand"])
    assert rc == 2


# ---- the release policy: the checksum has to be usable ------------------------------------------

def test_verify_catalogue_accepts_the_catalogue_it_was_built_from(built, tmp_path):
    """Decided: the repository ships the checksum and the release ships the FASTA.
    That is only a policy if the checksum can be used, so verification is a command."""
    out, _, _ = built
    build = _load("52_build_benchmark.py")
    assert build.verify_catalogue(out, tmp_path / "ref.fasta") == 0


def test_verify_catalogue_rejects_a_different_one(built, tmp_path):
    other = tmp_path / "other.fasta"
    other.write_text(">P1 payload\nMGARND\n", encoding="utf-8")
    build = _load("52_build_benchmark.py")
    assert build.verify_catalogue(out_dir := built[0], other) == 1
    assert (out_dir / "MANIFEST.json").is_file(), "verification must not touch the artifact"


def test_verify_catalogue_needs_a_manifest(tmp_path):
    build = _load("52_build_benchmark.py")
    (tmp_path / "f.fasta").write_text(">A\nMG\n", encoding="utf-8")
    assert build.verify_catalogue(tmp_path, tmp_path / "f.fasta") == 2


def test_the_manifest_pins_the_catalogues_own_source(built):
    """The Swiss-Prot release label was never recorded, so the source file's sha256 is what pins it,
    and that is the reason the release ships the FASTA: a rebuild from a later release would change the
    catalogue, hence the residue count, hence the significance floor, and could not be checked against
    the published checksum."""
    out, _, _ = built
    cat = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))["catalogue"]
    assert "release" in cat["distribution"] and "verify-catalogue" in cat["distribution"]
    # `built_by` has to say what produced the catalogue. It used to say it by naming a script
    # that the public tree does not ship, so the assertion follows the wording out.
    assert "exact Smith-Waterman" in cat["built_by"] and "Swiss-Prot" in cat["built_by"]
    assert "source" in cat


# ------------------------------------------------------- what a holdout actually leaves


def test_best_identity_left_is_the_best_surviving_relative_not_the_label():
    ids = [1.0, 0.989, 0.871, 0.502, 0.268]
    assert bm.best_identity_left(ids, 0.90) == 0.871
    assert bm.best_identity_left(ids, 0.70) == 0.502
    assert bm.best_identity_left(ids, 0.35) == 0.268
    assert bm.best_identity_left(ids, 0.25) is None, "nothing left is not a zero"


def test_best_identity_left_is_one_at_the_no_holdout_control():
    """At 1.00 the payload's own entry is the answer, so the value is its self-identity whatever the
    family looks like, which is the sense in which that cell measures string matching."""
    assert bm.best_identity_left([], 1.00) == 1.0
    assert bm.best_identity_left([0.268], 1.00) == 1.0


def test_holdout_phrase_cannot_be_rendered_without_the_quantity():
    """The refusal is structural: the quantity is a required argument, so there is no call that
    produces a bare nominal threshold. Same move as `mean_rank_phrase` for an embedding mean rank."""
    with pytest.raises(TypeError):
        bm.holdout_phrase(0.70)
    assert "0.268" in bm.holdout_phrase(0.70, 0.268)
    assert "0.432" in bm.holdout_phrase(0.70, 0.268), "the gap below the label is part of the phrase"
    assert "unscorable" in bm.holdout_phrase(0.25, None)
    with pytest.raises(ValueError):
        # Above the holdout means the targets and the mask are not one gated file.
        bm.holdout_phrase(0.70, 0.88)


def test_holdout_label_overstated_is_derived_from_the_sweep_not_chosen():
    assert bm.holdout_label_overstated(0.70, 0.268) is True
    assert bm.holdout_label_overstated(0.70, 0.502) is False
    assert bm.holdout_label_overstated(0.90, 0.70) is True, "0.70 is itself a step of the sweep"
    assert bm.holdout_label_overstated(0.25, 0.01) is False, "no step below the bottom one"
    assert bm.holdout_label_overstated(1.00, 1.0) is False, "the control holds nothing out"
    assert bm.holdout_label_overstated(0.70, None) is False


def test_the_shipped_homolog_table_reproduces_the_p09751_finding():
    """the working record on the published benchmark's own data rather than on a fixture, because the finding
    is what the rule exists for: `P09751`'s family has a hole, so the 0.70 holdout hands it a 27%
    relative and its 14 constructs read as a dilution failure when they are a remoteness failure.

    It also fixes 65.4's one arithmetic slip: the payloads in 0.47-0.70 are **22** of 24, not 21. The
    section's other counts stand, `P09751` first at 0.268, `P0DV26` next at 0.340, "the other 23
    between 0.34 and 0.70", and 1 + 1 + 22 is 24 where 1 + 1 + 21 was 23.
    """
    root = Path(__file__).resolve().parents[1]
    hom, items = root / "benchmark/v1/homologs.csv", root / "benchmark/v1/items.csv"
    if not (hom.is_file() and items.is_file()):
        pytest.skip("benchmark/v1 not built")
    from collections import defaultdict

    ids: dict[str, list[float]] = defaultdict(list)
    with open(hom, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            ids[r["payload_accession"]].append(float(r["identity"]))
    with open(items, newline="", encoding="utf-8-sig") as fh:
        accs = sorted({r["payload_accession"] for r in csv.DictReader(fh) if r["tier"] == "synth_real"})
    assert len(accs) == 24
    left = {a: bm.best_identity_left(ids.get(a, []), 0.70) for a in accs}
    assert len(ids["P09751"]) == 16
    assert round(left["P09751"], 4) == 0.2682
    assert round(left["P0DV26"], 4) == 0.3403
    assert sorted(left, key=lambda a: left[a])[:2] == ["P09751", "P0DV26"]
    assert sum(1 for v in left.values() if 0.47 <= v <= 0.70) == 22
    # And the rule flags more than the one payload 65.4 named: three at 0.70, four at 0.90.
    over = {t: sorted(a for a in accs
                      if bm.holdout_label_overstated(t, bm.best_identity_left(ids.get(a, []), t)))
            for t in bm.THRESHOLDS}
    assert over[0.70] == ["B4SI11", "P09751", "P0DV26"]
    assert len(over[0.90]) == 4
    assert over[1.00] == [] and over[0.25] == []


def test_script_10_records_what_the_holdout_left_in_every_row():
    """The column is the deliverable of the rule: without it a reader of the CSV can only re-stratify
    on the label, which is the quantity 65.4 showed mixes regimes."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "10_twilight.py").read_text(
        encoding="utf-8")
    assert "best_identity_left=" in src, "the twilight CSV must carry the column"
    assert "best_identity_left" in src.split("def main")[1].split("holdout_label_overstated")[0]
    assert '"holdout_label_overstated"' in src, "and the summary JSON must carry the flagged payloads"


# ----------------------------------------------- the random-filler tier, and the id collision it brings


def test_the_random_filler_tier_is_available_and_is_not_published_by_default():
    """`synth_random` is the worst case for a mean-pooled embedding arm where `synth_real` is the worst
    case for alignment, so the builder has to be able to assemble it. It is **not**
    in the default tier set, because adding it changes `items.csv` and `queries.faa` and therefore the
    sha256s the published MANIFEST carries, and the reference set is my decision.
    """
    build = _load("52_build_benchmark.py")
    names = [t[0] for t in build.TIERS]
    assert names == ["hand", "corpus", "synth_real", "synth_random"]
    assert build.PUBLISHED_TIERS == ("hand", "corpus", "synth_real")
    assert "synth_random" not in build.PUBLISHED_TIERS
    by_name = {t[0]: t for t in build.TIERS}
    # The two synthetic tiers differ only in the filler, so the thing that distinguishes them is which
    # input directory and manifest they read: never the free-text label (this project's binding-limit register).
    assert by_name["synth_random"][2] == "data/synthetic/synthetic_manifest.csv"
    assert by_name["synth_real"][2] == "data/synthetic_real/synthetic_manifest.csv"
    assert by_name["synth_random"][1] != by_name["synth_real"][1]


def _collide_fixture(tmp_path):
    """Two tiers holding the same construct id with different sequences: `synth_real` against
    `synth_random` in miniature."""
    payload = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
               "QTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFG")
    ref = tmp_path / "ref.fasta"
    with open(ref, "w", encoding="utf-8") as fh:
        fh.write(f">P1 payload\n{payload}\n")
        fh.write(f">D1 decoy\n{'MG' + 'ARNDCQEGHILKMFPSTWYV' * 6}\n")
    hom = tmp_path / "homologs.csv"
    with open(hom, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["payload_accession", "homolog_accession", "score", "identity", "coverage", "length"])
        w.writerow(["P1", "P1", 800.0, 1.0, 1.0, len(payload)])
    man = tmp_path / "manifest.csv"
    with open(man, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "payload_accession"])
        w.writerow(["plas1.gbk", "P1"])
    tiers = []
    for tier, filler in (("synth_real", "WWWWWWWWWWCCCCCCCCCCHHHHHHHHHHYYYYYYYYYY" * 2),
                         ("synth_random", "GGGGGGGGGGSSSSSSSSSSAAAAAAAAAATTTTTTTTTT" * 2)):
        rd = tmp_path / tier
        rd.mkdir()
        with open(rd / "orfs.faa", "w", encoding="utf-8") as fh:
            fh.write(f">plas1|orf1\n{filler + payload + filler}\n")
        with open(rd / "pilot_summary.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["plasmid_id", "source_file"])
            w.writerow(["plas1", "plas1.gbk"])
        tiers.append((tier, str(rd), str(man), f"{tier} in miniature"))
    return ref, hom, tuple(tiers)


def test_the_builder_refuses_two_tiers_that_share_construct_ids(tmp_path):
    """Measured 21 Oct, and the reason this is a refusal: a four-tier build before the check existed
    produced `queries.faa` with 1,102 records and 766 distinct headers, and a baseline with 432 rows:
    the three-tier count, `synth_real` gone. Adding a tier deleted one, silently."""
    build = _load("52_build_benchmark.py")
    ref, hom, tiers = _collide_fixture(tmp_path)
    build.TIERS = tiers
    out = tmp_path / "bench"
    rc = build.main(["--out-dir", str(out), "--reference", str(ref), "--homologs", str(hom),
                     "--tiers", "synth_real", "synth_random"])
    assert rc == 2, "a colliding tier set must not build"
    assert not out.exists(), "and must write nothing at all"


def test_namespaced_ids_make_the_colliding_tiers_addressable(tmp_path):
    build = _load("52_build_benchmark.py")
    ref, hom, tiers = _collide_fixture(tmp_path)
    build.TIERS = tiers
    out = tmp_path / "bench"
    rc = build.main(["--out-dir", str(out), "--reference", str(ref), "--homologs", str(hom),
                     "--tiers", "synth_real", "synth_random", "--namespace-ids"])
    assert rc == 0
    with open(out / "items.csv", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2 and len({r["construct_id"] for r in rows}) == 2
    assert sorted(r["construct_id"] for r in rows) == ["synth_random:plas1", "synth_real:plas1"]
    heads = [l[1:].split()[0] for l in (out / "queries.faa").read_text(encoding="utf-8").splitlines()
             if l.startswith(">")]
    assert heads == ["synth_real:plas1", "synth_random:plas1"], "the FASTA must stay a 1:1 key"
    # Joined per record, since the writer wraps at 60 columns.
    blocks = (out / "queries.faa").read_text(encoding="utf-8").split(">")[1:]
    seqs = ["".join(b.splitlines()[1:]) for b in blocks]
    assert len(seqs) == 2 and len(set(seqs)) == 2,         "both sequences must survive, which one duplicate id would not allow"
    mf = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert mf["construct_ids_namespaced"] is True, "a scorer must not have to guess the convention"
    assert mf["n_items"] == 2


def test_a_non_colliding_tier_set_is_never_namespaced(tmp_path):
    """The flag is inert where it is not needed, so the published three-tier build cannot acquire
    namespaced ids by someone passing it out of habit."""
    build = _load("52_build_benchmark.py")
    ref, hom, tiers = _collide_fixture(tmp_path)
    build.TIERS = tiers
    out = tmp_path / "bench"
    assert build.main(["--out-dir", str(out), "--reference", str(ref), "--homologs", str(hom),
                       "--tiers", "synth_real", "--namespace-ids"]) == 0
    mf = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert mf["construct_ids_namespaced"] is False
    with open(out / "items.csv", newline="", encoding="utf-8-sig") as fh:
        assert [r["construct_id"] for r in csv.DictReader(fh)] == ["plas1"]

# ---- A7: provenance ------------------------------------------------------------------------------

def test_orf_coordinates_unpacks_the_composite_id():
    build = _load("52_build_benchmark.py")
    got = build.orf_coordinates("addgene_108653_pCSIIbsr|orf011|+|3790..6906")
    assert got == dict(plasmid_id="addgene_108653_pCSIIbsr", orf_strand="+",
                       orf_start=3790, orf_end=6906, orf_wraps_origin=0)


def test_orf_coordinates_flags_an_origin_crossing_orf():
    """The whole reason the flag exists: `end - start + 1` is negative for these.

    These plasmids are circular and the ORF caller is a circular six-frame scanner, so an ORF can begin
    near the end of the sequence and finish past position 1. On the published build that is 5 items,
    and the naive length for `corpus:ncbi_AF218363_1` is -8,001 nt. Nothing raises; a reader simply
    gets a wrong number. `orf_len` is right for all of them because it comes from the translated
    sequence, so the flag's job is to stop a second length being derived.
    """
    got = build_coords("ncbi_AF218363_1|orf003|-|8833..831")
    assert got["orf_wraps_origin"] == 1
    assert got["orf_end"] - got["orf_start"] + 1 < 0, "the naive length really is negative"


def build_coords(orf_id):
    return _load("52_build_benchmark.py").orf_coordinates(orf_id)


@pytest.mark.skipif(not (ROOT / "benchmark" / "v1" / "items.csv").is_file(),
                    reason="needs the published benchmark")
def test_the_published_items_carry_their_provenance():
    """Every column A7 added is present, and the deposited tiers all name a deposit."""
    with open(ROOT / "benchmark" / "v1" / "items.csv", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    added = ["source_file", "source_id", "source_url", "filler_source",
             "plasmid_id", "orf_strand", "orf_start", "orf_end", "orf_wraps_origin"]
    for col in added:
        assert col in rows[0], col
    deposited = [r for r in rows if r["tier"] in ("hand", "corpus")]
    assert deposited, "no deposited tier in this build"
    assert all(r["source_file"] for r in deposited), "a deposited item with no source file"
    assert all(r["source_id"] for r in deposited), "a deposited item with no source id"
    # the generated tiers have no deposit to name, and an empty value there is the answer, not a gap
    generated = [r for r in rows if r["tier"].startswith("synth")]
    if generated:
        assert not any(r["source_id"] for r in generated), "a generated item invented a source"


@pytest.mark.skipif(not (ROOT / "benchmark" / "v1" / "items.csv").is_file(),
                    reason="needs the published benchmark")
def test_the_wrap_flag_agrees_with_the_coordinates_on_every_published_item():
    """The flag is derived, so it can only be wrong by being stale. This is what makes it trustworthy."""
    with open(ROOT / "benchmark" / "v1" / "items.csv", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        assert int(r["orf_wraps_origin"]) == int(int(r["orf_end"]) < int(r["orf_start"])), \
            r["construct_id"]
        # and the coordinates must agree with the composite key they were unpacked from
        _plasmid, _idx, strand, span = r["orf_id"].split("|")
        a, b = span.split("..")
        assert (strand, int(a), int(b)) == (r["orf_strand"], int(r["orf_start"]), int(r["orf_end"]))


@pytest.mark.skipif(not (ROOT / "benchmark" / "v1" / "items.csv").is_file(),
                    reason="needs the published benchmark")
def test_orf_len_is_not_derivable_from_the_coordinates_where_the_orf_wraps():
    """Pins the reason the flag is published rather than left implicit.

    For a wrapping ORF the coordinate-derived length is negative, so a reader who computes it instead
    of reading `orf_len` gets nonsense rather than an error. If this ever stops being true, because
    the caller starts emitting unwrapped coordinates, say, the flag has changed meaning and the
    README paragraph about it is wrong.
    """
    with open(ROOT / "benchmark" / "v1" / "items.csv", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    wrapped = [r for r in rows if int(r["orf_wraps_origin"])]
    assert wrapped, "no wrapping ORF in this build; the README's count of 5 needs rechecking"
    for r in wrapped:
        naive_nt = int(r["orf_end"]) - int(r["orf_start"]) + 1
        assert naive_nt < 0
        assert int(r["orf_len"]) > 0, "orf_len itself must stay correct"

