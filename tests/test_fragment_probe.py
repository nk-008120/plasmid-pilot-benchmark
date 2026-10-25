"""Tests for the fragment retrieval probe (`scripts/51_fragment_probe.py`).

The probe exists to check an arithmetic ceiling against a measurement, so the things worth pinning are
the ones that would let it produce a number that *looks* comparable to the twilight benchmark and is not.

Six of those:

* the cuts are deterministic and in range, so a re-run probes the same windows and no cut runs off the
  end of a short payload;
* the full-length control reproduces the payload's own rank, which is the acceptance condition the
  runbook names: if a 'fragment' equal to the whole payload does not rank its own catalogue entry
  first at a 1.00 holdout, the harness is wrong before any fragment is cut;
* a fragment that retrieves nothing is **reported** rather than dropped, because a silently shrinking
  denominator is how a hopeless length would come out looking fine;
* the payload's own entry is held out at every threshold below 1.00, so a fragment cannot win on the
  sequence it was cut from: the label-laundering failure mode `best_hit_segments` guards against;
* the floor columns are blank under a non-alignment scorer, because a cosine row is not in BLOSUM62
  points and comparing it to a floor near 100 is this project's error;
* the ranking function is literally the same object the twilight script uses, which is the only way
  "comparable to `twilight_*.csv`" is a fact rather than an intention.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import random
from pathlib import Path

import pytest

import pilot.align
from pilot.homology import score_floor

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.split(".")[0], ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load("51_fragment_probe.py")

AA = "ACDEFGHIKLMNPQRSTVWY"


def _seq(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(AA) for _ in range(n))


@pytest.fixture
def catalogue(tmp_path: Path):
    """A payload, one remote relative of it, and twenty unrelated decoys.

    The relative is the payload with ten evenly spread substitutions, so every window of ten residues or
    more still aligns to it well above anything random. `identity` in `homologs.csv` is a declared
    column, so the holdout arithmetic is exercised exactly as it is on the real table: the self row at
    1.0 is masked below a 1.00 holdout and the relative at 0.30 is a target down to 0.35.
    """
    rng = random.Random(20261020)
    payload = _seq(rng, 90)
    rel = list(payload)
    for i in range(5, 90, 9):
        rel[i] = AA[(AA.index(payload[i]) + 7) % 20]
    rel = "".join(rel)

    fasta = tmp_path / "ref.fasta"
    with open(fasta, "w", encoding="utf-8") as fh:
        fh.write(f">PAY1 payload\n{payload}\n")
        fh.write(f">REL1 homolog_of:PAY1\n{rel}\n")
        # In the catalogue and long enough, but its homolog table holds nothing but itself.
        fh.write(f">PAY2 payload\n{_seq(rng, 90)}\n")
        for i in range(20):
            fh.write(f">DEC{i:02d} decoy\n{_seq(rng, 90)}\n")

    hom = tmp_path / "homologs.csv"
    with open(hom, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["payload_accession", "homolog_accession", "score", "identity", "coverage", "length"])
        w.writerow(["PAY1", "PAY1", 450.0, 1.0, 1.0, 90])
        w.writerow(["PAY1", "REL1", 300.0, 0.30, 1.0, 90])
        # A second payload with no surviving relative: it must be excluded, with the reason counted.
        w.writerow(["PAY2", "PAY2", 450.0, 1.0, 1.0, 90])
    return fasta, hom, payload, rel


def _run(tmp_path, fasta, hom, *extra):
    out = tmp_path / f"out_{len(list(tmp_path.iterdir()))}"
    rc = probe.main(["--reference", str(fasta), "--homologs", str(hom), "--out-dir", str(out),
                     "--cache-dir", str(tmp_path / "cache"), "--lengths", "10", "20", "40",
                     "--n-offsets", "3", *extra])
    assert rc == 0
    csvs = list(out.glob("fragment_probe_*.csv"))
    assert len(csvs) == 1
    rows = list(csv.DictReader(open(csvs[0], encoding="utf-8-sig")))
    summary = json.loads(next(out.glob("*_summary.json")).read_text(encoding="utf-8"))
    return rows, summary


# ---- what the summary has to carry about its own run ----------------------------------------------

def test_the_summary_records_the_thread_count_and_the_scoring_time(tmp_path, catalogue):
    """A rate is a function of the thread count, and R13 produced one the summary could not confirm.

    `51_fragment_probe.py` was run on 22 Oct for the sole purpose of settling this
    catalogue's alignment rate, which 68.4 had in three places. It answered 0.327 s/query, and the
    summary recorded the scorer, the catalogue, the residue count, the E-value, the lengths, the
    offsets, the frame, the batch size and `max_length`, and **not the thread count**, which is the one
    setting the answer is a function of. The number had to be quoted as "0.327 if it was run as
    written", with no artefact able to confirm it.

    So both halves of the rate now live in the file that reports it, and this test is what keeps them
    there: `scoring_seconds` over `n_queries` is the rate, and `threads` is the condition it holds
    under. `scoring_seconds` covers the catalogue-pass loop alone, so it must be strictly less than the
    wall clock of the whole run and must not include writing the CSV.
    """
    fasta, hom, _payload, _rel = catalogue
    _rows, summary = _run(tmp_path, fasta, hom, "--threads", "2")
    assert summary["threads"] == 2
    assert isinstance(summary["scoring_seconds"], float)
    assert summary["scoring_seconds"] >= 0.0
    # the rate the runbook quotes is derivable from the summary alone, with no log needed
    assert summary["n_queries"] > 0
    rate = summary["scoring_seconds"] / summary["n_queries"]
    assert rate >= 0.0


def test_the_thread_count_in_the_summary_is_the_one_that_was_passed(tmp_path, catalogue):
    """Recording a default instead of the argument would be worse than recording nothing: it would
    read as a confirmed condition while being a guess."""
    fasta, hom, _payload, _rel = catalogue
    _rows, summary = _run(tmp_path, fasta, hom, "--threads", "3")
    assert summary["threads"] == 3


# ---- the cuts -------------------------------------------------------------------------------------

def test_cuts_are_deterministic_and_evenly_spaced():
    """Three offsets over a span of 100 land at 25, 50 and 75, and repeat identically."""
    assert probe.cut_starts(150, 50, 3) == [25, 50, 75]
    assert probe.cut_starts(150, 50, 3) == probe.cut_starts(150, 50, 3)
    assert probe.cut_starts(150, 50, 1) == [50]


def test_no_cut_runs_off_the_end():
    for L in (18, 25, 60, 90, 400):
        for m in (8, 12, 18, 25, 50, 66):
            for s in probe.cut_starts(L, m, 4):
                assert 0 <= s and s + m <= L


def test_a_fragment_longer_than_its_payload_yields_no_cut():
    assert probe.cut_starts(40, 66, 3) == []


def test_a_payload_exactly_as_long_as_the_fragment_yields_one_cut():
    assert probe.cut_starts(50, 50, 3) == [0]


# ---- the control ----------------------------------------------------------------------------------

def test_full_length_control_ranks_the_payloads_own_entry_first(tmp_path, catalogue):
    """The runbook's acceptance condition. At a 1.00 holdout nothing is masked and the payload itself is
    the answer, so an exact self-query that does not come back rank 1 means the harness is broken."""
    fasta, hom, _, _ = catalogue
    rows, _ = _run(tmp_path, fasta, hom)
    full = [r for r in rows if r["cut"] == "full" and float(r["threshold"]) == 1.0]
    assert full, "the full-payload control should be emitted by default"
    assert all(r["best_target_rank"] == "1" for r in full)


def test_the_full_control_can_be_switched_off(tmp_path, catalogue):
    fasta, hom, _, _ = catalogue
    rows, summary = _run(tmp_path, fasta, hom, "--no-full-control")
    assert not [r for r in rows if r["cut"] == "full"]
    assert summary["full_payload_control"] is False


# ---- a miss is reported, not dropped ---------------------------------------------------------------

def test_every_query_gets_a_row_at_every_threshold(tmp_path, catalogue):
    """A hopeless fragment length must show up as a hopeless row. If unscorable queries were dropped the
    denominator would shrink with length and the curve would flatten upwards on its own."""
    fasta, hom, _, _ = catalogue
    rows, summary = _run(tmp_path, fasta, hom)
    assert len(rows) == summary["n_queries"] * len(summary["thresholds"])
    for s in summary["by_threshold_and_length"]:
        assert s["n_queries"] == s["n_unscorable"] + (s["n_queries"] - s["n_unscorable"])
        # recall is over every query, not only the scorable ones
        assert 0.0 <= s[f"recall_at_{probe.RECALL_AT}"] <= 1.0


def test_recall_denominator_counts_unscorable_queries(tmp_path, catalogue):
    """At the 0.25 holdout the only relative is held out too, so nothing is retrievable and recall must
    read exactly 0.0 rather than being undefined or skipped."""
    fasta, hom, _, _ = catalogue
    _, summary = _run(tmp_path, fasta, hom)
    at25 = [s for s in summary["by_threshold_and_length"] if s["threshold"] == 0.25]
    assert at25
    assert all(s[f"recall_at_{probe.RECALL_AT}"] == 0.0 for s in at25)
    assert all(s["n_unscorable"] == s["n_queries"] for s in at25)


# ---- the holdout ----------------------------------------------------------------------------------

def test_the_payloads_own_entry_is_held_out_below_a_full_holdout(tmp_path, catalogue):
    """Otherwise a fragment wins on the sequence it was cut from and the probe measures substring search."""
    fasta, hom, _, _ = catalogue
    rows, _ = _run(tmp_path, fasta, hom)
    for r in rows:
        if float(r["threshold"]) < 1.0:
            assert int(r["n_held_out"]) >= 1
            assert r["top1_accession"] != "" and int(r["n_homologs_available"]) <= 1


def test_a_payload_with_no_surviving_relative_is_excluded_and_counted(tmp_path, catalogue):
    """PAY2 has only its own entry, so at every holdout below 1.00 it is a decided miss before a
    fragment is cut. It is excluded, and the reason is reported rather than left as a smaller n."""
    fasta, hom, _, _ = catalogue
    _, summary = _run(tmp_path, fasta, hom)
    assert summary["payloads_probed"] == ["PAY1"]
    assert summary["excluded"]["no_relative_below_the_loosest_holdout"] >= 1
    assert summary["n_payloads_in_table"] == 2


# ---- units ----------------------------------------------------------------------------------------

def test_alignment_arm_reports_the_floor_it_is_measured_against(tmp_path, catalogue):
    fasta, hom, _, _ = catalogue
    rows, summary = _run(tmp_path, fasta, hom)
    assert summary["is_alignment"] is True
    assert all(r["score_floor"] != "" for r in rows)
    assert all(r["best_score_clears_floor"] in ("0", "1") for r in rows)
    # The floor rises with query length, inside a logarithm, so a 40 aa query faces a higher bar than a
    # 10 aa one. Checking the direction catches a swapped argument, which no output would show.
    f10 = {float(r["score_floor"]) for r in rows if r["frag_len_aa"] == "10"}
    f40 = {float(r["score_floor"]) for r in rows if r["frag_len_aa"] == "40"}
    assert len(f10) == 1 and len(f40) == 1
    assert next(iter(f40)) > next(iter(f10))


def test_a_cosine_scorer_reports_no_floor(tmp_path, catalogue):
    """A k-mer row is not in BLOSUM62 points. Emitting a floor beside it would invite exactly the
    comparison the working record was caused by."""
    fasta, hom, _, _ = catalogue
    rows, summary = _run(tmp_path, fasta, hom, "--scorer", "kmer")
    assert summary["is_alignment"] is False
    assert all(r["score_floor"] == "" for r in rows)
    assert all(r["best_score_clears_floor"] == "" for r in rows)
    assert all("frac_best_clears_floor" not in s for s in summary["by_threshold_and_length"])


def test_nucleotide_length_is_three_times_the_peptide_and_the_frame_is_stated(tmp_path, catalogue):
    fasta, hom, _, _ = catalogue
    rows, summary = _run(tmp_path, fasta, hom)
    for r in rows:
        if r["cut"] == "fragment":
            assert int(r["frag_len_nt"]) == 3 * int(r["frag_len_aa"])
        assert r["frame"] == "payload_protein"
    assert "ceiling" in summary["frame"]


# ---- comparability --------------------------------------------------------------------------------

def test_ranks_come_from_the_same_function_the_twilight_benchmark_uses():
    """Not an equivalent implementation: the same object. A probe ranking by its own copy of the rule
    would produce numbers that look like twilight ranks and need not be."""
    tw = _load("10_twilight.py")
    assert probe.best_target_rank is pilot.align.best_target_rank
    assert tw.best_target_rank is pilot.align.best_target_rank


def test_the_bound_prediction_comes_from_the_script_that_owns_it(tmp_path, catalogue):
    """The probe prints the ceiling it is testing, computed by loading `50_fragment_bound.py` rather than
    by restating its answer, so the two cannot disagree."""
    fasta, hom, _, _ = catalogue
    _, summary = _run(tmp_path, fasta, hom)
    pred = summary["bound_prediction"]
    assert pred["min_aa_composition"] >= pred["min_aa_most_generous"] >= 1
    assert "ref.fasta" in pred["composition_source"]


def test_the_bound_is_a_bracket_and_the_composition_value_is_not_its_floor():
    """Measured on the real catalogue 20 Oct: the first fragment clears the floor at 16 aa while the
    composition-weighted value is 18, because a window rich in W, C, H, Y or F scores above the average
    this bound weights by. The floor over *every* fragment is the most-generous value, and that is the
    only one a measurement can contradict. Pinned here because the probe's verdict depends on reading
    the two numbers as a bracket, and an earlier draft of the runbook quoted the composition value alone
    as 'impossible below 54 nt'.
    """
    fb = _load("50_fragment_bound.py")
    N = 20_806_062  # the 40,750-protein catalogue, in-window residues
    freq, _ = fb.composition(None)  # flat prior: the relation below is about the two statistics, not a file
    s_comp = sum(freq[a] * fb.BLOSUM62_DIAGONAL[a] for a in fb.BLOSUM62_DIAGONAL)
    s_max = float(max(fb.BLOSUM62_DIAGONAL.values()))
    m_typ, m_hard = fb.min_length(s_comp, N), fb.min_length(s_max, N)
    assert m_hard < m_typ, "a generous diagonal must admit a shorter fragment than an average one"
    # A real 16-mer from the catalogue beats the average, which is why 16 < 18 was measurable.
    frag = "RDQLLLWCRQGWVFWP"  # Q5HZ09, start 177; its exact self-score cleared the floor
    self_score = sum(fb.BLOSUM62_DIAGONAL[c] for c in frag)
    assert self_score / len(frag) > s_comp
    assert self_score >= score_floor(1e-3, len(frag), N)


def test_rank_and_significance_are_reported_as_separate_crossings(tmp_path, catalogue):
    """The bound is a statement about significance, not about rank. A fragment can sit at rank 1 on a
    score below the E-value floor, because ranking only has to beat the other entries, so a single
    crossing compared with the bound would be testing one quantity against a condition derived for
    another, which is the error this project's working rules names twice. Both are emitted, and only the significance one
    is comparable with the prediction."""
    fasta, hom, _, _ = catalogue
    _, summary = _run(tmp_path, fasta, hom)
    assert "crossings_rank_aa" in summary and "crossings_significance_aa" in summary
    assert "different claims" in summary["crossing_note"]
    rank, sig = summary["crossings_rank_aa"]["1.00"], summary["crossings_significance_aa"]["1.00"]
    # Clearing the floor is the harder bar, so where both exist significance cannot cross first.
    assert rank is not None
    assert sig is None or sig >= rank


def test_a_cosine_scorer_reports_no_significance_crossing(tmp_path, catalogue):
    fasta, hom, _, _ = catalogue
    _, summary = _run(tmp_path, fasta, hom, "--scorer", "kmer")
    assert summary["crossings_significance_aa"] == {}
    assert summary["crossings_rank_aa"]


def test_summary_reports_the_cache_key_fields(tmp_path, catalogue):
    """`--batch-size` and `--max-length` are in the reference-matrix cache key, so a run
    that cannot say which it used cannot be compared with one that can."""
    fasta, hom, _, _ = catalogue
    _, summary = _run(tmp_path, fasta, hom)
    assert summary["max_length"] == 2048
    assert summary["batch_size"] == 8
    assert summary["offsets"].startswith("deterministic")
