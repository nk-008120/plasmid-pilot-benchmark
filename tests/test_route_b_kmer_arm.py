"""Guards for `scripts/49_route_b_kmer_arm.py`, the CPU-feasible mean-pooled arm on route B.

Three things here are load-bearing and none of them would announce a failure in the output:

* **The metric order.** this project's working rules fixes `recall@10` and MRR as primary for an *embedding* arm and
  allows mean rank only beside its count of rank >= 10,000 failures.
  A report that silently reverted to mean-first would still look like a report.
* **The binding counts.** `length_binding` exists because the k-mer embedder's limits are different
  limits from ESM-2's: no `max_length` at all, but a silent drop of every non-standard character.
  A count of 0 is a measurement here, so the test pins the 0 as well as the non-zero.
* **The pre-registered census.** 53.7 predicted *more* k = 1 falsifiers under mean-pooling. A census
  that miscounted ties as wins would turn a confirmation into an artefact, so the three-way split is
  asserted on hand-built rows where the answer is obvious by inspection.
"""
import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rb_kmer", ROOT / "scripts/49_route_b_kmer_arm.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

_strata_spec = importlib.util.spec_from_file_location("rb_strata",
                                                      ROOT / "scripts/45_route_b_strata.py")
strata_mod = importlib.util.module_from_spec(_strata_spec)
_strata_spec.loader.exec_module(strata_mod)


# ------------------------------------------------------------------ the stratification is the project's

def test_strata_and_arms_are_the_same_objects_script_45_reports():
    """Two scripts reporting the same tier must not drift into two definitions of a stratum or of an
    arm. 30.4 froze what `auto` and `bhs` mean; this is the cheap guard that they stay frozen."""
    assert mod.F_STRATA == strata_mod.F_STRATA == (0.50, 0.70, 0.95)
    assert mod.STRATUM_NAMES == strata_mod.STRATUM_NAMES
    assert mod.REPORT_ARMS == strata_mod.REPORT_ARMS
    for f in (0.0, 0.49, 0.5, 0.69, 0.7, 0.94, 0.95, 1.0):
        assert mod.stratum(f) == strata_mod.stratum(f)


def test_recall_depth_is_the_depth_i_fixed():
    """depth 10, a stated assumption about review practice and not a fitted value."""
    assert mod.RECALL_DEPTH == 10


def test_the_plasmid_id_maps_back_to_the_manifests_source_id():
    assert mod.source_id_of("addgene-plasmid-176139-sequence-353012") == "addgene_176139"
    assert mod.source_id_of("something_odd") == "something_odd"


# ------------------------------------------------------------------ 32.2's metric order

def test_an_embedding_arms_metrics_lead_with_recall_and_mrr():
    """The key order is the print order, so it is the thing that enforces 32.2 in practice."""
    keys = list(mod.embedding_metrics([1, 2, 3]))
    assert keys[:3] == ["n", "recall_at_10", "mrr"]
    assert keys.index("recall_at_10") < keys.index("mean_rank")
    assert keys.index("mrr") < keys.index("mean_rank")


def test_a_mean_rank_never_appears_without_its_failure_count():
    """The bimodal case 32.2 is about: median 1, mean in the thousands, and the reason is two rows."""
    got = mod.embedding_metrics([1, 1, 1, 1, 25000, 29699])
    assert got["median_rank"] == 1
    assert got["n_ge_10000"] == 2
    assert got["mean_rank"] > 9000
    assert got["recall_at_10"] == round(4 / 6, 3)
    assert "n_ge_10000" in got  # the pairing 32.2 requires, pinned so a refactor cannot drop it


def test_macro_averages_over_payloads_and_not_over_constructs():
    """27.4: 46 pairs carry 19 payloads, one of them in four backbones, so a construct average
    silently weights the repeated cassette four times."""
    by_payload = {"P1": [1, 1, 1, 1], "P2": [100]}
    got = mod.macro(by_payload)
    assert got["n_payloads"] == 2
    assert got["mean_rank"] == 50.5          # (1 + 100) / 2, not (1+1+1+1+100)/5 = 20.8
    assert got["recall_at_10"] == 0.5


# ------------------------------------------------------------------ the binding counts

def test_the_kmer_embedder_has_no_length_cap_and_the_zero_is_reported():
    """this project's table carries two ESM-2 `max_length` rows precisely because they bound on the
    model rather than on the search. The k-mer path has no such cap, and 'no cap' has to be printed
    as a measured 0 rather than left out: a limit nobody reports is the whole defect class."""
    got = mod.length_binding(["A" * 10, "A" * 5000])
    assert got["embedder_max_length"] is None
    assert got["rows_truncated_by_a_length_cap"] == 0
    assert got["rows"] == 2
    assert got["residues"] == 5010
    assert got["max_len"] == 5000


def test_non_standard_characters_are_counted_because_the_embedder_drops_them_silently():
    """`KmerEmbedder.embed` filters to the 20-letter alphabet with no warning, so an X-rich entry is
    embedded as a shorter sequence than it is. Small, invisible, and biting on a minority: the
    signature this project's binding-limit register is built around."""
    got = mod.length_binding(["ACDEFGHIKL", "ACDXXEFGHI", "ACDEFGHIKU"])
    assert got["rows_with_non_standard_characters"] == 2
    assert got["characters_dropped"] == 3
    assert got["rows_shorter_than_k"] == 0


def test_a_sequence_shorter_than_k_is_counted_as_an_all_zero_row():
    got = mod.length_binding(["AC", "ACD"], k=3)
    assert got["rows_shorter_than_k"] == 1


# ------------------------------------------------------------------ the extrapolation

def test_two_agreeing_rates_license_a_linear_projection_from_the_slower_one():
    """this project's closing paragraph: the 10x-wrong estimate came from one probe and no check that
    the cost was linear. Two points, and the projection is taken from the slowest so it is an upper
    bound rather than a hope."""
    points = [{"residues": 1000, "seconds": 1.0, "us_per_residue": 1000.0},
              {"residues": 4000, "seconds": 4.1, "us_per_residue": 1025.0}]
    got = mod.extrapolate(points, 10_000)
    assert got["linear_in_residues"] is True
    assert got["rate_spread"] == 1.025
    # the SLOWER rate (1025 us x 10,000 residues = 10.25 s), not the mean and not the faster; the
    # projection is rounded to a tenth of a second because nothing downstream can use more than that
    assert got["projected_seconds"] == round(10.25, 1)


def test_a_rate_that_moves_with_n_is_flagged_rather_than_projected_through():
    points = [{"residues": 1000, "seconds": 1.0, "us_per_residue": 1000.0},
              {"residues": 4000, "seconds": 8.0, "us_per_residue": 2000.0}]
    got = mod.extrapolate(points, 10_000)
    assert got["linear_in_residues"] is False
    assert got["rate_spread"] == 2.0


# ------------------------------------------------------------------ 53.7's pre-registered census

def _row(pid, k, whole, per_seg):
    return {"plasmid_id": pid, "k_bhs": k, "rank_whole": whole, "rank_per_segment": per_seg}


def test_the_k1_census_splits_three_ways_and_names_the_falsifiers():
    """A k = 1 falsifier is per_segment STRICTLY better than whole (53.1's counting rule). A tie is
    not a falsifier, which is the distinction that makes 53.1 a refutation of six rather than of 264."""
    rows = [_row("tie", 1, 5, 5), _row("whole_wins", 1, 2, 66),
            _row("falsifier", 1, 66, 1), _row("k2", 2, 44, 23)]
    got = mod.k1_census(rows)
    assert got["k=1"] == {"n": 3, "tie": 1, "whole_wins": 1, "per_segment_wins": 1,
                          "falsifiers": ["falsifier"]}
    assert got["k>=2"]["n"] == 1
    assert got["k>=2"]["per_segment_wins"] == 1
    assert got["k>=2"]["whole_wins"] == 0


def test_the_census_counts_k_ge_2_losses_because_that_is_53_2s_standing_claim():
    """53.2: 225 constructs, three tiers, two catalogues, zero constructs where k >= 2 and `whole`
    still won. If that ever breaks it is the louder result, so the column is counted, not assumed."""
    got = mod.k1_census([_row("a", 3, 1, 900)])
    assert got["k>=2"]["whole_wins"] == 1


def test_paired_rows_keeps_only_pairs_with_both_arms_scored():
    """A construct the embedding arm could not score at all must not enter the census as a tie."""
    rank_rows = [
        {"threshold": "0.25", "plasmid_id": "p1", "payload_accession": "P1", "arm": "whole",
         "best_target_rank": "7"},
        {"threshold": "0.25", "plasmid_id": "p1", "payload_accession": "P1",
         "arm": "bhs_per_segment", "best_target_rank": "1"},
        {"threshold": "0.25", "plasmid_id": "p2", "payload_accession": "P2", "arm": "whole",
         "best_target_rank": "3"},
        {"threshold": "0.25", "plasmid_id": "p2", "payload_accession": "P2",
         "arm": "bhs_per_segment", "best_target_rank": ""},
        {"threshold": "0.50", "plasmid_id": "p1", "payload_accession": "P1", "arm": "whole",
         "best_target_rank": "1"},
    ]
    got = mod.paired_rows(rank_rows, "0.25", {("p1", "P1"): 1, ("p2", "P2"): 2})
    assert [r["plasmid_id"] for r in got] == ["p1"]
    assert got[0] == {"plasmid_id": "p1", "payload_accession": "P1", "k_bhs": 1,
                      "rank_whole": 7, "rank_per_segment": 1}


def test_paired_rows_keys_on_the_pair_so_a_second_pass_cannot_overwrite_the_first():
    """The bug this test exists for, found by the segment-count cross-check and not by inspection.
    A manifest carries one payload per file, so route B's three POLB deposits are scored twice under
    the SAME `plasmid_id`: P06746 in the primary pass, P18887 in the second. Keyed on the construct
    alone, the second pass overwrote the first and the two pairs' segment counts summed to 6; keyed
    on the pair, both survive as themselves. This is 50.5's defect: a choice between two inputs made
    where no output showed it."""
    rank_rows = []
    for acc, whole, per_seg in (("P06746", "16", "132"), ("P18887", "30", "1")):
        rank_rows += [
            {"threshold": "0.25", "plasmid_id": "polb", "payload_accession": acc,
             "arm": "whole", "best_target_rank": whole},
            {"threshold": "0.25", "plasmid_id": "polb", "payload_accession": acc,
             "arm": "bhs_per_segment", "best_target_rank": per_seg},
        ]
    got = mod.paired_rows(rank_rows, "0.25", {("polb", "P06746"): 3, ("polb", "P18887"): 3})
    assert len(got) == 2
    assert {r["payload_accession"]: r["rank_per_segment"] for r in got} == {"P06746": 132,
                                                                           "P18887": 1}


def test_paired_rows_drops_a_pair_with_no_known_segment_count():
    rank_rows = [
        {"threshold": "0.25", "plasmid_id": "p1", "payload_accession": "P1", "arm": "whole",
         "best_target_rank": "7"},
        {"threshold": "0.25", "plasmid_id": "p1", "payload_accession": "P1",
         "arm": "bhs_per_segment", "best_target_rank": "1"},
    ]
    assert mod.paired_rows(rank_rows, "0.25", {}) == []


def _spans(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["plasmid_id", "payload_accession", "emitted"])
        w.writeheader()
        w.writerows(rows)
    return path


def test_count_segments_counts_emitted_spans_only(tmp_path):
    """`k_bhs` is the number of segments the segmenter EMITTED, not the number of spans it traced:
    53.6's `best_hit_emitted` exists because the two differ. Counting traced spans would inflate k
    and move constructs out of the k = 1 cell the prediction is about."""
    p = _spans(tmp_path / "bhs_spans.csv",
               [{"plasmid_id": "p1", "payload_accession": "P1", "emitted": "1"},
                {"plasmid_id": "p1", "payload_accession": "P1", "emitted": "0"},
                {"plasmid_id": "p2", "payload_accession": "P2", "emitted": "1"},
                {"plasmid_id": "p2", "payload_accession": "P2", "emitted": "1"}])
    assert mod.count_segments([p]) == {("p1", "P1"): 1, ("p2", "P2"): 2}


def test_count_segments_does_not_sum_the_two_passes_of_one_construct(tmp_path):
    """The same `plasmid_id` appears in both passes with different payloads. Summed, those three
    constructs read k = 6 and leave the k = 1 and k >= 2 cells they belong in."""
    a = _spans(tmp_path / "primary.csv",
               [{"plasmid_id": "polb", "payload_accession": "P06746", "emitted": "1"},
                {"plasmid_id": "polb", "payload_accession": "P06746", "emitted": "1"},
                {"plasmid_id": "polb", "payload_accession": "P06746", "emitted": "1"}])
    b = _spans(tmp_path / "second.csv",
               [{"plasmid_id": "polb", "payload_accession": "P18887", "emitted": "1"},
                {"plasmid_id": "polb", "payload_accession": "P18887", "emitted": "1"},
                {"plasmid_id": "polb", "payload_accession": "P18887", "emitted": "1"}])
    assert mod.count_segments([a, b]) == {("polb", "P06746"): 3, ("polb", "P18887"): 3}


def test_count_segments_records_a_pair_whose_every_span_was_rejected(tmp_path):
    """The fallback case: nothing cleared the floor, so k = 0. It must appear as 0 rather than be
    absent, or a construct that fell back to the whole ORF would vanish from the census."""
    p = _spans(tmp_path / "bhs_spans.csv",
               [{"plasmid_id": "p1", "payload_accession": "P1", "emitted": "0"}])
    assert mod.count_segments([p]) == {("p1", "P1"): 0}


def test_read_k_bhs_is_keyed_on_the_pair_too(tmp_path):
    """So the alignment baseline and this run's own count are compared key-for-key."""
    p = tmp_path / "detail.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["plasmid_id", "payload_accession", "k_bhs"])
        w.writeheader()
        w.writerows([{"plasmid_id": "polb", "payload_accession": "P06746", "k_bhs": "3"},
                     {"plasmid_id": "p2", "payload_accession": "P2", "k_bhs": "1"}])
    assert mod.read_k_bhs(p) == {("polb", "P06746"): 3, ("p2", "P2"): 1}


# ------------------------------------------------------------------ the hybrid's own premise

def test_the_script_documents_that_a_cosine_segmenter_cannot_work():
    """`--bhs-scorer align` is not a convenience. `best_hit_segments` stops against a floor in raw
    BLOSUM62 units near 100, so a cosine scorer in [-1, 1] stops at k = 0 on every construct and the
    arm degenerates to `whole`. If this script's docstring ever stops saying so, the next reader will
    run it without the flag and get a silently empty per-segment arm."""
    assert "--bhs-scorer align" in mod.__doc__
    assert "BLOSUM62" in mod.__doc__


def test_the_script_says_what_a_kmer_arm_does_not_say_about_esm2():
    """The one sentence that keeps this from being quoted as the thesis's number."""
    doc = mod.__doc__
    assert "not** the ESM-2 number" in doc or "not the ESM-2 number" in doc
