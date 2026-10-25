import numpy as np
import pytest

from pilot.embed import KmerEmbedder
from pilot.retrieval import PlasmidResult, evaluate, rank_against, summarise

REFS = ["P001", "P002", "P003", "P004"]


def res(pid, arch, best, orf_rank, payload="P001"):
    return PlasmidResult(pid, arch, payload, 3, best, "o1", orf_rank, "o1", [])


def test_rank_is_one_when_the_payload_scores_highest():
    sims = np.array([[0.9, 0.1, 0.2, 0.3]], dtype=np.float32)
    hits, best, best_orf, _ = evaluate(sims, ["o1"], REFS, "P001")
    assert (best, best_orf) == (1, "o1")
    assert hits[0].top1_accession == "P001" and hits[0].payload_score == pytest.approx(0.9)


def test_rank_counts_everything_scoring_at_least_as_high():
    sims = np.array([[0.5, 0.9, 0.7, 0.6]], dtype=np.float32)   # P001 is 4th
    _, best, _, _ = evaluate(sims, ["o1"], REFS, "P001")
    assert best == 4


def test_ties_take_the_worst_position_so_results_are_never_flattered():
    sims = np.array([[0.8, 0.8, 0.8, 0.1]], dtype=np.float32)
    _, best, _, _ = evaluate(sims, ["o1"], REFS, "P001")
    assert best == 3        # not 1


def test_best_rank_is_the_best_across_a_plasmids_orfs():
    sims = np.array([[0.1, 0.9, 0.8, 0.7], [0.95, 0.1, 0.2, 0.3]], dtype=np.float32)
    _, best, best_orf, _ = evaluate(sims, ["o1", "o2"], REFS, "P001")
    assert (best, best_orf) == (1, "o2")


def test_payload_orf_rank_tracks_the_named_orf_only():
    sims = np.array([[0.95, 0.1, 0.2, 0.3], [0.1, 0.9, 0.8, 0.7]], dtype=np.float32)
    _, best, _, orf_rank = evaluate(sims, ["o1", "o2"], REFS, "P001", payload_orf_id="o2")
    assert best == 1 and orf_rank == 4       # the best ORF is not the insert-matching one


def test_no_payload_or_unknown_accession_yields_no_rank_rather_than_a_false_failure():
    sims = np.array([[0.9, 0.1, 0.2, 0.3]], dtype=np.float32)
    for acc in ("", "NOT_IN_REF"):
        hits, best, _, _ = evaluate(sims, ["o1"], REFS, acc)
        assert best is None and hits[0].payload_rank is None
        assert hits[0].top1_accession == "P001"   # top-1 is still reported


def test_plasmid_with_no_orfs_is_handled():
    sims, ids = rank_against({}, ["MKV" * 20], ["P001"], KmerEmbedder())
    assert sims.shape == (0, 1) and ids == []
    hits, best, _, _ = evaluate(sims, ids, ["P001"], "P001")
    assert hits == [] and best is None


def test_summarise_excludes_unscored_plasmids_instead_of_counting_them_as_failures():
    rs = [res("a", "single_cds", 1, 1), res("b", "tagged_fusion", 3, 40),
          res("c", "no_payload", None, None, payload="")]
    s = summarise(rs)
    assert (s["n_plasmids"], s["n_scored"], s["n_no_payload"]) == (3, 2, 1)
    assert s["best_rank_top1"] == 1 and s["best_rank_top5"] == 2
    assert s["payload_orf_top1"] == 1 and s["payload_orf_top5"] == 1   # b's insert ORF ranks 40
    assert s["by_archetype"]["tagged_fusion"]["best_top5"] == 1
    assert s["by_archetype"]["tagged_fusion"]["orf_top5"] == 0


def test_end_to_end_a_sequence_retrieves_itself_against_decoys():
    """The whole point, with the stub embedder: an ORF that IS a reference protein must rank it first."""
    rng = np.random.default_rng(0)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    target = "".join(rng.choice(list(aa), 300))
    decoys = ["".join(rng.choice(list(aa), 300)) for _ in range(40)]
    refs, accs = [target] + decoys, ["TARGET"] + [f"D{i}" for i in range(40)]
    sims, ids = rank_against({"orf1": target}, refs, accs, KmerEmbedder())
    _, best, _, _ = evaluate(sims, ids, accs, "TARGET")
    assert best == 1


def test_contamination_pushes_the_payload_down_the_ranking():
    """A fusion ORF (payload + unrelated protein) should rank the payload worse than the clean ORF does."""
    rng = np.random.default_rng(1)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    payload = "".join(rng.choice(list(aa), 250))
    junk = "W" * 250                                   # compositionally very unlike the payload
    decoys = ["".join(rng.choice(list(aa), 250)) for _ in range(40)]
    refs, accs = [payload] + decoys, ["TARGET"] + [f"D{i}" for i in range(40)]
    e = KmerEmbedder()
    clean = evaluate(*rank_against({"clean": payload}, refs, accs, e), accs, "TARGET")[1]
    fused = evaluate(*rank_against({"fused": payload + junk}, refs, accs, e), accs, "TARGET")[1]
    assert clean == 1
    assert fused >= clean            # contamination never helps


def test_precomputed_reference_matrix_matches_embedding_inline():
    """Regression: the reference set was re-embedded per plasmid, 23x the work on a GPU run."""
    from pilot.retrieval import embed_reference

    e = KmerEmbedder()
    refs = ["MKVLAAGIVGLN" * 8, "WWWWPPPPGGGG" * 8, "ACDEFGHIKLMN" * 8]
    orfs = {"o1": "MKVLAAGIVGLN" * 8}
    inline, _ = rank_against(orfs, refs, ["A", "B", "C"], e)
    cached, _ = rank_against(orfs, refs, ["A", "B", "C"], e, embed_reference(refs, e))
    assert np.allclose(inline, cached)
