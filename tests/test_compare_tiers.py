"""`scripts/17_compare_tiers.py` must not quietly compare incomparable runs.

The script is loaded by path because `scripts/` is not an importable package.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("compare_tiers", ROOT / "scripts" / "17_compare_tiers.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compare_tiers"] = mod
    spec.loader.exec_module(mod)
    return mod


ct = pytest.importorskip("matplotlib") and _load()


def _write(d: Path, name: str, tier, n_ref=100):
    body = {"scorer": "sw-x", "tier": tier, "n_reference": n_ref,
            "by_threshold": {"1.00": {"whole_top1": 1, "whole_n": 1, "oracle_top1": 1, "oracle_n": 1}}}
    (d / name).write_text(json.dumps(body), encoding="utf-8")


def test_tiered_summaries_are_found_and_keyed_by_tier(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    _write(a, "twilight_sw-x_hand_summary.json", "hand")
    _write(b, "twilight_sw-x_corpus_summary.json", "corpus")
    found, skipped = ct.discover_summaries([a, b], "sw-x")
    assert sorted(found) == ["corpus", "hand"]
    assert skipped == []


def test_untiered_summary_is_skipped_not_renamed_after_its_directory(tmp_path):
    """The pre-`--tier` 15.13 run shares the scorer stem but used a different reference set."""
    d = tmp_path / "baseline2"
    d.mkdir()
    _write(d, "twilight_sw-x_hand_summary.json", "hand")
    _write(d, "twilight_sw-x_summary.json", "", n_ref=8328)
    found, skipped = ct.discover_summaries([d], "sw-x")
    assert list(found) == ["hand"]
    assert "baseline2" not in found
    assert [p.name for p in skipped] == ["twilight_sw-x_summary.json"]


def test_blank_tier_field_counts_as_untiered(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    _write(d, "twilight_sw-x_blank_summary.json", "   ")
    found, skipped = ct.discover_summaries([d], "sw-x")
    assert found == {}
    assert len(skipped) == 1


def test_a_different_scorer_is_not_picked_up(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    _write(d, "twilight_kmer3-1024_hand_summary.json", "hand")
    found, skipped = ct.discover_summaries([d], "sw-x")
    assert (found, skipped) == ({}, [])


class TestTwilightMaskAndTargetsAreSeparable:
    """`10_twilight.py` uses the homolog file for two different jobs: deciding which proteins count as a
    correct answer, and deciding which are removed from the catalogue at each holdout. They are the same
    file only while the target set is unfiltered. Once a gate filters targets, a close relative the gate
    dropped would stay in the catalogue as a distractor while not counting as a target either."""

    @staticmethod
    def _load():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "twilight", ROOT / "scripts" / "10_twilight.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["twilight"] = mod
        spec.loader.exec_module(mod)
        return mod

    def _write(self, path, rows):
        import csv as _csv

        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = _csv.DictWriter(fh, fieldnames=["payload_accession", "homolog_accession", "score",
                                                "identity", "coverage", "length"])
            w.writeheader()
            for acc, ident in rows:
                w.writerow({"payload_accession": "P1", "homolog_accession": acc, "score": 100.0,
                            "identity": ident, "coverage": 0.9, "length": 300})

    def test_both_files_load_independently(self, tmp_path):
        tw = self._load()
        full, gated = tmp_path / "full.csv", tmp_path / "gated.csv"
        self._write(full, [("A", 0.95), ("B", 0.40), ("C", 0.20)])
        self._write(gated, [("A", 0.95), ("C", 0.20)])
        assert len(tw.load_homologs(full)["P1"]) == 3
        assert len(tw.load_homologs(gated)["P1"]) == 2

    def test_the_mask_can_remove_a_protein_the_target_set_does_not_contain(self, tmp_path):
        """B at 0.40 identity is dropped by the gate. At a 0.25 holdout it must still be masked out of
        the catalogue, or it sits there competing while being unreachable as a correct answer."""
        tw = self._load()
        full, gated = tmp_path / "full.csv", tmp_path / "gated.csv"
        self._write(full, [("A", 0.95), ("B", 0.40), ("C", 0.20)])
        self._write(gated, [("A", 0.95), ("C", 0.20)])
        hs_mask = tw.load_homologs(full)["P1"]
        hs_tgt = tw.load_homologs(gated)["P1"]
        masked = {h.accession for h in hs_mask if h.identity > 0.25}
        targets = {h.accession for h in hs_tgt if h.identity <= 0.25}
        assert masked == {"A", "B"}
        assert targets == {"C"}
        assert "B" not in targets and "B" in masked
