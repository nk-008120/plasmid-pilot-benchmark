"""Tests for `pilot.complete`: the fragment-completion head and the contamination measurement.

The module's reason for existing is a caveat rather than a result, so what these tests pin is mostly the
*measuring* apparatus: that an identity is computed against residues and not against tokens, that the
baseline a completion has to beat is the composition baseline and not 1/20, and that the completer can
only emit the 20 standard residues. The one test that loads a model is environment-gated, because the
checkpoints are a 2.6 GB cache and not a dependency.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pilot.complete import (
    STANDARD_AA,
    Completion,
    MaskedLmCompleter,
    composition_baseline,
    identity,
    most_frequent_residue,
)

SMALL = "facebook/esm2_t12_35M_UR50D"


def _cached(name: str) -> bool:
    d = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + name.replace("/", "--"))
    return d.is_dir()


def test_identity_is_positionwise_and_refuses_a_length_mismatch():
    assert identity("ACDE", "ACDE") == 1.0
    assert identity("ACDE", "ACDQ") == 0.75
    assert identity("", "") == 0.0
    with pytest.raises(ValueError):
        # A ragged comparison is the coverage-denominator mistake again: both sides of a ratio have to
        # be computed the same way.
        identity("ACDE", "ACD")


def test_the_baseline_a_completion_must_beat_is_composition_not_one_in_twenty():
    """On a Swiss-Prot-derived catalogue the commonest residue alone is near 0.10, so a completion at
    0.12 identity has demonstrated nothing. Quoting 1/20 would flatter it by a factor of two."""
    seqs = ["AAAAAAAAAL", "AAAAAAAACD"]
    assert most_frequent_residue(seqs) == "A"
    assert composition_baseline(seqs) == pytest.approx(0.85)
    assert composition_baseline([]) == 0.0
    # Non-standard letters are excluded from both sides, not counted as a residue that was predicted.
    assert most_frequent_residue(["XXXXXA"]) == "A"


def test_completion_carries_what_is_needed_to_judge_it():
    c = Completion(payload_accession="P09751", fragment_len=33, n_completed=12, mode="iterative",
                   completed="ACDEFGHIKLMN", truth="ACDEFGHIKLMQ")
    assert c.identity_to_truth == pytest.approx(11 / 12)


def test_the_module_states_why_a_model_swap_is_not_a_holdout():
    """The contamination argument is load-bearing and lives in the module docstring rather than only in
    CONTEXT, because the next person to reach for a second model will read the code first."""
    import pilot.complete as mod
    doc = mod.__doc__ or ""
    assert "UniRef" in doc and "Swiss-Prot" in doc
    assert "scorer-independence" in doc
    for other in ("ESM-C", "ProtT5", "ProGen2"):
        assert other in doc, f"{other} is in the same training-corpus family and must be named"
    assert "novel payload" in doc, "the falsifying case has to be named before the run "


def test_complete_rejects_an_unknown_mode_without_loading_a_model():
    c = MaskedLmCompleter(SMALL, device="cpu")
    with pytest.raises(ValueError):
        c.complete("ACDE", 4, mode="beam")
    assert c._model is None, "an argument check must not pay for a 2.6 GB load"
    assert c.complete("ACDE", 0) == "", "nothing to complete is not an error either"
    assert c._model is None


@pytest.mark.skipif(not _cached(SMALL), reason="ESM-2 35M not in the local HuggingFace cache")
def test_the_completer_emits_only_standard_residues_in_both_modes():
    """The head is free to return `X`, `B`, `<mask>` or `<pad>`; an identity scored against those would
    be an identity against tokens that are not residues. The restriction is what makes the measurement
    a measurement."""
    c = MaskedLmCompleter(SMALL, device="cpu")
    prefix = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
    for mode in ("single", "iterative"):
        got = c.complete(prefix, 6, mode=mode)
        assert len(got) == 6
        assert set(got) <= set(STANDARD_AA), f"{mode} emitted a non-residue: {got!r}"
