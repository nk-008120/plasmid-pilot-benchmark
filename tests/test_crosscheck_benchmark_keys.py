"""`scripts/55_crosscheck_benchmark.py`: the two ways it could agree with nothing.

Loaded by path because `scripts/` is not an importable package, as in `tests/test_bhs_protocol_sample.py`.

`tests/test_crosscheck_per_segment.py` opens by saying that a cross-check which cannot report a
disagreement is worse than none, because it is quoted as evidence. Script 55 is quoted as evidence in
`benchmark/README.md`, the public claim that a reviewer's score and this project's differ by method
and by nothing else rests on its *120 of 120 identical*, and on 22 Oct it was found to have the same
defect in the other direction: it could not report an **absence** of comparison.

Both halves are pinned here.

1. **The key.** `52_build_benchmark.py --namespace-ids` addresses an item as `tier:construct_id`,
   because the two synthetic tiers are the same 336 constructs. `10_twilight.py` writes
   the bare id and knows nothing of that convention, so the published CSV and a namespaced `items.csv`
   are keyed on different strings. `published_id` is the only place that is reconciled.
2. **The refusal.** With the keys mismatched the script compared 0 of 120 cells, counted all 120 as
   "not in the published CSV", and printed `Every compared cell is identical`, because the success
   branch tested `not mismatches`, and zero cells produce zero mismatches. The verdict is now gated on
   `compared` and the function returns 2.

The second test reads the source rather than running `main`, and that is deliberate: `main` needs the
40,750-protein catalogue and a few minutes of exact Smith-Waterman, so a test that ran it would be
skipped in exactly the environments where this regression would be reintroduced.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "55_crosscheck_benchmark.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("crosscheck_benchmark", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_a_namespaced_id_maps_back_to_the_published_key(mod):
    assert mod.published_id("hand:addgene_73037", "hand") == "addgene_73037"
    assert mod.published_id("synth_random:sc_0001", "synth_random") == "sc_0001"
    assert mod.published_id("synth_real:sc_0001", "synth_real") == "sc_0001"


def test_the_two_synthetic_tiers_share_a_published_key_which_is_the_whole_point(mod):
    """They are the same 336 construct ids; namespacing is what tells them apart inside the benchmark.

    So de-namespacing must send them back to the *same* key, because each tier is compared against its
    own published CSV and that CSV uses the bare id.
    """
    assert (mod.published_id("synth_real:sc_0001", "synth_real")
            == mod.published_id("synth_random:sc_0001", "synth_random"))


def test_an_unnamespaced_id_is_returned_unchanged(mod):
    """The default build is not namespaced and must keep reproducing v1 byte for byte."""
    assert mod.published_id("addgene_73037", "hand") == "addgene_73037"
    assert mod.published_id("ncbi_M88535_1", "corpus") == "ncbi_M88535_1"


def test_only_this_tiers_prefix_is_stripped(mod):
    """A construct id that merely contains a colon is not a namespaced id and must survive intact."""
    assert mod.published_id("corpus:ncbi_M88535_1", "hand") == "corpus:ncbi_M88535_1"
    assert mod.published_id("odd:name", "hand") == "odd:name"


def test_every_tier_the_builder_can_assemble_has_a_published_csv(mod):
    """A `--tier` the builder emits but this script cannot check is a silent gap in the arbiter."""
    spec = importlib.util.spec_from_file_location("build_benchmark",
                                                  ROOT / "scripts" / "52_build_benchmark.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    buildable = {t[0] for t in builder.TIERS}
    assert buildable <= set(mod.TWILIGHT), f"no published CSV for {buildable - set(mod.TWILIGHT)}"


def test_the_success_message_is_gated_on_having_compared_something():
    """The 22 Oct defect, pinned at the source because `main` costs minutes and a catalogue.

    What made it dangerous is that nothing looked wrong: the run printed its item count, its cell
    count, a binding dict with no anomalies, and then the success paragraph. The only visible symptom
    was a line reading `not in the published CSV   120`, which the success text does not mention.
    """
    src = SCRIPT.read_text(encoding="utf-8")
    assert "compared = agree + len(mismatches)" in src
    assert "if not compared:" in src
    refusal = src.index("if not compared:")
    verdict = src.index('verdict = "IDENTICAL" if not mismatches')
    success = src.index("Every compared cell is identical")
    assert refusal < verdict < success, "the zero-cell refusal must precede the verdict"
    assert "return 2" in src[refusal:verdict], "the refusal must exit non-zero, not merely print"
