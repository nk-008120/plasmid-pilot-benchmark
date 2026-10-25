"""`54_baseline_submission.py`'s `--arm-kind` and `--source`: the flags that stop an edit clobbering a row.

the run board said, until 20 Oct, to add the embedding runs to `SOURCES` and pass `arm_kind="embedding"`.
That is accurate about what has to change and wrong about how, because three things in that script were
hard-coded beside `SOURCES`: the `report()` call's arm kind, the metadata's arm kind, and **the output
filename**. Following the instruction as written would have produced embedding numbers inside
`baseline_alignment.json`, labelled as alignment numbers, on top of the published alignment baseline.

So the arm kind is a flag, the sources are a flag, and the filename is derived from the arm kind rather
than written twice. This file pins the three properties that make that safe:

* the flagless invocation still produces the published alignment baseline, field for field;
* `--arm-kind embedding` writes its own file and **cannot** touch the alignment one;
* `--source` parses the way the help text says, including a Windows drive letter, which a naive
  `split(":")` would eat.

The integration tests are environment-gated: they need `benchmark/v1/items.csv` and the three finished
twilight CSVs, which are gitignored under `results/`. They run on the ThinkPad and skip in a worktree,
like the other environment-gated tests in this suite.

**One field is excluded from the comparison and it matters that it is named rather than tolerated.**
`no_run_needed.read_only_cost_seconds` is how long this script itself took, so it differs between any two
runs (0.515 against 0.533 when the flags were added). A test asserting byte-identity would have failed on
the clock and been "fixed" by loosening it, which is how a real difference gets waved through later. It
is excluded here by name, and everything else must match exactly.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark" / "v1"
TIMING = "read_only_cost_seconds"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "baseline_submission", ROOT / "scripts" / "54_baseline_submission.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sources_present(mod) -> bool:
    return all((ROOT / p).is_file() for _, p, _ in mod.SOURCES)


def _strip_timing(d: dict) -> dict:
    out = json.loads(json.dumps(d))
    out.get("no_run_needed", {}).pop(TIMING, None)
    return out


# --------------------------------------------------------------------------- parse_sources, no I/O

def test_no_source_flag_means_the_built_in_sources(mod):
    """The whole no-op guarantee rests on this identity, so it is asserted and not assumed."""
    assert mod.parse_sources(None) is mod.SOURCES
    assert mod.parse_sources([]) is mod.SOURCES


def test_a_source_carries_its_label_path_and_runtime(mod):
    assert mod.parse_sources(["corpus=results/x.csv:499"]) == (("corpus", "results/x.csv", 499.0),)


def test_a_runtime_is_optional(mod):
    """A runtime nobody recorded is better absent than guessed: the two embedding run logs do not
    print a total, so `--source` must be usable without one."""
    assert mod.parse_sources(["corpus=results/x.csv"]) == (("corpus", "results/x.csv", 0.0),)


def test_a_windows_drive_letter_survives(mod):
    r"""`C:\NK\x.csv` split on the FIRST colon gives label `C` and path `\NK\x.csv`. Splitting on the
    last, and rejecting a one-character head, is what keeps an absolute Windows path usable, and this
    is a Windows-only project, so the naive version would have broken on the first real path."""
    assert mod.parse_sources([r"corpus=C:\NK\x.csv"]) == (("corpus", r"C:\NK\x.csv", 0.0),)
    assert mod.parse_sources([r"corpus=C:\NK\x.csv:762.5"]) == (("corpus", r"C:\NK\x.csv", 762.5),)


def test_a_spec_with_no_label_is_refused(mod):
    """Refuse rather than guess. A path handed in without a label would otherwise become a tier named
    after part of itself and appear in the benchmark's provenance under that name."""
    with pytest.raises(ValueError):
        mod.parse_sources(["results/x.csv"])


def test_several_sources_keep_their_order(mod):
    got = mod.parse_sources(["a=1.csv:1", "b=2.csv:2"])
    assert [t for t, _, _ in got] == ["a", "b"]


# --------------------------------------------------------------------------- the no-op guard

@pytest.mark.skipif(not (BENCH / "items.csv").is_file(),
                    reason="needs benchmark/v1 (run scripts/52_build_benchmark.py)")
def test_the_flagless_run_reproduces_the_published_alignment_baseline(mod, tmp_path):
    """The guard that makes the refactor safe. Everything but this script's own wall clock must match
    the committed file, which is the artefact `benchmark/README.md` and `53_score_submission.py`
    both point at."""
    if not _sources_present(mod):
        pytest.skip("needs the three finished twilight CSVs under results/")
    bd = tmp_path / "v1"
    shutil.copytree(BENCH, bd)
    assert mod.main(["--benchmark", str(bd)]) in (0, None)
    fresh = json.loads((bd / "baseline_alignment.json").read_text(encoding="utf-8"))
    published = json.loads((BENCH / "baseline_alignment.json").read_text(encoding="utf-8"))
    assert _strip_timing(fresh) == _strip_timing(published)
    assert fresh["arm_kind"] == "alignment"
    # and the excluded field is genuinely present in both, so the exclusion cannot hide its removal
    assert TIMING in fresh["no_run_needed"] and TIMING in published["no_run_needed"]


@pytest.mark.skipif(not (BENCH / "items.csv").is_file(),
                    reason="needs benchmark/v1 (run scripts/52_build_benchmark.py)")
def test_an_embedding_run_writes_its_own_file_and_leaves_the_alignment_one_alone(mod, tmp_path):
    """The defect the old instruction would have caused, asserted as impossible. The embedding run here
    is given the ALIGNMENT csvs on purpose: this test is about which file gets written, not about which
    numbers go in it, and using the inputs that exist keeps it from needing a GPU run to be meaningful."""
    if not _sources_present(mod):
        pytest.skip("needs the three finished twilight CSVs under results/")
    bd = tmp_path / "v1"
    shutil.copytree(BENCH, bd)
    before = (bd / "baseline_alignment.json").read_bytes()
    srcs = []
    for tier, path, secs in mod.SOURCES:
        srcs += ["--source", f"{tier}={path}:{secs}"]
    assert mod.main(["--benchmark", str(bd), "--arm-kind", "embedding", *srcs]) in (0, None)
    assert (bd / "baseline_embedding.json").is_file()
    assert (bd / "baseline_embedding.csv").is_file()
    assert (bd / "baseline_alignment.json").read_bytes() == before
    emb = json.loads((bd / "baseline_embedding.json").read_text(encoding="utf-8"))
    assert emb["arm_kind"] == "embedding"
    assert emb["scorer"] == mod.SCORER_NOTE["embedding"]
    assert "max-length 2048" in emb["scorer"], "the cache key belongs in the provenance "


@pytest.mark.skipif(not (BENCH / "items.csv").is_file(),
                    reason="needs benchmark/v1 (run scripts/52_build_benchmark.py)")
def test_a_suffix_writes_a_separate_baseline_without_touching_canonical_files(mod, tmp_path):
    if not _sources_present(mod):
        pytest.skip("needs the three finished twilight CSVs under results/")
    bd = tmp_path / "v1"
    shutil.copytree(BENCH, bd)
    canonical = {name: (bd / name).read_bytes() for name in
                 ("baseline_alignment.json", "baseline_embedding.json") if (bd / name).is_file()}
    srcs = []
    for tier, path, secs in mod.SOURCES:
        srcs += ["--source", f"{tier}={path}:{secs}"]
    assert mod.main(["--benchmark", str(bd), "--arm-kind", "embedding", "--suffix", "postr10",
                     *srcs]) in (0, None)
    assert (bd / "baseline_embedding_postr10.json").is_file()
    assert (bd / "baseline_embedding_postr10.csv").is_file()
    assert {name: (bd / name).read_bytes() for name in canonical} == canonical
    postr10 = json.loads((bd / "baseline_embedding_postr10.json").read_text(encoding="utf-8"))
    assert postr10["arm_kind"] == "embedding"


def test_suffix_rejects_path_components(mod):
    with pytest.raises(SystemExit):
        mod.main(["--suffix", "../postr10"])


# --------------------------------------------------------------------------- segmentation provenance

def _spans(d: Path, *, with_threshold: bool, n: int = 3) -> Path:
    """A minimal `bhs_spans_*.csv`. Only the header and `plasmid_id` are read, so only they are real."""
    cols = ["tier", "plasmid_id", "iteration"] + (["threshold"] if with_threshold else [])
    lines = [",".join(cols)]
    for i in range(n):
        row = ["hand", f"p{i}", "1"] + (["0.25"] if with_threshold else [])
        lines.append(",".join(row))
    p = d / "bhs_spans_fake.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return p


def test_a_threshold_column_means_the_segmentation_is_per_holdout(mod, tmp_path):
    _spans(tmp_path, with_threshold=True)
    got = mod.segmentation_provenance(str(tmp_path / "twilight_x.csv"))
    assert got["per_threshold"] is True
    assert got["lower_bound_thresholds"] == []
    assert got["n_constructs"] == 3


def test_no_threshold_column_marks_every_looser_holdout_a_lower_bound(mod, tmp_path):
    """this project's defect, declared rather than discovered. One segmentation reused across the sweep
    manufactures reported LOSS at every holdout above the strictest, so those cells bound the cost of
    decomposing from below and are not measurements of it."""
    _spans(tmp_path, with_threshold=False)
    got = mod.segmentation_provenance(str(tmp_path / "twilight_x.csv"))
    assert got["per_threshold"] is False
    assert got["lower_bound_thresholds"] == [1.0, 0.9, 0.7, 0.5, 0.35]
    assert min(mod.THRESHOLDS) not in got["lower_bound_thresholds"], "the strictest holdout is clean"
    # The note has to say which mask produced the rows. It used to say it by citing a section
    # of the working record, which does not ship, so the assertion follows the wording out.
    assert "reused at every looser holdout" in got["reason"]


def test_a_missing_span_file_is_unknown_and_not_clean(mod, tmp_path):
    """`None` rather than `True`: a run whose segmentation cannot be placed has not been cleared of
    anything, and reading an absent file as a pass is how this project's denominator bug travelled."""
    got = mod.segmentation_provenance(str(tmp_path / "twilight_x.csv"))
    assert got["per_threshold"] is None
    assert got["spans_file"] is None
    assert got["lower_bound_thresholds"] == []


def test_the_two_arm_lists_partition_the_arms_the_baseline_emits(mod):
    """A new arm must be classified, not defaulted. `whole` and `oracle` query a fixed span and cannot
    inherit a segmentation defect; the other four can, and that difference is the whole point of the
    block. An arm added without a decision would silently read as unsegmented."""
    assert not set(mod.SEGMENTED_ARMS) & set(mod.UNSEGMENTED_ARMS)
    assert set(mod.SEGMENTED_ARMS) | set(mod.UNSEGMENTED_ARMS) == {
        "whole", "oracle", "auto", "auto_per_segment", "bhs", "bhs_per_segment"}
    assert set(mod.HEADLINE_ARMS) <= set(mod.SEGMENTED_ARMS) | set(mod.UNSEGMENTED_ARMS)


@pytest.mark.skipif(not (BENCH / "baseline_alignment.json").is_file(),
                    reason="needs the published baseline")
def test_the_published_baselines_declare_their_segmentation(mod):
    """The artifact a reviewer reads has to carry the limit, not just the script that wrote it.

    Derived from each file's own sources rather than enumerated, and the change is deliberate. This
    read `lower_bound_thresholds == [1.0, 0.9, 0.7, 0.5, 0.35]`, which asserted that the **canonical**
    baseline carries the the working record defect: true on the day it was written and false from the moment
    the user promoted the re-measured pair on 22 Oct. It is the same shape as the arm-list test fixed
    the same day: a test that pins today's state while its docstring states a rule.

    The rule is that the block must agree with the sources. A pair whose every source segmented under
    its own per-threshold mask must report no lower-bound thresholds and must not cite the working record as a
    live caveat; a pair with any pre-fix source must name every threshold above the strictest and say
    why. Both files are checked, so the promotion cannot silently leave one behind.
    """
    seen = 0
    for name in ("baseline_alignment.json", "baseline_embedding.json",
                 "baseline_alignment_prer10.json", "baseline_embedding_prer10.json"):
        f = BENCH / name
        if not f.is_file():
            continue
        seen += 1
        d = json.loads(f.read_text(encoding="utf-8"))
        seg = d["segmentation"]
        assert seg["per_source"], name
        assert all(s["per_threshold"] is not None for s in seg["per_source"]), name
        clean = all(s["per_threshold"] for s in seg["per_source"])
        if clean:
            assert seg["lower_bound_thresholds"] == [], name
            assert "per-threshold mask" in seg["note"], name
            assert d["is_canonical"] is True, f"{name}: a clean pair should be the canonical one"
        else:
            # every threshold above the strictest, which is where reuse manufactures loss
            assert seg["lower_bound_thresholds"] == sorted(
                (t for t in mod.THRESHOLDS if t > min(mod.THRESHOLDS)), reverse=True), name
            assert "built once at the strictest threshold" in seg["note"], name
            assert d["is_canonical"] is False, f"{name}: a pre-fix pair must not be canonical"
    assert seen, "no published baseline found at all; the skipif guard is wrong"
