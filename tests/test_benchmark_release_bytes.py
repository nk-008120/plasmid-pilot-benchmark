"""The shipped benchmark's bytes, and the checksums that describe them.

`benchmark/v1/MANIFEST.json` publishes a sha256 for four files. On 22 Oct all four described the
**Windows worktree** bytes: with `core.autocrlf=true` and no `.gitattributes`, git stored LF and
handed this machine CRLF, so every one of those checksums was over bytes that no clone on any other
platform receives. The release directives' step 3 is *rebuild, then re-verify the shipped checksums*,
and it could only ever have passed here.

Two halves, and both are needed, because either alone is a patch:

* `.gitattributes` pins `benchmark/** text eol=lf`, which fixes what git hands a reader.
* Scripts 52-55 write LF explicitly, `csv.writer`'s default `lineterminator` is `\\r\\n` and a text
  write on Windows translates `\\n`, which fixes what the build script hashes.

This file pins the observable both halves exist for: the shipped bytes are LF, and the published
checksum is a checksum of them. It is deliberately a test over the **artefact** rather than over the
writer, because the artefact is what is published and a future script is not the only way to break it.

**One file under `benchmark/` is not text, as of 22 Oct**: `presentation.pptx`, the deck, which ships
with the benchmark. A deflate stream contains CR bytes as data, so the LF rule below cannot be run
over it, and the first run after the deck was published is exactly how that was found out. The
exemption is an explicit suffix list rather than a guess at which files are text, and the list is
itself asserted against what is on disk, so a second binary cannot arrive and be waved through by a
rule written for the first one. `.gitattributes` is what keeps git from applying `text eol=lf` to it,
and `tests/test_benchmark_presentation.py` pins that rule and its position in the file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmark"
V1 = BENCH / "v1"


#: Suffixes under `benchmark/` whose bytes are not text and cannot be LF-checked. Every entry
#: needs a matching `binary` rule in `.gitattributes`, below the `benchmark/** text eol=lf` line.
BINARY_SUFFIXES = (".pptx",)


def _shipped_files() -> list[Path]:
    return sorted(p for p in BENCH.rglob("*") if p.is_file())


def _shipped_text_files() -> list[Path]:
    return [p for p in _shipped_files() if p.suffix not in BINARY_SUFFIXES]


@pytest.mark.parametrize("name", [
    "items.csv", "queries.faa", "homologs.csv", "catalogue_accessions.txt", "MANIFEST.json",
])
def test_no_carriage_return_in_shipped_file(name: str) -> None:
    """A CR here is a checksum nobody else can reproduce, not a cosmetic difference."""
    data = (V1 / name).read_bytes()
    assert b"\r" not in data, (
        f"{name} contains CR. git serves LF to every clone, so these bytes are not the bytes a "
        f"reader gets, and MANIFEST.json's sha256 describes the wrong ones."
    )


def test_every_text_file_under_benchmark_is_lf() -> None:
    """Not only the four that are hashed: the README is read before anything is verified."""
    text = _shipped_text_files()
    offenders = [str(p.relative_to(BENCH)) for p in text if b"\r" in p.read_bytes()]
    assert offenders == [], f"CRLF under benchmark/: {offenders}"
    # A count, not a verdict: an exemption list that grew to cover everything would otherwise
    # leave this test passing over nothing at all.
    assert len(text) >= 15, f"only {len(text)} text file(s) checked under benchmark/"


def test_the_binary_exemption_covers_exactly_what_is_on_disk() -> None:
    """The exemption is a list, so the list has to be right. A new binary under `benchmark/`
    fails here rather than quietly inheriting an exemption written for the deck."""
    binaries = sorted(str(p.relative_to(BENCH)).replace("\\", "/")
                      for p in _shipped_files() if p.suffix in BINARY_SUFFIXES)
    assert binaries == ["presentation.pptx"], binaries


def test_manifest_checksums_match_the_files_on_disk() -> None:
    """The defect this file exists for: the checksum described a file that is not shipped."""
    manifest = json.loads((V1 / "MANIFEST.json").read_text(encoding="utf-8"))
    checked = 0
    for name, meta in manifest["files"].items():
        if not (isinstance(meta, dict) and meta.get("sha256")):
            continue
        path = V1 / name
        assert path.exists(), f"MANIFEST.json records {name}, which is not in {V1}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == meta["sha256"], (
            f"{name}: manifest says {meta['sha256'][:12]}, file is {actual[:12]}"
        )
        if "bytes" in meta:
            assert meta["bytes"] == path.stat().st_size, f"{name}: recorded size is stale"
        checked += 1
    # 55_crosscheck_benchmark.py's own lesson: a check that compared nothing reported success.
    assert checked == 4, f"expected 4 checksummed files, verified {checked}"


def test_gitattributes_pins_the_benchmark_to_lf() -> None:
    """The other half. Without it a checkout reintroduces CRLF whatever the scripts write."""
    ga = (BENCH.parent / ".gitattributes").read_text(encoding="utf-8")
    assert "benchmark/** text eol=lf" in ga
    assert "*.sh text eol=lf" in ga, "the WSL helpers break on a CR and are the other live case"
