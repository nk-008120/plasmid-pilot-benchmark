"""The one binary the benchmark publishes, and the two ways it could arrive broken.

`benchmark/presentation.pptx` is written by the deck build rather than copied by hand, so the
published copy cannot drift from the one this project reads. That covers drift. It does not cover
the two things that would damage the file without anyone touching it, and both are pinned here.

**Line-ending conversion.** `.gitattributes` declares `benchmark/** text eol=lf`, which is right
for every other file in that directory and would corrupt a compressed archive: a CR byte inside a
deflate stream is not a line ending. What saves it is that `*.pptx binary` appears *after* that
rule and the last matching rule wins, so the ordering is load-bearing and is asserted below rather
than left to whoever edits the file next.

**A truncated or half-written build.** A deck build that fails partway leaves a file that exists,
has a plausible size, and is not a presentation. Opening it as an archive and reading the part
every presentation must contain is the cheapest check that would notice.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "benchmark" / "presentation.pptx"


def test_the_published_deck_is_present_and_is_not_a_stub():
    assert DECK.is_file(), f"{DECK} is published with the benchmark and must be in the tree"
    # The deck embeds measured figures, so it is megabytes. A build that died early is kilobytes.
    assert DECK.stat().st_size > 1_000_000, DECK.stat().st_size


def test_the_published_deck_is_an_intact_presentation():
    """A valid archive, and the one part that makes it a presentation rather than any zip."""
    with zipfile.ZipFile(DECK) as z:
        assert z.testzip() is None, "a corrupt member: check the line-ending rules below"
        names = set(z.namelist())
    assert "ppt/presentation.xml" in names
    assert "[Content_Types].xml" in names
    slides = [n for n in names if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
    assert len(slides) > 50, f"only {len(slides)} slides; the deck is 80"


def test_the_deck_carries_its_own_slides_text():
    """If the archive is intact but empty of text, the build wrote a shell of a deck."""
    with zipfile.ZipFile(DECK) as z:
        raw = z.read("ppt/slides/slide1.xml").decode("utf-8", "replace")
    assert "<a:t>" in raw, "slide 1 carries no text run"


@pytest.mark.parametrize("pattern", ["benchmark/** text eol=lf", "*.pptx binary"])
def test_both_line_ending_rules_are_declared(pattern):
    text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert pattern in text, f"{pattern!r} is missing from .gitattributes"


def test_the_binary_rule_comes_after_the_benchmark_rule():
    """Order is the whole of it: git applies the last matching rule, so `*.pptx binary` must sit
    below `benchmark/** text eol=lf` or the deck is checked out through a text filter."""
    lines = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    bench = next(i for i, ln in enumerate(lines) if ln.strip() == "benchmark/** text eol=lf")
    binary = next(i for i, ln in enumerate(lines) if ln.strip().startswith("*.pptx"))
    assert binary > bench, (
        f"*.pptx is declared at line {binary + 1} and benchmark/** at line {bench + 1}; "
        "the pptx rule must come second to win"
    )
