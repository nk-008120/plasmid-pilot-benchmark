"""Tests for the greyscale arm palette (`figstyle.ARM_DISPLAY_MONO`).

A second palette is a second place for the arm set to drift, which is the exact defect `figstyle`
exists to prevent, so the first test is that the two palettes carry the same arms, and the rest is
that the greyscale one still *encodes* what the colour one encodes.

The encoding is the point. `legend_note` promises a reader three things: that the distinguishing
channel says which segments a line is drawn from, that line style says how the hits were aggregated,
and that a solid and a dashed line of the same appearance differ only in aggregation. A greyscale
palette that merely replaced the hex values would break the third of those the moment two arms landed
on the same grey, and nothing in a figure would show it.
"""
from __future__ import annotations

import pytest

from pilot.arms import REPORT_ARM_NAMES
from pilot.figstyle import ARM_DISPLAY, ARM_DISPLAY_MONO, arm_display, legend_note

PAIRS = [("auto", "auto_per_segment"), ("bhs", "bhs_per_segment")]


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def test_both_palettes_carry_exactly_the_report_arms():
    """The drift guard. A new arm must appear in both palettes or in neither."""
    assert set(ARM_DISPLAY) == set(ARM_DISPLAY_MONO)
    assert set(ARM_DISPLAY) == set(REPORT_ARM_NAMES)


def test_only_the_colour_differs():
    """Label, line style, width, marker size and hollowness are the encoding; they must be shared.

    If the greyscale palette drifted on `fmt` or `hollow`, the two figures would be making different
    claims about aggregation while both citing `legend_note`.
    """
    for arm, colour_style in ARM_DISPLAY.items():
        mono_style = ARM_DISPLAY_MONO[arm]
        assert (mono_style.label, mono_style.fmt, mono_style.lw, mono_style.ms, mono_style.hollow) == \
               (colour_style.label, colour_style.fmt, colour_style.lw, colour_style.ms,
                colour_style.hollow), arm


def test_every_mono_colour_is_actually_grey():
    """R = G = B, which is what makes it survive a greyscale print rather than merely look muted."""
    for arm, st in ARM_DISPLAY_MONO.items():
        r, g, b = _rgb(st.colour)
        assert r == g == b, f"{arm} is {st.colour}, which is not a grey"


def test_paired_arms_share_a_grey_in_both_palettes():
    """Pooled and per-segment are one arm under two aggregations, so they must look like one arm."""
    for a, b in PAIRS:
        assert ARM_DISPLAY[a].colour == ARM_DISPLAY[b].colour
        assert ARM_DISPLAY_MONO[a].colour == ARM_DISPLAY_MONO[b].colour


def test_the_four_segment_sources_are_distinguishable_greys():
    """Four sources of segments, four greys, far enough apart to tell apart on paper.

    0x2A is about 16% of the range. Below that, two lines on a photocopy are one line, and the reader
    silently loses an arm rather than seeing a figure that looks wrong.
    """
    greys = sorted({_rgb(ARM_DISPLAY_MONO[a].colour)[0]
                    for a in ("whole", "auto", "bhs", "oracle")})
    assert len(greys) == 4, "two segment sources collapsed onto the same grey"
    gaps = [b - a for a, b in zip(greys, greys[1:])]
    assert min(gaps) >= 0x2A, f"greys too close: {greys}"


def test_grey_value_orders_with_how_much_of_the_frame_is_removed():
    """The caption claims lighter means less removed, so the palette has to actually do that.

    whole (nothing removed) is lightest, then the motif segments, then the best-hit segments, then the
    labelled payload span. A caption that described an order the colours did not have would be worse
    than no caption.
    """
    order = ["whole", "auto", "bhs", "oracle"]
    values = [_rgb(ARM_DISPLAY_MONO[a].colour)[0] for a in order]
    assert values == sorted(values, reverse=True), dict(zip(order, values))


def test_the_caption_changes_with_the_palette():
    """A grey figure captioned "colour =..." tells the reader to look for something not there."""
    assert "colour =" in legend_note()
    assert "colour =" not in legend_note(mono=True)
    assert "grey value" in legend_note(mono=True)
    for note in (legend_note(), legend_note(mono=True)):
        assert "line style = how they were aggregated" in note
        assert "solid = max-pooled" in note


def test_the_accessor_returns_the_right_palette():
    assert arm_display(False) is ARM_DISPLAY
    assert arm_display(True) is ARM_DISPLAY_MONO
    assert arm_display() is ARM_DISPLAY


@pytest.mark.parametrize("arm", list(REPORT_ARM_NAMES))
def test_mono_colours_are_six_digit_hex(arm):
    st = ARM_DISPLAY_MONO[arm]
    assert st.colour.startswith("#") and len(st.colour) == 7


def test_no_fifth_grey_clears_the_spacing_rule():
    """`tile` is the one arm whose grey is under 0x2A from a neighbour, and that is forced, not chosen.

    The four greys above are fixed by every figure already drawn, so the question is whether any fifth
    value could have been 0x2A from all of them. This proves it could not, rather than asserting it:
    the only admissible band is lighter than 0xF1, which is invisible on white. So the separation that
    does the work for `tile` on paper is the marker (`^` against `s`, `o` and `D`), and the four-arm
    spacing rule above is left exactly as it was rather than weakened to accommodate a fifth line.
    """
    fixed = sorted(_rgb(ARM_DISPLAY_MONO[a].colour)[0] for a in ("whole", "auto", "bhs", "oracle"))
    admissible = [v for v in range(256) if all(abs(v - f) >= 0x2A for f in fixed)]
    assert admissible, "if this is ever empty the premise below is wrong, not the conclusion"
    assert min(admissible) > 0xF1, f"a usable fifth grey exists and should be used: {admissible[:4]}"

    tile = _rgb(ARM_DISPLAY_MONO["tile"].colour)[0]
    assert min(abs(tile - f) for f in fixed) >= 0x20, "0x20 is the floor below which even the marker "
    assert ARM_DISPLAY_MONO["tile"].fmt.endswith("^")
    assert ARM_DISPLAY_MONO["tile_per_segment"].fmt.endswith("^")
    assert {ARM_DISPLAY_MONO[a].fmt[-1] for a in ("whole", "auto", "bhs", "oracle")}.isdisjoint({"^"})
