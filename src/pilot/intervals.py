"""Interval arithmetic on a (possibly circular) plasmid.

Everything is 0-based, half-open, in the plasmid's forward coordinates. A feature that crosses the
origin of a circular plasmid is represented as two non-wrapping segments, e.g. [(6900, 7000), (0, 120)].
"""
from __future__ import annotations

Segment = tuple[int, int]


def segments(start: int, length: int, plasmid_length: int) -> list[Segment]:
    """Split an interval that may run past the end of the plasmid into non-wrapping segments."""
    end = start + length
    if end <= plasmid_length:
        return [(start, end)]
    return [(start, plasmid_length), (0, end - plasmid_length)]


def overlap_bp(a: list[Segment], b: list[Segment]) -> int:
    """Total base pairs shared between two segment lists."""
    total = 0
    for a0, a1 in a:
        for b0, b1 in b:
            total += max(0, min(a1, b1) - max(a0, b0))
    return total


def union_bp(seg_lists: list[list[Segment]]) -> int:
    """Base pairs covered by the union of several segment lists (overlaps counted once)."""
    spans = sorted(s for segs in seg_lists for s in segs)
    total = 0
    cur_s = cur_e = None
    for s, e in spans:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def gap_bp(a: list[Segment], b: list[Segment], plasmid_length: int, circular: bool) -> int:
    """Smallest distance in bp between two segment lists (0 if they overlap or touch)."""
    best = None
    for a0, a1 in a:
        for b0, b1 in b:
            if min(a1, b1) > max(a0, b0):
                return 0
            gap = max(b0 - a1, a0 - b1)
            if circular:
                gap = min(gap, (b0 - a1) % plasmid_length, (a0 - b1) % plasmid_length)
            best = gap if best is None else min(best, gap)
    return best if best is not None else plasmid_length


def crosses_origin(segs: list[Segment], plasmid_length: int) -> bool:
    """True if a feature runs across the origin of a circular plasmid.

    Needs one segment ending at the last base and another starting at the first base; a spliced CDS whose
    exons are simply separated by an intron does not qualify.
    """
    return len(segs) > 1 and any(e == plasmid_length for _, e in segs) and any(s == 0 for s, _ in segs)
