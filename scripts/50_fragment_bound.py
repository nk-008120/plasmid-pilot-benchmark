#!/usr/bin/env python3
"""How short can a fragment be and still be identifiable at all? Derived, not measured.

    python scripts/50_fragment_bound.py
    python scripts/50_fragment_bound.py --reference results/gate_all173/reference_twilight.fasta

**What this is for.** The proposal's first extension is short fragments, 25 to 200 nt. Before any of a
borrowed GPU window is spent on that range, this asks the free question: *which part of it can clear a
significance gate at all?*

The gate inverts to a score floor with no free variable left once the query and the database are fixed:

    S_min = (ln(K m N) + ln(1/E)) / lambda          (`pilot.homology.score_floor`)

measured at 99.5 to 116.1 across this project's 173 payloads. A fragment's *best possible* score is
bounded above by aligning it to a perfect copy of itself, which is the sum of the BLOSUM62 diagonal over
its own residues. So the shortest identifiable fragment is the smallest `m` with

    m * s_diag >= S_min(m, N)

and because `m` enters the floor only inside a logarithm, that equation has one crossing and the answer
is a clean length rather than a region. **Nothing here is a prediction about retrieval.** It is a
ceiling: a fragment below this length cannot reach significance even against an identical database
entry, so no embedder, scorer or index can rescue it. Fragments above it are *admissible*, not
necessarily found.

**Two diagonal scores are reported, and the gap between them is the honest uncertainty.**

- `max` uses the largest BLOSUM62 diagonal, which is W at 11. A query of 66 tryptophans is not a
  biological sequence; this column exists to show that even the most generous arithmetic available
  does not rescue the short end.
- `composition` weights each diagonal by how often that residue actually occurs in this project's own
  catalogue, which is the realistic case and the one to quote. It is computed from the reference FASTA
  rather than a textbook table, so it belongs to this benchmark and says so.

**The nucleotide-to-residue step is where a reader will object, so it is made explicit.** 25 to 200 nt
is 8 to 66 codons. A fragment in unknown frame translates six ways and the longest clean open stretch is
shorter still, so the residue counts here are an upper bound on what a nucleotide fragment yields. That
makes the conclusion stronger, not weaker: the real admissible threshold in nucleotides is at least
3x the residue answer, and in practice more.

Writes `fragment_bound.csv` and `fragment_bound.json` to `--out-dir`.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.reference import iter_fasta  # noqa: E402
from pilot.homology import K_PARAM, LAMBDA, score_floor  # noqa: E402

# BLOSUM62 diagonal, the score of a residue aligned to itself. Written out rather than read from
# parasail so this script runs without the optional alignment dependency, and checked against it when
# it is present (see `_check_against_parasail`).
BLOSUM62_DIAGONAL = {
    "A": 4, "R": 5, "N": 6, "D": 6, "C": 9, "Q": 5, "E": 5, "G": 6, "H": 8, "I": 4,
    "L": 4, "K": 5, "M": 5, "F": 6, "P": 7, "S": 4, "T": 5, "W": 11, "Y": 7, "V": 4,
}

# The catalogues in play, from this project's register. Residue totals, not row counts: N in the E-value
# is the summed residue count of the database actually searched.
CATALOGUES = [
    ("5,016-protein (early decomposition catalogue)", 2_600_000),
    ("15,205-protein", 7_800_000),
    ("25,402-protein (route B, family-structured)", 14_614_744),
    ("40,750-protein (gate_all173, current elsewhere)", 20_806_731),
    ("Swiss-Prot, for comparison", 207_099_657),
]

MAX_EVALUE = 1e-3


def _check_against_parasail() -> str:
    """Confirm the hard-coded diagonal against the matrix the benchmark actually aligns with.

    The table above is written out so this script runs without the optional alignment dependency, and
    a hard-coded copy of a matrix is exactly the kind of thing this repository has been bitten by. So
    when parasail is present it is checked, residue by residue, against `parasail.blosum62` itself.
    The index comes from the matrix's own `mapper`, an ASCII-indexed lookup, rather than from a guess
    about column order.
    """
    try:
        import parasail  # noqa: PLC0415
    except ImportError:  # pragma: no cover - exercised only where parasail is absent
        return "parasail absent; diagonal NOT cross-checked"
    m = parasail.blosum62
    bad = [f"{aa}: table {want}, parasail {int(m.matrix[m.mapper[ord(aa)]][m.mapper[ord(aa)]])}"
           for aa, want in BLOSUM62_DIAGONAL.items()
           if int(m.matrix[m.mapper[ord(aa)]][m.mapper[ord(aa)]]) != want]
    return ("diagonal verified against parasail.blosum62 for all 20 residues" if not bad
            else "MISMATCH against parasail.blosum62: " + "; ".join(bad))


def composition(path: Path | None) -> tuple[dict[str, float], str]:
    """Residue frequencies of this project's own catalogue, or a flat prior when none is given."""
    if path is None or not path.is_file():
        n = len(BLOSUM62_DIAGONAL)
        return {a: 1.0 / n for a in BLOSUM62_DIAGONAL}, "flat prior (no reference given)"
    c: Counter[str] = Counter()
    for e in iter_fasta(path):
        c.update(ch for ch in e.seq if ch in BLOSUM62_DIAGONAL)
    tot = sum(c.values())
    return {a: c[a] / tot for a in BLOSUM62_DIAGONAL}, f"{path.name}, {tot:,} residues"


def min_length(s_diag: float, db_residues: int, max_evalue: float = MAX_EVALUE,
               limit: int = 100_000) -> int | None:
    """Smallest query length whose best possible score reaches the floor. None if it never does."""
    for m in range(1, limit + 1):
        if m * s_diag >= score_floor(max_evalue, m, db_residues):
            return m
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path,
                    default=Path("results/gate_all173/reference_twilight.fasta"),
                    help="catalogue whose residue composition weights the diagonal")
    ap.add_argument("--out-dir", type=Path, default=Path("results/fragment_bound"))
    ap.add_argument("--max-evalue", type=float, default=MAX_EVALUE)
    args = ap.parse_args(argv)

    freq, freq_src = composition(args.reference)
    s_comp = sum(freq[a] * BLOSUM62_DIAGONAL[a] for a in BLOSUM62_DIAGONAL)
    s_max = max(BLOSUM62_DIAGONAL.values())

    print(f"E <= {args.max_evalue:g}, lambda {LAMBDA}, K {K_PARAM}")
    print(f"BLOSUM62 diagonal: {_check_against_parasail()}")
    print(f"composition from {freq_src}")
    print(f"  expected self-score per residue: {s_comp:.3f}   most generous possible: {s_max}")
    print()

    rows = []
    print(f"{'catalogue':<48}{'floor':>8}{'min aa':>8}{'min nt':>8}{'min aa':>8}{'min nt':>8}")
    print(f"{'':<48}{'at m=50':>8}{'composition':>16}{'most generous':>16}")
    print("-" * 88)
    for name, N in CATALOGUES:
        m_comp = min_length(s_comp, N, args.max_evalue)
        m_max = min_length(float(s_max), N, args.max_evalue)
        floor50 = score_floor(args.max_evalue, 50, N)
        print(f"{name:<48}{floor50:>8.1f}{m_comp:>8}{m_comp * 3:>8}{m_max:>8}{m_max * 3:>8}")
        rows.append(dict(catalogue=name, db_residues=N, floor_at_m50=round(floor50, 2),
                         min_aa_composition=m_comp, min_nt_composition=m_comp * 3,
                         min_aa_most_generous=m_max, min_nt_most_generous=m_max * 3))

    # What this does to the proposal's stated range, which is the whole point of running it.
    main_cat = next(r for r in rows if r["db_residues"] == 20_806_731)
    lo_nt, hi_nt = 25, 200
    adm_lo = max(lo_nt, main_cat["min_nt_composition"])
    print()
    print(f"The proposal's range is {lo_nt} to {hi_nt} nt. Against the 40,750-protein catalogue the")
    print(f"realistic bound is {main_cat['min_nt_composition']} nt, so the admissible part of that range is")
    if adm_lo > hi_nt:
        print("  EMPTY: no fragment in the stated range can reach significance under alignment.")
        verdict = "empty"
    else:
        frac = (hi_nt - adm_lo + 1) / (hi_nt - lo_nt + 1)
        print(f"    {adm_lo} to {hi_nt} nt, which is {100 * frac:.0f}% of it.")
        verdict = f"{adm_lo}-{hi_nt} nt"
    print()
    print("Read this as a ceiling and not as a prediction. It assumes the fragment aligns to a perfect")
    print("copy of itself, which no holdout permits, and it ignores frame: a nucleotide fragment in")
    print("unknown frame yields a shorter clean peptide than its length in codons. Both make the real")
    print("threshold higher than the number above, never lower.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "fragment_bound.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out_dir / "fragment_bound.json").write_text(json.dumps({
        "max_evalue": args.max_evalue, "lambda": LAMBDA, "K": K_PARAM,
        "composition_source": freq_src,
        "self_score_per_residue_composition": round(s_comp, 4),
        "self_score_per_residue_most_generous": s_max,
        "diagonal_check": _check_against_parasail(),
        "proposal_range_nt": [lo_nt, hi_nt],
        "admissible_part_of_proposal_range": verdict,
        "rows": rows,
        "what_this_is_not": ("a retrieval prediction. It is an upper bound on what any scorer could "
                             "achieve, assuming a perfect self-match and ignoring reading frame."),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out_dir}/fragment_bound.csv and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
