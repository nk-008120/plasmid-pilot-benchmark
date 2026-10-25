import random

import pytest
from Bio.Seq import Seq

from pilot.genbank import STOP_CODONS
from pilot.orfs import find_orfs

from conftest import build_plasmid, random_dna


@pytest.mark.parametrize("strand", [1, -1])
@pytest.mark.parametrize("wrap", [False, True])
def test_planted_orf_is_found_exactly(strand, wrap):
    seq, planted = build_plasmid(7, [{"label": "g", "strand": strand, "n_codons": 120}], rotate_to_wrap=wrap)
    p = planted[0]
    assert p.wraps == wrap
    orfs = find_orfs(seq, circular=True, min_aa=100)
    hits = [o for o in orfs if o.protein == p.protein]
    assert len(hits) == 1
    o = hits[0]
    assert (o.strand, o.start, o.length_nt, o.wraps) == (strand, p.start, p.length_nt, wrap)


def test_linear_sequence_does_not_wrap():
    seq, planted = build_plasmid(3, [{"label": "g", "strand": 1, "n_codons": 80}])
    orfs = find_orfs(seq, circular=False, min_aa=60)
    assert any(o.protein == planted[0].protein for o in orfs)
    assert not any(o.wraps for o in orfs)


def test_orf_across_origin_is_missed_if_plasmid_treated_as_linear():
    seq, planted = build_plasmid(7, [{"label": "g", "strand": 1, "n_codons": 120}], rotate_to_wrap=True)
    assert not any(o.protein == planted[0].protein for o in find_orfs(seq, circular=False, min_aa=100))


def test_alternative_start_codon_is_read_as_met():
    cds = "GTG" + "AAA" * 60 + "TAA"
    seq = "CCCTAA" + cds + "CCC"
    orf = [o for o in find_orfs(seq, circular=False, min_aa=50, start_codons=("ATG", "GTG")) if o.length_nt == len(cds)][0]
    assert orf.protein == "M" + "K" * 60


def test_earliest_start_in_a_stop_free_run_is_used():
    cds = "ATG" + "AAA" * 5 + "ATG" + "AAA" * 60 + "TAA"
    seq = "CCCTAA" + cds + "CCC"
    orfs = find_orfs(seq, circular=False, min_aa=50)
    assert [o.length_nt for o in orfs] == [len(cds)]


def _naive(seq, circular, starts, min_aa):
    """Deliberately simple reference implementation, written independently of the scanner."""
    L = len(seq)
    out = set()
    for strand, s in ((1, seq), (-1, str(Seq(seq).reverse_complement()))):
        def codon(i):
            if circular:
                return "".join(s[(i + k) % L] for k in range(3))
            return s[i : i + 3] if 0 <= i and i + 3 <= L else None

        for p in range(L if circular else L - 2):
            if codon(p) not in starts:
                continue
            n, k = None, 0
            while 3 * (k + 1) <= L:
                c = codon(p + 3 * k)
                if c is None:
                    break
                if c in STOP_CODONS:
                    n = 3 * (k + 1)
                    break
                k += 1
            if n is None or n // 3 - 1 < min_aa:
                continue
            earlier, j = False, 1
            while 3 * j + n <= L:
                c = codon(p - 3 * j)
                if c is None or c in STOP_CODONS:
                    break
                if c in starts:
                    earlier = True
                    break
                j += 1
            if earlier:
                continue
            out.add((strand, p if strand == 1 else (L - p - n) % L, n))
    return out


@pytest.mark.parametrize("circular", [True, False])
@pytest.mark.parametrize("length", [297, 298, 299, 300, 301])
@pytest.mark.parametrize("starts", [("ATG",), ("ATG", "GTG", "TTG")])
def test_scanner_matches_independent_reference(circular, length, starts):
    total = 0
    for seed in range(6):
        seq = random_dna(random.Random(seed * 31 + length), length)
        got = {(o.strand, o.start, o.length_nt) for o in find_orfs(seq, circular, min_aa=3, start_codons=starts)}
        assert got == _naive(seq, circular, frozenset(starts), 3)
        total += len(got)
    assert total > 20  # the comparison is not vacuous
