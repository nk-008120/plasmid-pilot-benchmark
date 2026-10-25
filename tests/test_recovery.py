from pilot.genbank import AnnotatedCDS, Feature, Plasmid
from pilot.intervals import crosses_origin, gap_bp, overlap_bp, segments, union_bp
from pilot.orfs import ORF
from pilot.recovery import analyse

L = 3000


def plasmid(features=()):
    return Plasmid("p", "p.gb", "p", "A" * L, "circular", False, list(features))


def cds(label, protein, start, strand=1, invalid=""):
    n = len(protein) * 3 + 3
    return AnnotatedCDS(label, strand, segments(start, n, L), protein, n, True, invalid)


def orf(protein, start, strand=1):
    n = len(protein) * 3 + 3
    return ORF(strand, start, n, start + n > L, protein)


A = "M" + "K" * 79  # 80 aa


def status_of(c, orfs, min_aa=50):
    return analyse(plasmid(), [c], orfs, min_aa)[0][0]


def test_exact():
    r = status_of(cds("a", A, 100), [orf(A, 100)])
    assert r.status == "exact" and r.orf_minus_cds_aa == 0 and r.coverage == 1.0


def test_contained_reports_extension_on_each_side():
    r = status_of(cds("a", A, 160), [orf("MSS" + A + "GG", 100)])  # A sits inside a longer ORF (a fusion)
    assert (r.status, r.n_extra_aa, r.c_extra_aa, r.orf_minus_cds_aa) == ("contained", 3, 2, 5)
    r2 = status_of(cds("a", A, 160), [orf("T" * 20 + A, 100)])
    assert (r2.status, r2.n_extra_aa, r2.c_extra_aa) == ("contained", 20, 0)


def test_truncated_when_orf_starts_after_annotation():
    r = status_of(cds("a", A, 100), [orf(A[5:], 115)])
    assert r.status == "truncated" and r.orf_minus_cds_aa == -5


def test_missed_when_no_related_orf_even_if_overlapping_or_wrong_strand():
    assert status_of(cds("a", A, 100), [orf("M" + "R" * 79, 100)]).status == "missed"
    assert status_of(cds("a", A, 100), [orf(A, 100, strand=-1)]).status == "missed"
    assert status_of(cds("a", A, 100), []).status == "missed"


def test_exclusions_keep_denominators_honest():
    assert status_of(cds("a", "M" + "K" * 20, 100), []).status == "below_min_aa"
    assert status_of(cds("a", "", 100, invalid="internal_stop"), []).status == "invalid"


def test_prefers_exact_over_contained():
    r = status_of(cds("a", A, 100), [orf("T" * 10 + A, 70), orf(A, 100)])
    assert r.status == "exact" and r.orf_index == 1


def test_orf_classes_and_context():
    c = cds("a", A, 100)
    promoter = Feature("promoter", "T7 promoter", 1, [(2000, 2300)], "2001..2300")
    pl = plasmid([promoter])
    matched = orf(A, 100)
    shadow = orf("M" + "L" * 59, 130, strand=-1)  # antisense, inside the annotated CDS
    loose = orf("M" + "G" * 59, 2010)  # sits in the promoter, no CDS
    far = orf("M" + "S" * 59, 1500)
    cres, ores = analyse(pl, [c], [matched, shadow, loose, far], 50)
    assert [r.klass for r in ores] == ["cds_match", "cds_shadow", "unannotated", "unannotated"]
    assert ores[0].matched_cds == ["a"]
    assert ores[2].feature_context == "promoter:T7 promoter"
    assert ores[3].feature_context == ""


def test_interval_helpers():
    assert segments(2900, 200, 3000) == [(2900, 3000), (0, 100)]
    assert overlap_bp([(2900, 3000), (0, 100)], [(2950, 3000), (0, 10)]) == 60
    assert union_bp([[(0, 10)], [(5, 20)], [(30, 40)]]) == 30
    assert gap_bp([(0, 10)], [(2990, 3000)], 3000, circular=True) == 0
    assert gap_bp([(0, 10)], [(2990, 3000)], 3000, circular=False) == 2980
    assert gap_bp([(0, 10)], [(25, 30)], 3000, circular=True) == 15


def test_crosses_origin_distinguishes_wrap_from_introns():
    assert crosses_origin([(2900, 3000), (0, 100)], 3000)
    assert crosses_origin([(0, 100), (2900, 3000)], 3000)
    assert not crosses_origin([(100, 400), (700, 900)], 3000)  # spliced exons, not a wrap
    assert not crosses_origin([(100, 400)], 3000)


def test_non_atg_start_inside_longer_orf_is_contained_not_missed():
    """Regression: the Met substitution used to break substring matching.

    A gene with a GTG start reads "M..." on its own but "V..." inside a longer ORF, so matching on the
    reported protein scored a plainly-found gene as `missed` (seen on real lacI annotations).
    """
    gene_raw = "V" + "K" * 99  # GTG start -> Val when translated in context
    gene = AnnotatedCDS("lacI-like", 1, segments(160, 303, L), "M" + gene_raw[1:], 303, True, "", gene_raw)
    long_raw = "M" + "T" * 20 + gene_raw
    long_orf = ORF(1, 100, len(long_raw) * 3 + 3, False, long_raw, long_raw)
    r = analyse(plasmid(), [gene], [long_orf], 50)[0][0]
    assert r.status == "contained"
    assert (r.n_extra_aa, r.c_extra_aa) == (21, 0)


def test_objects_without_protein_raw_still_match_on_protein():
    """Callers that build ORF/AnnotatedCDS positionally (as these tests do) must keep working."""
    assert status_of(cds("a", A, 100), [orf(A, 100)]).status == "exact"
