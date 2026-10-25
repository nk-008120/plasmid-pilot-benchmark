from pilot.archetype import classify_archetype, is_backbone_marker, is_tag_label
from pilot.genbank import AnnotatedCDS, Feature, Plasmid
from pilot.intervals import segments

L = 6000


def pl(features=()):
    return Plasmid("p", "p.gb", "p", "A" * L, "circular", False, list(features))


def cds(label, n_aa, start, strand=1):
    segs = segments(start, n_aa * 3 + 3, L)
    return AnnotatedCDS(label, strand, segs, "M" + "A" * (n_aa - 1), n_aa * 3 + 3, True, "")


def feat(kind, label, start, end, strand=1):
    return Feature(kind, label, strand, [(start, end)], f"{start + 1}..{end}")


def test_label_helpers():
    assert all(is_backbone_marker(x) for x in ["AmpR", "KanR", "NeoR/KanR", "lacZα", "repA"])
    assert not any(is_backbone_marker(x) for x in ["EGFP", "Cas9", "repressor"])
    assert all(is_tag_label(x) for x in ["6xHis", "3xFLAG", "V5", "TEV site", "GS linker"])
    assert not is_tag_label("histone H2B") and not is_tag_label("HA", protein_len=566)
    assert is_tag_label("GST", protein_len=218)


def test_single_cds_vs_backbone_heavy():
    gfp = cds("EGFP", 239, 1000)
    assert classify_archetype(pl([feat("CDS", "EGFP", 1000, 1720)]), [gfp]).archetype == "backbone_heavy"  # 12% of plasmid
    assert classify_archetype(pl(), [gfp], backbone_threshold=0.10).archetype == "single_cds"  # threshold is tunable
    big = cds("GeneX", 1200, 1000)
    call = classify_archetype(pl(), [big])
    assert call.archetype == "single_cds" and call.payload_fraction > 0.5


def test_selection_markers_are_not_payload():
    call = classify_archetype(pl(), [cds("EGFP", 239, 1000), cds("AmpR", 286, 4000, -1), cds("KanR", 270, 5000)])
    assert call.n_payload_cds == 1


def test_multi_gene_cassette():
    call = classify_archetype(pl(), [cds("GeneA", 400, 500), cds("GeneB", 350, 2500)])
    assert call.archetype == "multi_gene_cassette" and call.n_payload_cds == 2


def test_tag_next_to_payload_makes_tagged_fusion_and_far_tag_does_not():
    gene = cds("GeneA", 400, 500)
    near = classify_archetype(pl([feat("misc_feature", "6xHis", 1720, 1738)]), [gene])
    assert near.archetype == "tagged_fusion" and near.tag_labels == ["6xHis"]
    far = classify_archetype(pl([feat("misc_feature", "6xHis", 4000, 4018)]), [gene])
    assert far.archetype == "single_cds"


def test_big_solubility_partner_is_tag_not_second_gene():
    gst, gene = cds("GST", 218, 100), cds("GeneA", 400, 760)  # GST ends at 757
    features = [feat("CDS", "GST", 100, 757), feat("CDS", "GeneA", 760, 1963)]
    call = classify_archetype(pl(features), [gst, gene])
    assert call.n_payload_cds == 1 and call.archetype == "tagged_fusion"


def test_tag_across_origin_is_found_on_circular_plasmid():
    gene = cds("GeneA", 400, 5000)  # ends near the origin
    call = classify_archetype(pl([feat("misc_feature", "FLAG", 10, 34)]), [gene])
    assert call.archetype == "tagged_fusion"
