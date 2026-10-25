import gzip

import pytest

from pilot.reference import build, iter_fasta, write_fasta


def write_sp(path, entries, gz=False):
    text = "".join(f">sp|{a}|{a}_TEST {d}\n{s}\n" for a, d, s in entries)
    if gz:
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        path.write_text(text)


def make(n, prefix="D"):
    return [(f"{prefix}{i:05d}", f"decoy {i}", "ACDEFGHIKLMNPQRSTVWY" * 5) for i in range(n)]


@pytest.mark.parametrize("gz", [False, True])
def test_iter_fasta_handles_plain_and_gzip(tmp_path, gz):
    p = tmp_path / ("sp.fasta.gz" if gz else "sp.fasta")
    write_sp(p, [("P12345", "Thing one OS=Homo sapiens", "MKVLAAG" * 10)], gz=gz)
    [e] = list(iter_fasta(p))
    assert e.accession == "P12345" and e.description.startswith("Thing one")
    assert len(e.seq) == 70


def test_payloads_are_always_included_and_never_sampled_as_decoys(tmp_path):
    p = tmp_path / "sp.fasta"
    write_sp(p, [("P00001", "payload", "MKVLAAG" * 20)] + make(200))
    entries, missing = build(p, {"P00001"}, n_decoys=10, seed=1)
    assert missing == []
    accs = [e.accession for e in entries]
    assert accs[0] == "P00001" and accs.count("P00001") == 1
    assert len(entries) == 11


def test_missing_payload_is_reported_not_silently_dropped(tmp_path):
    p = tmp_path / "sp.fasta"
    write_sp(p, make(50))
    _, missing = build(p, {"P00001", "P00002"}, n_decoys=5)
    assert missing == ["P00001", "P00002"]


def test_sampling_is_seeded_and_the_seed_matters(tmp_path):
    p = tmp_path / "sp.fasta"
    write_sp(p, make(500))
    a = [e.accession for e in build(p, set(), 20, seed=1)[0]]
    assert a == [e.accession for e in build(p, set(), 20, seed=1)[0]]
    assert a != [e.accession for e in build(p, set(), 20, seed=2)[0]]


def test_length_filter_excludes_decoys_but_never_payloads(tmp_path):
    p = tmp_path / "sp.fasta"
    write_sp(p, [("P00001", "short payload", "MKV"), ("D1", "short decoy", "MKV"),
                 ("D2", "ok decoy", "ACDEFGHIKL" * 10)])
    entries, missing = build(p, {"P00001"}, n_decoys=10, min_len=50)
    accs = [e.accession for e in entries]
    assert missing == [] and "P00001" in accs and "D1" not in accs and "D2" in accs


def test_write_fasta_roundtrips(tmp_path):
    p = tmp_path / "sp.fasta"
    write_sp(p, [("P00001", "payload", "MKVLAAGIVG" * 13)])
    entries, _ = build(p, {"P00001"}, 0)
    out = tmp_path / "ref" / "reference.fasta"
    write_fasta(entries, out)
    [back] = list(iter_fasta(out))
    assert back.accession == "P00001" and back.seq == entries[0].seq
