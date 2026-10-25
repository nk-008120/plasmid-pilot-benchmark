"""`--database` on `scripts/19_find_homologs_all.py`: the queries and the search database are separate.

The flag exists so the holdout mask can be computed *exhaustively against the twilight catalogue* rather
than discovered in Swiss-Prot. The mask's question is not "what are this payload's homologs in biology"
but "which of these 40,750 catalogue residents is too close to the payload to leave in the reference",
which is a property of the catalogue and so is answerable by scoring every resident. That removes the
depth parameter, the significance gate and the rank cap from the mask entirely.

The independence is tested by a protein that exists in only one of the two files, in each direction:
`ONLY_IN_SP` is identical to the payload but present only in the Swiss-Prot file, and `ONLY_IN_DB` is
identical to the payload but present only in the database. Both would be a 100% identity hit if they were
searched, so each one's presence or absence in the output is an unambiguous answer about which file was
searched, rather than something inferred from a count.
"""
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pilot.homology import MAX_LEN, MIN_LEN, iter_swissprot  # noqa: E402

pytest.importorskip("parasail")


def _load():
    spec = importlib.util.spec_from_file_location(
        "find_homologs_all_db", ROOT / "scripts" / "19_find_homologs_all.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["find_homologs_all_db"] = mod
    spec.loader.exec_module(mod)
    return mod


fha = _load()

PAYLOAD = ("MKAIFVLKGSLDRDLVEHFKQYGHEVVTNPLDAKALAEMMKNADALVVRSATKVTADVFAQAKNLKIVARAGVGLDNIDLDAAIKRGIYVVNTPGANS"
           "NAVAELVIGHMLSLARHIPQAHASMKEGRWDRKAYVGTELHGKTLGVIGLGRIGSEVARIAKAFGMNVIAYDPYISPERAEQIGVELVSLDELLAR")
UNRELATED = ("MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLLPRRGPRLGVRATRKTSERSQPRGRRQPIPKARRPEGRTWAQPGYPWPLYGNEGCGWAGWL"
             "LSPRGSRPSWGPTDPRRRSRNLGKVIDTLTCGFADLMGYIPLVGAPLGGAARALAHGVRVLEDGVNYATGNLPGCSFSIFLLALLSCLTVPASAYQV")
SHORT_PEPTIDE = "MKWVTFISLLFLFSSAYS"


def _fasta(path: Path, records: list[tuple[str, str]]) -> Path:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for header, seq in records:
            fh.write(f">{header}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i : i + 60] + "\n")
    return path


def _manifest(path: Path, accessions: list[str]) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["filename", "payload_accession"])
        w.writeheader()
        for i, a in enumerate(accessions):
            w.writerow({"filename": f"construct_{i}.gb", "payload_accession": a})
    return path


def _run(tmp_path: Path, extra: list[str], sp_records, db_records=None) -> dict:
    """Run main() and return {payload: {homolog_accession: identity}} from the written CSV."""
    sp = _fasta(tmp_path / "sp.fasta", sp_records)
    man = _manifest(tmp_path / "manifest.csv", ["PAY001"])
    out = tmp_path / "out"
    argv = ["--manifest", str(man), "--swissprot", str(sp), "--out-dir", str(out),
            "--decoys", str(tmp_path / "no_such_decoys.fasta"), "--threads", "1",
            "--top-n", "100000"]
    if db_records is not None:
        argv += ["--database", str(_fasta(tmp_path / "db.fasta", db_records))]
    argv += extra
    assert fha.main(argv) == 0
    got: dict = {}
    with open(out / "homologs.csv", newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            got.setdefault(r["payload_accession"], {})[r["homolog_accession"]] = float(r["identity"])
    return got


def test_database_is_searched_and_swissprot_is_not():
    """The whole point of the flag: queries from one file, subjects from the other."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        got = _run(
            tmp_path, [],
            sp_records=[("PAY001 payload", PAYLOAD), ("ONLY_IN_SP homolog_of:PAY001", PAYLOAD),
                        ("FILLER1 filler", UNRELATED)],
            db_records=[("ONLY_IN_DB homolog_of:PAY001", PAYLOAD), ("FILLER2 filler", UNRELATED)])
    hits = got.get("PAY001", {})
    assert "ONLY_IN_DB" in hits, "the --database file was not searched"
    assert "ONLY_IN_SP" not in hits, "--swissprot was searched even though --database was given"
    assert hits["ONLY_IN_DB"] == pytest.approx(1.0), "an identical sequence must come back at 100%"


def test_query_still_comes_from_swissprot_when_database_is_given():
    """The payload is absent from the database, so it cannot retrieve itself, but it is still searched.

    This is the Q9TT33 case made small: a 41 aa payload sits in the catalogue because the build adds its
    payloads unconditionally, and is then excluded from it as a database by the 50 aa lower bound.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        got = _run(
            tmp_path, [],
            sp_records=[("PAY001 payload", PAYLOAD), ("FILLER1 filler", UNRELATED)],
            db_records=[("REL001 homolog_of:PAY001", PAYLOAD), ("FILLER2 filler", UNRELATED)])
    assert "PAY001" not in got.get("PAY001", {}), "the payload is not in the database, so not a subject"
    assert "REL001" in got.get("PAY001", {}), "the payload was still used as the query"


def test_omitting_database_leaves_every_existing_invocation_unchanged():
    """Without --database, --swissprot is both query source and database, as before the flag existed."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        got = _run(
            tmp_path, [],
            sp_records=[("PAY001 payload", PAYLOAD), ("ONLY_IN_SP homolog_of:PAY001", PAYLOAD),
                        ("FILLER1 filler", UNRELATED)])
    hits = got.get("PAY001", {})
    assert "ONLY_IN_SP" in hits, "the default path must still search --swissprot"
    assert "PAY001" in hits, "the payload retrieves itself when it is in the database"


def test_catalogue_headers_parse_to_bare_accessions():
    """`iter_swissprot` on a twilight catalogue, which is the reader `--database` relies on.

    A catalogue header is ">A0A017SE81 homolog_of:P71079" and a Swiss-Prot one is ">sp|P42212|GFP_AEQVI".
    The accession is `split()[0]` in the first case and needs the pipes split in the second, so this is
    checked rather than assumed: a reader that returned "A0A017SE81 homolog_of:P71079" as the accession
    would silently produce a mask whose entries match nothing.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = _fasta(Path(td) / "cat.fasta",
                      [("A0A017SE81 homolog_of:P71079", PAYLOAD),
                       ("P42212 payload", UNRELATED),
                       ("A0A017SQ41 homolog_of:P07702", PAYLOAD)])
        got = [acc for acc, _ in iter_swissprot(path, min_len=1, max_len=10 ** 9)]
    assert got == ["A0A017SE81", "P42212", "A0A017SQ41"]


def test_the_database_window_still_applies_to_the_database_file():
    """A short catalogue resident is excluded as a subject, which is why the probe's count is 1, not 0."""
    import tempfile

    assert len(SHORT_PEPTIDE) < MIN_LEN
    with tempfile.TemporaryDirectory() as td:
        path = _fasta(Path(td) / "cat.fasta",
                      [("Q9TT33 payload", SHORT_PEPTIDE), ("A0A017SE81 homolog_of:Q9TT33", PAYLOAD)])
        kept = [acc for acc, _ in iter_swissprot(path, min_len=MIN_LEN, max_len=MAX_LEN)]
        unfiltered = [acc for acc, _ in iter_swissprot(path, min_len=1, max_len=10 ** 9)]
    assert kept == ["A0A017SE81"]
    assert unfiltered == ["Q9TT33", "A0A017SE81"]
