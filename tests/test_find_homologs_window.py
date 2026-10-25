"""An out-of-window payload must not be inserted into the catalogue it is excluded from.

`scripts/19_find_homologs_all.py` used to end its per-payload loop with an unconditional
`keep.setdefault(acc,...)`, so every payload was written into `reference_twilight.fasta` whether or not
it satisfied the 50-5,000 aa window that filtered everything else in that file. The result was a
catalogue that its own reader rejects: Q9TT33 (41 aa) sat in `results/gate_all173/reference_twilight.fasta`
as a record, and reading that same file back as a `--database` dropped it, so the one payload that could
not retrieve itself was also the one the catalogue claimed to contain.

The window is not widened to fix this. `pilot.homology.iter_swissprot`'s own stated reason for the lower
bound is that below 50 residues an alignment has too few columns to support a percent-identity estimate,
so admitting a 41 aa payload would put meaningless identity values inside the gate rather than outside it.
Excluding it makes that payload's constructs unscorable, which is the honest outcome.
"""
import csv
import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pilot.homology import MAX_LEN, MIN_LEN  # noqa: E402
from pilot.reference import iter_fasta  # noqa: E402

pytest.importorskip("parasail")


def _load():
    spec = importlib.util.spec_from_file_location(
        "find_homologs_all_win", ROOT / "scripts" / "19_find_homologs_all.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["find_homologs_all_win"] = mod
    spec.loader.exec_module(mod)
    return mod


fha = _load()

LONG = ("MKAIFVLKGSLDRDLVEHFKQYGHEVVTNPLDAKALAEMMKNADALVVRSATKVTADVFAQAKNLKIVARAGVGLDNIDLDAAIKRGIYVVNTPGANS"
        "NAVAELVIGHMLSLARHIPQAHASMKEGRWDRKAYVGTELHGKTLGVIGLGRIGSEVARIAKAFGMNVIAYDPYISPERAEQIGVELVSLDELLAR")
SHORT = "MKWVTFISLLFLFSSAYSRGVFRR"
UNRELATED = ("MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLLPRRGPRLGVRATRKTSERSQPRGRRQPIPKARRPEGRTWAQPGYPWPLYGNEGCGWAGWL"
             "LSPRGSRPSWGPTDPRRRSRNLGKVIDTLTCGFADLMGYIPLVGAPLGGAARALAHGVRVLEDGVNYATGNLPGCSFSIFLLALLSCLTVPASAYQV")


def _fasta(path, records):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for header, seq in records:
            fh.write(f">{header}\n{seq}\n")
    return path


def _manifest(path, accessions):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["filename", "payload_accession"])
        w.writeheader()
        for i, a in enumerate(accessions):
            w.writerow({"filename": f"c{i}.gb", "payload_accession": a})
    return path


def _setup(tmp, extra=()):
    sp = _fasta(tmp / "sp.fasta", [("PAYLONG payload", LONG), ("PAYSHORT payload", SHORT),
                                   ("REL001 x", LONG), ("FILL1 x", UNRELATED)])
    man = _manifest(tmp / "m.csv", ["PAYLONG", "PAYSHORT"])
    out = tmp / "out"
    argv = ["--manifest", str(man), "--swissprot", str(sp), "--out-dir", str(out),
            "--decoys", str(tmp / "none.fasta"), "--threads", "1", "--top-n", "100000", *extra]
    return argv, out


def test_short_payload_is_searched_but_kept_out_of_the_reference():
    assert len(SHORT) < MIN_LEN
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        argv, out = _setup(tmp)
        assert fha.main(argv) == 0
        accs = {e.accession for e in iter_fasta(out / "reference_twilight.fasta")}
        with open(out / "payload_lengths.csv", newline="", encoding="utf-8-sig") as fh:
            searched = {r["payload_accession"] for r in csv.DictReader(fh)}
    assert "PAYSHORT" in searched, "the short payload must still be searched"
    assert "PAYSHORT" not in accs, "the short payload must not be inserted into the catalogue"
    assert "PAYLONG" in accs, "an in-window payload is still written as its own record"


def test_the_written_reference_survives_being_read_back_as_a_database():
    """The defect in one assertion: every record in the catalogue must satisfy the window."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        argv, out = _setup(tmp)
        assert fha.main(argv) == 0
        lengths = [len(e.seq) for e in iter_fasta(out / "reference_twilight.fasta")]
    assert lengths, "the reference must not be empty"
    assert all(MIN_LEN <= n <= MAX_LEN for n in lengths), \
        f"a record outside the {MIN_LEN}-{MAX_LEN} window reached the catalogue: {sorted(lengths)[:5]}"


def test_strict_payload_window_makes_it_an_error_instead():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        argv, out = _setup(tmp, extra=["--strict-payload-window"])
        assert fha.main(argv) == 3
        assert not (out / "reference_twilight.fasta").exists(), \
            "a refused build must not leave a partial catalogue behind"


def test_an_all_in_window_payload_set_is_unaffected():
    """The default path for every existing invocation: nothing is excluded and nothing is printed."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sp = _fasta(tmp / "sp.fasta", [("PAYLONG payload", LONG), ("REL001 x", LONG),
                                       ("FILL1 x", UNRELATED)])
        man = _manifest(tmp / "m.csv", ["PAYLONG"])
        out = tmp / "out"
        assert fha.main(["--manifest", str(man), "--swissprot", str(sp), "--out-dir", str(out),
                         "--decoys", str(tmp / "none.fasta"), "--threads", "1",
                         "--top-n", "100000", "--strict-payload-window"]) == 0
        accs = {e.accession for e in iter_fasta(out / "reference_twilight.fasta")}
    assert "PAYLONG" in accs
