"""Build the retrieval reference set from a Swiss-Prot FASTA.

The reference is the catalogue the pipeline searches when asked "what protein is this ORF?". It contains:

* the **payload proteins** named in the manifest's `payload_accession` column: these are the correct
  answers, and the run fails loudly if any is missing rather than silently scoring against an absent target;
* a seeded sample of **decoys** from the rest of Swiss-Prot, so retrieval is a real search rather than a
  choice among twenty candidates.

Decoy count is a knob with a real trade-off: too few and the task is trivially easy, too many and the
embedding run gets slow. 5,000 keeps an ESM-2 650M pass to a sub-hour GPU job while still being a
~250-fold harder task than the payloads alone.
"""
from __future__ import annotations

import gzip
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

ACC_RX = re.compile(r"^(?:sp|tr)\|([^|]+)\|")

# One window for the whole project. These used to differ (4,000 here, 2,000 in pilot.homology), so a
# protein could be a legitimate reference entry and still be invisible to the homolog search.
from .homology import MAX_LEN, MIN_LEN  # noqa: E402,PLC0415


@dataclass
class Entry:
    accession: str
    header: str
    seq: str

    @property
    def description(self) -> str:
        return self.header.split(" ", 1)[1] if " " in self.header else self.header


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path, encoding="utf-8")


def iter_fasta(path: Path) -> Iterator[Entry]:
    header, buf = None, []
    with _open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    m = ACC_RX.match(header)
                    yield Entry(m.group(1) if m else header.split()[0], header, "".join(buf))
                header, buf = line[1:], []
            else:
                buf.append(line.strip())
    if header is not None:
        m = ACC_RX.match(header)
        yield Entry(m.group(1) if m else header.split()[0], header, "".join(buf))


def build(
    swissprot: Path,
    payload_accessions: set[str],
    n_decoys: int = 5000,
    seed: int = 0,
    min_len: int = MIN_LEN,
    max_len: int = MAX_LEN,
) -> tuple[list[Entry], list[str]]:
    """Return (entries, missing_accessions). Decoys are reservoir-sampled in one streaming pass."""
    rng = random.Random(seed)
    payloads: dict[str, Entry] = {}
    reservoir: list[Entry] = []
    n_seen = 0
    for e in iter_fasta(swissprot):
        if e.accession in payload_accessions:
            payloads[e.accession] = e
            continue  # a payload is never also a decoy
        if not (min_len <= len(e.seq) <= max_len):
            continue
        n_seen += 1
        if len(reservoir) < n_decoys:
            reservoir.append(e)
        else:
            j = rng.randrange(n_seen)
            if j < n_decoys:
                reservoir[j] = e
    missing = sorted(payload_accessions - set(payloads))
    entries = [payloads[a] for a in sorted(payloads)] + reservoir
    return entries, missing


def write_fasta(entries: list[Entry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(f">{e.header}\n")
            for i in range(0, len(e.seq), 60):
                fh.write(e.seq[i : i + 60] + "\n")
