#!/usr/bin/env python3
"""Fetch a corpus of synthetic constructs from NCBI by targeted query, not by bulk download.

    python scripts/11_fetch_corpus.py --email you@example.com --target 400 --out-dir data/corpus

Why targeted rather than bulk: the GenBank synthetic division is distributed as whole-division flat files,
which avoids rate limits but costs gigabytes, most of it patent oligos that every downstream filter would
throw away. Several hundred records fetched by query is a few megabytes and arrives in minutes.

NCBI asks that automated access identify itself by email, and allows 3 requests/second without an API key
or 10 with one (free, from an NCBI account). This script honours whichever applies and retries on the
transient 429/500 responses that E-utilities returns under load.

Writes one `.gbk` per record plus `fetch_log.csv` recording what was asked for and what came back, so the
corpus is reproducible from the log alone.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# A construct worth keeping is circular-ish, big enough to carry a payload and a backbone, small enough
# not to be a genome, and annotated. The size window is the same one used for the pilot set.
DEFAULT_QUERY = (
    '("synthetic construct"[Organism] OR gbdiv_syn[PROP]) '
    'AND (plasmid[Title] OR vector[Title] OR cassette[Title]) '
    'AND 2000:25000[SLEN] '
    'AND biomol_genomic[PROP]'
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--email", required=True, help="NCBI requires a contact address for automated access")
    ap.add_argument("--api-key", default=None, help="optional NCBI API key; raises the rate limit to 10/s")
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--target", type=int, default=400, help="how many records to retrieve")
    ap.add_argument("--batch", type=int, default=100, help="records per efetch call")
    ap.add_argument("--out-dir", type=Path, default=Path("data/corpus"))
    ap.add_argument("--dry-run", action="store_true", help="report how many records match, fetch nothing")
    args = ap.parse_args(argv)

    try:
        from Bio import Entrez, SeqIO
    except ImportError:
        print("biopython is required: pip install biopython", file=sys.stderr)
        return 2

    Entrez.email = args.email
    if args.api_key:
        Entrez.api_key = args.api_key
    pause = 0.11 if args.api_key else 0.34  # stay under 10/s or 3/s respectively

    print(f"query: {args.query}\n")
    with Entrez.esearch(db="nuccore", term=args.query, retmax=0, usehistory="y") as h:
        res = Entrez.read(h)
    total = int(res["Count"])
    print(f"{total:,} records match")
    if args.dry_run:
        print("dry run; nothing fetched")
        return 0
    if total == 0:
        print("nothing to fetch: widen the query", file=sys.stderr)
        return 1

    want = min(args.target, total)
    webenv, qkey = res["WebEnv"], res["QueryKey"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log, kept, seen = [], 0, 0
    t0 = time.time()

    for start in range(0, want, args.batch):
        n = min(args.batch, want - start)
        text = None
        for attempt in range(5):
            try:
                with Entrez.efetch(db="nuccore", rettype="gbwithparts", retmode="text",
                                   retstart=start, retmax=n, webenv=webenv, query_key=qkey) as h:
                    text = h.read()
                break
            except Exception as exc:  # noqa: BLE001 (E-utilities fails transiently under load)
                wait = 2 ** attempt
                print(f"  batch {start}: {type(exc).__name__}, retry in {wait}s ({attempt + 1}/5)")
                time.sleep(wait)
        if text is None:
            print(f"  batch {start}: gave up", file=sys.stderr)
            continue

        tmp = args.out_dir / f"_batch_{start}.gb"
        tmp.write_text(text, encoding="utf-8")
        for rec in SeqIO.parse(tmp, "genbank"):
            seen += 1
            n_cds = sum(1 for f in rec.features if f.type == "CDS")
            name = "".join(c if c.isalnum() or c in "-_" else "_" for c in rec.id)
            path = args.out_dir / f"ncbi_{name}.gbk"
            SeqIO.write(rec, path, "genbank")
            kept += 1
            log.append(dict(accession=rec.id, filename=path.name, length_bp=len(rec.seq), n_cds=n_cds,
                            topology=rec.annotations.get("topology", ""),
                            description=rec.description[:200]))
        tmp.unlink(missing_ok=True)
        print(f"    {min(start + n, want):>5}/{want} fetched, {kept} written  ({time.time() - t0:.0f}s)",
              flush=True)
        time.sleep(pause)

    with open(args.out_dir / "fetch_log.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    mb = sum(p.stat().st_size for p in args.out_dir.glob("*.gbk")) / 1e6
    print(f"\n{kept} records in {args.out_dir}  ({mb:.1f} MB)")
    print(f"with CDS features: {sum(1 for r in log if r['n_cds'] > 0)}/{kept}")
    print(f"circular: {sum(1 for r in log if r['topology'] == 'circular')}/{kept}")
    print("wrote fetch_log.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
