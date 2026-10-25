#!/usr/bin/env python3
"""Rank each plasmid's true payload among the reference set, from the ORFs a naive pipeline produces.

    # validate the logic locally, no model download:
    python scripts/05_retrieval.py --results-dir results/baseline2 --embedder kmer
    # the real run, on a GPU machine:
    python scripts/05_retrieval.py --results-dir results/baseline2 --embedder facebook/esm2_t33_650M_UR50D

Needs `orfs.faa` and `pilot_summary.csv` from a validate run, plus `data/reference/reference.fasta`.
Writes `retrieval_<embedder>.csv` (per plasmid) and `retrieval_<embedder>_orfs.csv` (per ORF).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from pilot.embed import get_embedder
from pilot.reference import iter_fasta
from pilot.retrieval import PlasmidResult, embed_reference, evaluate, rank_against, summarise


def read_faa(path: Path) -> dict[str, dict[str, str]]:
    """orfs.faa -> {plasmid_id: {orf_id: protein}}; the plasmid id is the part of the ORF id before '|'."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for e in iter_fasta(path):
        oid = e.header.split()[0]
        out[oid.split("|")[0]][oid] = e.seq
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results/baseline2"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--reference", type=Path, default=Path("data/reference/reference.fasta"))
    ap.add_argument("--embedder", default="kmer", help="'kmer' (baseline/stub) or an ESM-2 checkpoint name")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args(argv)

    for p in (args.results_dir / "orfs.faa", args.results_dir / "pilot_summary.csv", args.reference):
        if not p.is_file():
            print(f"not found: {p}", file=sys.stderr)
            return 2

    with open(args.manifest, newline="", encoding="utf-8-sig", errors="replace") as fh:
        man = {r["filename"].strip(): r for r in csv.DictReader(fh) if (r.get("filename") or "").strip()}
    with open(args.results_dir / "pilot_summary.csv", newline="", encoding="utf-8-sig") as fh:
        summary = list(csv.DictReader(fh))
    orfs_by_plasmid = read_faa(args.results_dir / "orfs.faa")
    refs = list(iter_fasta(args.reference))
    ref_acc = [e.accession for e in refs]
    ref_seq = [e.seq for e in refs]

    kw = {} if args.embedder.startswith("kmer") else {"batch_size": args.batch_size}
    emb = get_embedder(args.embedder, **kw)
    print(f"embedder: {emb.name} | reference: {len(refs)} proteins | plasmids: {len(summary)}")

    t0 = time.time()
    print("embedding the reference set once...", flush=True)
    ref_matrix = embed_reference(ref_seq, emb)
    print(f"  done in {time.time() - t0:.1f}s")
    results: list[PlasmidResult] = []
    for row in summary:
        pid = row["plasmid_id"]
        m = man.get(row["source_file"], {})
        payload = (m.get("payload_accession") or "").strip()
        arch = (row.get("archetype_manual") or row.get("archetype_auto") or "").strip()
        seqs = orfs_by_plasmid.get(pid, {})
        sims, ids = rank_against(seqs, ref_seq, ref_acc, emb, ref_matrix)
        hits, best, best_orf, orf_rank = evaluate(sims, ids, ref_acc, payload, row.get("insert_best_orf_id", ""))
        results.append(PlasmidResult(pid, arch, payload, len(ids), best, best_orf, orf_rank,
                                     row.get("insert_best_orf_id", ""), hits))
        tag = "-" if best is None else str(best)
        print(f"    {pid[:40]:<42}{arch[:20]:<22}payload={payload or '(none)':<9}orfs={len(ids):>3}  best_rank={tag}")

    agg = summarise(results)
    stem = emb.name.replace("/", "_")
    with open(args.results_dir / f"retrieval_{stem}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["plasmid_id", "archetype", "payload_accession", "n_orfs", "best_rank", "best_rank_orf",
                    "payload_orf_rank", "payload_orf_id"])
        for r in results:
            w.writerow([r.plasmid_id, r.archetype, r.payload_accession, r.n_orfs,
                        "" if r.best_rank is None else r.best_rank, r.best_rank_orf,
                        "" if r.payload_orf_rank is None else r.payload_orf_rank, r.payload_orf_id])
    with open(args.results_dir / f"retrieval_{stem}_orfs.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["plasmid_id", "orf_id", "payload_rank", "payload_score", "top1_accession", "top1_score"])
        for r in results:
            for h in r.hits:
                w.writerow([r.plasmid_id, h.orf_id, "" if h.payload_rank is None else h.payload_rank,
                            "" if h.payload_score is None else round(h.payload_score, 4),
                            h.top1_accession, round(h.top1_score, 4)])
    agg["embedder"] = emb.name
    agg["n_reference"] = len(refs)
    agg["runtime_seconds"] = round(time.time() - t0, 1)
    (args.results_dir / f"retrieval_{stem}_summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    n = agg["n_scored"]
    print(f"\nScored {n} plasmids ({agg['n_no_payload']} have no protein payload)")
    print(f"  best ORF: top1 {agg['best_rank_top1']}/{n}  top5 {agg['best_rank_top5']}/{n}  top10 {agg['best_rank_top10']}/{n}")
    print(f"  insert-matching: top1 {agg['payload_orf_top1']}/{n}  top5 {agg['payload_orf_top5']}/{n}  top10 {agg['payload_orf_top10']}/{n}")
    print("\n  by archetype (best / insert-matching, top1):")
    for a, d in sorted(agg["by_archetype"].items()):
        print(f"        {a:<22}n={d['n']:<3}best {d['best_top1']}/{d['n']}   insert-orf {d['orf_top1']}/{d['n']}")
    print(f"\nruntime {agg['runtime_seconds']}s; wrote retrieval_{stem}*.csv/json to {args.results_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
