#!/usr/bin/env python3
"""Measure what decomposition is worth: rank the payload from the whole ORF, from motif-cut segments, and
from the oracle span.

    python scripts/06_decompose.py --scorer kmer    # k-mer baseline; no model download, seconds
    python scripts/06_decompose.py --scorer align   # exact Smith-Waterman control (needs parasail)
    python scripts/06_decompose.py --scorer facebook/esm2_t33_650M_UR50D   # the real run, on a GPU machine

Needs `orfs.faa` and `pilot_summary.csv` from a validate run, plus `data/reference/reference.fasta`.
Writes `decompose_<scorer>.csv` (per plasmid), `_segments.csv` (per segment) and `_summary.json`. The
alignment arm writes the same columns under a different stem, so the two are directly comparable: that
comparison is the proposal's premise ("alignment misses what an embedding catches") put to the test rather
than asserted.

Arms, per plasmid (see `pilot.decompose` for why):
  whole       the payload-bearing ORF as called, the current baseline
  auto        motif decomposition, answer-blind; the plasmid scores as its best segment
  auto_noint  the same with the interior-tag rule off, an ablation, to show what that rule buys
  oracle      the locally-aligned payload span, a ceiling, not a method

The payload-bearing ORF is chosen by oracle alignment, so these numbers isolate decomposition from ORF
*selection*. Whether the deployable insert-size heuristic picked the same ORF is reported separately in
`orf_selection_agrees`, because conflating the two failures would hide both.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from pilot.align import get_scorer, model_kwargs, rank_one
from pilot.arms import MIN_AA, pick_payload_orf
from pilot.decompose import decompose, dilution
from pilot.reference import iter_fasta



def read_faa(path: Path) -> dict[str, dict[str, str]]:
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
    ap.add_argument("--scorer", "--embedder", dest="scorer", default="kmer",
                    help="'kmer', 'align' (Smith-Waterman control), or an ESM-2 checkpoint name")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1022,
                    help="residues per sequence handed to an ESM-2 checkpoint; longer ones are "
                         "truncated. Ignored by the k-mer and alignment arms. It is part of the "
                         "reference-matrix cache key, so changing it recomputes rather than reusing.")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/embed_cache"),
                    help="where to cache the embedded reference matrix; '' disables caching")
    ap.add_argument("--min-len", type=int, default=MIN_AA)
    ap.add_argument("--confident-cos", type=float, default=0.80,
                    help="score at which a segment's top hit counts as a confident identification. The "
                         "default suits cosine; the alignment arm's raw SW scores are on a different "
                         "scale, so pass a bit-score-like threshold (e.g. 200) when using --scorer align.")
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
    ref_by_acc = {e.accession: e.seq for e in refs}

    kw = model_kwargs(args.scorer, batch_size=args.batch_size, max_length=args.max_length)
    t0 = time.time()
    print(f"scorer: {args.scorer} | reference: {len(refs)} proteins")
    print("preparing the reference set once...", flush=True)
    scorer = get_scorer(args.scorer, [e.seq for e in refs],
                        cache_dir=(args.cache_dir or None), **kw)
    print(f"  done in {time.time() - t0:.1f}s ({scorer.name})", flush=True)

    rows, seg_rows = [], []
    n_orfs_total = n_segments_total = 0
    for row in summary:
        pid = row["plasmid_id"]
        m = man.get(row["source_file"], {})
        acc = (m.get("payload_accession") or "").strip()
        arch = (row.get("archetype_manual") or row.get("archetype_auto") or "").strip()
        orfs = orfs_by_plasmid.get(pid, {})
        n_orfs_total += sum(1 for s in orfs.values() if len(s) >= args.min_len)
        payload_seq = ref_by_acc.get(acc, "")
        target = ref_acc.index(acc) if acc in ref_by_acc else None

        oid, span = pick_payload_orf(orfs, payload_seq)
        # query-count inflation is a property of the pipeline, so count segments for every ORF
        for s in orfs.values():
            if len(s) >= args.min_len:
                n_segments_total += len(decompose(s, min_len=args.min_len)) or 1
        if not oid:
            rows.append(dict(plasmid_id=pid, archetype=arch, payload_accession=acc, orf_id="",
                             orf_aa="", oracle_span="", oracle_identity="", oracle_coverage="",
                             dilution="", whole_rank="", whole_cos="", auto_rank="", auto_cos="",
                             auto_n_segments="", auto_best_segment="", auto_noint_rank="",
                             oracle_rank="", oracle_cos="", orf_selection_agrees=""))
            print(f"    {pid[:44]:<46}(no protein payload)")
            continue

        orf_seq = orfs[oid]
        w_rank, w_cos, _, _ = rank_one(orf_seq, scorer, ref_acc, target)
        o_rank, o_cos = (None, None)
        if span:
            o_rank, o_cos, _, _ = rank_one(span.seq, scorer, ref_acc, target)

        best = {}
        for arm, kwargs in (("auto", {}), ("auto_noint", {"cut_at_internal_tags": False})):
            segs = decompose(orf_seq, min_len=args.min_len, **kwargs)
            ranked = []
            for i, sg in enumerate(segs, 1):
                r, c, t1, t1c = rank_one(sg.seq, scorer, ref_acc, target)
                ranked.append((r, c, sg, t1, t1c))
                if arm == "auto":
                    seg_rows.append([pid, oid, i, sg.start, sg.end, sg.length, ";".join(sg.removed),
                                     "" if r is None else r, "" if c is None else round(c, 4), t1,
                                     round(t1c, 4)])
            good = [x for x in ranked if x[0] is not None]
            if good:
                r, c, sg, _, _ = min(good, key=lambda x: (x[0], -(x[1] or 0)))
                best[arm] = (r, c, len(segs), f"{sg.start}..{sg.end}")
            else:
                best[arm] = (None, None, len(segs), "")

        a_rank, a_cos, a_n, a_seg = best["auto"]
        ni_rank = best["auto_noint"][0]
        rows.append(dict(
            plasmid_id=pid, archetype=arch, payload_accession=acc, orf_id=oid, orf_aa=len(orf_seq),
            oracle_span=f"{span.start}..{span.end}" if span else "",
            oracle_identity=span.identity if span else "", oracle_coverage=span.payload_coverage if span else "",
            dilution=dilution(orf_seq, span) if span else "",
            whole_rank="" if w_rank is None else w_rank, whole_cos="" if w_cos is None else round(w_cos, 4),
            auto_rank="" if a_rank is None else a_rank, auto_cos="" if a_cos is None else round(a_cos, 4),
            auto_n_segments=a_n, auto_best_segment=a_seg,
            auto_noint_rank="" if ni_rank is None else ni_rank,
            oracle_rank="" if o_rank is None else o_rank, oracle_cos="" if o_cos is None else round(o_cos, 4),
            orf_selection_agrees="yes" if oid == row.get("insert_best_orf_id", "") else "no",
        ))
        print(f"    {pid[:44]:<46}dil={rows[-1]['dilution']:<7}rank whole={w_rank} auto={a_rank}  "
              f"oracle={o_rank}  score {rows[-1]['whole_cos']}->{rows[-1]['auto_cos']}->{rows[-1]['oracle_cos']}",
              flush=True)

    stem = scorer.name.replace("/", "_")
    out = args.results_dir / f"decompose_{stem}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(args.results_dir / f"decompose_{stem}_segments.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["plasmid_id", "orf_id", "segment_i", "start_aa", "end_aa", "length_aa", "removed",
                    "payload_rank", "payload_cos", "top1_accession", "top1_cos"])
        w.writerows(seg_rows)

    scored = [r for r in rows if r["whole_rank"] != ""]
    def topk(key, k):
        return sum(1 for r in scored if r[key] != "" and int(r[key]) <= k)
    agg = {
        "scorer": scorer.name, "n_reference": len(refs), "n_plasmids": len(rows), "n_scored": len(scored),
        "orf_count_min_50aa": n_orfs_total, "segment_count_after_decomposition": n_segments_total,
        "query_inflation": round(n_segments_total / n_orfs_total, 3) if n_orfs_total else None,
        "orf_selection_agrees": sum(1 for r in scored if r["orf_selection_agrees"] == "yes"),
        "mean_dilution": round(float(np.mean([r["dilution"] for r in scored])), 4) if scored else None,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    for arm in ("whole", "auto", "auto_noint", "oracle"):
        for k in (1, 5, 10):
            agg[f"{arm}_top{k}"] = topk(f"{arm}_rank", k)
    for arm in ("whole", "auto", "oracle"):
        cs = [r[f"{arm}_cos"] for r in scored if r.get(f"{arm}_cos") not in ("", None)]
        agg[f"mean_cos_{arm}"] = round(float(np.mean(cs)), 4) if cs else None
    # Decomposition does not only rescue the payload: a segment downstream of a 2A peptide gets its own
    # ranked query, and a gene that the whole-ORF embedding averaged into invisibility can light up. For
    # screening that is the more consequential direction: a second coding sequence hidden inside a long
    # fusion ORF is exactly what a whole-construct query would miss.
    surfaced = []
    for s_ in seg_rows:
        pid_, oid_, _, a_, b_, len_, _, _, _, top_, topc_ = s_
        payload_of = next((r["payload_accession"] for r in rows if r["plasmid_id"] == pid_), "")
        if topc_ < args.confident_cos or top_ == payload_of:
            continue
        # How well did the *whole* ORF do on this same protein? That difference is the masking: a gene
        # this segment identifies at rank 1 may sit hundreds of places down the whole-ORF ranking.
        whole_seq = orfs_by_plasmid.get(pid_, {}).get(oid_, "")
        tgt_ = ref_acc.index(top_)
        wr, wc, _, _ = rank_one(whole_seq, scorer, ref_acc, tgt_)
        surfaced.append(dict(plasmid_id=pid_, segment=f"{a_}..{b_}", length_aa=len_, accession=top_,
                             segment_rank=1, segment_cos=topc_,
                             whole_orf_rank=wr, whole_orf_cos=round(wc, 4) if wc is not None else None))
    agg_extra = {
        "confident_cos_threshold": args.confident_cos,
        "nonpayload_genes_surfaced_by_decomposition": surfaced,
        # Low payload coverage means the caller never produced a full-length payload ORF, so the arms for
        # that plasmid measure ORF calling, not decomposition. Kept explicit so the CSV is self-explaining.
        "low_oracle_coverage": [
            dict(plasmid_id=r["plasmid_id"], coverage=r["oracle_coverage"])
            for r in scored if float(r["oracle_coverage"]) < 0.9
        ],
    }
    improved = [r for r in scored if r["auto_rank"] != "" and int(r["auto_rank"]) < int(r["whole_rank"])]
    worsened = [r for r in scored if r["auto_rank"] != "" and int(r["auto_rank"]) > int(r["whole_rank"])]
    agg["auto_improved_rank"] = [r["plasmid_id"] for r in improved]
    agg["auto_worsened_rank"] = [r["plasmid_id"] for r in worsened]
    agg["auto_matched_oracle_span"] = [
        r["plasmid_id"] for r in scored
        if r["auto_best_segment"] and r["oracle_span"] and _close(r["auto_best_segment"], r["oracle_span"])
    ]
    agg.update(agg_extra)
    (args.results_dir / f"decompose_{stem}_summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    n = len(scored)
    print(f"\nScored {n} plasmids with a protein payload; mean dilution {agg['mean_dilution']}")
    print(f"    {'arm':<12}{'top1':>6}{'top5':>6}{'top10':>7}   mean score to payload ({scorer.name})")
    for arm in ("whole", "auto", "auto_noint", "oracle"):
        mc = agg.get(f"mean_cos_{arm}")
        print(f"    {arm:<12}{agg[f'{arm}_top1']:>4}/{n}{agg[f'{arm}_top5']:>4}/{n}{agg[f'{arm}_top10']:>5}/{n}"
              f"      {'-' if mc is None else mc}")
    print(f"\n  decomposition turns {agg['orf_count_min_50aa']} ORFs into "
          f"{agg['segment_count_after_decomposition']} queries ({agg['query_inflation']}x)")
    print(f"  auto improved rank for: {agg['auto_improved_rank'] or 'none'}")
    print(f"  auto worsened rank for: {agg['auto_worsened_rank'] or 'none'}")
    print(f"  auto span ~= oracle span for {len(agg['auto_matched_oracle_span'])}/{n}")
    print(f"  ORF selection agreed with the insert heuristic for {agg['orf_selection_agrees']}/{n}")
    if agg["low_oracle_coverage"]:
        print("  payload ORF incompletely called (these arms measure ORF calling, not decomposition):")
        for d in agg["low_oracle_coverage"]:
            print(f"        {d['plasmid_id'][:44]:<46}payload coverage {d['coverage']}")
    if surfaced:
        print(f"  non-payload genes surfaced by decomposition at cos >= {args.confident_cos}:")
        for d in surfaced:
            print(f"        {d['plasmid_id'][:44]:<46}{d['segment']:<14}{d['accession']}    "
                  f"segment rank 1 (cos {d['segment_cos']})  vs whole-ORF rank {d['whole_orf_rank']}  "
                  f"(cos {d['whole_orf_cos']})")
    print(f"\nruntime {agg['runtime_seconds']}s; wrote {out.name} + _segments.csv + _summary.json")
    return 0


def _close(a: str, b: str, tol: int = 15) -> bool:
    """Do two 'start..end' spans agree to within `tol` residues at both ends?"""
    try:
        a1, a2 = (int(x) for x in a.split("..")); b1, b2 = (int(x) for x in b.split(".."))
    except ValueError:
        return False
    return abs(a1 - b1) <= tol and abs(a2 - b2) <= tol


if __name__ == "__main__":
    raise SystemExit(main())
