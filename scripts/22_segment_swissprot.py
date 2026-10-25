#!/usr/bin/env python3
"""What do decomposition's segments actually hit, when every answer is available to be hit?

    python scripts/22_segment_swissprot.py --swissprot %USERPROFILE%\\Downloads\\uniprot_sprot.fasta.gz

**The caveat this removes.** `06_decompose.py` scored segments against a built catalogue: 5,016 proteins,
assembled as the payloads plus Swiss-Prot decoys. When a segment's top hit was not the payload, that was
read as decomposition surfacing a co-encoded gene, but the catalogue could not contain most of the parts
these constructs carry, so a segment whose true identity was absent had to hit *something*, and whatever it
hit was recorded. Against the full Swiss-Prot the correct answer is present by definition: every tag,
marker, reporter and resistance gene in a synthetic construct is a real, reviewed protein. So the question
becomes answerable rather than suggestive: which of the previously surfaced hits survive once the right
answer exists to compete?

**No confidence gate.** `06_decompose.py` used `--confident-cos` to decide which hits counted. A threshold
on a raw Smith-Waterman score is not a confidence statement, and picking one would decide the answer in
advance. This reports the alignment statistics for every segment, identity, coverage, score, the hit's
own length and description, and lets them speak. A 40-residue segment matching a 1,400 aa protein over
30 columns at 95% identity and 0.02 coverage is visibly not an identification; nothing needs to rule it out
by fiat.

**The whole-ORF rank is the comparison that matters.** For each segment's top hit, the script also reports
where that same accession ranks when the *undecomposed* ORF is the query. That is the quantity the
decomposition claim rests on: a co-encoded gene that ranks 1 from its own segment and 400th
from the whole ORF is one the whole-ORF query hides.

Threaded across cores: parasail releases the GIL, so the 559,680-entry sweep parallelises while all
threads share one copy of the database.

Writes `segments_swissprot.csv` and a per-construct summary to --out-dir.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from pilot.arms import build_arms
from pilot.homology import MAX_LEN, MIN_LEN
from pilot.reference import iter_fasta

GAP_OPEN, GAP_EXTEND = 11, 1


def read_swissprot(path: Path, min_len: int = MIN_LEN, max_len: int = MAX_LEN):
    """(accession, description, sequence). `pilot.homology.iter_swissprot` drops the description, and the
    description is most of the point here: an accession alone does not say whether a hit is a reporter."""
    op = gzip.open if path.suffix == ".gz" else open
    acc = desc = None
    cur: list[str] = []
    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                seq = "".join(cur)
                if acc and min_len <= len(seq) <= max_len:
                    yield acc, desc, seq
                head = line[1:].rstrip("\n")
                acc = head.split("|")[1] if head.count("|") >= 2 else head.split()[0]
                rest = head.split(None, 1)[1] if " " in head else ""
                desc = rest.split(" OS=")[0].strip() if rest else ""
                cur = []
            else:
                cur.append(line.strip())
    seq = "".join(cur)
    if acc and min_len <= len(seq) <= max_len:
        yield acc, desc, seq


def score_all_threaded(query: str, seqs: list[str], threads: int) -> np.ndarray:
    import parasail  # noqa: PLC0415

    m = parasail.blosum62
    n = len(seqs)
    out = np.zeros(n, dtype=np.float32)
    if not query:
        return out

    def run(lo: int, hi: int) -> None:
        f = parasail.sw_striped_sat
        for i in range(lo, hi):
            out[i] = f(query, seqs[i], GAP_OPEN, GAP_EXTEND, m).score

    if threads <= 1:
        run(0, n)
    else:
        import threading  # noqa: PLC0415

        bounds = [(n * k // threads, n * (k + 1) // threads) for k in range(threads)]
        ws = [threading.Thread(target=run, args=b, daemon=True) for b in bounds]
        for w in ws:
            w.start()
        for w in ws:
            w.join()
    return out


def stats(query: str, subject: str) -> tuple[float, float, float]:
    """(score, identity, coverage) for one pair; coverage is against the query, as in pilot.homology."""
    import parasail  # noqa: PLC0415

    r = parasail.sw_trace_striped_sat(query, subject, GAP_OPEN, GAP_EXTEND, parasail.blosum62)
    comp = r.traceback.comp
    if not comp:
        return float(r.score), 0.0, 0.0
    aligned = len(comp)
    return float(r.score), comp.count("|") / aligned, min(aligned / len(query), 1.0)


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
    ap.add_argument("--reference", type=Path, default=Path("data/reference/reference.fasta"),
                    help="only used to look up each payload's sequence, not as the search catalogue")
    ap.add_argument("--swissprot", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=Path("results/segments_swissprot"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="substring of plasmid_id; restricts the run to it")
    args = ap.parse_args(argv)

    for p in (args.swissprot, args.manifest, args.reference, args.results_dir / "orfs.faa"):
        if not p.is_file():
            print(f"not found: {p}", file=sys.stderr)
            return 2

    t0 = time.time()
    with open(args.manifest, newline="", encoding="utf-8-sig", errors="replace") as fh:
        man = {r["filename"].strip(): r for r in csv.DictReader(fh) if (r.get("filename") or "").strip()}
    with open(args.results_dir / "pilot_summary.csv", newline="", encoding="utf-8-sig") as fh:
        summary = list(csv.DictReader(fh))
    orfs_by_plasmid = read_faa(args.results_dir / "orfs.faa")
    payload_seq = {e.accession: e.seq for e in iter_fasta(args.reference)}

    print(f"reading {args.swissprot.name} ...", flush=True)
    entries = list(read_swissprot(args.swissprot))
    accs = [a for a, _, _ in entries]
    descs = {a: d for a, d, _ in entries}
    seqs = [s for _, _, s in entries]
    lens = {a: len(s) for a, _, s in entries}
    idx_of = {a: i for i, a in enumerate(accs)}
    print(f"    {len(entries):,} proteins in {time.time() - t0:.0f}s; "
          f"searching on {args.threads} threads\n", flush=True)

    rows: list[dict] = []
    done = 0
    for srow in summary:
        pid = srow["plasmid_id"]
        acc = (man.get(srow["source_file"], {}).get("payload_accession") or "").strip()
        if not acc or acc not in payload_seq:
            continue
        if args.only and args.only not in pid:
            continue
        arms = build_arms(pid, orfs_by_plasmid.get(pid, {}), payload_seq[acc])
        if arms is None:
            continue
        done += 1
        if args.limit and done > args.limit:
            break
        t = time.time()
        whole_row = score_all_threaded(arms.whole, seqs, args.threads)
        whole_order = np.argsort(-whole_row, kind="stable")
        rank_of = {int(j): r + 1 for r, j in enumerate(whole_order[:50_000])}
        w_top = int(whole_order[0])
        print(f"    {pid[:46]:<48}payload {acc}  ORF {len(arms.whole)}aa  "
              f"{len(arms.auto)} segments  whole-ORF top1 {accs[w_top]} ({descs[accs[w_top]][:40]})",
              flush=True)

        for seg_i, seg in enumerate(arms.auto, 1):
            if not seg.seq:
                continue
            srow_scores = score_all_threaded(seg.seq, seqs, args.threads)
            top = int(np.argmax(srow_scores))
            top_acc = accs[top]
            sc, ident, cov = stats(seg.seq, seqs[top])
            # where the same protein sits when the undecomposed ORF is the query
            w_rank = rank_of.get(top, None)
            w_sc, w_ident, w_cov = stats(arms.whole, seqs[top])
            pay_rank = rank_of.get(idx_of.get(acc, -1), None)
            rows.append(dict(
                plasmid_id=pid, payload_accession=acc, orf_aa=len(arms.whole), segment_i=seg_i,
                start_aa=seg.start, end_aa=seg.end, length_aa=len(seg.seq),
                top_accession=top_acc, top_description=descs.get(top_acc, ""),
                top_length_aa=lens.get(top_acc, 0),
                segment_score=round(sc, 1), segment_identity=round(ident, 4),
                segment_coverage=round(cov, 4),
                whole_orf_rank_of_top=("" if w_rank is None else w_rank),
                whole_orf_score_of_top=round(w_sc, 1),
                whole_orf_identity_of_top=round(w_ident, 4),
                whole_orf_coverage_of_top=round(w_cov, 4),
                is_payload=int(top_acc == acc),
                payload_rank_from_whole_orf=("" if pay_rank is None else pay_rank),
            ))
            print(f"      seg {seg_i}  {seg.start:>5}-{seg.end:<5}  {len(seg.seq):>4}aa -> {top_acc:<8}"
                  f"{descs.get(top_acc, '')[:44]:<46}id {ident:.2f} cov {cov:.2f}    "
                  f"whole-ORF rank {w_rank}", flush=True)
        print(f"      ({time.time() - t:.0f}s)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / (f"segments_swissprot_{args.only}.csv" if args.only else "segments_swissprot.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_payload = sum(r["is_payload"] for r in rows)
    print(f"\n{len(rows)} segments over {done} constructs -> {out}")
    print(f"  segments whose top hit is the payload itself: {n_payload}")
    print(f"  segments whose top hit is something else:     {len(rows) - n_payload}")
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
