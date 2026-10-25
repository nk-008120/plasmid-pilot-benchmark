"""Can a mean-pooled embedding arm run on route B *on this machine*, and what does it say if it does?

    python scripts/49_route_b_kmer_arm.py --cost          # phase 1: price the catalogue build
    python scripts/49_route_b_kmer_arm.py --report         # phase 2: read the run back

**What is blocked and why.** this project records the one thing route B's alignment half does not
settle: *no embedding arm has run on it*, and the thesis is about mean-pooled embeddings. The blocker
is concrete rather than conceptual: route B scores against `results/route_b_homologs/reference_twilight.fasta`,
a **25,402**-protein catalogue (this project's eighteenth row), because 0 of route B's 21 payloads are
payload entries in the 40,750-protein `gate_all173` catalogue every other run uses. An ESM-2 reference
matrix for 25,402 rows exists nowhere in `data/embed_cache/`, which holds 5,016 / 15,205 / 40,750 /
40,750, and building one needs a GPU this machine does not have.

**What this script does instead, and what that is worth.** `pilot.embed.KmerEmbedder` mean-pools too.
It is a hashed 3-mer frequency vector, L2-normalised, scored by cosine, so it exercises the *same
mechanism the thesis names*: one vector for the whole frame, in which a payload and its co-encoded
neighbours are mixed before anything is compared. It is **not** the ESM-2 number and must never be
quoted as one: a k-mer vector knows nothing about homology beyond shared 3-mers, so a failure under it
is overdetermined (mean-pooling *or* composition-blindness could cause it) while a *rescue* under it is
attributable to decomposition alone, because the retrieval scorer is identical across arms. Read the
arms against each other, never the tier against gate_all173 (20.2: this catalogue is family-structured
and ~1.6x harder than its size suggests).

**Phase 1 is a cost measurement and is reported whether or not phase 2 ever runs.** this project's closing paragraph is the reason it exists: a 2-3 h estimate for the alignment run came from a probe
sharing two physical cores with another session, and the real figure was 772.4 s. So the timings here
are taken on random subsamples, extrapolated on *measured* residue throughput rather than on row
count, and the script prints what else was running. A k-mer embedding is O(residues) in pure Python,
one blake2b per k-mer, so residues, not rows, is the quantity that scales.

**Phase 2 reads back the run and reports it the way this project's working rules requires of an embedding arm.** 26.3
stratifies by payload fraction f and forbids a pooled scalar across strata; 32.2 makes `recall@10` and
MRR primary for an *embedding* arm and allows mean rank only beside its count of rank >= 10,000
failures, because mean-pooling makes the rank distribution bimodal where alignment's is not. It also
runs this project's pre-registered test: **more k = 1 falsifiers under mean-pooling, not fewer.**

The retrieval run itself is `scripts/10_twilight.py --scorer kmer --bhs-scorer align`, unchanged. The
hybrid flag is not optional and not a convenience: `best_hit_segments` stops against a floor in raw
BLOSUM62 units, so a cosine segmenter compares numbers in [-1, 1] against a floor near 100 and the arm
degenerates to the whole ORF at k = 0 (10_twilight's `resolve_bhs_scorer`). Keeping segmentation on
exact Smith-Waterman also makes the comparison against the working record *paired*: the same segments, the same
holdouts, the same catalogue, and the retrieval scorer as the only thing that changed.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from pilot.decompose import span_rows_at

ROOT = Path(__file__).resolve().parents[1]

F_STRATA = (0.50, 0.70, 0.95)
STRATUM_NAMES = ("<0.50", "0.50-0.70", "0.70-0.95", ">=0.95")
REPORT_ARMS = ("whole", "auto", "bhs", "oracle", "auto_per_segment", "bhs_per_segment")
# The subsample sizes phase 1 times. Two points, not one, because a single point cannot tell a linear
# cost from a superlinear one and the whole purpose is an extrapolation that is allowed to be trusted.
COST_ROWS = (500, 2000)
RECALL_DEPTH = 10          # fixed by the user 19 Oct; a stated assumption, not a fit


# ---------------------------------------------------------------------------- phase 1: cost

def length_binding(seqs: list[str], k: int = 3) -> dict:
    """Every limit the k-mer embedder imposes, counted whether or not it binds.

    `KmerEmbedder.embed` has **no `max_length`**: that is `Esm2Embedder`'s parameter and the reason
    this project's binding-limit register carries two rows for it (3,889 of 40,750 at 1022; 504 at 2048). The k-mer
    path truncates nothing. What it *does* do is silently drop every character outside the 20-letter
    standard alphabet, and emit an all-zero (hence, after L2, an all-zero) row for any sequence left
    shorter than k. Both are limits in this project's sense: invisible in the output, biting on a minority.
    """
    aa = set("ACDEFGHIKLMNPQRSTVWY")
    lens = [len(s) for s in seqs]
    dropped_rows, dropped_chars, short_rows = 0, 0, 0
    for s in seqs:
        kept = sum(1 for c in s.upper() if c in aa)
        if kept < len(s):
            dropped_rows += 1
            dropped_chars += len(s) - kept
        if kept < k:
            short_rows += 1
    return {"rows": len(seqs), "residues": sum(lens),
            "min_len": min(lens) if lens else 0, "max_len": max(lens) if lens else 0,
            "embedder_max_length": None,
            "rows_truncated_by_a_length_cap": 0,
            "rows_with_non_standard_characters": dropped_rows,
            "characters_dropped": dropped_chars,
            "rows_shorter_than_k": short_rows}


def time_embed(embedder, seqs: list[str], n: int, seed: int = 0) -> dict:
    """Time one random subsample. Random, not the first n: the catalogue is grown from homolog sets, so
    its head is one payload's family and its lengths are not a sample of the whole."""
    rng = random.Random(seed)
    sub = rng.sample(seqs, min(n, len(seqs)))
    res = sum(len(s) for s in sub)
    t = time.perf_counter()
    embedder.embed(sub)
    dt = time.perf_counter() - t
    return {"rows": len(sub), "residues": res, "seconds": round(dt, 3),
            "us_per_residue": round(1e6 * dt / max(res, 1), 4),
            "ms_per_row": round(1e3 * dt / max(len(sub), 1), 3)}


def extrapolate(points: list[dict], total_residues: int) -> dict:
    """Project the full build from measured residue throughput, and report the two points' agreement.

    The agreement *is* the evidence that the extrapolation is admissible: two subsamples whose
    microsecond-per-residue rates differ by a few percent say the cost is linear in residues over a
    4x range, which is what licenses projecting across the remaining 7x. A rate that moves with n
    would say the opposite, and the projection would have to be reported as a lower bound instead.
    """
    rates = [p["us_per_residue"] for p in points]
    slowest = max(rates)
    spread = (max(rates) / min(rates)) if min(rates) > 0 else float("inf")
    return {"points": points,
            "rate_spread": round(spread, 3),
            "linear_in_residues": spread <= 1.25,
            "projected_seconds": round(slowest * total_residues / 1e6, 1),
            "projected_from": "the slowest measured rate, so the projection is an upper bound"}


def run_cost(args) -> int:
    from pilot.embed import get_embedder  # noqa: PLC0415 (phase 2 needs neither import)
    from pilot.reference import iter_fasta  # noqa: PLC0415

    refs = list(iter_fasta(args.reference))
    seqs = [e.seq for e in refs]
    emb = get_embedder(args.embedder)

    print(f"phase 1: costing a {emb.name} reference matrix for {args.reference}")
    print(f"  contention: {args.contention}")
    binding = length_binding(seqs, k=getattr(emb, "k", 3))
    print("  BINDING COUNTS for the catalogue and the embedder's own length handling:")
    for key, val in binding.items():
        print(f"        {key:<40}  {val}")
    print("    (the embedder has no max_length at all, so the this project's working rules rows for ESM-2's 1022 and "
          "2048 caps have no counterpart here; 0 is a measurement, not an omission)")

    points = [time_embed(emb, seqs, n, seed=args.seed) for n in COST_ROWS]
    for p in points:
        print(f"  timed  rows={p['rows']:>6}  residues={p['residues']:>9,}    {p['seconds']:>8.3f}s  "
              f"{p['us_per_residue']:>7.4f} us/residue  {p['ms_per_row']:>7.3f} ms/row")
    proj = extrapolate(points, binding["residues"])
    shape = ("linear in residues" if proj["linear_in_residues"]
             else "NOT linear: treat the projection as a lower bound")
    print(f"  rate spread across the subsamples: {proj['rate_spread']}x ({shape})")
    print(f"  PROJECTED full build: {proj['projected_seconds']:.1f}s for {binding['rows']:,} rows / "
          f"{binding['residues']:,} residues, from {proj['projected_from']}")

    out = {"reference": str(args.reference), "embedder": emb.name, "contention": args.contention,
           "binding": binding, "cost": proj}
    if args.measure_full:
        t = time.perf_counter()
        emb.embed(seqs)
        full = round(time.perf_counter() - t, 1)
        err = (full - proj["projected_seconds"]) / max(proj["projected_seconds"], 1e-9)
        print(f"  MEASURED full build: {full:.1f}s; the projection erred by {err:+.1%}")
        out["measured_full_seconds"] = full
        out["projection_error"] = round(err, 4)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cost.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out_dir / 'cost.json'}")
    return 0


# ---------------------------------------------------------------------------- phase 2: the report

def stratum(f: float) -> str:
    if f < F_STRATA[0]:
        return STRATUM_NAMES[0]
    if f < F_STRATA[1]:
        return STRATUM_NAMES[1]
    if f < F_STRATA[2]:
        return STRATUM_NAMES[2]
    return STRATUM_NAMES[3]


def source_id_of(plasmid_id: str) -> str:
    parts = plasmid_id.split("-")
    return "addgene_" + parts[2] if len(parts) > 2 else plasmid_id


def read_f_by_pair(paths) -> dict:
    out = {}
    for p in paths:
        with open(p, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r.get("f"):
                    out[(r["source_id"], r["payload_accession"])] = float(r["f"])
    return out


def read_ranks(paths) -> list:
    rows = []
    for p in paths:
        with open(p, newline="", encoding="utf-8-sig") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def embedding_metrics(ranks: list[int]) -> dict:
    """The metric order this project's working rules fixes for an **embedding** arm (user, 18 Oct).

    `recall@10` and MRR are primary; mean rank is reported only beside `n_ge_10000`, because under
    mean-pooling the rank distribution is bimodal and the mean stops describing the centre: the
    corpus `whole` mean of 1,088 was median 1 plus 15 constructs of one payload at ~29,699. The dict
    is ordered so that anything printing it in key order prints the primaries first.
    """
    n = len(ranks)
    return {"n": n,
            "recall_at_10": round(sum(1 for r in ranks if r <= RECALL_DEPTH) / n, 3),
            "mrr": round(statistics.fmean(1.0 / r for r in ranks), 3),
            "median_rank": statistics.median(ranks),
            "mean_rank": round(statistics.fmean(ranks), 2),
            "n_ge_10000": sum(1 for r in ranks if r >= 10_000),
            "top1": sum(1 for r in ranks if r == 1)}


def macro(by_payload: dict) -> dict:
    """Average per payload, so one cassette in four backbones counts once (27.4: macro governs)."""
    rec = {a: sum(1 for r in rs if r <= RECALL_DEPTH) / len(rs) for a, rs in by_payload.items()}
    mrr = {a: statistics.fmean(1.0 / r for r in rs) for a, rs in by_payload.items()}
    per = {a: statistics.fmean(rs) for a, rs in by_payload.items()}
    return {"n_payloads": len(per),
            "recall_at_10": round(statistics.fmean(rec.values()), 3),
            "mrr": round(statistics.fmean(mrr.values()), 3),
            "mean_rank": round(statistics.fmean(per.values()), 2)}


def k1_census(rows: list[dict]) -> dict:
    """this project's pre-registered test, counted exactly as 53.1 counted it under alignment.

    A **k = 1 falsifier** is a construct whose segmenter emitted one segment and whose per-segment
    query still beats the whole-ORF query. 53.1 refuted the gate's k = 1 branch by finding six under
    exact Smith-Waterman; 53.7 predicted *more* under mean-pooling, because the residues a single
    segment discards are the ones competing for the rank and a pooled embedding mixes the whole frame.

    `rows` carry `k_bhs`, `rank_whole` and `rank_per_segment`. The k >= 2 column is counted beside it
    for free, since 53.2's claim, no construct in 189 where k >= 2 and `whole` still won, is the
    other half of the same table and would be the louder result if it broke.
    """
    out = {}
    for label, pick in (("k=1", lambda k: k == 1), ("k>=2", lambda k: k >= 2)):
        sub = [r for r in rows if pick(r["k_bhs"])]
        out[label] = {
            "n": len(sub),
            "tie": sum(1 for r in sub if r["rank_whole"] == r["rank_per_segment"]),
            "whole_wins": sum(1 for r in sub if r["rank_per_segment"] > r["rank_whole"]),
            "per_segment_wins": sum(1 for r in sub if r["rank_per_segment"] < r["rank_whole"]),
            "falsifiers": sorted(r["plasmid_id"] for r in sub
                                 if r["rank_per_segment"] < r["rank_whole"]),
        }
    return out


def paired_rows(rank_rows: list[dict], threshold: str, k_by_pair: dict) -> list[dict]:
    """One row per **(construct, payload)** at one holdout: `whole`, `bhs_per_segment` and k_bhs.

    The key is the pair and not the construct, and that is not fussiness. A manifest carries one
    payload per file, so route B's three POLB deposits are scored twice, P06746 in the primary pass
    and P18887 in the second (44_route_b_bench.py), under the *same* `plasmid_id`. Keying on the
    construct alone silently overwrote one pass with the other and inflated those three constructs'
    segment counts to 6; it is 50.5's defect exactly, a choice between two inputs made invisibly, and
    it was caught only because the segment-count cross-check printed a disagreement.

    k comes from the alignment detail CSV **and is cross-checked against this run's own segment count**
    by the caller, not assumed: `--bhs-scorer align` means the segments are the same object, so if the
    two counts ever disagree the hybrid is not what it claims and the pre-registered test is void.
    """
    want = {"whole": "rank_whole", "bhs_per_segment": "rank_per_segment", "oracle": "rank_oracle"}
    by_pair: dict[tuple, dict] = defaultdict(dict)
    for r in rank_rows:
        if r["threshold"] != threshold or r["arm"] not in want or r["best_target_rank"] == "":
            continue
        by_pair[(r["plasmid_id"], r["payload_accession"])][want[r["arm"]]] = \
            int(r["best_target_rank"])
    out = []
    for (pid, acc), v in sorted(by_pair.items()):
        if "rank_whole" in v and "rank_per_segment" in v and (pid, acc) in k_by_pair:
            out.append(dict(plasmid_id=pid, payload_accession=acc,
                            k_bhs=k_by_pair[(pid, acc)], **v))
    return out


def pair_key(row: dict) -> tuple:
    """The join key everywhere in this script: a (construct, payload) pair and never a construct.

    A loop variable named `key` once shadowed this when it was a lambda, and the shadowing surfaced
    only at the last line of a 400-second run. A module-level function cannot be shadowed by a `for`
    in a function body, which is the whole reason it is one.
    """
    return (row["plasmid_id"], row["payload_accession"])


def read_k_bhs(path: Path) -> dict:
    """(construct, payload) -> k_bhs from a `26_bhs_protocol.py --detail-csv` artefact."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return {(r["plasmid_id"], r["payload_accession"]): int(r["k_bhs"])
                for r in csv.DictReader(fh) if r.get("k_bhs")}


def count_segments(paths, threshold: float | None = None) -> dict:
    """k per (construct, payload) from this run's own `bhs_spans_*.csv`: the count of EMITTED spans,
    which is what `k_bhs` means. Keyed by the pair for the reason `paired_rows` documents.

    `threshold` picks one segmentation out of a span file written on or after 20 Oct, which holds one
    per (construct, threshold). Without it such a file would be counted as a single segmentation and
    k_bhs would come out up to six times too large; `span_rows_at` refuses rather than guess (the working record). A file written before then holds one segmentation and `threshold` is ignored.
    """
    out: Counter = Counter()
    seen = set()
    for p in paths:
        if not Path(p).is_file():
            continue
        with open(p, newline="", encoding="utf-8-sig") as fh:
            rows = span_rows_at(list(csv.DictReader(fh)), threshold)
        for r in rows:
            key = (r["plasmid_id"], r["payload_accession"])
            seen.add(key)
            if r["emitted"] == "1":
                out[key] += 1
    for key in seen:
        out.setdefault(key, 0)
    return dict(out)


def run_report(args) -> int:
    f_by_pair = read_f_by_pair(args.manifest)
    rows = read_ranks(args.rank_csv)
    scored = [r for r in rows if r["best_target_rank"] != ""]
    k_align = read_k_bhs(args.align_detail)
    # the same holdout the alignment detail CSV was written at, so the two k columns stay paired
    k_here = count_segments(args.spans_csv, float(args.k_threshold))

    print(f"phase 2: route B under {args.arm_label}, read back from "
          + ", ".join(p.name for p in args.rank_csv))
    print(f"  catalogue: {args.reference} (25, 402 proteins) | "
          f"segmenter: exact Smith-Waterman (--bhs-scorer align) | retrieval: {args.arm_label}")
    print(f"  rank rows {len(rows)} ({len(scored)} scorable); f known for {len(f_by_pair)} pairs")

    # The hybrid's load-bearing claim, checked rather than trusted.
    shared = sorted(set(k_align) & set(k_here))
    disagree = [p for p in shared if k_align[p] != k_here[p]]
    print(f"  BINDING, (construct, payload) pairs whose segment count differs from the alignment "
          f"run's: {len(disagree)} of {len(shared)}  {disagree}")
    print(f"  BINDING, pairs with a k-mer rank but no alignment baseline (so outside the paired "
          f"census): {len(set(k_here) - set(k_align))}  "
          f"{sorted(p for p in set(k_here) - set(k_align))}")
    print(f"  BINDING, route B payloads reachable in the catalogue at all: "
          f"{len({r['payload_accession'] for r in scored})} distinct payloads with at least one "
          f"scorable rank, of {len({a for _, a in f_by_pair})} in the manifests")
    counts: dict = defaultdict(set)
    for r in scored:
        key = (source_id_of(r["plasmid_id"]), r["payload_accession"])
        if key in f_by_pair:
            counts[stratum(f_by_pair[key])].add(key)
    print("  stratum sizes (pairs): "
          + "  ".join(f"{s}  {len(counts[s])}" for s in STRATUM_NAMES))
    missing = sorted({(source_id_of(r["plasmid_id"]), r["payload_accession"]) for r in scored
                      if (source_id_of(r["plasmid_id"]), r["payload_accession"]) not in f_by_pair})
    print(f"  BINDING, pairs with a rank but no f: {len(missing)}  {missing}")
    print("  NEVER pool these rows into one scalar (26.3). recall@10 and MRR are primary here and "
          "mean rank is quoted only beside its >=1e4 count (32.2).\n")

    out: dict = {"arm_label": args.arm_label, "thresholds": {},
                 "segment_count_disagreements": disagree}
    for thr in args.thresholds:
        at = [r for r in scored if r["threshold"] == thr]
        print(f"=== holdout {thr}  " + "=" * 70)
        print("{:<12}{:<20}{:>4}{:>8}{:>7}{:>6}{:>9}{:>7}{:>6} | {:>8}{:>8}{:>7}{:>9}".format(
            "f stratum", "arm", "n", "r@10", "MRR", "med", "mean", ">=1e4", "top1",
            "macro n", "r@10", "MRR", "mean"))
        out["thresholds"][thr] = {}
        for s in STRATUM_NAMES:
            rows_s = [r for r in at
                      if (source_id_of(r["plasmid_id"]), r["payload_accession"]) in f_by_pair
                      and stratum(f_by_pair[(source_id_of(r["plasmid_id"]),
                                             r["payload_accession"])]) == s]
            if not rows_s:
                continue
            out["thresholds"][thr][s] = {}
            for arm in REPORT_ARMS:
                got = [r for r in rows_s if r["arm"] == arm]
                if not got:
                    continue
                ranks = [int(r["best_target_rank"]) for r in got]
                by_payload = defaultdict(list)
                for r in got:
                    by_payload[r["payload_accession"]].append(int(r["best_target_rank"]))
                mi, ma = embedding_metrics(ranks), macro(by_payload)
                out["thresholds"][thr][s][arm] = {"micro": mi, "macro": ma}
                print("{:<12}{:<20}{:>4}{:>8}{:>7}{:>6}{:>9}{:>7}{:>6} | "
                      "{:>8}{:>8}{:>7}{:>9}".format(
                          s, arm, mi["n"], mi["recall_at_10"], mi["mrr"], mi["median_rank"],
                          mi["mean_rank"], mi["n_ge_10000"], mi["top1"],
                          ma["n_payloads"], ma["recall_at_10"], ma["mrr"], ma["mean_rank"]))
            print()

    # ---- this project's pre-registered test
    print("=== the working record, pre-registered: MORE k = 1 falsifiers under mean-pooling, not fewer "
          + "=" * 5)
    here = paired_rows(scored, args.k_threshold, k_here)
    there = []
    with open(args.align_detail, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("rank_whole") and r.get("rank_per_segment"):
                there.append(dict(plasmid_id=r["plasmid_id"],
                                  payload_accession=r["payload_accession"],
                                  k_bhs=int(r["k_bhs"]),
                                  rank_whole=int(r["rank_whole"]),
                                  rank_per_segment=int(r["rank_per_segment"])))
    both = {pair_key(r) for r in here} & {pair_key(r) for r in there}
    cen_here = k1_census([r for r in here if pair_key(r) in both])
    cen_there = k1_census([r for r in there if pair_key(r) in both])
    print(f"  paired on the {len(both)} (construct, payload) pairs scorable under both, at the "
          f"{args.k_threshold} holdout, with the SAME segments")
    print("{:<26}{:>5}{:>6}{:>12}{:>18}".format("run", "n", "tie", "whole wins",
                                                "per_segment wins"))
    for label, cen in (("alignment ", cen_there), (args.arm_label, cen_here)):
        for cell in ("k=1", "k>=2"):
            c = cen[cell]
            print("{:<26}{:>5}{:>6}{:>12}{:>18}".format(
                f"{label}  {cell}", c["n"], c["tie"], c["whole_wins"], c["per_segment_wins"]))
    n_here = cen_here["k=1"]["per_segment_wins"]
    n_there = cen_there["k=1"]["per_segment_wins"]
    verdict = ("CONFIRMED" if n_here > n_there else
               "REFUTED (fewer)" if n_here < n_there else "NOT CONFIRMED (equal)")
    print(f"  k = 1 falsifiers: alignment {n_there}, {args.arm_label}  {n_here} -> {verdict}")
    print(f"  k = 1 falsifiers under {args.arm_label}: {cen_here['k=1']['falsifiers'] or 'none'}")
    out["pre_registered_53_7"] = {"holdout": args.k_threshold, "n_paired": len(both),
                                  "alignment": cen_there, args.arm_label: cen_here,
                                  "verdict": verdict}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "route_b_kmer_strata.json").write_text(json.dumps(out, indent=1),
                                                           encoding="utf-8")
    with open(args.out_dir / "k1_paired.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["plasmid_id", "payload_accession", "k_bhs",
                                           "rank_whole", "rank_per_segment", "rank_oracle",
                                           "rank_whole_align", "rank_per_segment_align"])
        w.writeheader()
        align_by_pair = {pair_key(r): r for r in there}
        for r in here:
            a = align_by_pair.get(pair_key(r), {})
            w.writerow({**{k: r.get(k, "") for k in
                           ("plasmid_id", "payload_accession", "k_bhs", "rank_whole",
                            "rank_per_segment", "rank_oracle")},
                        "rank_whole_align": a.get("rank_whole", ""),
                        "rank_per_segment_align": a.get("rank_per_segment", "")})
    print(f"wrote {args.out_dir / 'route_b_kmer_strata.json'} and "
          f"{args.out_dir / 'k1_paired.csv'}")
    return 0


# ---------------------------------------------------------------------------- cli

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cost", action="store_true", help="phase 1: price the catalogue build")
    ap.add_argument("--report", action="store_true", help="phase 2: read the finished run back")
    ap.add_argument("--embedder", default="kmer")
    ap.add_argument("--measure-full", action="store_true",
                    help="phase 1: also embed the whole catalogue and print the projection's error. "
                         "Only worth it once: the matrix the run uses is built and cached by "
                         "10_twilight.py, which keys the cache on the embedder and not the device.")
    ap.add_argument("--contention", default="unrecorded",
                    help="what else was running while the timings were taken. this project's 10x bad "
                         "estimate came from omitting this, so it is a required field in spirit and "
                         "is written into cost.json.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference", type=Path,
                    default=ROOT / "results/route_b_homologs/reference_twilight.fasta")
    ap.add_argument("--rank-csv", type=Path, nargs="+", default=[
        ROOT / "results/route_b_kmer/twilight_kmer3-1024_route_b_kmer.csv",
        ROOT / "results/route_b_kmer/twilight_kmer3-1024_route_b_kmer_second.csv"])
    ap.add_argument("--spans-csv", type=Path, nargs="+", default=[
        ROOT / "results/route_b_kmer/bhs_spans_kmer3-1024_route_b_kmer.csv",
        ROOT / "results/route_b_kmer/bhs_spans_kmer3-1024_route_b_kmer_second.csv"])
    ap.add_argument("--align-detail", type=Path,
                    default=ROOT / "results/route_b_gate/detail_h25.csv",
                    help="the alignment run's per-construct k_bhs and paired ranks, "
                         "which is the baseline 53.7's prediction is measured against")
    ap.add_argument("--manifest", type=Path, nargs="+", default=[
        ROOT / "data/manifest_route_b.csv", ROOT / "data/manifest_route_b_second.csv"])
    ap.add_argument("--thresholds", nargs="+", default=["0.25", "0.5", "1.0"])
    ap.add_argument("--k-threshold", default="0.25",
                    help="the holdout the k = 1 census runs at; 0.25 is where the alignment detail "
                         "CSV was written, and a census at a different one would not be paired")
    ap.add_argument("--arm-label", default="kmer3-1024")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/route_b_kmer")
    args = ap.parse_args(argv)

    if not (args.cost or args.report):
        ap.error("pass --cost, --report, or both")
    rc = 0
    if args.cost:
        rc = run_cost(args) or rc
    if args.report:
        rc = run_report(args) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
