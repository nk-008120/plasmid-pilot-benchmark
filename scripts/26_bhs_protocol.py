#!/usr/bin/env python3
"""Why a segment arm can find the right span and gain nothing: the aggregation, not the segmentation.

    python scripts/26_bhs_protocol.py --results-dir results/gate173_bhs_hand \
        --reference results/gate_all173/reference_twilight.fasta \
        --homologs results/gate_all173/homologs.csv --threshold 0.25

`10_twilight.py` scores a segment arm by **max-pooling**: each reference gets the best score any segment
gave it, which is what a pipeline submitting every segment as one combined query would see. That choice
is invisible in the output and it is not neutral. A distractor segment, GFP at 98% identity, say,
contributes high scores to every GFP-family entry in the catalogue, and those entries then outrank the
payload's remote homologs just as they did under the whole-ORF query. Max-pooling therefore **puts back
the rank competition decomposition was supposed to remove**, which is the mechanism the working record
identifies as the whole effect.

So this script re-scores the same segments under three protocols, to split the gap into its parts:

* **pooled**: max over segments per reference. What `10_twilight.py` reports. Deployable.
* **per-segment**: every segment is its own query with its own ranking, and the payload counts as found
  if any segment's own ranking finds it. Also deployable (a screen returns one hit list per query), but
  it costs k hit lists to triage instead of one, so its precision cost is real and is reported as k.
  This is the protocol the working record used when it found 6 buried genes surfacing to rank 1.
* **oracle-selected**: the single segment that best overlaps the oracle span, used alone. **This uses
  the label and is a diagnostic, not an arm.** It answers the one question the other two cannot: if
  segment *selection* were free, would these segments reach the ceiling? That separates "the segmenter
  cut in the wrong place" from "the segmenter cut correctly and the protocol threw it away".

Writes nothing. Prints a paired table over the constructs scorable in every protocol.
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from pilot.align import get_scorer
from pilot.arms import build_arms
from pilot.homology import MAX_LEN, MIN_LEN, Homolog, score_floor
from pilot.reference import iter_fasta


def read_faa(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for e in iter_fasta(path):
        oid = e.header.split()[0]
        out[oid.split("|")[0]][oid] = e.seq
    return out


def scorable_frame(twilight_csv: Path, threshold: float) -> list[str]:
    """Construct ids scorable under EVERY arm at `threshold`, read from a finished twilight CSV.

    This is the sampling frame, and it is read from a completed run rather than recomputed, so a probe
    and the full run are drawn from the same population the control run reported. A construct counts as
    scorable only when no arm left `best_target_rank` empty: an arm that cannot be scored would drop the
    construct from the paired table anyway, so sampling constructs that are going to be dropped would
    quietly shrink a probe below its stated n.
    """
    per: dict[str, list[str]] = defaultdict(list)
    with open(twilight_csv, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if abs(float(r["threshold"]) - threshold) < 1e-9:
                per[r["plasmid_id"]].append(r["best_target_rank"])
    return sorted(pid for pid, v in per.items() if v and all(x != "" for x in v))


def select_constructs(available: list[str], *, only: list[str] | None = None,
                      frame: list[str] | None = None, sample: int | None = None,
                      seed: int = 0) -> list[str] | None:
    """Which constructs this run scores: everything, an explicit list, or a seeded sample of a frame.

    Returns None for "no restriction", so the caller can tell "scope everything" from "scope to an empty
    set" instead of silently running a whole tier when a filter matched nothing.

    The sample is drawn from the **sorted** intersection of the frame and what the run actually has, so
    it depends only on the seed and the two sets: not on CSV row order, filesystem order, or dict
    insertion order. A cost probe whose sample moved between invocations would not be a probe of
    anything.
    """
    if only is not None:
        return sorted(set(only) & set(available))
    if frame is None:
        return None
    pool = sorted(set(frame) & set(available))
    if sample is None or sample >= len(pool):
        return pool
    return sorted(random.Random(seed).sample(pool, sample))


def read_id_file(path: Path) -> list[str]:
    """One id per line; blank lines and `#` comments ignored."""
    out = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def load_filler_exclusions(filler_homologs: Path, manifest_rows: list[dict],
                           id_field: str = "construct_id") -> tuple[dict[str, dict[str, float]], dict]:
    """`construct_id -> {catalogue accession: highest identity to any of that construct's fillers}`.

    This is condition B of this project's rebuild: a filler's own relatives are held out of the search
    database for that construct, at the same identity threshold the payload's relatives are held out at.
    The identity kept per accession is the **maximum** over the construct's fillers, because the exclusion
    is a union, an entry close to either filler is removed, and taking the maximum is what makes a
    single threshold comparison implement that union.

    Returns the map and a stats dict, because the thing that must be reported here is not the setting but
    what it binds on: how many entries each construct loses, and therefore how far its score
    floor moves. `N` in `S_min = (ln(K m N) + ln(1/E))/lambda` is the residue count of the database
    ACTUALLY searched, so condition B's floor is genuinely different from condition C's for a reason that
    has nothing to do with the fillers. A 9.95x database ratio moved the floor 8.61 points earlier in this
    project, so this is not a rounding concern.
    """
    rel: dict[str, dict[str, float]] = defaultdict(dict)
    with open(filler_homologs, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            d = rel[r["filler_accession"]]
            acc, ident = r["homolog_accession"], float(r["identity"])
            if ident > d.get(acc, -1.0):
                d[acc] = ident
    out: dict[str, dict[str, float]] = {}
    n_no_filler = 0
    for row in manifest_rows:
        accs = [a for a in (row.get("filler_accessions") or "").split("|") if a]
        if not accs:
            n_no_filler += 1
            out[row[id_field]] = {}
            continue
        merged: dict[str, float] = {}
        for a in accs:
            for acc, ident in rel.get(a, {}).items():
                if ident > merged.get(acc, -1.0):
                    merged[acc] = ident
        out[row[id_field]] = merged
    stats = {"n_fillers_with_relatives": len(rel), "n_constructs": len(out),
             "n_constructs_without_filler": n_no_filler,
             "n_constructs_with_exclusions": sum(1 for v in out.values() if v)}
    return out, stats


class CountingScorer:
    """Pass-through wrapper that counts `score_all` calls and the query residues they carried.

    The cost of this protocol is not one number per construct, it is k full-catalogue passes whose cost
    scales with query length, so a wall-clock figure alone cannot be extrapolated to a different set of
    constructs. Counting the calls and the residues makes an extrapolation a measurement of the cost
    model rather than an assumption about it ("report binding, not settings").
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.name = inner.name
        self.calls = 0
        self.residues = 0

    def score_all(self, query: str):
        self.calls += 1
        self.residues += len(query)
        return self._inner.score_all(query)


def rank_of_best_target(row: np.ndarray, tgt: np.ndarray, mask: np.ndarray) -> int | None:
    if tgt.size == 0:
        return None
    r = row.astype(np.float64, copy=True)
    r[mask] = -np.inf
    best = float(np.max(r[tgt]))
    if not np.isfinite(best):
        return None
    return int((r >= best).sum())


def construct_masks(hs, filler_ident: dict[str, float], idx_of: dict[str, int],
                    acc: str, thr: float) -> tuple[np.ndarray, np.ndarray]:
    """`(scoring_mask, segmenter_mask)` for one construct at one holdout. They are NOT the same array.

    The scoring mask is the holdout: this payload's relatives above `thr`, unioned with condition B's
    filler relatives above the same `thr`. That is the database the arms are ranked in, and the payload's
    own entry does **not** belong in it, because at `thr >= 1.0` the payload is itself a target
    (`identity <= thr`) and masking it would delete the question rather than answer it.

    The segmenter mask is that, plus the payload's own catalogue entry, for the reason
    `pilot.arms.segment_arms` gives and `10_twilight.py:segmentation_mask` implements the same way
. Two cases, and only the second does anything:

    * at every `thr < 1.0` the union is a **provable no-op**: a payload's self-identity is 1.0 and
      `1.0 > thr`, so the scoring mask already holds it. Pinned by a test over the real homolog tables
      rather than argued.
    * at `thr >= 1.0` the predicate `identity > thr` is **empty**, so without the union the segmenter is
      handed an unmasked catalogue containing the payload at 100% identity. It then claims exactly the
      payload's span in iteration 1 and reproduces `oracle` while still reading as label-free. this project measured that on this script's one `--threshold 1.0` run: `n_payload_masked` 0 on all 43
      constructs and the segmenter's best hit **was** the payload on **28 of 43**, against 0 of 43 at
      0.25 and 0.50.

    So the asymmetry is the whole content of this function, and splitting the two masks is what lets the
    1.00 column be answered at all. `db_residues` is left on the scoring mask: the segmenter searching
    one protein fewer moves its floor by `ln` of one part in 40,750, which is below the floor's own
    rounding and is not worth a second denominator (this project's working rules on both sides of a ratio).
    """
    payload_excl = {idx_of[h.accession] for h in hs if h.identity > thr and h.accession in idx_of}
    fill_excl = {idx_of[a] for a, ident in filler_ident.items() if ident > thr and a in idx_of}
    scoring = payload_excl | fill_excl
    own = {idx_of[acc]} if acc in idx_of else set()
    return (np.array(sorted(scoring), dtype=int),
            np.array(sorted(scoring | own), dtype=int))


def stats(d: dict[str, int], keys: list[str]) -> tuple[int, float, float]:
    v = [d[k] for k in keys]
    return sum(1 for x in v if x == 1), sum(1 / x for x in v) / len(v), sum(v) / len(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--homologs", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--scorer", default="align")
    ap.add_argument("--max-evalue", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=1,
                    help="OS threads for the Smith-Waterman score pass. parasail releases the GIL, so "
                         "this is real parallelism; the arrays are bitwise identical to the serial path "
                         "(tests/test_align_threads.py), so it changes the clock and nothing else.")
    ap.add_argument("--filler-homologs", type=Path, default=None,
                    help="CONDITION B. From scripts/32_filler_homologs.py. For each construct, every "
                         "catalogue entry more than --threshold identical to one of ITS OWN fillers is "
                         "removed from the search database, exactly as the payload's relatives are. "
                         "This is an explicitly reported exclusion ON TOP of the holdout mask, not a "
                         "replacement for it, and --mask-homologs is still not passed.")
    ap.add_argument("--condition", default="",
                    help="a label written into the detail and cost CSVs: random | real_masked | "
                         "real_competing. Keeps the three arms' artefacts distinguishable.")
    ap.add_argument("--only", type=Path, default=None,
                    help="file of plasmid_ids, one per line, to restrict the run to")
    ap.add_argument("--frame", type=Path, default=None,
                    help="a finished twilight CSV; the sampling frame is its constructs scorable under "
                         "every arm at --threshold. Required by --sample.")
    ap.add_argument("--sample", type=int, default=None,
                    help="draw this many constructs from --frame, seeded by --seed. Omit to run the "
                         "whole frame.")
    ap.add_argument("--seed", type=int, default=0,
                    help="the sample's seed; the sample is otherwise independent of file ordering")
    ap.add_argument("--detail-csv", type=Path, default=None,
                    help="write one row per construct: every protocol's rank, the labelled payload "
                         "fraction f, the segment count, and the fraction of the ORF covered by the "
                         "segmenter's own best single hit. Scoring this protocol costs an hour, so the "
                         "analysis is deliberately separated from it: `30_bhs_strata.py` reads this "
                         "file and needs no re-scoring to re-table or re-threshold anything.")
    ap.add_argument("--cost-csv", type=Path, default=None,
                    help="write per-construct wall-clock, score_all calls and query residues here, so "
                         "the cost of a probe can be extrapolated from its own cost model")
    args = ap.parse_args(argv)
    if args.sample is not None and args.frame is None:
        ap.error("--sample needs --frame")

    refs = list(iter_fasta(args.reference))
    ref_acc = [e.accession for e in refs]
    idx_of = {a: i for i, a in enumerate(ref_acc)}
    ref_by_acc = {e.accession: e.seq for e in refs}
    seqs = [e.seq for e in refs]
    db_residues = sum(len(e.seq) for e in refs if MIN_LEN <= len(e.seq) <= MAX_LEN)

    with open(args.manifest, newline="", encoding="utf-8-sig", errors="replace") as fh:
        man = {r["filename"].strip(): r for r in csv.DictReader(fh) if (r.get("filename") or "").strip()}
    with open(args.results_dir / "pilot_summary.csv", newline="", encoding="utf-8-sig") as fh:
        summary = list(csv.DictReader(fh))
    orfs_by_plasmid = read_faa(args.results_dir / "orfs.faa")
    hom: dict[str, list[Homolog]] = defaultdict(list)
    with open(args.homologs, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            hom[r["payload_accession"]].append(
                Homolog(r["homolog_accession"], float(r["score"]), float(r["identity"]),
                        float(r["coverage"]), int(r["length"])))

    thr = args.threshold
    scorer = CountingScorer(get_scorer(args.scorer, seqs, cache_dir=None, threads=args.threads))
    print(f"catalogue: {args.reference}, {len(refs):,} proteins, {db_residues:,} residues in the "
          f"{MIN_LEN}-{MAX_LEN} aa window | holdout {thr:.0%} | scorer {scorer.name}")
    print(f"targets and holdout mask: {args.homologs} (the same gated file, split by the holdout)")
    if args.threads > 1:
        print(f"score pass over {args.threads} threads (bitwise identical to serial; see "
              f"tests/test_align_threads.py)")

    filler_excl: dict[str, dict[str, float]] = {}
    if args.filler_homologs:
        man_rows = list(csv.DictReader(
            open(args.manifest, newline="", encoding="utf-8-sig", errors="replace")))
        key = "construct_id" if man_rows and "construct_id" in man_rows[0] else "filename"
        filler_excl, fstats = load_filler_exclusions(args.filler_homologs, man_rows, id_field=key)
        print(f"CONDITION B: filler relatives from {args.filler_homologs}  "
              f"({fstats['n_fillers_with_relatives']} fillers have relatives); "
              f"{fstats['n_constructs_with_exclusions']} of {fstats['n_constructs']} constructs carry an "
              f"exclusion set, {fstats['n_constructs_without_filler']} carry no filler at all")

    available = [srow["plasmid_id"] for srow in summary]
    frame = scorable_frame(args.frame, thr) if args.frame else None
    wanted = select_constructs(available, only=read_id_file(args.only) if args.only else None,
                               frame=frame, sample=args.sample, seed=args.seed)
    if wanted is None:
        print(f"scope: every construct in {args.results_dir.name} ({len(available)})\n")
    else:
        print(f"scope: {len(wanted)} of {len(available)} constructs"
              + (f"; frame {len(frame)} scorable at {thr:.0%} in {args.frame.name}" if frame else "")
              + (f"; sampled {args.sample} at seed {args.seed}" if args.sample else "") + "\n")
    wanted_set = None if wanted is None else set(wanted)
    costs: list[dict] = []
    details: list[dict] = []

    out: dict[str, dict[str, int]] = {p: {} for p in ("whole", "auto_pooled", "auto_per_segment",
                                                      "pooled", "per_segment", "oracle_selected",
                                                      "oracle")}
    nseg: dict[str, int] = {}
    detail = []
    for srow in summary:
        pid = srow["plasmid_id"]
        if wanted_set is not None and pid not in wanted_set:
            continue
        acc = (man.get(srow["source_file"], {}).get("payload_accession") or "").strip()
        if not acc or acc not in ref_by_acc:
            continue
        t_con, calls0, res0 = time.time(), scorer.calls, scorer.residues
        hs = hom.get(acc, [])
        payload_excl = {idx_of[h.accession] for h in hs
                        if h.identity > thr and h.accession in idx_of}
        # Condition B: this construct's own fillers' relatives, above the SAME identity threshold the
        # payload's relatives are held out at. Reported per construct, never silent.
        fill_excl = {idx_of[a] for a, ident in filler_excl.get(pid, {}).items()
                     if ident > thr and a in idx_of}
        excl, seg_excl = construct_masks(hs, filler_excl.get(pid, {}), idx_of, acc, thr)
        # `N` is the database ACTUALLY searched, so removing entries moves the floor. Computed per
        # construct rather than once, which is the whole reason condition B's floor differs from C's.
        db_res_here = db_residues
        if fill_excl:
            db_res_here = db_residues - sum(
                len(seqs[i]) for i in (payload_excl | fill_excl)
                if MIN_LEN <= len(seqs[i]) <= MAX_LEN)
        arms = build_arms(pid, orfs_by_plasmid.get(pid, {}), ref_by_acc[acc],
                          ref_seqs=seqs, scorer=scorer, db_residues=db_res_here,
                          ref_accessions=ref_acc, max_evalue=args.max_evalue, exclude_idx=seg_excl)
        if arms is None:
            continue
        tgt_all = [idx_of[h.accession] for h in hs
                   if h.identity <= thr and h.accession in idx_of]
        # A target that is ALSO a filler relative is gone from the database, so it cannot be retrieved.
        # That changes the target set between conditions B and C, which would break the pairing silently,
        # so it is counted and reported rather than discovered later.
        tgt = np.array([i for i in tgt_all if i not in fill_excl], dtype=int)
        n_targets_lost = len(tgt_all) - int(tgt.size)
        mask = excl
        seg_rows = [(s, scorer.score_all(s.seq)) for s in arms.bhs if s.seq]
        if not seg_rows:
            continue
        r_whole = rank_of_best_target(scorer.score_all(arms.whole), tgt, mask)
        # the SAME two protocols applied to the motif arm, which is the control that decides whether the
        # protocol or the segmenter is doing the work
        auto_rows = [scorer.score_all(s.seq) for s in arms.auto if s.seq]
        a_pooled = (rank_of_best_target(np.max(np.vstack(auto_rows), axis=0), tgt, mask)
                    if auto_rows else None)
        a_per = [rank_of_best_target(r, tgt, mask) for r in auto_rows]
        a_per_ok = [x for x in a_per if x is not None]
        a_best = min(a_per_ok) if a_per_ok else None
        r_oracle = rank_of_best_target(scorer.score_all(arms.oracle), tgt, mask) if arms.oracle else None
        pooled = rank_of_best_target(np.max(np.vstack([r for _, r in seg_rows]), axis=0), tgt, mask)
        per_seg = [rank_of_best_target(r, tgt, mask) for _, r in seg_rows]
        per_seg_ok = [x for x in per_seg if x is not None]
        best_per_seg = min(per_seg_ok) if per_seg_ok else None
        # oracle-selected: the segment overlapping the oracle span most. USES THE LABEL: diagnostic.
        osp = arms.span
        sel = None
        if osp is not None:
            ov = [(max(0, min(s.end, osp.end) - max(s.start, osp.start) + 1), i)
                  for i, (s, _) in enumerate(seg_rows)]
            best_ov, bi = max(ov)
            if best_ov > 0:
                sel = rank_of_best_target(seg_rows[bi][1], tgt, mask)
        vals = dict(whole=r_whole, auto_pooled=a_pooled, auto_per_segment=a_best,
                    pooled=pooled, per_segment=best_per_seg, oracle_selected=sel,
                    oracle=r_oracle)
        if any(v is None for v in vals.values()):
            continue
        for k, v in vals.items():
            out[k][pid] = v
        nseg[pid] = len(seg_rows)
        detail.append((pid, acc, vals, len(seg_rows), per_seg, len(auto_rows)))
        bres = arms.bhs_result
        # The segmenter's own best single hit: the trace entry with the highest alignment score, and the
        # fraction of the ORF its claimed span covers. This is the label-free counterpart of f: a
        # screener can compute it, because it asks only "how much of this ORF did my best catalogue hit
        # account for?" and never consults the payload. Taken as the max over the trace rather than as
        # iteration 1 because that is what the quantity means; `best_hit_is_iter1` records whether the
        # two coincide, which they should, instead of assuming it.
        best = max(bres.trace, key=lambda t: t.score) if (bres and bres.trace) else None
        orf_n = len(arms.orf_seq)
        details.append(dict(
            plasmid_id=pid, payload_accession=acc, orf_len=orf_n,
            oracle_len=(arms.span.length if arms.span else ""),
            f=(round(arms.span.length / orf_n, 6) if arms.span and orf_n else ""),
            k_bhs=len(seg_rows), k_auto=len(auto_rows),
            bhs_iterations=(None if bres is None else bres.n_iterations),
            bhs_stop=(None if bres is None else bres.stop_reason),
            best_hit_accession=("" if best is None else best.ref_accession),
            best_hit_score=("" if best is None else round(best.score, 1)),
            best_hit_start=("" if best is None else best.start),
            best_hit_end=("" if best is None else best.end),
            best_hit_cov=("" if best is None or not orf_n
                          else round((best.end - best.start + 1) / orf_n, 6)),
            best_hit_is_iter1=("" if best is None else int(best.iteration == 1)),
            best_hit_emitted=("" if best is None else int(best.emitted)),
            bhs_fellback=(0 if (bres and bres.segments) else 1),
            condition=args.condition,
            n_filler_excluded=len(fill_excl),
            n_payload_masked=len(payload_excl),
            n_segmenter_masked=int(seg_excl.size),
            payload_entry_masked_for_segmenter=int(acc in idx_of),
            n_targets=int(tgt.size),
            n_targets_lost_to_filler_masking=n_targets_lost,
            db_residues_searched=db_res_here,
            floor=(None if bres is None else round(bres.floor, 2)),
            floor_shift_vs_full_db=round(
                score_floor(args.max_evalue, max(len(arms.orf_seq), 1), db_res_here)
                - score_floor(args.max_evalue, max(len(arms.orf_seq), 1), db_residues), 3),
            **{f"rank_{k}": v for k, v in vals.items()},
            per_segment_ranks=" ".join("-" if x is None else str(x) for x in per_seg)))
        costs.append(dict(plasmid_id=pid, payload_accession=acc, orf_len=len(arms.orf_seq),
                          seconds=round(time.time() - t_con, 2),
                          score_all_calls=scorer.calls - calls0,
                          query_residues=scorer.residues - res0,
                          bhs_iterations=(None if bres is None else bres.n_iterations),
                          bhs_stop=(None if bres is None else bres.stop_reason),
                          k_bhs=len(seg_rows), k_auto=len(auto_rows)))
        c = costs[-1]
        print(f"    {pid[:40]:<42}{acc:<9}orf={c['orf_len']:>5}  iters={c['bhs_iterations']}    "
              f"k={c['k_bhs']}  calls={c['score_all_calls']:>3}  qres={c['query_residues']:>6}    "
              f"{c['seconds']:>8.2f}s", flush=True)

    keys = sorted(set.intersection(*[set(d) for d in out.values()]))
    n = len(keys)
    print(f"PAIRED on the {n} constructs scorable under every protocol.\n")
    print(f"{'protocol':<18}{'label-free':<12}{'top-1':>8}{'MRR':>9}{'mean rank':>12}"
          f"{'d top-1':>9}{'d MRR':>9}{'d mean':>9}   queries")
    base = stats(out["whole"], keys)
    rows_meta = [
        ("whole", "yes", "1"),
        ("auto_pooled", "yes", "1 (max over motif segs)"),
        ("auto_per_segment", "yes", "k_auto, each ranked"),
        ("pooled", "yes", "1 (max over k)"),
        ("per_segment", "yes", "k, each ranked"),
        ("oracle_selected", "NO, label", "1 of k, chosen by label"),
        ("oracle", "NO, label", "1, the answer's span"),
    ]
    for name, free, q in rows_meta:
        t1, mrr, mean = stats(out[name], keys)
        print(f"{name:<18}{free:<12}{t1:>5}/{n:<2}{mrr:>9.3f}{mean:>12.2f}"
              f"{t1 - base[0]:>+9}{mrr - base[1]:>+9.3f}{mean - base[2]:>+9.2f}      {q}")

    print(f"\nsegments per construct: mean {sum(nseg[k] for k in keys) / n:.2f}, "
          f"max {max(nseg[k] for k in keys)}, "
          f"distribution { {v: sum(1 for k in keys if nseg[k] == v) for v in sorted({nseg[k] for k in keys})} }")

    print(f"\n{'construct':<42}{'payload':<9}{'whole':>7}{'autoSeg':>8}{'pooled':>8}{'perSeg':>8}"
          f"{'oracSel':>9}{'oracle':>8}{'k':>3}{'kAuto':>6}   per-segment ranks (bhs)")
    for pid, acc, v, k, per, ka in sorted(detail, key=lambda x: x[0]):
        if pid not in set(keys):
            continue
        print(f"{pid[:40]:<42}{acc:<9}{v['whole']:>7}{v['auto_per_segment']:>8}{v['pooled']:>8}"
              f"{v['per_segment']:>8}{v["oracle_selected"]:>9}{v["oracle"]:>8}{k:>3}{ka:>6}      "
              f"{[x if x is not None else '-' for x in per]}")

    if args.detail_csv and details:
        # Only the paired constructs: a row whose protocols are not all scorable is not in any table
        # above, so writing it here would let a later analysis quietly use a different n than the
        # headline does. That is the denominator bug in a new costume.
        keep = [d for d in details if d["plasmid_id"] in set(keys)]
        args.detail_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.detail_csv, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(keep[0]))
            w.writeheader()
            w.writerows(keep)
        n_i1 = sum(1 for d in keep if d["best_hit_is_iter1"] == 1)
        print(f"\nwrote {args.detail_csv}: {len(keep)} paired constructs of {len(details)} scored; "
              f"the best-scoring hit is iteration 1 for {n_i1} of {len(keep)}")

    if filler_excl and details:
        ex = [d for d in details]
        nz = [d for d in ex if d["n_filler_excluded"]]
        lost = sum(d["n_targets_lost_to_filler_masking"] for d in ex)
        print(f"\nCONDITION B's CONFOUND, reported rather than avoided:")
        print(f"  filler-relative exclusions: {len(nz)} of {len(ex)} constructs affected; per construct "
              f"min {min((d['n_filler_excluded'] for d in nz), default=0)}, "
              f"max {max((d['n_filler_excluded'] for d in nz), default=0)}")
        dbs = sorted({d["db_residues_searched"] for d in ex})
        print(f"  distinct databases actually searched: {len(dbs)}  "
              f"(residues {dbs[0]:,} to {dbs[-1]:,}; full catalogue {db_residues:,})")
        shifts = [d["floor_shift_vs_full_db"] for d in ex]
        print(f"  score-floor shift vs the full database: min {min(shifts):+.3f}, max {max(shifts):+.3f}  "
              f"points: NEGATIVE means a LOWER floor, so the segmenter iterates MORE in B than in C "
              f"for a reason unrelated to the fillers")
        print(f"  targets removed because they were also a filler relative: {lost} across all constructs"
              + ("" if lost == 0 else "  <: B and C are NOT scored against the same target set for "
                                      "these; they must be excluded from the paired comparison"))

    if costs:
        tot = sum(c["seconds"] for c in costs)
        qres = sum(c["query_residues"] for c in costs)
        scored_ids = {c["plasmid_id"] for c in costs}
        print(f"\nCOST, measured on the {len(costs)} constructs actually scored "
              f"({len(scored_ids & set(keys))} of them in the paired table):")
        print(f"  wall-clock {tot:.1f}s total, {tot / len(costs):.1f}s per construct "
              f"(min {min(c['seconds'] for c in costs):.1f}, max {max(c['seconds'] for c in costs):.1f})")
        print(f"  score_all calls {sum(c['score_all_calls'] for c in costs)}, query residues {qres:,}, "
              f"so {tot / max(1, qres) * 1e6:.1f}s per million query residues against "
              f"{db_residues:,} catalogue residues")
        it = [c["bhs_iterations"] for c in costs if c["bhs_iterations"] is not None]
        if it:
            print(f"  bhs iterations {{{', '.join(f'{k}: {it.count(k)}' for k in sorted(set(it)))}}}  "
                  f"(mean {sum(it) / len(it):.2f}, max {max(it)})")
        kk = [c["k_bhs"] for c in costs]
        print(f"  segments per construct {{{', '.join(f'{k}: {kk.count(k)}' for k in sorted(set(kk)))}}}  "
              f"(mean {sum(kk) / len(kk):.2f}, max {max(kk)}); "
              f"{sum(1 for x in kk if x > 1)} of {len(kk)} produce more than one segment")
        if args.cost_csv:
            args.cost_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(args.cost_csv, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=list(costs[0]))
                w.writeheader()
                w.writerows(costs)
            print(f"  wrote {args.cost_csv}")

    # the one question this script exists to answer
    print("\nWHERE THE GAP GOES (paired, n=%d):" % n)
    for name, label in (("auto_per_segment", "the motif arm, each segment ranked on its own"),
                        ("pooled", "max-pooling the segments"),
                        ("per_segment", "each segment ranked on its own"),
                        ("oracle_selected", "the right segment, chosen by the label")):
        t1, mrr, mean = stats(out[name], keys)
        o1, omrr, omean = stats(out["oracle"], keys)
        print(f"    {label:<42} top-1 {t1:>2}/{n}  MRR {mrr:.3f}  mean {mean:>7.2f}"
              f"   | ceiling {o1}/{n}, {omrr:.3f}, {omean:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
