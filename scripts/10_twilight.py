#!/usr/bin/env python3
"""The twilight-zone benchmark: can the payload's family still be found when nothing close is in the catalogue?

    python scripts/10_twilight.py --scorer align    # the alignment control
    python scripts/10_twilight.py --scorer kmer     # the embedding baseline
    python scripts/10_twilight.py --scorer facebook/esm2_t33_650M_UR50D   # the real run
    python scripts/10_twilight.py --scorer facebook/esm2_t33_650M_UR50D --bhs-scorer align   # the hybrid

Needs `scripts/09_find_homologs.py` to have been run first.

**Why this exists.** The Step 2 benchmark kept each payload in the reference set, so the right answer sat
there at 100% identity and exact Smith-Waterman scored 19/20. That measures string matching, not screening.
Here the reference set is progressively stripped of the payload's close relatives: at a holdout of 50%,
every protein more than 50% identical to the payload is removed, and success means retrieving one of the
genuine homologs that remain. Sweeping the holdout down to 25% walks the query into the twilight zone,
where alignment is expected to fail and a function-based method is supposed to keep working. That claim is
the proposal's, and this script is what turns it into a curve.

Writes `twilight_<scorer>.csv`, `_summary.json`, and `figures/fig_d4_twilight_curve.*`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# `best_target_rank` moved to `pilot.align` on 20 Oct so that `scripts/51_fragment_probe.py` ranks by
# the same function rather than its own copy of it. Imported under its own name, so this module still
# exposes `best_target_rank` to the tests that reach for it here.
from pilot.align import AlignmentScorer, best_target_rank, get_scorer, model_kwargs
from pilot.arms import (
    ARM_NAMES,
    PER_SEGMENT_ARMS,
    REPORT_ARM_NAMES,
    build_arms,
    segment_arms,
)
from pilot.benchmark import best_identity_left, holdout_label_overstated
from pilot.homology import MAX_LEN, MIN_LEN, Homolog, score_floor
from pilot.reference import iter_fasta

THRESHOLDS = (1.00, 0.90, 0.70, 0.50, 0.35, 0.25)
# The top_k grid the prefilter recall curve is reported on, for a two-stage scorer (`--scorer eba:...`).
# Not a tunable: it is the x-axis of the measurement that says whether a shortlist can be small enough
# for the two-stage design to pay (open item 32). Free: the curve is derived from
# one recorded rank per construct per arm.
PREFILTER_RECALL_AT = (200, 500, 2000, 5000, 10000)


def read_faa(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for e in iter_fasta(path):
        oid = e.header.split()[0]
        out[oid.split("|")[0]][oid] = e.seq
    return out


def load_homologs(path: Path) -> dict[str, list[Homolog]]:
    out: dict[str, list[Homolog]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            out[r["payload_accession"]].append(
                Homolog(r["homolog_accession"], float(r["score"]), float(r["identity"]),
                        float(r["coverage"]), int(r["length"])))
    return out


def pool_scores(arm_rows: list[np.ndarray]) -> np.ndarray:
    """Max-pooled score row: the best score each reference gets from any of the arm's queries.

    This is what `auto` and `bhs` have always meant in the twilight CSVs, and every comparison in
    the working record reads them that way. It is pinned by a test rather than left to a later refactor's
    judgement, because a change of aggregation here would be a change of meaning with no change of
    column name, which is the one defect in this repo's history that no output would have shown.
    """
    return arm_rows[0] if len(arm_rows) == 1 else np.max(np.vstack(arm_rows), axis=0)


def per_segment_rank(arm_rows: list[np.ndarray], target_idx: np.ndarray,
                     mask_idx: np.ndarray) -> int | None:
    """Best rank any single query reaches when each is ranked in its own hit list.

    The same segments, the same scorer and the same score rows as `pool_scores`; only the aggregation
    differs, so this is pure ranking and works for a cosine scorer exactly as it does for alignment.
    Pooling re-admits every score a distractor segment earned into one ranked list, so the co-encoded
    gene's family floods the top of it exactly as it did under the whole-ORF query. Measured three
    times, on three tiers: 19.5 mean-rank points on the hand tier, +31 against
    `whole` where per-segment is -589 on the corpus, and pooled mean rank 348.88 against
    per-segment 3.41 on identical segments in the real-filler synthetic condition C
    (`results/synthetic_real/C_detail_238.csv`, not yet in CONTEXT).

    It costs no catalogue pass: the rows are already computed for the pooled arm, and each one is ranked
    by the same single comparison pass `best_target_rank` already makes. A one-query arm is identical
    under both aggregations, which is why the run prints how many constructs produce more than one.

    `None` only when no query could be scored at all, matching `best_target_rank`'s convention and
    `scripts/26_bhs_protocol.py`'s `min` over the segment ranks that are not None.
    """
    ranks = []
    for row in arm_rows:
        # an explicit loop, not a comprehension: an assignment expression inside a comprehension binds
        # in the ENCLOSING scope, which is how a `:=` once rebound this script's `rows` results list
        r = best_target_rank(row, target_idx, mask_idx)
        if r is not None:
            ranks.append(r)
    return min(ranks) if ranks else None


def macro_avg(per_payload: list[list[int]], fn, nd: int) -> float | None:
    """`fn` applied within each payload, then averaged over payloads, each payload counting once.

    410 corpus constructs carry 138 distinct payloads and one appears in 37 of them, so a micro average
    is 37 repetitions of one answer. Both are always reported.
    """
    if not per_payload:
        return None
    return round(sum(fn(v) for v in per_payload) / len(per_payload), nd)


def resolve_bhs_scorer(mode: str, scorer, ref_seqs: list[str], threads: int = 1):
    """The scorer that *segments*, which need not be the scorer that *retrieves*.

    `best_hit_segments` stops when the best remaining hit falls below `homology.score_floor`, which is
    the significance gate inverted into **raw BLOSUM62 score units**. Hand it a cosine scorer and it
    compares numbers in [-1, 1] against a floor of 99.5-116.1, so every construct stops at k = 0 and the
    arm silently degenerates to the whole ORF. That is the mechanism behind this project's `--no-bhs`,
    and it is what blocked the measurement the thesis is about: whether the label-free decomposition step
    rescues an *embedding* screen the way it rescues an alignment one (38.1 is exact Smith-Waterman only).

    `mode="align"` keeps segmentation on exact Smith-Waterman, where the floor is derived and validated,
    while retrieval runs under `--scorer`. The segmenter still never sees the label, `build_arms`
    passes it the ORF, the catalogue and the mask, never `payload_seq`, so the hybrid is label-free
    throughout, which is the arrangement declared at 34.6.

    **The no-op is by object identity, not by equivalence.** When the run scorer already aligns, this
    returns that very object, so `--scorer align --bhs-scorer align` cannot drift from `--scorer align`:
    there is no second code path for it to drift along. Open item 42.
    """
    if mode == "run" or isinstance(scorer, AlignmentScorer):
        return scorer
    return get_scorer("align", ref_seqs, threads=threads)


def _scorer_from_span_name(stem: str, tier: str) -> str:
    """The segmenting scorer a span file was written by, recovered from its name.

    `10_twilight.py` names the file `bhs_spans_<scorer>_<tier>.csv`, and neither part is quoted, so the
    tier has to come off the end explicitly rather than by splitting on the last underscore: scorer
    names contain underscores too (`esm2_t33_650M_UR50D`). Reported rather than trusted: it goes into
    the summary as provenance and nothing branches on it.
    """
    name = stem[len("bhs_spans_"):] if stem.startswith("bhs_spans_") else stem
    if tier and name.endswith("_" + tier):
        name = name[: -(len(tier) + 1)]
    return name


def segmentation_mask(mask_homologs: dict[str, list[Homolog]], idx_of: dict[str, int],
                      acc: str, thr: float, *, protocol: str = "per-threshold",
                      thresholds: tuple[float, ...] = THRESHOLDS) -> np.ndarray:
    """The catalogue entries the SEGMENTER may not claim a span against, at this holdout.

    The same predicate the scoring mask uses, `identity > thr`, so segmentation and scoring see
    the same catalogue. That identity is the whole of this project's fix; before 20 Oct this was built
    once at `min(THRESHOLDS)` and reused across the sweep, which on the corpus tier at the 0.70
    holdout manufactured 14,717 rank points of loss and 6 of gain.

    Union the payload's own accession, and note exactly what that union is and is not:

    * at every `thr < 1.0` it is provably a **no-op**, because a payload's self-identity is 1.0 and
      `1.0 > thr`, so the predicate has already masked it. Pinned by a test over every payload of
      both homolog tables rather than argued.
    * at `thr >= 1.0` the predicate is **empty**, and an empty segmentation mask is the leak
      `best_hit_segments` documents as a correctness requirement rather than a tuning knob: the
      segmenter would find the payload sitting in the catalogue at 100% identity, claim exactly its
      span in iteration 1, and reproduce `oracle` while appearing to be label-free. So this union is
      what makes the 1.00 column answerable at all. `26_bhs_protocol.py` has no such guard, and its
      one `--threshold 1.0` run leaks exactly this way on 28 of 43 constructs.

    The 1.00 holdout stays a genuinely different question either way: its *scoring* mask is empty and
    its target set is the payload accession alone, which is what makes ranks non-monotonic across the
    sweep. That is the benchmark's design, not an artefact to smooth out.
    """
    cut = min(thresholds) if protocol == "strictest" else thr
    idx = {idx_of[h.accession] for h in mask_homologs.get(acc, [])
           if h.identity > cut and h.accession in idx_of}
    if acc in idx_of:
        idx.add(idx_of[acc])
    return np.array(sorted(idx), dtype=int)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results/baseline2"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--reference", type=Path, default=Path("data/reference/reference_twilight.fasta"))
    ap.add_argument("--homologs", type=Path, default=Path("data/reference/homologs.csv"),
                    help="defines the TARGETS: which proteins count as retrieving the payload's family")
    ap.add_argument("--mask-homologs", type=Path, default=None,
                    help="defines the MASK: which proteins are removed from the catalogue at each "
                         "holdout. Defaults to --homologs, which is right when one file does both jobs. "
                         "They must differ when the target set is filtered (an E-value gate, say): the "
                         "mask has to stay the full set, or close relatives the gate dropped would be "
                         "left in the catalogue as distractors while not counting as targets either.")
    ap.add_argument("--scorer", default="align")
    ap.add_argument("--max-evalue", type=float, default=1e-3,
                    help="significance gate for the bhs arm's stopping rule. Inverted to a raw-score "
                         "floor against the residue count of THIS catalogue, not Swiss-Prot's: the "
                         "search space differs by 9.95x, which is 8.6 raw-score points, and borrowing "
                         "the Swiss-Prot floor would stop the iteration early and silently.")
    ap.add_argument("--no-bhs", action="store_true",
                    help="skip the iterative best-hit arm. It costs one full-catalogue pass per "
                         "iteration per construct, so this is here for quick reruns of the other arms.")
    ap.add_argument("--tile-window", type=int, default=None, metavar="AA",
                    help="build the fixed-window `tile` arm with windows of this many residues. OFF by "
                         "default, so a run that does not pass it writes exactly the file it wrote "
                         "before the arm existed. The tiling consults nothing, not the motif "
                         "catalogue, not the reference catalogue, not the holdout mask, so it is "
                         "scored once for the whole sweep like `whole` and `auto`, never per threshold. "
                         "this project priced 300, 200 and 100 over the 336 synth_real ORFs at 1,816 / "
                         "2,788 / 5,749 windows, and cost follows the window COUNT, so the step is what "
                         "sets the price and not the size.")
    ap.add_argument("--tile-step", type=int, default=None, metavar="AA",
                    help="stride between windows; defaults to half --tile-window, which is the overlap "
                         "all three of 68.4's priced policies use. A step larger than the window is "
                         "refused rather than clamped, because it would leave residues in no query.")
    ap.add_argument("--bhs-scorer", choices=("run", "align"), default="run",
                    help="which scorer segments. 'run' (the default, and the behaviour before this flag "
                         "existed) segments with --scorer, which is correct only while that scorer's "
                         "scores are in the units the stopping floor is derived in: raw BLOSUM62. "
                         "'align' keeps segmentation on exact Smith-Waterman while retrieval runs under "
                         "--scorer: the label-free hybrid, and the only way an embedding "
                         "arm can be scored with the segmenter on at all (open item 42). With "
                         "--scorer align the two resolve to the same object, so the flag is then a no-op "
                         "by construction rather than by agreement.")
    ap.add_argument("--bhs-mask", choices=("per-threshold", "strictest"), default="per-threshold",
                    help="which holdout mask the SEGMENTER searches under. 'per-threshold' (the default, "
                         "and the only correct setting) masks `identity > thr` at each threshold of the "
                         "sweep, so segmentation and scoring see the same catalogue. 'strictest' is the "
                         "behaviour before 20 Oct: one mask built at min(THRESHOLDS) and reused at every "
                         "looser holdout, which this project measured as manufacturing 14,717 rank points "
                         "of loss and 6 of gain on the corpus tier at the 0.70 holdout: conservative "
                         "in NEITHER direction. It is kept reachable only so those figures stay "
                         "reproducible; never report a number from it. The two settings are identical at "
                         "min(THRESHOLDS) by construction, which is why the project's headline holdout "
                         "was never affected.")
    ap.add_argument("--bhs-spans", type=Path, default=None,
                    help="replay the segmentation recorded in a previous run's bhs_spans CSV instead of "
                         "recomputing it. Segmentation costs one full-catalogue alignment pass per "
                         "iteration per construct and depends on the SEGMENTING scorer, the catalogue "
                         "AND the holdout mask: not on --scorer. So a hybrid run can reuse an "
                         "alignment run's segmentation over the same catalogue at the same holdouts and "
                         "spend its GPU window on retrieval. The file must have been written by a run "
                         "with the same --tier, over the same constructs, against the same catalogue, "
                         "and under the same masking protocol; the first two are refused here, the third "
                         "shows up as a ReplayMismatch, and the fourth is refused on the file's "
                         "`threshold` column. A span file written before 20 Oct has no such column: it "
                         "holds one strictest-threshold segmentation per construct and is accepted only "
                         "under --bhs-mask strictest. There is no repair path and no warning: a span file "
                         "that does not reproduce its own invariants is describing a different run.")
    ap.add_argument("--tier", default="", help="label written into every row and into the output "
                                               "filename, so the three tiers' results stay distinguishable")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=1,
                    help="OS threads for the Smith-Waterman score pass, as in 26_bhs_protocol.py. "
                         "parasail releases the GIL, so this is real parallelism and the threads share "
                         "one copy of the catalogue. The arrays are bitwise identical to the serial "
                         "path (tests/test_align_threads.py asserts array_equal, not allclose), so it "
                         "changes the clock and nothing else, and the scorer's name deliberately "
                         "does not carry the thread count, so two runs at different settings write the "
                         "same filenames and the same columns. Ignored by the embedding arms.")
    ap.add_argument("--max-length", type=int, default=1022,
                    help="residues per sequence handed to an ESM-2 checkpoint; longer ones are "
                         "truncated. Ignored by the k-mer and alignment arms. It is part of the "
                         "reference-matrix cache key, so changing it recomputes rather than reusing.")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/embed_cache"),
                    help="where to cache the embedded reference matrix; '' disables caching")
    args = ap.parse_args(argv)

    for p in (args.results_dir / "orfs.faa", args.reference, args.homologs):
        if not p.is_file():
            print(f"not found: {p}  (run scripts/09_find_homologs.py first)", file=sys.stderr)
            return 2

    with open(args.manifest, newline="", encoding="utf-8-sig", errors="replace") as fh:
        man = {r["filename"].strip(): r for r in csv.DictReader(fh) if (r.get("filename") or "").strip()}
    with open(args.results_dir / "pilot_summary.csv", newline="", encoding="utf-8-sig") as fh:
        summary = list(csv.DictReader(fh))
    orfs_by_plasmid = read_faa(args.results_dir / "orfs.faa")
    refs = list(iter_fasta(args.reference))
    ref_acc = [e.accession for e in refs]
    idx_of = {a: i for i, a in enumerate(ref_acc)}
    ref_by_acc = {e.accession: e.seq for e in refs}
    homologs = load_homologs(args.homologs)
    mask_homologs = load_homologs(args.mask_homologs) if args.mask_homologs else homologs
    if args.mask_homologs:
        print(f"  targets from {args.homologs.name}, holdout mask from {args.mask_homologs.name}",
              flush=True)

    # ---- a replayed segmentation, loaded and guarded before any scorer is built -----------------
    # Loaded here, before the catalogue is embedded, so a mismatched file costs seconds rather than a
    # reference-matrix build. Two refusals live here; the third invariant check is per construct, in
    # `replay_best_hit_segments`, and the fourth (the holdout mask size) is in the loop below.
    replay_by_pid: dict[str, list[dict]] | None = None
    replay_meta: dict = {}
    if args.bhs_spans is not None:
        if args.no_bhs:
            print("--bhs-spans and --no-bhs contradict each other", file=sys.stderr)
            return 2
        if not args.bhs_spans.is_file():
            print(f"not found: {args.bhs_spans}", file=sys.stderr)
            return 2
        raw = args.bhs_spans.read_bytes()
        with open(args.bhs_spans, newline="", encoding="utf-8-sig") as fh:
            span_rows = list(csv.DictReader(fh))
        tiers = {r["tier"] for r in span_rows}
        if tiers != {args.tier}:
            print(f"--bhs-spans was written for tier(s) {sorted(tiers)}, this run is --tier "
                  f"{args.tier!r}. Refusing.", file=sys.stderr)
            return 2
        # Keyed on (construct, threshold), because segmentation is per threshold as of 20 Oct. A file
        # without a `threshold` column predates that and holds ONE segmentation per construct, built at
        # the strictest threshold in the sweep: replaying it under the per-threshold protocol would
        # reintroduce this project's defect while labelling the run as fixed, so it is refused rather than
        # broadcast. This is the loud consequence the fix has for every hybrid run already on disk.
        has_thr = bool(span_rows) and "threshold" in span_rows[0]
        if not has_thr and args.bhs_mask != "strictest":
            print(f"{args.bhs_spans.name} has no `threshold` column, so it carries one "
                  f"strictest-threshold segmentation per construct (the pre-20-Oct protocol, CONTEXT "
                  f"58). Replaying it under --bhs-mask per-threshold would reproduce that defect under a "
                  f"name that says otherwise. Re-derive the spans with a --bhs-mask per-threshold "
                  f"alignment run, or pass --bhs-mask strictest to reproduce the old numbers "
                  f"deliberately. Refusing.", file=sys.stderr)
            return 2
        replay_by_pid = defaultdict(list)
        if has_thr:
            for r in span_rows:
                replay_by_pid[(r["plasmid_id"], float(r["threshold"]))].append(r)
        else:
            # One segmentation for the whole sweep, so every threshold replays the same rows. Only
            # reachable under --bhs-mask strictest, where that is exactly what the protocol asks for.
            by_pid: dict[str, list[dict]] = defaultdict(list)
            for r in span_rows:
                by_pid[r["plasmid_id"]].append(r)
            for pid_, rs in by_pid.items():
                for t in THRESHOLDS:
                    replay_by_pid[(pid_, t)] = rs
        scored_pids = {r["plasmid_id"] for r in summary}
        extra = sorted({k[0] for k in replay_by_pid} - scored_pids)
        if extra:
            print(f"--bhs-spans covers {len(extra)} construct(s) this run does not score, e.g. "
                  f"{extra[:3]}. Refusing: a span file from a different construct set cannot be "
                  f"partially reused.", file=sys.stderr)
            return 2
        replay_meta = {
            "path": str(args.bhs_spans),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "rows": len(span_rows),
            "constructs_covered": len({k[0] for k in replay_by_pid}),
            "thresholds_covered": sorted({k[1] for k in replay_by_pid}),
            "per_threshold_segmentation": has_thr,
            "segmenting_scorer_from_filename": _scorer_from_span_name(args.bhs_spans.stem, args.tier),
        }
        print(f"  bhs segmentation REPLAYED from {args.bhs_spans.name}: {len(span_rows):,} rows over "
              f"{len({k[0] for k in replay_by_pid}):,} constructs "
              f"({'per-threshold' if has_thr else 'one strictest-threshold segmentation, broadcast'}), "
              f"sha256 {replay_meta['sha256'][:16]}. No "
              f"segmentation pass will run.", flush=True)
        # A construct absent from the file had zero iterations in the run that wrote it; that is normal
        # and is reported rather than inferred, because the count is the reader's check on the file.
        print(f"    {len(scored_pids) - len({k[0] for k in replay_by_pid}):,} of {len(scored_pids):,} scored constructs "
              f"are absent from the file, which means they claimed nothing above the floor.", flush=True)

    kw = model_kwargs(args.scorer, batch_size=args.batch_size, max_length=args.max_length)
    t0 = time.time()
    print(f"scorer: {args.scorer} | twilight reference: {len(refs):,} proteins", flush=True)
    # `threads` is operational rather than a model option, so it is passed beside `model_kwargs`
    # (which drops everything for the model-free arms) and read only by the alignment arm.
    scorer = get_scorer(args.scorer, [e.seq for e in refs],
                        cache_dir=(args.cache_dir or None), threads=args.threads, **kw)
    if args.threads > 1:
        print(f"  score pass over {args.threads} threads (bitwise identical to serial; see "
              f"tests/test_align_threads.py)", flush=True)
    print(f"  ready in {time.time() - t0:.0f}s ({scorer.name})", flush=True)

    # The bhs stopping rule is a significance gate inverted to a score floor, and `N` in the E-value is
    # the summed residue count of the database ACTUALLY SEARCHED. That is this catalogue. The catalogue
    # file holds 40,750 entries but its own reader's 50 aa window drops one (the working record, Q9TT33 at
    # 41 aa), so both counts are printed: the in-window count is the one that sets the floor.
    all_residues = sum(len(e.seq) for e in refs)
    in_window = [e for e in refs if MIN_LEN <= len(e.seq) <= MAX_LEN]
    db_residues = sum(len(e.seq) for e in in_window)
    bhs_ref_seqs = None if (args.no_bhs or replay_by_pid is not None) else [e.seq for e in refs]
    bhs_scorer = scorer
    if not args.no_bhs and replay_by_pid is None:
        t1 = time.time()
        bhs_scorer = resolve_bhs_scorer(args.bhs_scorer, scorer, bhs_ref_seqs, threads=args.threads)
        # Report the scorer that bound, not the flag that was passed: with `--scorer align` the two
        # resolve to one object and the run is identical to one without the flag, which is the claim
        # the acceptance test checks.
        if bhs_scorer is scorer:
            print(f"  bhs segmentation scorer: {bhs_scorer.name} (the run scorer)", flush=True)
        else:
            print(f"  bhs segmentation scorer: {bhs_scorer.name}, built in {time.time() - t1:.0f}s; "
                  f"retrieval stays on {scorer.name}", flush=True)
    if not args.no_bhs:
        print(f"  bhs database: {len(in_window):,} of {len(refs):,} entries in the {MIN_LEN}-{MAX_LEN} aa "
              f"window, {db_residues:,} residues (file total {all_residues:,}); "
              f"floor at E<={args.max_evalue:g} spans "
              f"{score_floor(args.max_evalue, 100, db_residues):.1f} (m=100) to "
              f"{score_floor(args.max_evalue, 3000, db_residues):.1f} (m=3000); "
              f"the Swiss-Prot floor would be "
              f"{score_floor(args.max_evalue, 100, 207_099_657) - score_floor(args.max_evalue, 100, db_residues):.2f}"
              f" points higher", flush=True)

    rows = []
    # One trace per (construct, threshold): the segmenter runs once per threshold now, so a construct
    # has as many segmentations as its homolog table gives it distinct holdout masks.
    bhs_traces: list[tuple] = []
    # How many queries each segment arm produced per construct. A limit's binding count is
    # reported rather than assumed: only a construct with more than one query can differ between the
    # pooled and per-segment arms, so this count is the exact reach of the new arms on THIS tier. It is
    # not transferable: k_auto is computed from the ORF sequence, so a tier whose ORFs change changes it.
    # `auto` is holdout-invariant by construction (it cuts on motifs in the ORF and never consults the
    # catalogue), so it is counted once; `bhs` is counted per threshold, because that is now what it is.
    k_by_arm: dict[str, list[int]] = defaultdict(list)
    k_bhs_by_thr: dict[float, list[int]] = defaultdict(list)
    # The binding count for the segmentation memo, segmentations actually run against
    # (construct, threshold) pairs asked for. A construct whose homolog table holds no identity between
    # two adjacent thresholds has the SAME mask at both, so its segmentation is computed once and reused
    #: sound precisely because it is the same mask, which is what this project's reuse across different
    # masks was not.
    counts = {"segmentations": 0, "pairs": 0}
    # What each holdout actually leaves, per payload (the working record, promoted to a rule in this project's working rules's
    # binding table): a threshold is an upper bound on availability and not a description of it, and
    # which of the two a row describes is a per-payload property no output here printed until 21 Oct.
    # Keyed on (threshold, payload) rather than per construct, because it is a property of the family.
    best_left: dict[tuple[float, str], float | None] = {}

    for srow in summary:
        pid = srow["plasmid_id"]
        acc = (man.get(srow["source_file"], {}).get("payload_accession") or "").strip()
        if not acc or acc not in ref_by_acc:
            continue
        # The three mask-independent arms, built and scored ONCE: `whole` and `oracle` are sequences and
        # `auto` cuts on motifs in the ORF, so no holdout can reach any of them. Only
        # `bhs` takes the mask, and it is built per threshold in the loop below.
        base = build_arms(pid, orfs_by_plasmid.get(pid, {}), ref_by_acc[acc],
                          tile_window=args.tile_window, tile_step=args.tile_step)
        if base is None:
            continue
        hs = homologs.get(acc, [])
        hs_mask = mask_homologs.get(acc, [])
        # Every arm is a list of queries; a segment arm's score row is the best each reference gets from
        # any of its segments, which is what a pipeline submitting every segment as its own query sees.
        # Written as a loop rather than a comprehension on purpose: an assignment expression inside a
        # comprehension binds in the ENCLOSING scope, so a `:=` here silently rebound the `rows` results
        # list and the run lost every construct but the last.
        scored: dict[str, np.ndarray] = {}
        seg_rows: dict[str, list[np.ndarray]] = {}
        # A two-stage scorer shortlists before it reranks, and a reference the shortlist drops can never
        # be recovered, so the arm's ceiling is its prefilter, not its dynamic programming. Nothing in
        # this harness reported that until 18 Oct, which is how `TOP_K = 200` survived while dropping the
        # answer for 97.1% of diluted `whole` queries. The prefilter row is memoised
        # inside the scorer, so recording it costs no extra prefilter pass, and a scorer without a
        # prefilter leaves these dicts empty and the columns blank.
        pf_scored: dict[str, np.ndarray] = {}
        pf_seg_rows: dict[str, list[np.ndarray]] = {}
        pf_fn = getattr(scorer, "prefilter_scores", None)

        def score_queries(qs: list[str]) -> tuple[list, list]:
            """One arm's score rows, and its prefilter rows when the scorer has a prefilter."""
            arm_rows = [scorer.score_all(q) for q in qs]
            pf_arm_rows = ([r for r in (pf_fn(q) for q in qs) if r is not None]
                           if (pf_fn is not None and qs) else [])
            return arm_rows, pf_arm_rows

        for arm, qs in base.queries().items():
            arm_rows, pf_arm_rows = score_queries(qs)
            if arm_rows:
                scored[arm] = pool_scores(arm_rows)
                # The per-segment arm re-uses these rows. No second scoring pass exists or should:
                # if this ever calls `score_all` again, the arm has stopped being free.
                if arm in PER_SEGMENT_ARMS:
                    seg_rows[PER_SEGMENT_ARMS[arm]] = arm_rows
                    k_by_arm[arm].append(len(arm_rows))
            if pf_arm_rows:
                pf_scored[arm] = pool_scores(pf_arm_rows)
                if arm in PER_SEGMENT_ARMS:
                    pf_seg_rows[PER_SEGMENT_ARMS[arm]] = pf_arm_rows

        # The segmentation memo, keyed on the mask itself and never on the threshold: two thresholds that
        # produce the same mask produce the same segmentation and the same score rows, so computing them
        # twice is waste rather than evidence. Keying on the mask means the key IS what the segmenter
        # consumes, so a hit cannot quietly cross two different masks, which is the mistake this whole
        # change exists to remove, and it would be a poor fix that reintroduced it as an optimisation.
        seg_cache: dict[tuple[int, ...], tuple] = {}

        def segmented(seg_mask: np.ndarray, thr: float) -> tuple:
            key = tuple(seg_mask.tolist())
            if key in seg_cache:
                return seg_cache[key] + (False,)
            if args.no_bhs:
                hit: tuple = (None, [], [])
            else:
                a = segment_arms(base, ref_seqs=bhs_ref_seqs, scorer=bhs_scorer,
                                 db_residues=db_residues, ref_accessions=ref_acc,
                                 max_evalue=args.max_evalue, exclude_idx=seg_mask,
                                 bhs_replay=(None if replay_by_pid is None
                                             else replay_by_pid.get((pid, thr), [])))
                counts["segmentations"] += 1
                hit = (a, *score_queries(a.queries().get("bhs", [])))
            seg_cache[key] = hit
            return hit + (True,)

        for thr in THRESHOLDS:
            mask = np.array([idx_of[h.accession] for h in hs_mask
                             if h.identity > thr and h.accession in idx_of], dtype=int)
            tgt = np.array([idx_of[h.accession] for h in hs
                            if h.identity <= thr and h.accession in idx_of], dtype=int)
            if thr >= 1.0:  # no holdout: the payload itself is the answer, as in the Step 2 benchmark
                mask, tgt = np.array([], dtype=int), np.array([idx_of[acc]], dtype=int)
            n_t = int(tgt.size)
            # Over the payload's in-catalogue homologs only: a relative the catalogue does not hold is
            # not available at any threshold, so counting it would overstate what the holdout leaves.
            bil = best_identity_left(
                (h.identity for h in hs if h.accession in idx_of), thr)
            best_left[(thr, acc)] = bil
            seg_mask = segmentation_mask(mask_homologs, idx_of, acc, thr,
                                         protocol=args.bhs_mask)
            if not args.no_bhs:
                counts["pairs"] += 1
            arms, bhs_rows, bhs_pf_rows, fresh = segmented(seg_mask, thr)
            thr_scored, thr_seg_rows = dict(scored), dict(seg_rows)
            thr_pf_scored, thr_pf_seg_rows = dict(pf_scored), dict(pf_seg_rows)
            if bhs_rows:
                thr_scored["bhs"] = pool_scores(bhs_rows)
                thr_seg_rows[PER_SEGMENT_ARMS["bhs"]] = bhs_rows
                k_bhs_by_thr[thr].append(len(bhs_rows))
            if bhs_pf_rows:
                thr_pf_scored["bhs"] = pool_scores(bhs_pf_rows)
                thr_pf_seg_rows[PER_SEGMENT_ARMS["bhs"]] = bhs_pf_rows
            if arms is not None and arms.bhs_result is not None:
                res = arms.bhs_result
                # The fourth invariant, and the one that catches a span file written against a DIFFERENT
                # catalogue or a different homolog table: the holdout mask size is recomputed here from
                # this run's own homologs, so it is independent of anything the file says. Checked per
                # threshold, because the mask it checks is now per threshold.
                if replay_by_pid is not None and res.trace:
                    declared = int(replay_by_pid[(pid, thr)][0]["n_catalogue_entries_masked"])
                    if declared != int(seg_mask.size):
                        print(f"--bhs-spans mismatch on {pid} at threshold {thr:g}: the file was "
                              f"written with {declared} catalogue entries masked, this run masks "
                              f"{seg_mask.size}. The span file belongs to a different catalogue, "
                              f"homolog table or masking protocol. Refusing.", file=sys.stderr)
                        return 2
                bhs_traces.append((pid, acc, arms, res, int(seg_mask.size), thr))
                # Only a fresh segmentation is logged. A cache hit is the same spans under the same mask
                # and printing it again would read as a second measurement.
                if fresh:
                    print(f"    bhs thr {thr:<4g} floor {res.floor:.1f}  masked {seg_mask.size}    "
                          f"iters {res.n_iterations}  stop {res.stop_reason}    "
                          + "  ".join(f"[{t.start}-{t.end}  {t.ref_accession} S={t.score:.0f}"
                                      f"{'' if t.emitted else ' <min'}]" for t in res.trace), flush=True)
            for arm in REPORT_ARM_NAMES:
                if arm in thr_scored:
                    r = best_target_rank(thr_scored[arm], tgt, mask)
                elif arm in thr_seg_rows:
                    r = per_segment_rank(thr_seg_rows[arm], tgt, mask)
                else:
                    continue
                # The same aggregation as the scorer's own row, so the prefilter rank is the rank of
                # the best surviving target *as this arm presents it*: pooled for a pooled arm, best
                # over segments for a per-segment arm. Comparing a per-segment arm against a `whole`
                # prefilter rank would be the sample-change-plus-effect-change error of this project's working rules.
                if arm in thr_pf_scored:
                    pr = best_target_rank(thr_pf_scored[arm], tgt, mask)
                elif arm in thr_pf_seg_rows:
                    pr = per_segment_rank(thr_pf_seg_rows[arm], tgt, mask)
                else:
                    pr = None
                rows.append(dict(tier=args.tier, threshold=thr, plasmid_id=pid, payload_accession=acc,
                                 arm=arm, n_homologs_available=n_t, n_held_out=int(mask.size),
                                 best_identity_left="" if bil is None else round(bil, 4),
                                 best_target_rank="" if r is None else r,
                                 prefilter_rank="" if pr is None else pr))
        print(f"    {pid[:44]:<46}{acc}  homologs={len(hs):>4}    "
              + "  ".join(f"{int(t * 100)}%:{sum(1 for h in hs if h.identity <= t)}"
                          for t in THRESHOLDS[1:])
              + ("" if args.no_bhs else f"  masks={len(seg_cache)}"),
              flush=True)

    stem = scorer.name.replace("/", "_") + (f"_{args.tier}" if args.tier else "")

    # Per-construct spans against the oracle span. This is the named falsifying case made checkable: a
    # miss can be "claimed the wrong thing" (a span that misses the oracle span) or "claimed a chimera"
    # (a first span straddling payload and distractor, leaving neither clean), and those need opposite
    # responses. Jaccard overlap against the oracle span is what separates them.
    if bhs_traces:
        spans_out = args.results_dir / f"bhs_spans_{stem}.csv"
        srows = []
        for pid, acc, arms, res, n_masked, thr in bhs_traces:
            osp = arms.span
            for t in res.trace:
                inter = 0 if osp is None else max(0, min(t.end, osp.end) - max(t.start, osp.start) + 1)
                union = (t.end - t.start + 1) + (0 if osp is None else osp.length) - inter
                srows.append(dict(
                    tier=args.tier, threshold=thr, plasmid_id=pid, payload_accession=acc,
                    orf_len=len(arms.orf_seq),
                    iteration=t.iteration, hit_accession=t.ref_accession, hit_score=round(t.score, 1),
                    floor=round(t.floor, 2), seg_start=t.start, seg_end=t.end,
                    seg_len=t.end - t.start + 1, identity=t.identity, hit_coverage=t.ref_coverage,
                    emitted=int(t.emitted), newly_masked=t.n_masked_now,
                    oracle_start="" if osp is None else osp.start,
                    oracle_end="" if osp is None else osp.end,
                    oracle_len="" if osp is None else osp.length,
                    overlap_residues=inter,
                    jaccard_with_oracle="" if union == 0 else round(inter / union, 4),
                    frac_of_oracle_covered=("" if osp is None or osp.length == 0
                                            else round(inter / osp.length, 4)),
                    frac_of_segment_in_oracle=round(inter / (t.end - t.start + 1), 4),
                    n_iterations=res.n_iterations, stop_reason=res.stop_reason,
                    n_catalogue_entries_masked=n_masked))
        with open(spans_out, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(srows[0]))
            w.writeheader()
            w.writerows(srows)
        iters = [t[3].n_iterations for t in bhs_traces]
        dist = {k: iters.count(k) for k in sorted(set(iters))}
        stops = {k: [t[3].stop_reason for t in bhs_traces].count(k)
                 for k in sorted({t[3].stop_reason for t in bhs_traces})}
        # Every count below is over (construct, threshold) pairs, not constructs: one construct
        # contributes one trace per threshold. Said rather than left to be inferred, because the same
        # words meant "per construct" before 20 Oct and a reader comparing two runs would otherwise read
        # a 6x rise in "constructs" as a tier change.
        print(f"\nbhs iterations per (construct, threshold): {dist}  (n={len(iters)}, "
              f"mean {sum(iters) / len(iters):.2f}, max {max(iters)})")
        print(f"bhs stop reasons: {stops}")
        fellback = [(t[0], t[5]) for t in bhs_traces if not t[3].segments]
        print(f"bhs segments emitted: {sum(len(t[2].bhs) for t in bhs_traces)}; "
              f"spans claimed: {len(srows)}; wrote {spans_out.name}")
        print(f"bhs fell back to the whole ORF on {len(fellback)} of {len(bhs_traces)}  "
              f"(construct, threshold) pairs (nothing cleared the floor): "
              f"{', '.join(f'{f[0][:30]}@{f[1]:g}' for f in fellback) or 'none'}")
        how = "REPLAYED (no alignment pass)" if replay_by_pid is not None else "run"
        protocol = ("per-threshold" if args.bhs_mask == "per-threshold"
                    else "STRICTEST-MASK protocol, do not report")
        print(f"segmentations {how}: {counts['segmentations']} for {counts['pairs']}  "
              f"(construct, threshold) pairs: "
              f"{counts['pairs'] - counts['segmentations']} served from the per-construct mask memo, "
              f"because the mask is identical at adjacent thresholds there ({protocol} masking)")

    # The binding count for the per-segment arms. A single-query construct is byte-identical under both
    # aggregations, so this is how much of the tier the new arms can possibly move.
    for arm, ks in sorted(k_by_arm.items()):
        dist = {k: ks.count(k) for k in sorted(set(ks))}
        multi = sum(1 for k in ks if k > 1)
        print(f"{arm} queries per construct: {dist}  (n={len(ks)}, mean {sum(ks) / len(ks):.2f}, "
              f"max {max(ks)}); {multi} of {len(ks)} produce more than one, so "
              f"{arm}_per_segment can differ from {arm} on {multi}  "
              f"({100 * multi / len(ks):.1f}%) and on no others")
    # `bhs` gets its own table, per threshold, because the segmenter's mask is per threshold: k_bhs is a
    # function of the holdout and reporting one distribution for the sweep would hide that. the working record
    # measured the shape of the hiding: over-masking inflates k_bhs, and k_bhs is the input to the
    # decomposition gate the write-up carries (53.8).
    if k_bhs_by_thr:
        # The quantity is QUERIES per construct, the same as the `auto` line above, so a construct that
        # fell back to the whole ORF counts as 1 and never as 0. Said in the header because the working record
        # reports a k_bhs distribution counting EMITTED SPANS, where a fallback is 0, and the two
        # distributions are not the same numbers.
        print(f"\n{'holdout':>8}{'n':>6}{'mean':>9}{'max':>5}   bhs queries per construct "
              f"(a fallback to the whole ORF counts as 1, not 0)")
        for thr in THRESHOLDS:
            ks = k_bhs_by_thr.get(thr) or []
            if not ks:
                continue
            dist = {k: ks.count(k) for k in sorted(set(ks))}
            print(f"{thr:>8.0%}{len(ks):>6}{sum(ks) / len(ks):>9.2f}{max(ks):>5}      {dist}")

    # What each holdout actually LEAVES, per payload. this project's binding-limit register carries this as a rule
    # since a holdout is an upper bound on what is available, so the nominal threshold
    # names the mask and not the regime, and a stratified row read as "the arm loses n constructs to
    # dilution at 0.70" can be one payload and can be remoteness. Free: every identity below is
    # already in the homolog table this run read.
    if best_left:
        print()
        print(f"{'holdout':>8}{'payloads':>10}{'left':>7}{'min':>7}{'med':>7}{'max':>7}      "
              f"payloads whose label overstates by a full step of this sweep")
        for thr in THRESHOLDS:
            vals = {a: v for (t_, a), v in best_left.items() if t_ == thr}
            if not vals:
                continue
            have = sorted(v for v in vals.values() if v is not None)
            over = sorted(a for a, v in vals.items() if holdout_label_overstated(thr, v, THRESHOLDS))
            med = have[len(have) // 2] if have else float("nan")
            print(f"{thr:>8.2f}{len(vals):>10}{len(have):>7}"
                  f"{(have[0] if have else float('nan')):>7.3f}{med:>7.3f}"
                  f"{(have[-1] if have else float('nan')):>7.3f}      "
                  + (f"{len(over)}: " + ", ".join(f"{a} at {vals[a]:.3f}" for a in over[:6])
                     + ("..." if len(over) > 6 else "") if over else "none"))

    out = args.results_dir / f"twilight_{stem}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Constructs are not independent trials. A corpus of 410 labelled constructs carries 138 distinct
    # payloads, and one protein appears in 37 of them: different vectors around the same gene, which
    # sketch deduplication cannot see because the sequences genuinely differ. The benchmark asks whether
    # a protein can be identified, so 37 constructs sharing a payload are one answer repeated, not 37
    # trials. Both numbers are therefore reported: the micro average over constructs, and the macro
    # average over payloads, which weights each protein once.
    # `arms` is the arms that actually produced rows, not the arms this script knows about. It used to
    # be `REPORT_ARM_NAMES` verbatim, so a `--no-bhs` run advertised `bhs` and `bhs_per_segment` and a
    # reader had no way to tell an arm that scored zero from an arm that never ran (open item 31). The
    # full list stays beside it as `arms_known`, since a reader comparing two runs needs to see which
    # arms the *code* had at the time.
    arms_scored = [a for a in REPORT_ARM_NAMES if any(r["arm"] == a for r in rows)]
    agg: dict = {"scorer": scorer.name,
                 # The resolved segmentation scorer, not the flag: `--scorer align --bhs-scorer align`
                 # writes exactly what `--scorer align` writes, so the no-op covers the summary too.
                 "bhs_scorer": (None if args.no_bhs else
                                ("replayed:" + replay_meta["segmenting_scorer_from_filename"]
                                 if replay_by_pid is not None else bhs_scorer.name)),
                 # A result whose segmentation came from somewhere else says so in its own output.
                 # Without this, this project's register problem reappears one level down: two runs with
                 # identical filenames and columns, one of which never segmented anything.
                 "bhs_spans_replayed": (replay_meta or None),
                 "tier": args.tier, "n_reference": len(refs),
                 "thresholds": list(THRESHOLDS), "arms": arms_scored,
                 "arms_known": list(REPORT_ARM_NAMES), "no_bhs": bool(args.no_bhs),
                 "query_arms": list(ARM_NAMES), "per_segment_arms": dict(PER_SEGMENT_ARMS),
                 "queries_per_construct": {a: {str(k): ks.count(k) for k in sorted(set(ks))}
                                           for a, ks in sorted(k_by_arm.items())},
                 # Per payload, per threshold, so a reader can re-stratify any row of the CSV on what
                 # the holdout actually left rather than on its label. `null` means the
                 # holdout left nothing in the catalogue, which is unscorable and not a zero.
                 "best_identity_left": {
                     f"{thr:g}": {a: v for (t_, a), v in sorted(best_left.items()) if t_ == thr}
                     for thr in THRESHOLDS if any(t_ == thr for t_, _ in best_left)},
                 "holdout_label_overstated": {
                     f"{thr:g}": sorted(a for (t_, a), v in best_left.items()
                                        if t_ == thr and holdout_label_overstated(thr, v, THRESHOLDS))
                     for thr in THRESHOLDS if any(t_ == thr for t_, _ in best_left)},
                 "runtime_seconds": round(time.time() - t0, 1), "by_threshold": {}}
    if bhs_traces:
        iters = [t[3].n_iterations for t in bhs_traces]
        agg["bhs"] = {
            "max_evalue": args.max_evalue,
            "db_residues_in_window": db_residues,
            "db_residues_file_total": all_residues,
            "n_entries_in_window": len(in_window),
            "swissprot_residues_for_comparison": 207_099_657,
            "floor_points_below_swissprot": round(
                score_floor(args.max_evalue, 100, 207_099_657)
                - score_floor(args.max_evalue, 100, db_residues), 3),
            "floor_min": round(min(t[3].floor for t in bhs_traces), 2),
            "floor_max": round(max(t[3].floor for t in bhs_traces), 2),
            "iterations_mean": round(sum(iters) / len(iters), 3),
            "iterations_max": max(iters),
            "iterations_distribution": {str(k): iters.count(k) for k in sorted(set(iters))},
            "stop_reasons": {k: [t[3].stop_reason for t in bhs_traces].count(k)
                             for k in sorted({t[3].stop_reason for t in bhs_traces})},
            "segments_emitted": sum(len(t[2].bhs) for t in bhs_traces),
            "n_fell_back_to_whole_orf": sum(1 for t in bhs_traces if not t[3].segments),
            "spans_claimed": sum(t[3].n_iterations for t in bhs_traces),
            "mask_entries_min": min(t[4] for t in bhs_traces),
            "mask_entries_max": max(t[4] for t in bhs_traces),
            # The protocol, by name. Before 20 Oct this field was the literal `min(THRESHOLDS)`, which
            # recorded the *value* of a choice nobody knew was a choice; a reader of an old summary sees
            # a number and a reader of a new one sees which of two protocols produced the file.
            "mask_protocol": args.bhs_mask,
            "masked_at_threshold": (min(THRESHOLDS) if args.bhs_mask == "strictest"
                                    else "per-threshold"),
            # Every limit reports how much it bound. These two are the memo's reach: a
            # (construct, threshold) pair whose mask equals a neighbour's is segmented once.
            "segmentation_passes": counts["segmentations"],
            "segmentation_pairs": counts["pairs"],
            "traces": len(bhs_traces),
            # Queries per construct, per threshold: not emitted spans. this project's `k_bhs` counts
            # emitted spans, where a fallback to the whole ORF is 0; here it is 1 query. Named in full
            # so the two are not read as one series.
            "bhs_queries_per_construct_by_threshold": {
                f"{t:.2f}": {str(k): (k_bhs_by_thr.get(t) or []).count(k)
                             for k in sorted(set(k_bhs_by_thr.get(t) or []))}
                for t in THRESHOLDS if k_bhs_by_thr.get(t)},
        }
    print(f"\n{'holdout':>8}{'constr':>8}{'payld':>8}      "
          + "".join(f"{a + ' top1':>14}" for a in ARM_NAMES)
          + f"  | macro-averaged by payload ({'/'.join(ARM_NAMES)})")
    for thr in THRESHOLDS:
        at = [r for r in rows if r["threshold"] == thr]
        scorable = {r["plasmid_id"] for r in at if r["best_target_rank"] != ""}
        payloads = {r["payload_accession"] for r in at if r["best_target_rank"] != ""}
        d = {"n_scorable": len(scorable), "n_payloads": len(payloads)}
        for arm in REPORT_ARM_NAMES:
            got = [r for r in at if r["arm"] == arm and r["best_target_rank"] != ""]
            d[f"{arm}_top1"] = sum(1 for r in got if int(r["best_target_rank"]) == 1)
            d[f"{arm}_top10"] = sum(1 for r in got if int(r["best_target_rank"]) <= 10)
            d[f"{arm}_n"] = len(got)
            # Mean rank and MRR are the primary metrics: they are the only figures that
            # move monotonically with dilution and do not saturate. recall@10 is the operational
            # secondary; top-1 is kept as a column for continuity with sections 15-21 and never quoted
            # alone, which is why it no longer has a table to itself.
            ranks = [int(r["best_target_rank"]) for r in got]
            d[f"{arm}_mean_rank"] = round(sum(ranks) / len(ranks), 2) if ranks else None
            d[f"{arm}_mrr"] = round(sum(1 / x for x in ranks) / len(ranks), 3) if ranks else None
            # macro average: each payload contributes its own figure once, however many constructs carry it
            per_payload: dict[str, list[int]] = defaultdict(list)
            for r in got:
                per_payload[r["payload_accession"]].append(int(r["best_target_rank"]))
            pp = list(per_payload.values())
            d[f"{arm}_top1_macro"] = macro_avg(pp, lambda v: 100 * sum(1 for x in v if x == 1) / len(v), 1)
            d[f"{arm}_top10_macro"] = macro_avg(pp, lambda v: 100 * sum(1 for x in v if x <= 10) / len(v), 1)
            d[f"{arm}_mean_rank_macro"] = macro_avg(pp, lambda v: sum(v) / len(v), 2)
            d[f"{arm}_mrr_macro"] = macro_avg(pp, lambda v: sum(1 / x for x in v) / len(v), 3)
            d[f"{arm}_n_payloads"] = len(per_payload)
        agg["by_threshold"][f"{thr:.2f}"] = d
        micro = "".join(f"{d[a + '_top1']:>9}/{d[a + '_n']:<4}" for a in ARM_NAMES)
        macro = "".join(f"{(d[a + '_top1_macro'] if d[a + '_top1_macro'] is not None else 0):>8.1f}%"
                        for a in ARM_NAMES)
        print(f"{thr:>8.0%}{len(scorable):>8}{len(payloads):>8}      {micro}  |{macro}")
    # mean rank and MRR are primary, recall@10 is the operational secondary, top-1 is a
    # column, and micro and macro are both reported always: constructs are not independent trials.
    # The table above keeps its historic shape for the four query arms; this one carries every arm.
    print()
    print(f"{'holdout':>8}  {'arm':<17}{'n':>5}{'payld':>6}{'mean rank':>11}{'MRR':>8}"
          f"{'recall@10':>11}{'top1':>7}   | macro {'mean rank':>10}{'MRR':>8}{'recall@10':>11}{'top1':>8}")
    for thr in THRESHOLDS:
        d = agg["by_threshold"][f"{thr:.2f}"]
        for arm in REPORT_ARM_NAMES:
            n = d.get(f"{arm}_n") or 0
            if not n:
                continue
            print(f"{thr:>8.0%}  {arm:<17}{n:>5}{d[f'{arm}_n_payloads']:>6}"
                  f"{d[f'{arm}_mean_rank']:>11.2f}{d[f'{arm}_mrr']:>8.3f}"
                  f"{100 * d[f'{arm}_top10'] / n:>10.1f}%{d[f'{arm}_top1']:>7}"
                  f"   |       {d[f'{arm}_mean_rank_macro']:>10.2f}{d[f'{arm}_mrr_macro']:>8.3f}"
                  f"{d[f'{arm}_top10_macro']:>10.1f}%{d[f'{arm}_top1_macro']:>7.1f}%")
    # Prefilter recall, per arm, as a curve. Only a two-stage scorer fills `prefilter_rank`, so this
    # block is silent for the alignment and mean-pooled arms. For an EBA arm it is the ceiling: a
    # reference the shortlist dropped cannot reach rank 1 whatever the reranker does, so a `top_k` whose
    # recall is below 100% here bounds every number in the tables above (open item 32).
    pf_rows = [r for r in rows if r.get("prefilter_rank") not in (None, "")]
    if pf_rows:
        agg["prefilter"] = {"recall_at_grid": list(PREFILTER_RECALL_AT), "by_threshold": {}}
        print("\nprefilter recall (share of constructs whose best surviving target the prefilter ranks "
              "at or above k)")
        print(f"{'holdout':>8}  {'arm':<17}{'n':>5}"
              + "".join(f"{'k=' + str(k):>10}" for k in PREFILTER_RECALL_AT))
        for thr in THRESHOLDS:
            per_arm: dict[str, dict] = {}
            for arm in REPORT_ARM_NAMES:
                pr = [int(r["prefilter_rank"]) for r in pf_rows
                      if r["threshold"] == thr and r["arm"] == arm and r["n_homologs_available"]]
                if not pr:
                    continue
                at = {str(k): round(100 * sum(1 for x in pr if x <= k) / len(pr), 1)
                      for k in PREFILTER_RECALL_AT}
                per_arm[arm] = {"n": len(pr), "recall_at": at,
                                "median_prefilter_rank": int(sorted(pr)[len(pr) // 2]),
                                "max_prefilter_rank": max(pr)}
                print(f"{thr:>8.0%}  {arm:<17}{len(pr):>5}"
                      + "".join(f"{at[str(k)]:>9.1f}%" for k in PREFILTER_RECALL_AT))
            agg["prefilter"]["by_threshold"][f"{thr:.2f}"] = per_arm
        print("Read this before the tables above: it is an upper bound on them, per arm.")

    (args.results_dir / f"twilight_{stem}_summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"\nruntime {agg['runtime_seconds']}s; wrote {out.name} + _summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
