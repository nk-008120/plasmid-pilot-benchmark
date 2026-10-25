#!/usr/bin/env python3
"""Calibrate a significance floor for a cosine scorer, and test it on the segmenter (open item 25).

    python scripts/36_cosine_floor.py --embedder blosum --n-subjects 3000 --n-shuffles 8
    python scripts/36_cosine_floor.py --embedder facebook/esm2_t12_35M_UR50D --n-subjects 2000

**Why this exists.** The best-hit segmenter is this project's best method and it cannot run under an
embedding scorer, because its stopping rule is Karlin-Altschul inverted on *raw Smith-Waterman* scores
(open item 25). the working record narrowed the problem: `bhs_stop` is `below_floor` for
all 777 constructs on all four tiers, so one call decides the whole stopping rule and a replacement has
to reproduce one decision and nothing else.

**What is fitted, and what is not chosen.** `pilot.cosine_floor` fits `lambda` and `K` of the same
equation to the scorer in use, from scores against length-matched shuffled queries. So the floor keeps
both dependencies the alignment floor has, query length and database size, which a chosen cosine
threshold carries neither of. Nothing here is tuned against retrieval success: that would choose a
threshold on the data it is then evaluated on, which is this project's error.

**The acceptance test is the segment count, not retrieval.** Both segmenters, alignment and cosine,
are run on the **same sampled catalogue**, so the comparison is paired on one database rather than
across two, and agreement is reported on `iterations` (what the floor decides) and on `k` (what carries
the value). The full-catalogue figures are printed beside them as context only; they are a different
database and are not the comparator. Retrieval is reported last and labelled as not the criterion.

**A limit this task does not remove.** `best_hit_segments` claims each span with a BLOSUM Smith-Waterman
traceback against the winning reference (`oracle_span`), and `eba.py` has no traceback at all: only a
score. So this builds the *stopping rule* in cosine units while span claiming stays BLOSUM. That is a
hybrid and it is declared, not hidden: it is label-free either way, since the winner comes from the
catalogue, but it is not a pure cosine segmenter and must not be described as one.

Writes `cosine_floor.json`, `cosine_floor_detail.csv` and `cosine_floor.log`.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from pilot.align import get_scorer
from pilot.arms import build_arms
from pilot.cosine_floor import (BLOSUM62_EXPECTED_OVER_SD, VIABLE_EXPECTED_SCORE_RANGE,
                                 background_pair_stats, fit_evd, score_shuffles)
from pilot.decompose import best_hit_segments
from pilot.eba import SHIFT as EBA_SHIFT
from pilot.homology import MAX_LEN, MIN_LEN, score_floor
from pilot.positional import get_positional_embedder
from pilot.reference import iter_fasta


def read_faa(path: Path) -> dict[str, dict[str, str]]:
    """`{plasmid_id: {orf_id: seq}}`, byte-identical to `10_twilight.py`'s reader.

    **This must not be reimplemented.** An `orfs.faa` header is
    `>plasmid|orf001|+|1..1944 length_aa=647 class=unannotated`, and the harness keys the inner dict on
    the whitespace-stripped *whole* id, `plasmid|orf001|+|1..1944`, not on the ORF number. The id is
    not cosmetic: `arms.pick_payload_orf` sorts candidates on `(containment_count, oid)` and takes the
    top `PREFILTER_TOP`, so the string is a tie-breaker that can change which ORF is selected. A reader
    that keeps the description, or strips the plasmid prefix, silently picks a different ORF on ties.
    `tests/test_cosine_floor.py` pins this against a real-format header.
    """
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for e in iter_fasta(path):
        oid = e.header.split()[0]
        out[oid.split("|")[0]][oid] = e.seq
    return out


def agreement(rows: list[dict], variant_names: list[str]) -> dict:
    """Per-variant agreement between the cosine floor's segmentation and exact alignment's.

    Module level so it can be tested without a catalogue: the version inlined in `main` reported the
    **pooled** cosine distribution in every variant's block, because the key was built without the
    variant suffix. The counts and the confusion matrix were right throughout, so the defect was
    invisible in the agreement percentages and visible only in that a 95%-agreeing variant printed a
    mean its own confusion matrix contradicts.
    """

    def dist(key: str) -> dict:
        v = [r[key] for r in rows]
        return {"mean": round(statistics.fmean(v), 2), "median": statistics.median(v),
                "max": max(v), "histogram": {str(x): v.count(x) for x in sorted(set(v))}}

    out: dict[str, dict] = {}
    for q in ("k", "iterations"):
        for vname in variant_names:
            sfx = "" if vname == "pooled" else f"_{vname}"
            a = [r[f"align_{q}"] for r in rows]
            c = [r[f"cosine{sfx}_{q}"] for r in rows]
            exact = sum(1 for x, y in zip(a, c) if x == y)
            out[f"{q}{sfx}"] = {
                "exact": exact, "n": len(rows),
                "exact_pct": round(100 * exact / len(rows), 1),
                "within_1": sum(1 for x, y in zip(a, c) if abs(x - y) <= 1),
                "cosine_higher": sum(1 for x, y in zip(a, c) if y > x),
                "cosine_lower": sum(1 for x, y in zip(a, c) if y < x),
                "align": dist(f"align_{q}"), "cosine": dist(f"cosine{sfx}_{q}"),
                # the confusion matrix, not just the mean: a rule can match on average and be wrong
                # on every construct
                "confusion": {f"{x}->{y}": sum(1 for p, q2 in zip(a, c) if (p, q2) == (x, y))
                              for x, y in sorted({(p, q2) for p, q2 in zip(a, c)})}}
    return out


def stats_record(stats, shift: float) -> dict:
    """`PairStats` as JSON, with the 200-bin histogram replaced by what is derived from it.

    The histogram is an implementation detail of `ungapped_lambda`; the numbers a reader needs are the
    bound and the onset, so those are recorded and the bins are not.
    """
    d = {k: v for k, v in vars(stats).items() if k != "cosine_hist"}
    d["signal_to_noise"] = round(stats.signal_to_noise(shift), 4)
    d["snr_matched_shift"] = round(stats.snr_matched_shift(), 4)
    d["blosum62_analogous_shift"] = round(stats.blosum62_analogous_shift(), 4)
    d["ungapped_lambda"] = round(stats.ungapped_lambda(shift), 6)
    d["gumbel_onset"] = round(stats.gumbel_onset(shift), 4)
    d["n_cosine_hist_bins"] = len(stats.cosine_hist)
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results/esm_hand_2048"),
                    help="supplies orfs.faa and pilot_summary.csv for the tier under test")
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--reference", type=Path,
                    default=Path("results/gate_all173/reference_twilight.fasta"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/cosine_floor"))
    ap.add_argument("--embedder", default="blosum",
                    help="blosum, kmerpos, or a SMALL ESM-2 checkpoint. Not 650M: calibration is "
                         "n_queries * n_shuffles full catalogue passes.")
    ap.add_argument("--shift", type=float, default=None,
                    help="cosine shift. Default: the scorer's own (eba.SHIFT).")
    ap.add_argument("--n-subjects", type=int, default=3000,
                    help="catalogue sample the calibration and BOTH segmenters run against. The same "
                         "sample for both is what makes the comparison paired.")
    ap.add_argument("--n-shuffles", type=int, default=8, help="shuffled queries per construct")
    ap.add_argument("--n-calib-queries", type=int, default=12,
                    help="constructs whose ORFs are shuffled to build the null. Spread over the "
                         "length range, not the first n: lambda must not depend on length and that is "
                         "the goodness check.")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--max-evalue", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--scan-only", action="store_true",
                    help="print the background cosine and the viable shift range for the embedder, "
                         "then stop. No dynamic programming, seconds not hours. RUN THIS FIRST.")
    ap.add_argument("--fit-from", type=Path, default=None,
                    help="re-fit from a cached .npz written by an earlier run of this script. Zero "
                         "scoring. Use it to change --n-length-bins, --min-r2 or the tail window "
                         "without paying for the catalogue passes again.")
    ap.add_argument("--min-r2", type=float, default=0.97,
                    help="the tail regression's goodness bar. A fit below it is REPORTED, not raised: "
                         "the per-bin numbers are what say whether the pooled model is wrong or no "
                         "Gumbel regime exists.")
    ap.add_argument("--n-length-bins", type=int, default=3)
    ap.add_argument("--no-effective-length", action="store_true",
                    help="skip BLAST's finite-length correction. Only to show what it buys: without "
                         "it the apparent lambda drifts with query length even where every length's "
                         "own tail is exponential.")
    ap.add_argument("--max-minutes", type=float, default=45.0,
                    help="refuse to start if the timed single-pass estimate exceeds this. The run that "
                         "produced the working record took 2 h 45 m and was rejected at the end.")
    ap.add_argument("--top-k", type=int, default=0,
                    help="keep 0. A prefilter bands non-shortlisted references, so the tail being "
                         "fitted would be the band rather than the scorer (cosine_floor docstring).")
    args = ap.parse_args(argv)

    for p in (args.results_dir / "orfs.faa", args.results_dir / "pilot_summary.csv",
              args.reference, args.manifest):
        if not p.is_file():
            print(f"not found: {p}", file=sys.stderr)
            return 2
    if args.top_k:
        print("--top-k must be 0 for calibration; see the module docstring", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = open(args.out_dir / "cosine_floor.log", "w", encoding="utf-8")

    def say(line: str = "") -> None:
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    with open(args.manifest, newline="", encoding="utf-8-sig", errors="replace") as fh:
        man = {r["filename"].strip(): r for r in csv.DictReader(fh) if (r.get("filename") or "").strip()}
    with open(args.results_dir / "pilot_summary.csv", newline="", encoding="utf-8-sig") as fh:
        summary = list(csv.DictReader(fh))
    orfs_by_plasmid = read_faa(args.results_dir / "orfs.faa")

    full = [e for e in iter_fasta(args.reference) if MIN_LEN <= len(e.seq) <= MAX_LEN]
    ref_by_acc = {e.accession: e.seq for e in full}
    # The sample must contain every payload under test, or a construct is unscorable for reasons that
    # have nothing to do with the floor. Payloads first, then a seeded random fill.
    payloads = sorted({(man.get(r["source_file"], {}).get("payload_accession") or "").strip()
                       for r in summary} - {""})
    keep = [e for e in full if e.accession in set(payloads)]
    rest = [e for e in full if e.accession not in set(payloads)]
    random.Random(args.seed).shuffle(rest)
    refs = keep + rest[: max(args.n_subjects - len(keep), 0)]
    ref_seqs = [e.seq for e in refs]
    ref_acc = [e.accession for e in refs]
    db_residues = sum(len(s) for s in ref_seqs)
    say(f"catalogue sample: {len(refs):,} of {len(full):,} in-window proteins, "
        f"{db_residues:,} residues ({len(keep)} payloads kept, seed {args.seed})")
    say(f"  the full catalogue is {sum(len(e.seq) for e in full):,} residues, so this sample's floor "
        f"sits {score_floor(args.max_evalue, 500, sum(len(e.seq) for e in full)) - score_floor(args.max_evalue, 500, db_residues):.2f}  "
        f"points below the full-catalogue alignment floor at m=500")

    # constructs
    cons = []
    for srow in summary:
        pid = srow["plasmid_id"]
        acc = (man.get(srow["source_file"], {}).get("payload_accession") or "").strip()
        if not acc or acc not in ref_by_acc:
            continue
        arms = build_arms(pid, orfs_by_plasmid.get(pid, {}), ref_by_acc[acc])
        if arms is None or not arms.orf_seq:
            continue
        cons.append((pid, acc, arms.orf_seq))
    say(f"{len(cons)} scorable constructs, ORF lengths "
        f"{min(len(c[2]) for c in cons)}-{max(len(c[2]) for c in cons)} aa")

    # ---- pre-flight: milliseconds, and it is what would have saved the 18 Oct run ---------------
    stats = background_pair_stats(get_positional_embedder(
        args.embedder, **({} if args.embedder in ("blosum", "blosum62")
                          or args.embedder.startswith("kmerpos") else {"batch_size": 8})),
        [e.seq for e in full[:400]])
    shift_in_use = args.shift if args.shift is not None else EBA_SHIFT
    e_pair = stats.expected_pair_score(shift_in_use)
    lo_s, hi_s = stats.recommended_shift_range()
    say(f"\nbackground cosine for {stats.embedder_name or args.embedder}: mean "
        f"{stats.mean_cosine:.4f} (median {stats.median_cosine:.4f}, sd {stats.sd_cosine:.4f}) from "
        f"{stats.n_pairs:,} cross-protein residue pairs over {stats.n_proteins} proteins; "
        f"clustered SE {stats.se_clustered:.5f}")
    say(f"  shift in use: {shift_in_use:.4f}  ->  E[pair score] = {e_pair:+.4f}")
    say(f"  minimum viable shift (E[pair score] < 0): {stats.min_viable_shift:.4f}")
    say(f"  recommended sweep range (E[pair score] in {VIABLE_EXPECTED_SCORE_RANGE}): "
        f"{lo_s:.4f} to {hi_s:.4f}")
    say(f"  E[pair score] / sd = {stats.signal_to_noise(shift_in_use):+.4f} against BLOSUM62's "
        f"{BLOSUM62_EXPECTED_OVER_SD:+.4f}; SNR-matched shift {stats.snr_matched_shift():.4f}, "
        f"BLOSUM62-analogous (E/max) shift {stats.blosum62_analogous_shift():.4f}")
    bound = stats.ungapped_lambda(shift_in_use)
    onset = stats.gumbel_onset(shift_in_use)
    say(f"  ungapped lambda at this shift: {bound:.4f}  ->  ANY gapped lambda fitted below must "
        f"come out under it,\n    because gaps can only raise an optimal score and flatten its "
        f"tail (BLOSUM62: 0.318 ungapped, 0.267 gapped at 11/1)")
    say(f"  Gumbel onset lambda*sd = {onset:.4f}. Small means the asymptotics need long "
        f"alignments, so a fit over\n    short queries measures the approach to the asymptote "
        f"and not lambda, which shows up as lambda\n    depending on query length. Below "
        f"roughly 0.3 expect exactly that.")
    if e_pair >= 0:
        say("\n*** STOP. E[pair score] >= 0, so an optimal local alignment does not terminate: it grows "
            "with the\n    sequences, the score measures length rather than similarity, and the score "
            "distribution is\n    Gaussian rather than Gumbel. There is no E-value to invert and no "
            "floor to derive at this\n    shift, and no amount of sampling changes that. Re-run with "
            f"--shift inside the range above.")
        log.close()
        return 3
    if args.scan_only:
        say("\n--scan-only: stopping before any dynamic programming.")
        log.close()
        return 0

    spec = f"eba:{args.embedder}" + (f":shift={args.shift}" if args.shift is not None else "")
    kw = {"top_k": 0, "prefilter": "none", "precompute": False}
    if not (args.embedder in ("blosum", "blosum62") or args.embedder.startswith("kmerpos")):
        kw["batch_size"] = 8
    t0 = time.time()
    cos_scorer = get_scorer(spec, ref_seqs, **kw)
    say(f"cosine scorer: {cos_scorer.name}  (ready in {time.time() - t0:.0f}s)")

    # Calibration queries spread across the ORF length range, so lambda's length-independence is
    # testable rather than assumed.
    by_len = sorted(cons, key=lambda c: len(c[2]))
    step = max(len(by_len) // max(args.n_calib_queries, 1), 1)
    calib = [c[2] for c in by_len[::step]][: args.n_calib_queries]
    say(f"\ncalibrating on {len(calib)} shuffled ORFs x {args.n_shuffles} shuffles = "
        f"{len(calib) * args.n_shuffles} catalogue passes; lengths "
        f"{sorted({len(q) for q in calib})}")
    fit_kw = dict(n_boot=args.n_boot, min_r2=args.min_r2, n_length_bins=args.n_length_bins,
                  effective_length=not args.no_effective_length)
    cache = args.out_dir / "shuffle_scores.npz"

    if args.fit_from is not None:
        z = np.load(args.fit_from)
        rows = [z[k] for k in sorted((k for k in z.files if k.startswith("row")),
                                     key=lambda k: int(k[3:]))]
        lens = list(z["query_lens"])
        say(f"\nre-fitting from {args.fit_from}: {len(rows)} cached score rows, no scoring")
    else:
        # One timed pass, then a refusal rather than a surprise. the working record is 2 h 45 m spent
        # blind on a scoring system the pre-flight above now rejects in under a second; this is the
        # second guard, for the case where the scorer is fine and the budget is not.
        t = time.time()
        _ = cos_scorer.score_all(calib[len(calib) // 2])
        per_pass = time.time() - t
        total_passes = len(calib) * args.n_shuffles
        est = per_pass * total_passes / 60.0
        say(f"\none timed catalogue pass: {per_pass:.1f}s at m={len(calib[len(calib) // 2])} -> "
            f"{total_passes} passes is about {est:.0f} min "
            f"(cost is linear in query length x database residues, so halving --n-subjects halves it "
            f"and does NOT reduce the tail sample, which scales with --n-shuffles)")
        if est > args.max_minutes:
            say(f"*** refusing to start: {est:.0f} min exceeds --max-minutes {args.max_minutes:g}. "
                f"Lower --n-subjects (cheaper, same tail) or --n-shuffles (cheaper, smaller tail), "
                f"or raise --max-minutes deliberately.")
            log.close()
            return 4
        say(f"calibrating on {len(calib)} shuffled ORFs x {args.n_shuffles} shuffles")
        t0 = time.time()
        last = [0.0]

        def progress(done: int, total: int) -> None:
            if time.time() - last[0] > 60:
                last[0] = time.time()
                eta = (time.time() - t0) / max(done, 1) * (total - done) / 60.0
                print(f"        {done}/{total} ({time.time() - t0:.0f}s, ~{eta:.0f} min left)",
                      flush=True)

        rows, lens = score_shuffles(cos_scorer, calib, n_shuffles=args.n_shuffles, seed=args.seed,
                                    progress=progress)
        # Persisted BEFORE the fit, always. The fit is milliseconds and the scoring is hours.
        np.savez_compressed(cache, query_lens=np.asarray(lens),
                            **{f"row{i}": r for i, r in enumerate(rows)})
        say(f"  scored in {time.time() - t0:.0f}s; cached {len(rows)} rows to {cache.name}: "
            f"re-fit with --fit-from {cache} at no cost")

    fit = fit_evd(rows, lens, ref_seqs and [len(x) for x in ref_seqs], seed=args.seed,
                  scorer_name=cos_scorer.name, ungapped_lambda_bound=bound, **fit_kw)
    say(f"  lambda = {fit.lam:.6f}  95% CI (clustered on shuffled query) "
        f"[{fit.lam_ci[0]:.6f}, {fit.lam_ci[1]:.6f}]")
    say(f"  K      = {fit.k_param:.6g}   (an extrapolation of the tail intercept back to S = 0, so "
        f"it is\n           not independently interpretable: the quantity in use is the floor "
        f"below, and\n           lambda is what the data constrain)")
    say(f"  tail regression R^2 = {fit.r_squared} over E_hat in "
        f"[{fit.tail_lo}, {fit.tail_hi}] per query, {fit.n_tail_points} points, "
        f"{fit.n_pairs:,} pairs from {fit.n_queries} shuffled queries")
    say(f"  lambda by query-length bin ({fit.n_length_bins} bins): {fit.lam_by_query_len}")
    say(f"  R^2    by query-length bin: {fit.r2_by_query_len}")
    if fit.lams_above_bound:
        say(f"  *** above the ungapped bound {bound:.4f}: {fit.lams_above_bound} (0 = the pooled "
            f"fit). A gapped lambda\n      cannot exceed the ungapped one, so these are not "
            f"estimates of lambda. This is theory against\n      the fit and needs no R^2 "
            f"threshold; it is reported and does not change the pre-registered\n      verdict "
            f". ***")
    else:
        say(f"  every fitted lambda is under the ungapped bound {bound:.4f}, as theory requires")
    say(f"  finite-length correction: {'on' if fit.effective_length else 'OFF'}; mean effective query "
        f"length {fit.mean_effective_query_len:g} against a mean raw length of "
        f"{sum(lens) / len(lens):.0f}")
    if not fit.usable:
        say(f"\n*** THE FIT IS REJECTED, and this is a result rather than a crash:\n    "
            f"{fit.rejected_reason}")
        say(f"\n    The {len(rows)} score rows are cached at {cache if args.fit_from is None else args.fit_from}. "
            f"Re-fit with --fit-from and a different\n    --n-length-bins or --min-r2 at no cost; "
            f"nothing needs re-scoring. Report this as a negative\n    result with its mechanism "
            f"rather than widening the tail window until it passes.")
        out = {"catalogue_sample": {"n_proteins": len(refs), "db_residues": db_residues},
               "scorer": cos_scorer.name, "embedder": args.embedder, "shift": shift_in_use,
               "background_pair_stats": stats_record(stats, shift_in_use),
           "valid_for_promotion": ok2, "valid_for_promotion_reason": why2,
           "verdict_by_bin_count": scan, "lams_above_ungapped_bound": list(fit.lams_above_bound), "rejected": True,
               "rejected_reason": fit.rejected_reason,
               "fit": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(fit).items()}}
        (args.out_dir / "cosine_floor.json").write_text(json.dumps(out, indent=2, default=str),
                                                        encoding="utf-8")
        say(f"\nwrote cosine_floor.json and the score cache in {args.out_dir}")
        log.close()
        return 5
    ok2, why2 = fit.valid_for_promotion(args.min_r2)
    if ok2:
        say("  rule 2 (every length bin at or above "
            f"{args.min_r2}): PASSED, a floor may be promoted from this fit")
    else:
        say(f"  *** rule 2 (every length bin at or above {args.min_r2}): FAILED, "
            f"{why2}.\n      The fit is not refuted (`usable` asks whether a Gumbel regime exists at "
            f"all and tolerates\n      one bin), but no floor may be promoted from it. These are "
            f"different tests and reading\n      the first as the second is the working record. ***")
    # The verdict must not depend on a free parameter, so it is reported at every resolution the
    # sample supports: roughly 30 queries per bin, below which the check reports sampling noise.
    scan = []
    for nb in range(2, max(len(rows) // 30, 2) + 1):
        try:
            f2 = fit_evd(rows, lens, ref_seqs and [len(x) for x in ref_seqs], n_boot=0,
                         n_length_bins=nb, min_r2=args.min_r2,
                         effective_length=not args.no_effective_length,
                         ungapped_lambda_bound=bound)
        except ValueError:
            continue
        scan.append((nb, f2.usable, f2.valid_for_promotion(args.min_r2)[0],
                     round(100 * f2.lam_spread_across_lengths / f2.lam)))
    say("  verdict by bin count (bins, un-refuted, promotable, lambda spread %): " + str(scan))
    pct = 100 * fit.lam_spread_across_lengths / fit.lam
    say(f"  spread across length bins = {fit.lam_spread_across_lengths:.6f} ({pct:.1f}% of lambda)")
    say(f"  lambda is a property of the scoring system and must NOT depend on length. Read this with "
        f"n in\n  hand: {fit.n_queries} shuffled queries over {fit.n_length_bins} bins is "
        f"{fit.n_queries // max(fit.n_length_bins, 1)} per bin. Below roughly 30 per bin this "
        f"number\n  reports sampling noise, not the scorer: raise --n-shuffles before concluding "
        f"anything from it.")
    if pct > 25 and fit.n_queries >= 90:
        say("  *** the spread is large at a sample that can support the check: the fit has absorbed a "
            "length\n      effect into a constant that has none, and the floor will be wrong at the "
            "lengths furthest\n      from the calibration set. Report this rather than tuning it "
            "away. ***")
    say(f"  floor on this sample: "
        + ", ".join(f"m={m}: {fit.score_floor(args.max_evalue, m, db_residues):.3f}"
                    for m in (100, 300, 1000, 3000)))
    say(f"  the alignment floor for comparison (different units, same equation): "
        + ", ".join(f"m={m}: {score_floor(args.max_evalue, m, db_residues):.1f}"
                    for m in (100, 300, 1000, 3000)))

    # --- the acceptance test: both segmenters on the same sampled catalogue -----------------------
    # Three floors, not one: the pooled lambda and the per-bin extremes. A lambda that varies across
    # query-length bins is a precision problem, so the verdict has to be shown to survive the band
    # rather than quoted at the midpoint.
    lo_lam, hi_lam = fit.lam_range
    variants = [("pooled", fit)]
    if abs(hi_lam - lo_lam) > 1e-9:
        variants += [("lam_min", fit.with_lam(lo_lam)), ("lam_max", fit.with_lam(hi_lam))]
    say(f"\nsegmenting {len(cons)} constructs on the SAME {len(refs):,}-protein sample: once under "
        f"exact alignment and once per lambda variant {[n for n, _ in variants]}")
    say(f"  lambda band {lo_lam:.6f} to {hi_lam:.6f}; floors at m=500: "
        + ", ".join(f"{n}={f.score_floor(args.max_evalue, 500, db_residues):.3f}"
                    for n, f in variants))
    align_scorer = get_scorer("align", ref_seqs)
    rows = []
    for i, (pid, acc, orf) in enumerate(cons, 1):
        a = best_hit_segments(orf, ref_seqs, align_scorer, db_residues, ref_accessions=ref_acc,
                              max_evalue=args.max_evalue)
        row = dict(plasmid_id=pid, payload_accession=acc, orf_len=len(orf),
                   align_k=len(a.segments), align_iterations=len(a.trace),
                   align_stop=a.stop_reason, align_floor=round(a.floor, 3))
        for name, variant in variants:
            c = best_hit_segments(orf, ref_seqs, cos_scorer, db_residues, ref_accessions=ref_acc,
                                  max_evalue=args.max_evalue, floor_fn=variant.score_floor)
            suffix = "" if name == "pooled" else f"_{name}"
            row.update({f"cosine{suffix}_k": len(c.segments),
                        f"cosine{suffix}_iterations": len(c.trace),
                        f"cosine{suffix}_stop": c.stop_reason,
                        f"cosine{suffix}_floor": round(c.floor, 4)})
        row["k_delta"] = row["cosine_k"] - row["align_k"]
        row["iterations_delta"] = row["cosine_iterations"] - row["align_iterations"]
        rows.append(row)
        print(f"    {i:>3}/{len(cons)}  {pid[:36]:<38} align k={row['align_k']} it="
              f"{row['align_iterations']}  {row['align_stop']:<12} | "
              + " ".join(f"{n} k={row['cosine' + ('' if n == 'pooled' else '_' + n) + '_k']}"
                         for n, _ in variants), flush=True)

    with open(args.out_dir / "cosine_floor_detail.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    agree = agreement(rows, [n for n, _ in variants])
    for key, a in agree.items():
        q, vname = (key.split("_", 1) + ["pooled"])[:2] if "_" in key else (key, "pooled")
        say(f"\n{q} [{vname}]: exact agreement {a['exact']}/{a['n']} ({a['exact_pct']}%), "
            f"within 1 {a['within_1']}/{a['n']}; "
            f"cosine higher {a['cosine_higher']}, lower {a['cosine_lower']}")
        say(f"  align  {a['align']}")
        say(f"  cosine {a['cosine']}")
        say(f"  confusion (align -> cosine): {a['confusion']}")

    stops = {s: sum(1 for r in rows if r["cosine_stop"] == s)
             for s in sorted({r["cosine_stop"] for r in rows})}
    say(f"\ncosine stop reasons: {stops}")
    say(f"alignment stop reasons on the same sample: "
        f"{ {s: sum(1 for r in rows if r['align_stop'] == s) for s in sorted({r['align_stop'] for r in rows})} }")
    say("\nREAD THIS AS THE RESULT. Agreement on `iterations` is the stopping rule's own accuracy; "
        "agreement on\n`k` is whether that accuracy survives span emission. Retrieval is NOT the "
        "criterion and is not\nreported here: see the module docstring. If the distributions do not "
        "agree, that is the finding:\ndo not tune the null until they match.")

    out = {"catalogue_sample": {"n_proteins": len(refs), "db_residues": db_residues,
                                "n_in_window_full": len(full), "seed": args.seed,
                                "reference": str(args.reference)},
           "scorer": cos_scorer.name, "embedder": args.embedder, "shift": args.shift,
           "max_evalue": args.max_evalue, "n_constructs": len(rows),
           "fit": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(fit).items()},
           "background_pair_stats": stats_record(stats, shift_in_use),
           "valid_for_promotion": ok2, "valid_for_promotion_reason": why2,
           "verdict_by_bin_count": scan, "lams_above_ungapped_bound": list(fit.lams_above_bound),
           "fit_lambda_spread_across_lengths": fit.lam_spread_across_lengths,
           "agreement": agree, "cosine_stop_reasons": stops,
           "span_claiming": "BLOSUM traceback (oracle_span); eba.py has no traceback, declared hybrid"}
    (args.out_dir / "cosine_floor.json").write_text(json.dumps(out, indent=2, default=str),
                                                    encoding="utf-8")
    say(f"\nwrote cosine_floor.json, cosine_floor_detail.csv, cosine_floor.log in {args.out_dir}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
