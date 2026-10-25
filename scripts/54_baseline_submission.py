#!/usr/bin/env python3
"""The benchmark's baseline row: this project's own arms, re-expressed under the benchmark's rules.

    python scripts/54_baseline_submission.py --benchmark benchmark/v1

**Why this needs no run.** Every one of the 766 published items already has an exact Smith-Waterman rank
on disk, at all six holdout thresholds, for all six arms, measured against the same 40,750-protein
catalogue the benchmark publishes: 20,806,731 in-window residues in the benchmark's MANIFEST and in all
three runs' logs. So the baseline is a **re-expression of a finished measurement**, not a new one.
Regenerating it would cost 7 h 0 m on the ThinkPad (hand 400.6 s, corpus 7,044.8 s, synth_real
17,765.8 s); reading it costs a fifth of a second. The run that would be needed is the one the baseline
cannot supply: an embedding arm on the corpus and synth_real, which is a GPU job.

**What a baseline is for.** Without one, a reviewer can score a method and has nothing to compare it
with. With one, the artifact carries both the claim and its boundary: at the 25% holdout in the
`f < 0.50` stratum of `synth_real`, `whole` reaches `recall@10` 0.056 and `bhs_per_segment` 0.667, while
in the corpus `f >= 0.95` stratum the two are **identical**: decomposition does nothing where there is
nothing to decompose.

**The one thing that must be stated or the row reads as an error.** The benchmark keeps an unscorable
item in the denominator as a miss; this project's own internal summaries drop it. Same ranks, different
`n`, so **the baseline does not match the published numbers** and should not be expected to. This script
computes the gap rather than describing it, cell by cell, and writes it into the artifact beside the
figures: mean +0.119 `recall@10` over the headline cells, max +0.520, exactly equal in 21 of 48. Neither
rule is wrong; the benchmark's is the stricter one, because an item nobody can retrieve is a result.

Writes `baseline_{arm_kind}.json` and `.csv` into the benchmark directory; `--suffix` adds a separate
variant without replacing the canonical file.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.benchmark import DEPTH, STRATA, THRESHOLDS, load_items, report  # noqa: E402
from pilot.decompose import span_rows_at  # noqa: E402

#: Where each tier's finished alignment ranks live, and the runtime each cost, so the "no run needed"
#: claim carries its own evidence. All four ran `--scorer align` against the 40,750-protein catalogue.
#:
#: **Repointed 22 Oct at the post-fix runs, when I promoted them to canonical.** The
#: default has to write the canonical pair, or `54` with no flags produces something the published
#: directory does not contain, which is what `test_the_flagless_run_reproduces_the_published_
#: alignment_baseline` exists to catch, and what it caught here. The superseded pre-fix set is no
#: longer a default and is passed explicitly with `--suffix prer10`; the four source paths for it are
#: in `the release directives` step 2 and in the `_prer10` files' own provenance.
#:
#: `synth_random`'s 16,643.9 s is a **ThinkPad, 3-thread** figure where the other three are Omen,
#: 14-thread. The sum is therefore across two machines and is not a prediction for either; it is
#: reported rather than normalised, because a normalised runtime is a number nobody measured.
SOURCES: tuple[tuple[str, str, float], ...] = (
    ("hand", "results/maskfix_hand_perthr/twilight_sw-blosum62-11-1_hand.csv", 153.3),
    ("corpus", "results/maskfix_corpus/twilight_sw-blosum62-11-1_corpus.csv", 2671.2),
    ("synth_real", "results/maskfix_synthreal/twilight_sw-blosum62-11-1_synth_real.csv", 6769.0),
    ("synth_random", "results/maskfix_synthrandom/twilight_sw-blosum62-11-1_synthetic.csv", 16643.9),
)

#: The two arms the baseline leads with: the query a naive pipeline submits, and the label-free
#: decomposition this project proposes, ranked one segment at a time. The others are carried too:
#: `oracle` is a ceiling that uses the label and is not a method, and the pooled `auto`/`bhs` must never
#: be quoted as an arm's result.
HEADLINE_ARMS = ("whole", "bhs_per_segment")

#: The arms whose ranks depend on the segmenter, and therefore on the mask the segmenter was handed.
#: `whole` and `oracle` are NOT among them: each queries a fixed span against the catalogue masked for
#: scoring, and that mask was per-threshold throughout. So a segmentation defect reaches four of this
#: baseline's six arms and cannot reach the other two, which is why the two lists are written out.
SEGMENTED_ARMS = ("auto", "auto_per_segment", "bhs", "bhs_per_segment")
UNSEGMENTED_ARMS = ("oracle", "whole")


def segmentation_provenance(source: str) -> dict:
    """Was this source's segmentation built per holdout threshold? Read from the run's own span CSV.

    `10_twilight.py` built the segmenter's holdout mask **once**, at `min(THRESHOLDS)`, and
    reused it at every looser threshold. A payload whose whole homolog family is masked at 0.25 cannot
    be claimed at all, so the loop stops at `k = 1` holding the wrong segment, which manufactures
    reported **loss** at every threshold above the strictest one: 14,717 rank points of loss and 6 of
    gain, measured on the corpus. this project fixed it and added a `threshold` column to the span CSV
    *precisely* so that a later replay could not inherit the defect silently.

    That column is the evidence, and this function reads it rather than asserting anything. The reason
    it has to exist: this baseline publishes `bhs_per_segment` as a headline arm, so it owes a reader
    the difference between a cell that is a measurement and a cell that is a lower bound. The check is
    free, one header line per source, and this project's rule is that a limit which binds gets counted
    at construction time rather than discovered afterwards.

    A source with no span CSV beside it returns `per_threshold: None`: unknown is not clean. Note that a
    replayed run (`bhs_scorer = replayed:...` in its summary) carries its parent's segmentation, so its
    own span file is the parent's and this check reads the right thing either way.

    `n_span_rows` and `n_constructs` both describe ONE segmentation, the one at the strictest holdout
, which is why the per-threshold case goes through `pilot.decompose.span_rows_at` rather than
    counting the file: a six-threshold file holds six segmentations, and counting them as one is
    precisely the six-fold `k_bhs` that reader exists to refuse.

    `n_constructs_any_threshold` is the other denominator, kept apart from it under its own name because
    the two are not interchangeable and this project has twice paid for a denominator that travelled
. Their **difference is a measurement of this project's defect**: a construct that
    emits no segment at 0.25, because its whole homolog family is masked there, emits one at a looser
    holdout, so on a per-threshold file the second count exceeds the first by exactly the constructs
    the old single-mask run could not segment at all. On the corpus that is 406 against 308.
    """
    d = Path(source).parent
    spans = sorted(d.glob("bhs_spans_*.csv"))
    if not spans:
        return dict(spans_file=None, per_threshold=None, n_span_rows=None, n_constructs=None,
                    n_constructs_any_threshold=None, lower_bound_thresholds=[],
                    reason="no bhs_spans_*.csv beside the ranks, so the segmentation cannot be placed")
    f = spans[0]
    with open(f, newline="", encoding="utf-8-sig") as fh:
        rdr = csv.DictReader(fh)
        per_threshold = "threshold" in (rdr.fieldnames or [])
        rows = list(rdr)
    strictest = min(THRESHOLDS)
    one = span_rows_at(rows, strictest) if per_threshold else span_rows_at(rows)
    return dict(
        spans_file=f.name,
        per_threshold=per_threshold,
        n_span_rows=len(one),
        n_constructs=len({r["plasmid_id"] for r in one}),
        n_constructs_any_threshold=len({r["plasmid_id"] for r in rows}),
        lower_bound_thresholds=[] if per_threshold else [t for t in THRESHOLDS if t > strictest],
        reason=("the span CSV carries a `threshold` column, so each holdout was segmented under its own "
                "mask" if per_threshold else
                f"the span CSV has no `threshold` column and holds one segmentation per construct, so "
                f"the mask was built once at {strictest:g} and reused at every looser holdout "
                f""),
    )


#: What produced the ranks, one string per arm kind. It is metadata a reviewer reads to know what they
#: are comparing against, so it cannot be a single literal once this script serves two arms. The
#: embedding entry names the pooling, the width and the length cap, because all three are in the cache
#: key and a row measured under a different cap is a different measurement.
SCORER_NOTE = {
    "alignment": "sw-blosum62-11-1 (exact Smith-Waterman, parasail)",
    "embedding": ("esm2_t33_650M_UR50D, mean-pooled over the attention mask, cosine; "
                  "--batch-size 8 --max-length 2048"),
}


def parse_sources(specs: list[str] | None) -> tuple[tuple[str, str, float], ...]:
    """`LABEL=PATH[:SECONDS]` strings into the same shape as `SOURCES`, or `SOURCES` when none are given.

    Three reasons this is a flag rather than an edit to `SOURCES`, all of them found by writing the run board out line by line on 20 Oct:

    * the arm kind, the output filename and `SOURCES` are three separate hard-coded things, so editing
      one of them in place writes embedding numbers into `baseline_alignment.json` **over** the published
      alignment baseline, labelled as alignment numbers. A flag cannot do that, because the filename is
      derived from the arm kind.
    * a re-expression of finished runs should be a *command*, not a diff. T8a's lesson was that this
      baseline needs no run at all because the ranks are already on disk; a flag keeps that true for the
      next arm as well.
    * `SECONDS` is optional and only feeds `provenance` and `regeneration_cost_seconds`, which exist so
      the "no run needed" claim carries its own evidence. Omitting it costs that evidence and nothing
      else, so it must not be mandatory: a runtime nobody recorded is better left absent than guessed.

    The path is split on the LAST colon so a Windows drive letter survives, and a path with no colon at
    all is accepted with no runtime.
    """
    if not specs:
        return SOURCES
    out = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--source needs LABEL=PATH[:SECONDS], got {spec!r}")
        label, rest = spec.split("=", 1)
        secs = 0.0
        head, sep, tail = rest.rpartition(":")
        # a bare "C:" head means `rest` was a drive-qualified path with no runtime, not PATH:SECONDS
        if sep and head and len(head) > 1:
            try:
                secs = float(tail)
                rest = head
            except ValueError:
                pass
        out.append((label.strip(), rest.strip(), secs))
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", type=Path, default=Path("benchmark/v1"))
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--arm-kind", choices=("alignment", "embedding"), default="alignment",
                    help="which metric rule the rows are rendered under, and the name of the file they "
                         "are written to. for an embedding arm recall@10 and MRR are "
                         "primary and `mean_rank_phrase` REFUSES to render a mean rank without its "
                         "count of rank >= 10,000 failures. That refusal is the point of this flag and "
                         "must not be worked around if it fires: it is reporting a real bimodality.")
    ap.add_argument("--suffix", default="", metavar="SUFFIX",
                    help="append _SUFFIX to the output filenames, leaving the canonical baseline files "
                         "untouched")
    ap.add_argument("--source", action="append", default=None, metavar="LABEL=PATH[:SECONDS]",
                    help="replace the built-in SOURCES with these finished twilight CSVs, one flag per "
                         "tier. Omit it entirely and the three alignment runs are used exactly as "
                         "before, which is what keeps the published baseline reproducible by the same "
                         "command that first wrote it.")
    args = ap.parse_args(argv)
    if args.suffix and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.suffix):
        ap.error("--suffix must start with a letter or digit and contain only letters, digits, '_' or '-'")
    try:
        sources = parse_sources(args.source)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    stem = f"baseline_{args.arm_kind}" + (f"_{args.suffix}" if args.suffix else "")

    bd = args.benchmark
    if not (bd / "items.csv").is_file():
        print(f"not found: {bd / 'items.csv'}  (run scripts/52_build_benchmark.py)", file=sys.stderr)
        return 2

    t0 = time.time()
    items = {i.construct_id: i for i in load_items(bd / "items.csv")}
    # A benchmark whose selected tiers share construct ids publishes them as `tier:construct_id`, and
    # says so in its MANIFEST rather than leaving a reader to infer it from the ids (52's refusal). The
    # twilight CSVs this script reads carry the BARE id in `plasmid_id`, so the join has to be built the
    # same way the benchmark was. Read from the manifest and not guessed: a wrong guess here joins zero
    # rows, which the coverage check below reports but does not refuse.
    namespaced = False
    mf = bd / "MANIFEST.json"
    if mf.is_file():
        namespaced = bool(json.loads(mf.read_text(encoding="utf-8-sig"))
                          .get("construct_ids_namespaced"))
    # tier -> arm -> records, and tier -> arm -> raw rank lists per cell for the denominator comparison
    rec: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    cells: dict[tuple, list] = defaultdict(list)
    covered: dict[str, set] = defaultdict(set)
    missing_sources = []
    for tier, path, _secs in sources:
        p = Path(path)
        if not p.is_file():
            missing_sources.append(path)
            continue
        with open(p, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                key = f"{tier}:{r['plasmid_id']}" if namespaced else r["plasmid_id"]
                it = items.get(key)
                if it is None or it.tier != tier:
                    continue
                rank = int(r["best_target_rank"]) if r["best_target_rank"] != "" else None
                thr = float(r["threshold"])
                rec[tier][r["arm"]].append((it, thr, rank))
                cells[(tier, r["arm"], thr, it.stratum)].append(rank)
                covered[tier].add(key)
    if missing_sources:
        print(f"missing source CSV(s): {missing_sources}. The baseline would be partial; refusing.",
              file=sys.stderr)
        return 2

    # Coverage is the claim this whole script rests on, so it is counted rather than assumed.
    per_tier = defaultdict(set)
    for i in items.values():
        per_tier[i.tier].add(i.construct_id)
    coverage = {t: {"items": len(per_tier[t]), "with_ranks": len(covered[t] & per_tier[t])}
                for t in sorted(per_tier)}
    gaps = {t: v["items"] - v["with_ranks"] for t, v in coverage.items()}
    print(f"coverage: " + ", ".join(f"{t}  {v['with_ranks']}/{v['items']}"
                                   for t, v in coverage.items()))
    if any(gaps.values()):
        print(f"    {sum(gaps.values())} published item(s) have no rank on disk: {gaps}. Those cells are "
              f"absent from the baseline rather than counted as misses, because the absence is OURS "
              f"and not a method's failure. Say so wherever the row is quoted.")

    # ---- which cells are measurements and which are lower bounds ------------------------------
    seg = [dict(tier=t, **segmentation_provenance(p)) for t, p, _ in sources]
    lb_thr = sorted({thr for sp in seg for thr in sp["lower_bound_thresholds"]}, reverse=True)
    for sp in seg:
        flag = {True: "per-threshold", False: "BUILT ONCE", None: "unknown"}[sp["per_threshold"]]
        print(f"  segmentation [{sp['tier']}]: {flag}, {sp['reason']}")
    if lb_thr:
        print(f"  so {', '.join(f'{t:.0%}' for t in lb_thr)} x {len(SEGMENTED_ARMS)} segmented arms are "
              f"LOWER BOUNDS, not measurements. `whole` and `oracle` are unaffected at every threshold.")

    rows_out = []
    for tier in coverage:
        for arm in sorted(rec[tier]):
            for r in report(rec[tier][arm], depth=args.depth, arm_kind=args.arm_kind):
                rows_out.append(dict(tier=tier, arm=arm, **r))

    # ---- the denominator comparison, computed rather than described ----------------------------
    # Same ranks, two rules: the benchmark counts an unscorable item as a miss, the twilight summaries
    # drop it. A reader comparing the baseline with CONTEXT needs the size of that, not a warning.
    deltas = []
    for (tier, arm, thr, st), ranks in cells.items():
        if arm not in HEADLINE_ARMS or thr not in (0.25, 0.70):
            continue
        got = [r for r in ranks if r is not None]
        if not ranks or not got:
            continue
        hit = sum(1 for r in got if r <= args.depth)
        deltas.append(dict(tier=tier, arm=arm, threshold=thr, stratum=st, n=len(ranks),
                           n_scorable=len(got),
                           benchmark_rule=round(hit / len(ranks), 4),
                           twilight_rule=round(hit / len(got), 4),
                           delta=round(hit / len(got) - hit / len(ranks), 4)))
    ds = [d["delta"] for d in deltas]
    denom = dict(
        what=("the benchmark keeps an unscorable item in the denominator as a miss; this "
              "project's own internal summaries drop it. Identical ranks, different n"),
        cells_compared=len(deltas),
        mean_delta=round(sum(ds) / len(ds), 4) if ds else None,
        max_delta=round(max(ds), 4) if ds else None,
        cells_in_exact_agreement=sum(1 for d in ds if d == 0),
        which_is_stricter="the benchmark's: an item nobody can retrieve is a result, not an absence",
        per_cell=sorted(deltas, key=lambda d: -d["delta"]),
    )

    print(f"\n{'tier':<11}{'arm':<18}{'hold':>6}{'stratum':>12}{'n':>5}{'unsc':>6}"
          f"{'rec@' + str(args.depth):>8}{'macro':>8}{'MRR':>8}{'meanR':>10}")
    for r in rows_out:
        if r["arm"] not in HEADLINE_ARMS or r["threshold"] not in (0.70, 0.25):
            continue
        print(f"{r['tier']:<11}{r['arm']:<18}{r['threshold']:>6.0%}{r['stratum']:>12}{r['n']:>5}"
              f"{r['n_unscorable']:>6}{r[f'recall_at_{args.depth}']:>8.3f}"
              f"{(r[f'recall_at_{args.depth}_macro'] or 0):>8.3f}{(r['mrr'] or 0):>8.3f}"
              f"{(r['mean_rank'] or 0):>10.1f}")

    print(f"\ndenominator rule vs the published summaries: {denom['cells_compared']} cells, "
          f"mean {denom['mean_delta']:+.3f} recall@{args.depth}, max {denom['max_delta']:+.3f}, "
          f"exactly equal in {denom['cells_in_exact_agreement']}")

    out = dict(
        baseline="this project's own arms under the benchmark's rules",
        # A suffixed file is a VARIANT, and has to say so in itself: `benchmark/README.md` and
        # `53_score_submission.py` point at the unsuffixed name, so a reader who opens this one from
        # disk has no other way to know it is not the row a submission is scored against. Regenerating
        # the canonical baseline is a release decision and belongs.
        is_canonical=not args.suffix,
        variant=args.suffix or None,
        file_stem=stem,
        arm_kind=args.arm_kind, scorer=SCORER_NOTE[args.arm_kind],
        depth=args.depth, thresholds=list(THRESHOLDS),
        strata=[n for n, _, _ in STRATA],
        headline_arms=list(HEADLINE_ARMS),
        provenance=[dict(tier=t, source=p,
                         original_runtime_seconds=s if s else None) for t, p, s in sources],
        no_run_needed=dict(
            coverage=coverage,
            regeneration_cost_seconds=round(sum(s for _, _, s in sources), 1) or None,
            read_only_cost_seconds=None,  # filled below
            note=("the published ranks were measured against the same catalogue the benchmark ships: "
                  "40,750 entries and 20,806,731 in-window residues in the MANIFEST and in all three "
                  "runs' logs"),
        ),
        segmentation=dict(
            per_source=seg,
            segmented_arms=list(SEGMENTED_ARMS),
            unsegmented_arms=list(UNSEGMENTED_ARMS),
            lower_bound_thresholds=lb_thr,
            note=("a segmenter mask built once at the strictest threshold manufactures "
                  "reported loss at every looser one, so where `per_threshold` is false the four "
                  "segmented arms at those thresholds are LOWER BOUNDS on what decomposition does and "
                  "must not be quoted as its cost. `whole` and `oracle` are unaffected everywhere, "
                  "which is where this baseline's headline comparison lives. this project fixed the "
                  "mask; this project keeps a register of which cells have been re-run since"
                  if lb_thr else
                  "every source segmented under its own per-threshold mask, so no cell "
                  "here is a lower bound on that account"),
        ),
        arms_note=("`oracle` uses the label and is a ceiling, not a method. The pooled `auto` and `bhs` "
                   "are carried for continuity and must never be quoted as an arm's result"),
        denominator_difference=denom,
        rows=rows_out,
    )
    out["no_run_needed"]["read_only_cost_seconds"] = round(time.time() - t0, 3)

    with open(bd / f"{stem}.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2)
    with open(bd / f"{stem}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        flat = [{k: (",".join(v) if isinstance(v, list) else v) for k, v in r.items()}
                for r in rows_out]
        w = csv.DictWriter(fh, fieldnames=list(flat[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(flat)
    cost = out["no_run_needed"]["regeneration_cost_seconds"]
    print(f"\nwrote {bd}/{stem}.json and .csv ({len(rows_out)} rows) in "
          f"{out['no_run_needed']['read_only_cost_seconds']}s, against "
          + (f"{cost}s to regenerate the ranks" if cost else
             "an unrecorded cost to regenerate them: pass LABEL=PATH:SECONDS to record it; the "
             "number is `runtime_seconds` in each run's own *_summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
