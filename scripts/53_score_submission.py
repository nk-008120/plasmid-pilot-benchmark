#!/usr/bin/env python3
"""Score a submission against the published benchmark, under the project's own metric rules.

    python scripts/53_score_submission.py --benchmark benchmark/v1 --submission my_ranks.csv
    python scripts/53_score_submission.py --benchmark benchmark/v1 --submission scores.npy --arm embedding

**Two accepted submission forms**, scored by the same code:

* **`.csv`, a ranked list per item**: columns `construct_id,rank,accession`, any depth. Cheap to produce.
  A true rank deeper than the submitted list is **unknown, not large**, so the scorer counts how many
  cells that bound and says so rather than averaging over the rest as though they were found.
* **`.npy`, a score matrix**: one row per item in `items.csv` order, one column per entry in
  `catalogue_accessions.txt` order, higher is better. Exact ranks at every depth, and the only form in
  which a mean rank is fully defined. The shape is checked, because a transposed matrix is the one error
  here that would still produce plausible-looking numbers.

**`--arm` is not cosmetic.** `alignment` and `embedding` select which metrics are primary, because under
mean-pooling the rank distribution is bimodal where alignment's is not: the corpus mean
rank of 1,088 at the 90% holdout is median rank 1 plus fifteen constructs carrying one payload at
~29,699 of 40,750. For an embedding arm this scorer will not print a mean rank without its failure
count, and `recall@10` and MRR lead.

Every table is stratified by payload fraction and **no pooled scalar is printed**, because there is no
scalar at any depth that carries this result pooled (user, 17 Oct).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.benchmark import (  # noqa: E402
    DEPTH,
    Submission,
    load_items,
    load_submission,
    mean_rank_phrase,
    score_submission,
)
from pilot.homology import Homolog  # noqa: E402


def load_homologs(path: Path) -> dict[str, list[Homolog]]:
    out: dict[str, list[Homolog]] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["payload_accession"], []).append(
                Homolog(r["homolog_accession"], float(r["score"]), float(r["identity"]),
                        float(r["coverage"]), int(r["length"])))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", type=Path, default=Path("benchmark/v1"))
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--arm", choices=("alignment", "embedding"), default="alignment",
                    help="which metrics are primary;")
    ap.add_argument("--depth", type=int, default=DEPTH,
                    help="recall depth. 10 was fixed by the user 19 Oct and is a stated assumption "
                         "about review practice, not a measurement")
    ap.add_argument("--tier", default="", help="score one tier only. The three are never pooled")
    ap.add_argument("--out", type=Path, default=None, help="write the rows as JSON as well")
    args = ap.parse_args(argv)

    bd = args.benchmark
    for name in ("items.csv", "homologs.csv", "catalogue_accessions.txt"):
        if not (bd / name).is_file():
            print(f"not found: {bd / name}  (run scripts/52_build_benchmark.py)", file=sys.stderr)
            return 2
    if not args.submission.is_file():
        print(f"not found: {args.submission}", file=sys.stderr)
        return 2

    items = load_items(bd / "items.csv")
    if args.tier:
        items = [i for i in items if i.tier == args.tier]
        if not items:
            print(f"no items in tier {args.tier!r}", file=sys.stderr)
            return 2
    homologs = load_homologs(bd / "homologs.csv")
    catalogue = (bd / "catalogue_accessions.txt").read_text(encoding="utf-8").split()

    sub: Submission = load_submission(args.submission)
    if sub.kind == "matrix" and args.tier:
        print("a score matrix is indexed by items.csv row order, so --tier cannot subset it. "
              "Score the whole file and read the tier's rows, or submit a ranked list.",
              file=sys.stderr)
        return 2

    rows, bind = score_submission(sub, items, homologs, catalogue,
                                  depth=args.depth, arm_kind=args.arm)

    tiers = sorted({i.tier for i in items})
    print(f"benchmark {bd}  |  {len(items):,} items over {len(tiers)} tier(s): {', '.join(tiers)}")
    print(f"submission {args.submission.name}  |  form: {sub.kind}"
          + (f", depth {sub.depth_submitted}" if sub.kind == "ranked" else "")
          + f"  |  arm: {args.arm}")
    if len(tiers) > 1:
        # Derived from the build, not written as a number: the benchmark went from three tiers to
        # four on 22 Oct and this line still said three, two lines under a header that said four.
        print(f"NOTE: the {len(tiers)} tiers cover each other's weaknesses and are never pooled into "
              f"one number. Re-run with --tier to read one at a time.")

    # Binding counts first, because a table read without them can be a table about the submission's
    # depth rather than about the method.
    print("\nbinding counts (a count of zero means the limit is free and can be forgotten):")
    for k, v in bind.items():
        print(f"    {k:<34}{v}")
    if bind["rank_beyond_submitted_depth"]:
        print(f"  -> {bind['rank_beyond_submitted_depth']} cell(s) have no rank inside the submitted "
              f"list. Those are counted as misses in recall and MRR, which is right, and they are "
              f"EXCLUDED from the mean rank, which therefore describes only the cells that were found. "
              f"Quote it with this count beside it or submit a score matrix.")
    if bind["missing_from_submission"]:
        print(f"  -> {bind['missing_from_submission']} item(s) absent from the submission entirely. "
              f"Counted as misses, not dropped.")
    if bind["unknown_accessions"]:
        print(f"  -> {bind['unknown_accessions']} submitted accession(s) are not in the catalogue. They "
              f"occupy no rank; check you ranked against catalogue_accessions.txt.")

    d = args.depth
    # The mean-rank column is as wide as the widest phrase it must hold, measured from the rows
    # rather than chosen. `mean_rank_phrase` returns `1.00` for an alignment arm, `no scorable item`
    # for an empty cell and `1.00 (0 of 298 at rank >= 10,000)` for an embedding one. This carried a
    # constant 22, which fits the first two and is overrun by the third, and a format width is a
    # *minimum*, so the overrun did not wrap or raise: it silently ate the gap and printed the MRR
    # value joined to the mean rank as `1.00001.00`. Every number was right and the column was
    # unreadable, for the arm that is one of the two baselines published in `benchmark/v1`.
    # Widening the constant would only move the binding point, because that phrase grows with the
    # stratum's `n` and with `FAILURE_RANK`: a limit on the output that nothing printed. So the
    # width is derived here and the header below is padded from the same value, which is what keeps
    # the label and the cells on one right edge.
    mean_rank_cells = [mean_rank_phrase(r, arm_kind=args.arm) for r in rows]
    mr_w = max([len("mean rank")] + [len(c) for c in mean_rank_cells]) + 2
    print(f"\n{'holdout':>8}  {'stratum':<12}{'n':>5}{'payld':>6}{'unscor':>7}"
          f"{'recall@' + str(d):>11}{'MRR':>8}{'mean rank':>{mr_w}}{'top1':>6}"
          f"   | macro {'recall@' + str(d):>11}{'MRR':>8}")
    for r, mean_rank_cell in zip(rows, mean_rank_cells):
        print(f"{r['threshold']:>8.0%}  {r['stratum']:<12}{r['n']:>5}{r['n_payloads']:>6}"
              f"{r['n_unscorable']:>7}"
              f"{(r[f'recall_at_{d}'] if r[f'recall_at_{d}'] is not None else 0):>11.4f}"
              f"{(r['mrr'] or 0):>8.4f}{mean_rank_cell:>{mr_w}}{r['top1']:>6}"
              f"   |       {(r[f'recall_at_{d}_macro'] or 0):>11.4f}{(r['mrr_macro'] or 0):>8.4f}")

    print(f"\nPrimary metrics for an {args.arm} arm: {', '.join(rows[0]['primary_metrics'])}. "
          f"Top-1 is a column, never a headline.")
    print("No pooled row is printed and none should be computed: at the 25% holdout a pooled "
          "recall@10 moves +1.0% while the f < 0.50 stratum moves +40.0, because most items are "
          "saturated at depth 10.")

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(dict(benchmark=str(bd), submission=str(args.submission), arm=args.arm,
                           depth=d, tiers=tiers, binding=bind, rows=rows), fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
