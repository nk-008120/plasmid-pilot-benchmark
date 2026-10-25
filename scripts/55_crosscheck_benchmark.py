#!/usr/bin/env python3
"""Does scoring a submission reproduce `10_twilight.py`'s own ranks? Checked cell by cell, not on average.

    python scripts/55_crosscheck_benchmark.py --tier hand --threads 4

**What is being checked and why it is not obvious.** `benchmark/README.md` claims that a reviewer's
score and this project's differ by method and by nothing else. That claim has two halves. The metric
half is covered by unit tests. The other half is that `pilot.benchmark`'s holdout construction and
ranking reproduce what the measurement script does, and those are different code paths reaching the
same answer through `holdout_mask`, `targets_at` and `best_target_rank`. Two paths agreeing by
construction is an argument; two paths agreeing on every cell is a measurement.

**Why cell by cell.** An aggregate can agree while the cells do not: a rank moved up in one construct
and down in another leaves `recall@10` untouched. This project has been bitten by exactly that shape of
error (where a change of sample and a change of effect were summed into one number), so
the comparison here is per `(construct, threshold)` and the script **exits non-zero on a single
mismatch**. There is no tolerance parameter, because there is no reason for a tolerance to exist: the
ranks are integers produced from the same scores by the same rule.

**And it exits non-zero when it compares nothing**, which is a different guarantee and was not
here until 22 Oct. Run against a benchmark built with `--namespace-ids` this script matched 0 of
120 cells, every one skipped as "not in the published CSV", and printed `Every compared cell
is identical` anyway, because the success branch tested `not mismatches` and zero cells produce
zero mismatches. A vacuous pass on the one check standing behind `benchmark/README.md`'s public
claim is worse than a failure, so `cells_compared == 0` is now a refusal with its own verdict.

**It runs on the hand tier because that is what it costs.** 20 items against 40,750 proteins is a few
minutes of exact Smith-Waterman; the corpus and synth_real would be 1 h 57 m and 4 h 56 m to reproduce
from scratch, and the claim being checked is about the code, not about the tier. `--tier` will run any
of them if that ever matters.

**The `whole` arm, specifically.** `queries.faa` publishes `arms.orf_seq`, which *is* the `whole`
arm's query, so the benchmark's own published input is scorable without rebuilding any arms. Any other
arm would need the segmentation re-derived here, which would be checking two things at once.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.align import AlignmentScorer  # noqa: E402
from pilot.benchmark import Submission, load_items, rank_cells  # noqa: E402
from pilot.homology import Homolog  # noqa: E402
from pilot.reference import iter_fasta  # noqa: E402

TWILIGHT = {
    "hand": "results/persegment_hand_bhs/twilight_sw-blosum62-11-1_hand.csv",
    "corpus": "results/persegment_corpus/twilight_sw-blosum62-11-1_corpus.csv",
    "synth_real": "results/persegment_synthreal/twilight_sw-blosum62-11-1_synth_real.csv",
    "synth_random": "results/persegment_synthetic/twilight_sw-blosum62-11-1_synthetic.csv",
}


def published_id(construct_id: str, tier: str) -> str:
    """The key the published CSV uses, which is not the key a namespaced benchmark uses.

    `52_build_benchmark.py --namespace-ids` writes `tier:construct_id`, because the two synthetic
    tiers are the same 336 constructs and `construct_id` alone cannot address an item (the working record). `10_twilight.py` knows nothing about that convention and writes the bare id, so a
    namespaced benchmark matched literally against the published CSV finds **nothing**, which is
    not a disagreement and must not be read as one. Stripping the prefix here is what makes the
    two keys the same quantity again; the refusal below is what catches it if this is ever wrong.
    """
    prefix = f"{tier}:"
    return construct_id[len(prefix):] if construct_id.startswith(prefix) else construct_id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", type=Path, default=Path("benchmark/v1"))
    ap.add_argument("--reference", type=Path,
                    default=Path("results/gate_all173/reference_twilight.fasta"))
    ap.add_argument("--tier", default="hand", choices=sorted(TWILIGHT))
    ap.add_argument("--arm", default="whole", help="only `whole` needs no arms rebuilt here")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    bd = args.benchmark
    tw_path = Path(TWILIGHT[args.tier])
    for p in (bd / "items.csv", bd / "queries.faa", bd / "homologs.csv",
              bd / "catalogue_accessions.txt", args.reference, tw_path):
        if not p.is_file():
            print(f"not found: {p}", file=sys.stderr)
            return 2

    items = [i for i in load_items(bd / "items.csv") if i.tier == args.tier]
    if not items:
        print(f"no items in tier {args.tier!r}", file=sys.stderr)
        return 2
    queries = {e.accession: e.seq for e in iter_fasta(bd / "queries.faa")}
    missing = [i.construct_id for i in items if i.construct_id not in queries]
    if missing:
        print(f"{len(missing)} item(s) absent from queries.faa, e.g. {missing[:3]}", file=sys.stderr)
        return 2

    homologs: dict[str, list[Homolog]] = {}
    with open(bd / "homologs.csv", newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            homologs.setdefault(r["payload_accession"], []).append(
                Homolog(r["homolog_accession"], float(r["score"]), float(r["identity"]),
                        float(r["coverage"]), int(r["length"])))
    catalogue = (bd / "catalogue_accessions.txt").read_text(encoding="utf-8").split()

    refs = list(iter_fasta(args.reference))
    ref_acc = [e.accession for e in refs]
    if ref_acc != catalogue:
        print(f"the reference FASTA's accession order does not match "
              f"catalogue_accessions.txt ({len(ref_acc):,} vs {len(catalogue):,}). A matrix scored "
              f"against one cannot be read against the other; refusing.", file=sys.stderr)
        return 2

    # The published ranks this is checked against.
    expected: dict[tuple[str, float], int | None] = {}
    with open(tw_path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["arm"] != args.arm:
                continue
            expected[(r["plasmid_id"], float(r["threshold"]))] = (
                int(r["best_target_rank"]) if r["best_target_rank"] != "" else None)

    print(f"tier {args.tier}: {len(items)} items | arm {args.arm} | catalogue {len(refs):,} entries")
    print(f"published ranks from {tw_path.name}: {len(expected):,} cells")
    t0 = time.time()
    scorer = AlignmentScorer([e.seq for e in refs], threads=args.threads)
    rows = []
    for n, it in enumerate(items, 1):
        rows.append(scorer.score_all(queries[it.construct_id]))
        if n % 5 == 0 or n == len(items):
            print(f"  scored {n}/{len(items)} in {time.time() - t0:.0f}s", flush=True)
    matrix = np.vstack(rows)

    records, bind = rank_cells(Submission("matrix", matrix=matrix), items, homologs, catalogue)
    print(f"ranked {len(records):,} cells; binding: {bind}")

    agree, mismatches, absent = 0, [], 0
    for item, thr, got in records:
        key = (published_id(item.construct_id, item.tier), thr)
        if key not in expected:
            absent += 1
            continue
        want = expected[key]
        if got == want:
            agree += 1
        else:
            mismatches.append(dict(construct_id=item.construct_id, threshold=thr,
                                   benchmark=got, twilight=want, f=round(item.f, 4),
                                   stratum=item.stratum))

    print()
    print(f"cells compared        {agree + len(mismatches):,}")
    print(f"identical             {agree:,}")
    print(f"MISMATCHED            {len(mismatches):,}")
    if absent:
        print(f"not in the published CSV   {absent:,} (skipped, not counted as agreement)")
    for m in mismatches[:20]:
        print(f"      {m['construct_id']:<22} holdout {m['threshold']:.2f}    "
              f"benchmark {str(m['benchmark']):>8}  twilight {str(m['twilight']):>8}    "
              f"f={m['f']}  {m['stratum']}")

    compared = agree + len(mismatches)
    if not compared:
        print(f"\nNo cell was compared at all: all {absent:,} ranked cells are absent from "
              f"{tw_path.name}, so this run checked nothing and is not evidence of agreement. "
              f"The usual cause is a key mismatch rather than missing data: a namespaced "
              f"`items.csv` against a published CSV keyed on the bare `construct_id`. Refusing, "
              f"because a check that compares nothing must not read as a check that passed.",
              file=sys.stderr)
        if args.out:
            with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(dict(tier=args.tier, arm=args.arm, verdict="NOTHING COMPARED",
                               cells_compared=0, identical=0, mismatched=0,
                               not_in_published_csv=absent, published_source=str(tw_path),
                               binding=bind), fh, indent=2)
            print(f"wrote {args.out}")
        return 2

    verdict = "IDENTICAL" if not mismatches else "MISMATCHED"
    if mismatches:
        print(f"\n{len(mismatches)} cell(s) disagree. The benchmark's rank and the measurement "
              f"script's rank are not the same quantity, so `benchmark/README.md`'s claim that a "
              f"reviewer's score differs from ours by method alone does not hold as written. Fix that "
              f"before publishing the baseline.")
    else:
        print(f"\nEvery compared cell is identical. `pilot.benchmark`'s holdout construction and "
              f"ranking reproduce `10_twilight.py`'s, which is what the README claims and what the "
              f"baseline row depends on. Measured on {args.tier}, arm {args.arm}; the other tiers are "
              f"the same code on more items.")

    out = dict(tier=args.tier, arm=args.arm, verdict=verdict,
               cells_compared=agree + len(mismatches), identical=agree,
               mismatched=len(mismatches), not_in_published_csv=absent,
               mismatches=mismatches, binding=bind,
               published_source=str(tw_path), catalogue_entries=len(refs),
               scoring_seconds=round(time.time() - t0, 1))
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.out}")
    else:
        with open(bd / f"crosscheck_{args.tier}_{args.arm}.json", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {bd}/crosscheck_{args.tier}_{args.arm}.json")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
