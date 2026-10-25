#!/usr/bin/env python3
"""Exact Smith-Waterman homolog discovery for every tier's payloads, threaded across cores.

    python scripts/19_find_homologs_all.py --manifest data/manifest.csv \\
        --manifest data/manifest_auto.csv --manifest data/synthetic/synthetic_manifest.csv \\
        --swissprot %USERPROFILE%\\Downloads\\uniprot_sprot.fasta.gz \\
        --out-dir results/homologs_all173

**Why this exists rather than `09_find_homologs.py`.** That script reads one manifest and requires every
payload to be present in `data/reference/reference.fasta`, which holds the 20 hand-tier payloads and their
decoys. The other 157 payloads are not in it, so it cannot do the full set. This reads any number of
manifests and takes the payload sequences from Swiss-Prot directly.

**Why not `16_build_twilight_refs.py`.** That one is DIAMOND. the working record are the measurement of
what DIAMOND costs here: `--ultra-sensitive` returned 18 raw hits for GFP where exact alignment finds 258,
and at a 25% holdout the hand tier kept 858 homolog rows instead of 1,705 and 13 scorable constructs
instead of 19. This script is the exact route for the whole payload set, so the heuristic can be taken out
of the measurement path entirely.

**Threads, not processes.** parasail's kernels release the GIL, so the score-only pass parallelises with
OS threads (measured 2.0x on two) while all threads share the single 280 MB copy of Swiss-Prot. One
process per core would need a copy each, which does not fit in the memory this machine has spare.

**Coverage is measured against the payload**, via `pilot.homology.coverage_of`, the same function the
DIAMOND route now calls. That mismatch was a real bug: dividing by the subject length made
a 3,054 aa polyprotein look well covered by partial hits and put 71 phantom homologs on P04517.

**A payload outside the database window is searched but not written into the reference.** It used to be
inserted unconditionally, so a 41 aa payload (Q9TT33) sat in a catalogue that the catalogue's own reader
excluded when that catalogue was used as a `--database`. Such a payload is now counted and excluded, and
`--strict-payload-window` turns the exclusion into an error.

Writes `homologs.csv`, `reference_twilight.fasta` and `payload_lengths.csv` to `--out-dir`.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from pilot.homology import MAX_LEN, MIN_LEN, find_homologs, iter_swissprot
from pilot.reference import Entry, iter_fasta, write_fasta

THRESHOLDS = (0.90, 0.70, 0.50, 0.35, 0.25)


def select_only(spec: str, available: list[str]) -> list[str]:
    """The accessions named in `--only` that are actually searchable, in the given order.

    Separate from the argument parsing so the filter is testable without a Swiss-Prot file. Names that
    are not in the manifests, or not in this Swiss-Prot release, are dropped and reported rather than
    silently producing an empty run.
    """
    want = [a.strip() for a in spec.split(",") if a.strip()]
    have = set(available)
    seen: set[str] = set()
    out = []
    for a in want:
        if a in have and a not in seen:
            seen.add(a)
            out.append(a)
    return out


def payload_accessions(manifests: list[Path]) -> dict[str, set[str]]:
    """{accession: {manifest names that use it}}; the provenance matters when tiers share a payload."""
    out: dict[str, set[str]] = defaultdict(set)
    for m in manifests:
        if not m.is_file():
            print(f"  skipped, not found: {m}", file=sys.stderr)
            continue
        with open(m, newline="", encoding="utf-8-sig", errors="replace") as fh:
            for r in csv.DictReader(fh):
                acc = (r.get("payload_accession") or "").strip()
                if acc:
                    out[acc].add(m.name)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, action="append", required=True)
    ap.add_argument("--swissprot", type=Path, required=True,
                    help="where the PAYLOAD sequences come from. Also the search database unless "
                         "--database is given.")
    ap.add_argument("--database", type=Path, default=None,
                    help="search database, if it is not --swissprot. The two jobs are independent: "
                         "queries still come from --swissprot, so a payload absent from the database "
                         "is still searched. Pointing this at a twilight catalogue computes an "
                         "exhaustive holdout mask, which catalogue residents are too close to each "
                         "payload to be held out, rather than discovering homologs in Swiss-Prot. "
                         "The database window still applies on the way in.")
    ap.add_argument("--out-dir", type=Path, default=Path("results/homologs_all173"))
    ap.add_argument("--decoys", type=Path, default=Path("data/reference/reference.fasta"),
                    help="existing decoy set folded into the twilight reference, as in 16_build_twilight_refs")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--top-n", type=int, default=2000,
                    help="how many top-scoring hits get a traceback. the working record used 400, "
                         "but the cap binds: every payload in a smoke run filled it, so the homolog set "
                         "was truncated at the remote end: the same defect this project criticised "
                         "DIAMOND's --max-target-seqs for. Pass 400 to reproduce those runs exactly.")
    ap.add_argument("--min-identity", type=float, default=0.20)
    ap.add_argument("--min-coverage", type=float, default=0.40)
    ap.add_argument("--limit", type=int, default=0, help="first N payloads only, for a smoke run")
    ap.add_argument("--strict-payload-window", action="store_true",
                    help="fail instead of excluding a payload that is outside the database length "
                         "window. The default excludes it and prints the count; this makes a build "
                         "refuse to produce a catalogue that silently drops one of its own payloads.")
    ap.add_argument("--only", default="", help="comma-separated payload accessions to search, instead "
                                               "of every accession in the manifests. For probing one "
                                               "parameter on a few payloads without a full sweep.")
    ap.add_argument("--max-evalue", type=float, default=None,
                    help="significance gate. Setting it replaces --top-n entirely: the gate inverts to a "
                         "raw-score floor (105.4 to 115.7 across this payload range at 1e-3), so every "
                         "subject at or above the floor is traced back and truncation is impossible. "
                         "The count above the floor is then a measurement, not a limit.")
    ap.add_argument("--rank-by", default="evalue", choices=("evalue", "raw"),
                    help="statistic used to pick the top-n candidates for traceback. Raw Smith-Waterman "
                         "score is not length-normalised, so long sequences displace short genuine "
                         "homologs out of the pool before any gate applies.")
    args = ap.parse_args(argv)

    if not args.swissprot.is_file():
        print(f"not found: {args.swissprot}", file=sys.stderr)
        return 2

    t0 = time.time()
    accs = payload_accessions(args.manifest)
    print(f"{len(accs)} distinct payload accessions across {len(args.manifest)} manifests")
    for m in args.manifest:
        print(f"    {m.name:<34}{sum(1 for v in accs.values() if m.name in v):>5}")

    print(f"\nreading {args.swissprot.name} ...", flush=True)
    # The 50-5,000 aa window belongs on the *database*, where it keeps titins and polyproteins from
    # dominating a raw-score ranking for reasons unrelated to homology. Applying it to the payload lookup
    # as well silently drops any payload outside it: P04517 is 3,054 aa, and it is one of the records
    # this run exists to settle. So the file is read once without a window and filtered afterwards.
    everything = list(iter_swissprot(args.swissprot, min_len=1, max_len=10 ** 9))
    sp = dict(everything)
    if args.database is None:
        db_all = everything
        db_name = args.swissprot.name
    else:
        # Queries and database are independent. `iter_swissprot` is the right reader for a twilight
        # catalogue too: its headers are ">A0A017SE81 homolog_of:P71079", with no pipes, so the accession
        # is `split()[0]` exactly as for a Swiss-Prot record. There is a test for that rather than a note.
        print(f"reading database {args.database.name} ...", flush=True)
        db_all = list(iter_swissprot(args.database, min_len=1, max_len=10 ** 9))
        db_name = args.database.name
    entries = [(a, q) for a, q in db_all if MIN_LEN <= len(q) <= MAX_LEN]
    db_seq = dict(entries)
    print(f"    {len(sp):,} query entries read from {args.swissprot.name}", flush=True)
    print(f"    {len(entries):,} of {len(db_all):,}  {db_name} records in the {MIN_LEN}-{MAX_LEN} aa "
          f"database ({sum(len(q) for _, q in entries):,} residues) in {time.time() - t0:.0f}s",
          flush=True)
    if args.database is not None and len(db_all) != len(entries):
        dropped = sorted(((a, len(q)) for a, q in db_all if not (MIN_LEN <= len(q) <= MAX_LEN)),
                         key=lambda t: t[1])
        # A catalogue this script built contains its own payloads unconditionally (the `keep.setdefault`
        # at the end of the search loop), so a payload outside the window sits in the catalogue and is
        # then excluded from it when the catalogue is used as a database. Not a contradiction, but such a
        # payload cannot retrieve itself, and for an exhaustive mask that is the one pair missing by
        # construction rather than by measurement.
        print(f"    {len(dropped)} database records excluded by the window: "
              + ", ".join(f"{a} ({n}aa)" for a, n in dropped[:12])
              + ("..." if len(dropped) > 12 else ""), file=sys.stderr, flush=True)
    del everything, db_all

    missing = sorted(a for a in accs if a not in sp)
    if missing:
        print(f"    {len(missing)} payloads absent from this Swiss-Prot release: {missing[:10]}",
              file=sys.stderr)
    # An out-of-window payload is searched but is NOT written into the twilight reference, which is the
    # one place this used to be silently inconsistent: the reference was built by inserting every payload
    # unconditionally, so a 41 aa payload entered a catalogue that `iter_swissprot` then excluded from its
    # own 50-5,000 aa window when that catalogue was used as a database. The catalogue has to satisfy the
    # window it was filtered by, so the insert is now explicit and counted.
    #
    # The lower bound is the binding one and it is not arbitrary: below 50 residues an alignment has too
    # few columns to support a percent-identity estimate, which is `iter_swissprot`'s own stated reason.
    # Excluding such a payload makes its constructs unscorable, and that is the honest outcome: the
    # alternative is a catalogue record whose identity values do not mean anything. Widening the window
    # would not fix it, it would only move the meaningless identities inside the gate.
    oversize = sorted(a for a in accs if a in sp and not (MIN_LEN <= len(sp[a]) <= MAX_LEN))
    if oversize and args.strict_payload_window:
        print(f"    {len(oversize)} payload(s) outside the {MIN_LEN}-{MAX_LEN} aa window: "
              + ", ".join(f"{a} ({len(sp[a])}aa)" for a in oversize)
              + "\n  --strict-payload-window is set, so this is an error rather than an exclusion.",
              file=sys.stderr)
        return 3
    if oversize:
        print(f"    {len(oversize)} payload(s) outside the {MIN_LEN}-{MAX_LEN} aa window. Searched, but "
              f"EXCLUDED from the twilight reference, so their own constructs become unscorable: "
              + ", ".join(f"{a} ({len(sp[a])}aa)" for a in oversize), file=sys.stderr, flush=True)
    todo = sorted(a for a in accs if a in sp)
    if args.only:
        wanted = select_only(args.only, todo)
        missing = sorted({a.strip() for a in args.only.split(",") if a.strip()} - set(wanted))
        if missing:
            print(f"  --only names {len(missing)} accession(s) not in the manifests or not in this "
                  f"Swiss-Prot release: {missing}", file=sys.stderr)
        todo = wanted
    if args.limit:
        todo = todo[: args.limit]
    # longest first: the long payloads dominate the runtime, so starting them early keeps the tail short
    todo.sort(key=lambda a: -len(sp[a]))
    print(f"\nsearching {len(todo)} payloads against {len(entries):,}  {db_name} subjects on "
          f"{args.threads} threads, top-n {args.top_n}, gates identity >= {args.min_identity}  "
          f"coverage >= {args.min_coverage}\n", flush=True)

    rows: list[dict] = []
    keep: dict[str, Entry] = {}
    if args.decoys.is_file():
        for e in iter_fasta(args.decoys):
            keep[e.accession] = e
    lengths: list[dict] = []
    capped: list[str] = []
    for i, acc in enumerate(todo, 1):
        t = time.time()
        st: dict = {}
        hits = find_homologs(sp[acc], entries, top_n=args.top_n, min_identity=args.min_identity,
                             min_coverage=args.min_coverage, threads=args.threads,
                             rank_by=args.rank_by, stats=st, max_evalue=args.max_evalue)
        bins = {t_: sum(1 for h in hits if h.identity <= t_) for t_ in THRESHOLDS}
        # Truncation is a property of the search, not of the hit count. With a score floor there is no
        # cap, so a payload with 4,000 homologs is a measurement rather than a warning; comparing against
        # an unused `--top-n` would raise exactly the kind of false alarm this project keeps chasing.
        if st.get("truncated"):
            capped.append(acc)
        lengths.append(dict(payload_accession=acc, payload_aa=len(sp[acc]), n_homologs=len(hits),
                            n_considered=st.get("n_considered"), n_rejected=st.get("n_rejected"),
                            reject_fraction=st.get("reject_fraction"),
                            score_floor=st.get("score_floor"), n_above_floor=st.get("n_above_floor"),
                            truncated=st.get("truncated"),
                            at_top_n_cap=int(bool(st.get("truncated"))),
                            **{f"available_at_{int(t_ * 100)}": bins[t_] for t_ in THRESHOLDS},
                            seconds=round(time.time() - t, 1)))
        print(f"  [{i:>3}/{len(todo)}] {acc:<8}{len(sp[acc]):>5}aa{len(hits):>5} homologs  "
              + "  ".join(f"{int(t_ * 100)}%:{bins[t_]}" for t_ in THRESHOLDS)
              + (f"  floor {st['score_floor']:.0f} above {st['n_above_floor']:,}"
                 if st.get('score_floor') is not None else f"  reject {st.get('reject_fraction', 0):.0%}")
              + f"    {time.time() - t:.0f}s", flush=True)
        for h in hits:
            rows.append(dict(payload_accession=acc, homolog_accession=h.accession, score=h.score,
                             identity=h.identity, coverage=h.coverage, length=h.length))
            if h.accession in db_seq:
                keep.setdefault(h.accession, Entry(h.accession, f"{h.accession} homolog_of:{acc}",
                                                   db_seq[h.accession]))
        if MIN_LEN <= len(sp[acc]) <= MAX_LEN:
            keep.setdefault(acc, Entry(acc, f"{acc} payload", sp[acc]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "homologs.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out_len = args.out_dir / "payload_lengths.csv"
    with open(out_len, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lengths[0]))
        w.writeheader()
        w.writerows(lengths)
    out_fa = args.out_dir / "reference_twilight.fasta"
    write_fasta(sorted(keep.values(), key=lambda e: e.accession), out_fa)

    print(f"\n{len(rows):,} homolog rows for {len(todo)} payloads -> {out_csv}")
    print(f"twilight reference: {len(keep):,} proteins -> {out_fa}")
    in_ref = sum(1 for a in todo if a in keep)
    print(f"  payloads present in the reference as their own record: {in_ref}/{len(todo)}"
          + (f"  ({len(todo) - in_ref} excluded by the {MIN_LEN}-{MAX_LEN} aa window)"
             if in_ref != len(todo) else ""))
    for t_ in THRESHOLDS:
        n = sum(1 for r in lengths if r[f"available_at_{int(t_ * 100)}"] > 0)
        print(f"  payloads still scorable at a {t_:.0%} holdout: {n}/{len(todo)}")
    if capped:
        print(f"\n  {len(capped)} payloads were TRUNCATED at the --top-n {args.top_n} cap, so their "
              f"homolog sets are cut at the remote end: {capped[:12]}"
              f"{'...' if len(capped) > 12 else ''}")
    elif args.max_evalue is not None:
        top = sorted(lengths, key=lambda r: -r["n_homologs"])[:3]
        biggest = ", ".join(f"{r['payload_accession']}  {r['n_homologs']:,}" for r in top)
        print(f"\n  no truncation: candidates were selected by score floor, not by rank, so --top-n did "
              f"not apply. Largest homolog sets: {biggest}")
    print(f"\ntotal {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
