#!/usr/bin/env python3
"""Assemble the public benchmark: the items, the queries, the holdouts, and the catalogue's provenance.

    python scripts/52_build_benchmark.py --out-dir benchmark/v1

**What this is for.** The repository is being published for a reviewer, and every number in it depends on
which catalogue it was measured against, which holdout was applied, and which metric was read. Three
things this project has got wrong at least once each. So the benchmark ships as data plus rules rather
than as a claim: the items a method is handed, the answer key, and `pilot.benchmark`, which is the metric
rules as code. A reviewer's score and this project's differ by method and by nothing else.

**What a method is asked to do.** For each item it is handed one protein, the payload-bearing ORF as a
naive caller emitted it, and the catalogue, and it returns a ranking of catalogue accessions. The
benchmark then removes the payload's close relatives at each of six holdout thresholds and asks where the
best *surviving* relative landed. Nothing about the task requires the method to know what a plasmid is.

**What is deliberately not in the item table.** No archetype, no segment boundaries, no payload span,
nothing a decomposition step is supposed to discover. `f` is published because a benchmark has to say
what each item is, and it is a *description of the item*, computed before any retrieval runs: it uses
the label and can never be part of a screening rule.

**The catalogue is not shipped as a FASTA.** It is 40,750 entries and 20.8 M residues, it is derived from
UniProt, and a copy in git would be both large and unverifiable. What ships is the accession list in
catalogue order, the sha256 of the FASTA this project measured against, and the counts that set the
significance floor, so a rebuild is checkable rather than hoped about. `--copy-catalogue` writes the
FASTA too, for a release that needs to be self-contained.

Writes `items.csv`, `queries.faa`, `homologs.csv`, `catalogue_accessions.txt` and `MANIFEST.json`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.arms import build_arms  # noqa: E402
from pilot.benchmark import STRATA, THRESHOLDS, DEPTH, stratum_of  # noqa: E402
from pilot.homology import MAX_LEN, MIN_LEN, score_floor  # noqa: E402
from pilot.reference import iter_fasta  # noqa: E402

#: The tiers this builder can assemble, each covering the others' weakness, and **never pooled into
#: one number**. Which of them a given build ships is `--tiers`; see PUBLISHED_TIERS below.
#:. `results_dir` supplies `orfs.faa` and `pilot_summary.csv`; `manifest` supplies the
#: answer key, keyed on `filename` exactly as `10_twilight.py` keys it.
TIERS: tuple[tuple[str, str, str, str], ...] = (
    ("hand", "results/gate173_bhs_hand", "data/manifest.csv",
     "23 hand-audited deposits; ground truth is the depositor's stated insert. Realism, and the anchor"),
    ("corpus", "results/persegment_corpus", "data/manifest_auto.csv",
     "410 screened NCBI records, 138 distinct payloads; ground truth by gated alignment, 80% validated. "
     "Breadth and statistical power, and note the constructs are NOT independent trials"),
    ("synth_real", "results/persegment_synthreal", "data/synthetic_real/synthetic_manifest.csv",
     "336 synthetic constructs with exact boundaries and real filler. The only tier built INSIDE the "
     "diluted regime: median f 0.600, 64.3% below 0.70"),
    # The random-filler twin of the tier above: the SAME 336 construct ids, payload accessions,
    # `payload_aa` and `orf_aa`, differing only in the filler residues, so f is identical per construct
    # and dilution is matched residue for residue. It is a separate tier and not a
    # variant of `synth_real`, because the filler **reverses the sign of the dilution cost between
    # scorers**: random filler is invisible to a local alignment (`whole` bit-identical to `oracle` on
    # all 336) and costs mean-pooled ESM-2 `recall@10` 0.0% at a median rank of 19,170 against a chance
    # median of 20,375 (60.2). So a benchmark shipping only `synth_real` poses the worst case for
    # alignment and not for an embedding arm, and a robustness claim needs both (66.6).
    ("synth_random", "results/persegment_synthetic", "data/synthetic/synthetic_manifest.csv",
     "the same 336 constructs as synth_real with random-sequence filler instead of SwissProt filler, "
     "matched residue for residue on f. The worst case for a mean-pooled embedding arm, where "
     "synth_real is the worst case for alignment: 'adversarial' is scorer-dependent"),
)

#: The tiers `benchmark/v1` was PUBLISHED with, and the default for `--tiers` for exactly that reason.
#: `TIERS` is what the builder *can* assemble; this is what the shipped manifest's checksums were
#: computed over. Adding a tier to the default would change `items.csv`, `queries.faa` and
#: `homologs.csv`, hence all three sha256s in `MANIFEST.json`, hence the published benchmark, which is
#: a release decision. So a four-tier
#: build is an explicit `--tiers hand corpus synth_real synth_random`, and the default still rebuilds
#: v1 byte for byte.
PUBLISHED_TIERS: tuple[str, ...] = ("hand", "corpus", "synth_real")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def orf_coordinates(orf_id: str) -> dict:
    """Unpack `plasmid|orfNNN|strand|start..end` into named fields, with the wrap made explicit.

    The coordinates were always here: packed into `orf_id`, which is a composite key a reader has to
    know the shape of before they can use it. A7 is about not making them parse it.

    **`orf_wraps_origin` is the part that is not decoration.** These plasmids are circular and the ORF
    caller is a circular six-frame scanner, so an ORF can begin near the end of the sequence and finish
    past position 1. For those `end < start`, and the naive `end - start + 1` is **negative**: on the
    published build that is 5 items, and a reader computing lengths that way would get -8,001 nt for
    `corpus:ncbi_AF218363_1` with nothing raised. `orf_len` is right for all of them throughout,
    because it comes from the translated sequence and not from these coordinates, so this flag
    exists to stop a reader deriving a second and wrong length, not to fix a first one.
    """
    parts = orf_id.split("|")
    blank = dict(plasmid_id=parts[0], orf_strand="", orf_start="", orf_end="", orf_wraps_origin="")
    if len(parts) != 4 or ".." not in parts[3]:
        # Not a crash. A header shape this does not recognise means the coordinate columns are
        # unknown for that item, which is a thing to report and not a thing to guess or to die on;
        # `_, orf_index, _, _` would have taken the whole build down over one record's formatting.
        return blank
    _plasmid, _orf_index, strand, span = parts
    a, b = span.split("..")
    if not (a.isdigit() and b.isdigit()):
        return blank
    start, end = int(a), int(b)
    return dict(plasmid_id=parts[0], orf_strand=strand,
                orf_start=start, orf_end=end, orf_wraps_origin=int(end < start))


def read_faa(path: Path) -> dict[str, dict[str, str]]:
    """plasmid_id -> {orf_id: seq}, the same grouping `10_twilight.py` reads."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for e in iter_fasta(path):
        oid = e.header.split()[0]
        out[oid.split("|")[0]][oid] = e.seq
    return out


def verify_catalogue(bench_dir: Path, fasta: Path) -> int:
    """Is this FASTA the catalogue the benchmark was measured on? Checked three ways, not one.

    This is what makes "the repository ships a checksum, the release ships the FASTA" an actionable
    policy rather than a number sitting in a JSON file. The sha256 settles it on its own; the entry
    count and the in-window residue total are reported too because they are what a reader needs if it
    does **not** match: the residue count sets the significance floor, so a near-miss catalogue is a
    different benchmark rather than a slightly different one.
    """
    mpath = bench_dir / "MANIFEST.json"
    if not mpath.is_file():
        print(f"not found: {mpath}", file=sys.stderr)
        return 2
    if not fasta.is_file():
        print(f"not found: {fasta}", file=sys.stderr)
        return 2
    want = json.loads(mpath.read_text(encoding="utf-8"))["catalogue"]
    got_sha = _sha256(fasta)
    refs = list(iter_fasta(fasta))
    in_window = [e for e in refs if MIN_LEN <= len(e.seq) <= MAX_LEN]
    residues = sum(len(e.seq) for e in in_window)
    checks = [
        ("sha256", got_sha, want["sha256"]),
        ("entries", len(refs), want["entries"]),
        ("entries in window", len(in_window), want["entries_in_window"]),
        ("in-window residues", residues, want["residues"]),
    ]
    bad = [name for name, got, exp in checks if got != exp]
    for name, got, exp in checks:
        mark = "ok  " if got == exp else "BAD "
        print(f"    {mark}{name:<20}{got}" + ("" if got == exp else f"   expected {exp}"))
    if not bad:
        print("\nTHIS IS THE CATALOGUE THE BENCHMARK WAS MEASURED ON.")
        return 0
    print(f"\nNOT the published catalogue: {', '.join(bad)} differ. Scores computed against this file "
          f"are not comparable with the baseline, and if the residue count differs the significance "
          f"floor differs too, which makes it a different benchmark rather than a noisy one. The "
          f"authoritative copy is attached to the release.", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("benchmark/v1"))
    ap.add_argument("--reference", type=Path,
                    default=Path("results/gate_all173/reference_twilight.fasta"))
    ap.add_argument("--homologs", type=Path, default=Path("results/gate_all173/homologs.csv"))
    ap.add_argument("--tiers", nargs="+", default=list(PUBLISHED_TIERS),
                    choices=[t[0] for t in TIERS],
                    help="which tiers to assemble. The default is the three tiers v1 was "
                         "published with, so a default build reproduces its checksums; `synth_random` "
                         "is available and not default, because adding it changes every shipped "
                         "checksum and that is a release decision")
    ap.add_argument("--copy-catalogue", action="store_true",
                    help="also write the reference FASTA into the benchmark directory")
    ap.add_argument("--swissprot", type=Path, default=Path("data/swissprot/uniprot_sprot.fasta.gz"),
                    help="the catalogue's own source, checksummed into the MANIFEST when present. Its "
                         "release label was never recorded, so the file's sha256 is what pins it")
    ap.add_argument("--verify-catalogue", type=Path, default=None,
                    help="check a FASTA against the MANIFEST this directory already carries, and exit "
                         "non-zero if it is not the catalogue the benchmark was measured on. Build "
                         "nothing")
    ap.add_argument("--max-evalue", type=float, default=1e-3)
    ap.add_argument("--namespace-ids", action="store_true",
                    help="write every construct_id as `tier:construct_id`. Required when two selected "
                         "tiers share construct ids, which `synth_real` and `synth_random` do for all "
                         "336 of theirs: see the refusal below for why nothing else will do")
    args = ap.parse_args(argv)

    if args.verify_catalogue is not None:
        return verify_catalogue(args.out_dir, args.verify_catalogue)

    if not args.reference.is_file() or not args.homologs.is_file():
        print(f"not found: {args.reference} or {args.homologs}", file=sys.stderr)
        return 2

    refs = list(iter_fasta(args.reference))
    ref_by_acc = {e.accession: e.seq for e in refs}
    in_window = [e for e in refs if MIN_LEN <= len(e.seq) <= MAX_LEN]
    db_residues = sum(len(e.seq) for e in in_window)
    print(f"catalogue: {len(refs):,} entries, {len(in_window):,} in the {MIN_LEN}-{MAX_LEN} aa window, "
          f"{db_residues:,} residues")

    items: list[dict] = []
    queries: list[tuple[str, str]] = []
    tier_meta = []
    for tier, rdir, mpath, blurb in TIERS:
        if tier not in args.tiers:
            continue
        rd, mp = Path(rdir), Path(mpath)
        if not (rd / "orfs.faa").is_file() or not (rd / "pilot_summary.csv").is_file():
            print(f"    {tier}: SKIPPED, {rd} has no orfs.faa / pilot_summary.csv", file=sys.stderr)
            continue
        with open(mp, newline="", encoding="utf-8-sig", errors="replace") as fh:
            man = {r["filename"].strip(): r for r in csv.DictReader(fh)
                   if (r.get("filename") or "").strip()}
        with open(rd / "pilot_summary.csv", newline="", encoding="utf-8-sig") as fh:
            summary = list(csv.DictReader(fh))
        orfs_by_plasmid = read_faa(rd / "orfs.faa")

        # Each exclusion is a different reason an item cannot be a benchmark item, and each is counted
        # rather than left as a smaller n (this project's working rules: report binding, not settings).
        kept, no_acc, not_in_cat, no_orf, verbatim = 0, 0, 0, 0, 0
        for srow in summary:
            pid = srow["plasmid_id"]
            acc = (man.get(srow["source_file"], {}).get("payload_accession") or "").strip()
            if not acc:
                no_acc += 1
                continue
            if acc not in ref_by_acc:
                not_in_cat += 1
                continue
            # `build_arms` without a scorer or catalogue builds `whole`, `auto` and `oracle` and no
            # `bhs`, which is all this needs: f is `len(span) / len(orf_seq)` on the payload-bearing
            # ORF. Deriving it here rather than reading a lengths CSV keeps one definition of f in the
            # repository: the `hasattr` mistake of 17 Oct came from a second one.
            arms = build_arms(pid, orfs_by_plasmid.get(pid, {}), ref_by_acc[acc])
            if arms is None or arms.span is None:
                no_orf += 1
                continue
            mrow = man.get(srow["source_file"], {})
            items.append(dict(construct_id=pid, tier=tier, payload_accession=acc, orf_id=arms.orf_id,
                              orf_len=len(arms.orf_seq), oracle_len=arms.span.length,
                              f=round(arms.span.length / len(arms.orf_seq), 4),
                              stratum=stratum_of(arms.span.length / len(arms.orf_seq)),
                              # A7: where the sequence came from, READ from the tier manifest and
                              # not derived from the id. `ncbi_M88535_1` -> `M88535` is a rule, and
                              # a rule applied to 410 ids is one that can be wrong on a single one
                              # of them with nothing in any output to say so.
                              source_file=srow["source_file"],
                              source_id=(mrow.get("source_id") or "").strip(),
                              source_url=(mrow.get("source_url") or "").strip(),
                              # The generated tiers have no deposit, but `synth_real` is not
                              # therefore free of third-party sequence: its filler is SwissProt
                              # and its payload is a real protein in every tier. Passed through
                              # verbatim; what an empty value means is in the manifest, not here.
                              filler_source=(mrow.get("filler_source") or "").strip(),
                              **orf_coordinates(arms.orf_id)))
            queries.append((pid, arms.orf_seq))
            # The licence statement rests on this: a query that contains its payload's
            # catalogue entry verbatim contains UniProt sequence, whether the construct around
            # it was deposited or generated. Counted here rather than reasoned about, because
            # the obvious reasoning is wrong: the GENERATED tiers are the ones where it is
            # always true, since they are built around an exact payload span.
            if ref_by_acc[acc] in arms.orf_seq:
                verbatim += 1
            kept += 1
        # Three provenance limits, counted before they are reported: a count of zero makes one free
        # to forget, and a count above zero is something a reader will otherwise derive wrongly.
        tier_items = [i for i in items if i["tier"] == tier]
        wraps = sum(1 for i in tier_items if i["orf_wraps_origin"] == 1)
        unparsed = sum(1 for i in tier_items if i["orf_wraps_origin"] == "")
        no_src = sum(1 for i in tier_items if not i["source_id"])
        # A tier whose manifest has no `source_id` column has no deposits to name, which is a
        # different statement from "the lookup failed" and has to read as one: an expected absence
        # printed as a miss is how a reader learns to ignore the line.
        has_src_col = any("source_id" in r for r in man.values())

        tier_meta.append(dict(tier=tier, description=blurb, results_dir=str(rd), manifest=str(mp),
                              items=kept, excluded=dict(no_payload_accession=no_acc,
                                                        payload_absent_from_catalogue=not_in_cat,
                                                        no_orf_carries_the_payload=no_orf),
                              queries_containing_payload_verbatim=verbatim,
                              items_naming_a_deposit=(len(tier_items) - no_src) if has_src_col else 0,
                              manifest_has_source_id_column=has_src_col,
                              orfs_wrapping_origin=wraps,
                              orf_ids_not_parsed=unparsed))
        print(f"    {tier}: {kept} items (excluded: {no_acc} without a payload accession, "
              f"{not_in_cat} whose payload is not in the catalogue, {no_orf} with no ORF carrying it)")
        src = (f"{len(tier_items) - no_src} of {len(tier_items)} name a deposit" if has_src_col
               else "generated tier: no deposits, and this manifest has no source_id column")
        print(f"    provenance: {src}; {wraps} ORF(s) wrap the origin"
              + (" (end < start, so take a length from orf_len and not from the coordinates)"
                 if wraps else "")
              + (f"; {unparsed} orf_id(s) in an unrecognised shape, coordinates left empty"
                 if unparsed else ""))

    if not items:
        print("no items; nothing to build", file=sys.stderr)
        return 2

    # An item is addressed by `construct_id`: it is the key of `items.csv`, the header in `queries.faa`
    # and the key a submission returns rankings under. So two tiers that share construct ids cannot both
    # be in one benchmark under bare ids, and `synth_real` and `synth_random` share all 336 of theirs,
    # because they are the same constructs differing only in the filler.
    #
    # **This refuses rather than warning, because the silent version is not an inconvenience, it is a
    # missing tier.** Measured 21 Oct on a four-tier build made before this check existed:
    # `queries.faa` came out with 1,102 records and 766 distinct headers: 336 duplicate ids carrying
    # DIFFERENT sequences, so every reader that keys a FASTA on its id, `read_faa` in this repository
    # included, keeps the last and drops the first. `items.csv` had 1,102 rows and 766 distinct keys the
    # same way, and `54_baseline_submission.py` builds `{i.construct_id: i}` from it, so the tier
    # assembled FIRST was shadowed entirely and its rows were then skipped by that script's own
    # `it.tier != tier` guard: the four-tier baseline came out with **432 rows, exactly the three-tier
    # count, with `synth_real` absent**. Adding a tier deleted one. Nothing crashed and no count printed
    # the loss. This is this project's free-text `--tier` row one level up: there the ambiguity was which
    # tier a *number* belonged to, here it is which tier an *item* belongs to.
    clashing = sorted(k for k, v in Counter(i["construct_id"] for i in items).items() if v > 1)
    namespaced = False
    if clashing:
        if not args.namespace_ids:
            tiers_of = {}
            for i in items:
                tiers_of.setdefault(i["construct_id"], []).append(i["tier"])
            pairs = sorted({tuple(sorted(v)) for k, v in tiers_of.items() if len(v) > 1})
            print(f"REFUSING: {len(clashing)} construct id(s) appear in more than one selected tier, "
                  f"across {pairs}. `construct_id` is what items.csv is keyed on, what queries.faa "
                  f"headers are, and what a submission returns rankings under, so a build with "
                  f"duplicates silently merges the tiers: the FASTA alone would lose one sequence per "
                  f"collision. Example: {clashing[0]}. Re-run with --namespace-ids to write every id as "
                  f"`tier:construct_id`, or select tiers that do not collide.", file=sys.stderr)
            return 2
        queries = [(f"{i['tier']}:{pid}", seq) for i, (pid, seq) in zip(items, queries)]
        for i in items:
            i["construct_id"] = f"{i['tier']}:{i['construct_id']}"
        namespaced = True
        print(f"  ids namespaced as `tier:construct_id` ({len(clashing)} collision(s) resolved)")

    # Only the benchmark's own payloads travel, so the answer key is small enough to read and diff.
    wanted = {i["payload_accession"] for i in items}
    hom_rows = []
    with open(args.homologs, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["payload_accession"] in wanted:
                hom_rows.append(r)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "items.csv", "w", newline="", encoding="utf-8-sig") as fh:
        # LF, not csv's default CRLF: these bytes are hashed into MANIFEST.json and git
        # serves LF to every clone, so a CRLF file here publishes a checksum nobody can
        # reproduce. `.gitattributes` pins the other half.
        w = csv.DictWriter(fh, fieldnames=list(items[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(items)
    with open(out / "queries.faa", "w", encoding="utf-8", newline="\n") as fh:
        for pid, seq in queries:
            fh.write(f">{pid}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")
    with open(out / "homologs.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hom_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(hom_rows)
    with open(out / "catalogue_accessions.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(e.accession for e in refs) + "\n")
    if args.copy_catalogue:
        shutil.copy2(args.reference, out / "catalogue.fasta")

    strata_counts = {t["tier"]: Counter(i["stratum"] for i in items if i["tier"] == t["tier"])
                     for t in tier_meta}
    manifest = dict(
        benchmark="plasmid-pilot payload retrieval under a holdout",
        built_from_commit_note="record the commit when releasing; this file does not shell out to git",
        task=("given one protein per item (the payload-bearing ORF as a naive caller emitted it) and the "
              "catalogue, return a ranking of catalogue accessions. Scored on where the best relative "
              "SURVIVING each holdout threshold lands."),
        rules_as_code="src/pilot/benchmark.py",
        # A7, 22 Oct: every item says where its sequence came from and where on the plasmid
        # the ORF sits, so shipping `queries.faa` is documented rather than merely defended.
        provenance=dict(
            columns={
                "source_file": "the record this item was built from, as the tier manifest "
                               "names it. Empty for no item",
                "source_id": "the depositor-facing id, read from the tier manifest and never "
                             "derived from construct_id. Empty for the two generated tiers, "
                             "whose manifests have no such column",
                "source_url": "where the deposit can be fetched. Present for the hand tier; "
                              "the corpus manifest carries no URL column, so it is empty "
                              "there and source_id names the record",
                "filler_source": "`swissprot` for synth_real. Empty for synth_random, whose "
                                 "filler is random sequence by the tier definition, and for "
                                 "the deposited tiers, which have no filler concept",
                "plasmid_id": "construct_id before namespacing",
                "orf_strand / orf_start / orf_end": "1-based inclusive on the plasmid, the "
                                                    "same coordinates packed into orf_id",
                "orf_wraps_origin": "1 when the ORF crosses the origin of a circular plasmid, "
                                    "so end < start",
            },
            coordinates_note=("the plasmids are circular and the ORF caller is a circular "
                              "six-frame scanner, so `end - start + 1` is NEGATIVE where "
                              "orf_wraps_origin is 1. Take a length from `orf_len`, which is "
                              "derived from the translated sequence and is correct for every "
                              "item. The per-tier counts are in `tiers[].orfs_wrapping_origin`"),
            uniprot_note=("`queries.faa` is not free of third-party sequence and the licence "
                          "section says so. Each tier reports "
                          "`queries_containing_payload_verbatim`: the payload is a real "
                          "Swiss-Prot protein in every tier, and in the two GENERATED tiers it "
                          "appears exactly, because those constructs are built around an exact "
                          "payload span. So the synthetic tiers are the ones where a query "
                          "certainly contains UniProt sequence, which is the opposite of what "
                          "the word synthetic suggests"),
        ),
        # Two licences, because two parties own what is here. Machine-readable so a tool does not
        # have to parse README prose, and generated rather than hand-added for the reason the
        # distribution string above now carries its own comment.
        licence=dict(
            code_and_derived_data="MIT",
            covers="src/, scripts/, items.csv, queries.faa, homologs.csv and both baseline pairs: "
                   "this project's own derivation",
            catalogue="CC BY 4.0",
            catalogue_note="reference_twilight.fasta is derived from UniProt Swiss-Prot and carries "
                           "UniProt's CC BY 4.0 attribution. It is attached to the release, not "
                           "committed; this directory carries the accession list and the sha256",
            citation="CITATION.cff at the repository root"),
        scorer="scripts/53_score_submission.py",
        thresholds=list(THRESHOLDS), depth=DEPTH,
        strata=[{"name": n, "lo": lo, "hi": hi} for n, lo, hi in STRATA],
        metric_rules=[
            "mean rank and MRR are primary, stratified by f; recall@10 is the operational secondary; "
            "top-1 is a column and never a headline",
            "for an embedding arm recall@10 and MRR are primary, and a mean rank may be quoted only "
            "beside its count of rank >= 10,000 failures",
            "no pooled scalar across f strata, ever; pilot.benchmark.pooled_scalar raises",
            "micro and macro are both always reported: constructs are not independent trials",
            "an item that cannot be scored stays in the denominator as a miss",
            "never pool the tiers into one number",
        ],
        f_definition=("oracle_span_length / len(payload-bearing ORF), from pilot.arms.build_arms. NOT "
                      "the longest ORF: that mistake produced a wrong hand-tier distribution on "
                      "17 Oct that tier totals could not have caught"),
        tiers=tier_meta,
        n_items=len(items), n_payloads=len(wanted),
        # True when `construct_id` was rewritten as `tier:construct_id` because two selected tiers
        # share ids. It changes what a submission is keyed on, so it is recorded rather than inferred
        # from the ids: a scorer must not have to guess which convention a directory uses.
        construct_ids_namespaced=namespaced,
        strata_counts={k: dict(v) for k, v in strata_counts.items()},
        catalogue=dict(
            path=str(args.reference), sha256=_sha256(args.reference), entries=len(refs),
            entries_in_window=len(in_window), window_aa=[MIN_LEN, MAX_LEN], residues=db_residues,
            shipped_as="accession list and checksum" + (" and FASTA" if args.copy_catalogue else ""),
            # Reworded for a public artefact on 21 Oct by hand, in the generated JSON, where a
            # rebuild would have silently reverted it: MANIFEST.json has no source but this line.
            distribution=("decided 20 Oct: the repository carries the accession list and "
                          "this checksum, and the FASTA is attached to the release. Verify a copy with "
                          "`52_build_benchmark.py --verify-catalogue <fasta>`"),
            built_by="exact Smith-Waterman over Swiss-Prot; the search script is part of the working repository and is not shipped, because the catalogue is distributed as an accession list and a checksum",
            source=(dict(path=str(args.swissprot), sha256=_sha256(args.swissprot),
                         bytes=args.swissprot.stat().st_size,
                         release_label="NOT RECORDED when downloaded (~19 Oct). The sha256 above is "
                                       "what pins this input, and a different Swiss-Prot release "
                                       "would change the catalogue and therefore the significance "
                                       "floor. This is why the release ships the FASTA: a rebuild "
                                       "cannot be checked against the checksum without it")
                    if args.swissprot.is_file() else
                    dict(path=str(args.swissprot), note="absent at build time; not checksummed")),
            note=("the significance floor is a function of this residue count, so a different "
                  "catalogue is a different benchmark. This project has used eighteen reference sets "
                  "and keeps a register of them; this is the 40,750-protein gate_all173 one"),
            score_floor_at_m100=round(score_floor(args.max_evalue, 100, db_residues), 2),
            score_floor_at_m3000=round(score_floor(args.max_evalue, 3000, db_residues), 2),
            max_evalue=args.max_evalue,
        ),
        files={},
    )
    for name in ("items.csv", "queries.faa", "homologs.csv", "catalogue_accessions.txt"):
        p = out / name
        manifest["files"][name] = dict(bytes=p.stat().st_size, sha256=_sha256(p))
    with open(out / "MANIFEST.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n{len(items):,} items over {len(tier_meta)} tier(s), {len(wanted)} distinct payloads, "
          f"{len(hom_rows):,} homolog rows")
    print(f"{'tier':<12}" + "".join(f"{n:>12}" for n, _, _ in STRATA))
    for t in tier_meta:
        c = strata_counts[t["tier"]]
        print(f"{t['tier']:<12}" + "".join(f"{c.get(n, 0):>12}" for n, _, _ in STRATA))
    print(f"\nwrote {out}/: items.csv, queries.faa, homologs.csv, catalogue_accessions.txt, "
          f"MANIFEST.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
