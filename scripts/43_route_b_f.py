#!/usr/bin/env python3
"""Payload fraction `f` for the route B deposits, measured from the downloaded GenBank files.

    python scripts/43_route_b_f.py

**What this settles.** this project recorded that the route B picks were *not* selected for payload
fraction and could not have been: 44.2's architecture flags are deposit-level, so the harvest could say
"this deposit contains a 2A somewhere" and never "this protein is the 2A-linked one". Group A
(KLF4, MYC, SOX2, OCT4, NEUROD1, ISL1, GATA4) was chosen on the *bet* that reprogramming cassettes land
at low f, and 48.3 says in as many words that the bet is unmeasured. This script measures it.

**The quantity is the project's own, computed the project's own way.** `f = oracle_span / called_ORF`
, via `orfs.find_orfs` then `arms.pick_payload_orf`, which is the same path
`10_twilight.py` takes. this project's working rules 25.1 is explicit that f uses `len(arms.orf_seq)`, the payload-bearing
ORF, and **not** the longest ORF: that mistake produced a wrong hand-tier f distribution on 17 Oct
that the tier totals could not have caught. Using `pick_payload_orf` rather than re-deriving the choice
is how this script avoids repeating it.

**Why the payload sequence comes from the route B catalogue and not the gate_all173 one.** Route B's 125
candidates were searched separately, so their Swiss-Prot entries live in
`results/route_b_homologs/reference_twilight.fasta`. That catalogue **self-tags** payloads
(`>O43474 homolog_of:O43474`) where `gate_all173` leaves a payload's own entry untagged, a difference
between two rows of this project's register, and the reason the reader here keys on the accession before
the tag rather than on the absence of a tag.

**Nothing here is a screening rule.** f uses the label by construction and is a description of a
benchmark item (26.3). Every limit gets its binding count printed, deposits whose
GenBank yields no ORF carrying the payload are counted and named rather than dropped silently, because
that count is the honest answer to "how much of the route B harvest is scorable at all", which 48.3
lists as unsettled.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.arms import MIN_AA, pick_payload_orf  # noqa: E402
from pilot.genbank import load_plasmids  # noqa: E402
from pilot.orfs import find_orfs  # noqa: E402

# this project's strata, reused unchanged so a route B number reads against the same boundaries as a
# hand-tier or corpus one.
F_STRATA = (0.50, 0.70, 0.95)

# The lowest span identity among the 20 known-true hand-audited payload pairs, measured by
# `--validate` on this same code path. It is a floor read off true pairs, not a threshold anyone tuned,
# and it has NO margin: n = 20, so the true minimum is an estimate from a small sample and a real pair
# at 0.84 would be misclassified. Quote it with its n.
#
# **Coverage is deliberately not used for this.** The same validation measures known-true payload
# coverage down to 0.077: P04517, the TEV polyprotein whose construct carries only its protease
# domain, which is this project's construct seen from the query side. A coverage floor would throw
# away the most interesting true case in the project. Identity separates true pairs from spurious
# alignments; coverage does not, and the two are not interchangeable just because both are fractions.
KNOWN_TRUE_MIN_IDENTITY = 0.856
KNOWN_TRUE_N = 20

# **The route B spans fall into three classes, and the boundary between the first two is a measured
# empty gap rather than a chosen number.** Sorted span identities on this harvest run
# 0.236, 0.236, 0.391, then 0.779, 0.782, 0.822, then 0.856 upward: nothing at all between 0.391 and
# 0.779. So:
#   below 0.50  the payload is ABSENT. Verified for all three by an unconstrained six-frame search:
#               T7 RNAP's best match anywhere in 236780/236781 is 54 aa at identity 0.241, and 238174
#               is a CRISPRa *guide* plasmid that was never expected to carry a KLF4 ORF. These are not
#               ORF-caller failures; the gene is not in the record.
#   0.50-0.856  the payload is present as an ENGINEERED VARIANT: a SOX2/SOX17 chimera, a Gs-coupled
#               rM3D designer receptor. Real sequence, deliberately not the Swiss-Prot protein.
#   >= 0.856    canonical, at or above the hand tier's known-true floor.
# The middle class is a judgement about what counts as the payload,
#  so it is reported separately and never silently merged into either neighbour.
ABSENT_BELOW_IDENTITY = 0.50


def read_payload_seqs(path: Path, wanted: set[str]) -> dict[str, str]:
    """`{accession: sequence}` for the wanted accessions, from a catalogue FASTA.

    **Two catalogues, three header styles, and this is a measured trap rather than a hypothetical one.**
    A first version of this reader took the header's first token, which works on the route B catalogue
    and silently resolves **zero of 16** payloads on gate_all173:

        route B catalogue   >O43474 homolog_of:O43474      self-tagged, accession is the first token
        either catalogue    >P83690 homolog_of:P42212      a homolog, not a payload
        gate_all173 >sp|P42212|GFP_AEQVI Green... Swiss-Prot style, accession is field 1 of 3

    The failure mode is the dangerous one: nothing raises, the payload set comes back empty and every
    construct becomes "unscorable" for a reason that has nothing to do with the construct. Two rows of
    this project's register, two formats; a reader that assumes one is a denominator carried between
    catalogues in a new costume.

    Selecting on "has no `homolog_of` tag" is also wrong, in the other direction: it returns nothing on
    the route B catalogue, where payloads *are* tagged with themselves.
    """
    out: dict[str, list[str]] = {}
    cur: str | None = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(">"):
                head = line[1:].strip()
                tok = head.split()[0]
                # Swiss-Prot / TrEMBL style `db|ACCESSION|NAME`, else the bare first token.
                acc = tok.split("|")[1] if tok.count("|") >= 2 else tok
                cur = acc if acc in wanted else None
                if cur:
                    out.setdefault(cur, [])
            elif cur:
                out[cur].append(line.strip())
    return {k: "".join(v) for k, v in out.items()}


def stratum(f: float, bounds: tuple[float, ...] = F_STRATA) -> str:
    keys = ([f"<{bounds[0]:.2f}"]
            + [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(bounds, bounds[1:])]
            + [f">={bounds[-1]:.2f}"])
    return keys[sum(1 for b in bounds if f >= b)]


def validate(args) -> int:
    """Run the identical path on the 23 hand-audited constructs, where the answer is the depositor's.

    Two things come out of this and both are needed before the route B numbers mean anything:

    1. **Does this script reproduce the hand tier's known f distribution?** this project records the
       hand tier at median f 0.934 with 3 of 15 below 0.70. A script that disagrees with that is
       measuring something else, and 25.1 is the erratum from the last time an f distribution was
       computed down a slightly different path.
    2. **What does a known-true payload pair look like in identity and coverage?** The route B run turns
       up spans at identity 0.236 and coverage 0.159, which are not plausible payload hits: one of
       them is `238174`, a CRISPRa *guide* plasmid that carries no KLF4 ORF at all. The 23 hand pairs
       are true by construction, so their *minimum* identity and coverage is a measured floor rather
       than a threshold anyone chose. That is the distinction this project's working rules draws between deriving a limit
       and tuning one.
    """
    man = {r["filename"]: r for r in csv.DictReader(open(args.hand_manifest, encoding="utf-8-sig"))}
    wanted = {r["payload_accession"] for r in man.values() if r["payload_accession"]}
    seqs = read_payload_seqs(args.hand_payloads, wanted)
    print("VALIDATE against the 23 hand-audited constructs")
    print(f"  manifest rows: {len(man)}   payload accessions: {len(wanted)}      "
          f"sequences resolved: {len(seqs)}")

    out = []
    for fn, r in sorted(man.items()):
        acc = r["payload_accession"]
        fp = args.hand_dir / fn
        if not acc or not fp.is_file() or acc not in seqs:
            continue
        pl = load_plasmids(fp)[0]
        orfs = {f"orf{i}": o.protein
                for i, o in enumerate(find_orfs(pl.seq, circular=True, min_aa=args.min_aa))}
        oid, span = pick_payload_orf(orfs, seqs[acc], args.min_aa)
        if not oid or span is None:
            out.append((fn, acc, None, None, None, None))
            continue
        orf_aa = len(orfs[oid])
        out.append((fn, acc, span.length / orf_aa, span.identity, span.payload_coverage, orf_aa))

    scored = [o for o in out if o[2] is not None]
    fs = [o[2] for o in scored]
    print(f"  scorable: {len(scored)} of {len(out)}")
    print(f"\n  f: median {statistics.median(fs):.3f}  mean {statistics.fmean(fs):.3f}    "
          f"below 0.70: {sum(1 for x in fs if x < 0.70)} of {len(fs)}")
    print("  this project records the hand tier at median 0.934, 3 of 15 below 0.70 "
          "(on the 15 scorable at the 25% holdout, a different denominator from this one)")

    ids = sorted(o[3] for o in scored)
    covs = sorted(o[4] for o in scored)
    print(f"\n  known-true span identity: min {ids[0]:.3f}  p10 {ids[len(ids)//10]:.3f}    "
          f"median {statistics.median(ids):.3f}")
    print(f"  known-true payload coverage: min {covs[0]:.3f}  p10 {covs[len(covs)//10]:.3f}    "
          f"median {statistics.median(covs):.3f}")
    print("\n  the five lowest-coverage known-true pairs (these set the floor, and the route B "
          "spurious hits must sit below them):")
    for fn, acc, f, i, c, orf in sorted(scored, key=lambda o: o[4])[:5]:
        print(f"    cov={c:.3f} id={i:.3f} f={f:.3f}    {acc}    {fn[:48]}")
    return 0



def choose_records(kind_of: dict[tuple[str, str], str],
                   bp_of: dict[tuple[str, str], int]) -> dict[str, str]:
    """Pick one sequence record per deposit: Addgene-Verified first, then the longest.

    `kind_of` maps (deposit, sequence_id) -> provenance; `bp_of` maps the same key to the record's
    length in bp, for the records actually on disk. A deposit whose records are all absent from
    `bp_of` still gets a choice, so a missing download degrades to the old behaviour rather than
    dropping the deposit.

    The length tie-break is the fix for the working record. 236780 and 236781 each carry two
    Addgene-Verified Full records that are **not the same molecule**, 19,523 against 20,956 bp and
    20,369 against 21,498 bp, 3 and 5 annotated CDS against 21 and 22, and taking whichever sorted
    first measured the partial record and reported T7 RNAP absent from a construct that contains it
    at identity 0.998. Length is a property of the record, never of the result, so this rule cannot
    select for the answer.
    """
    best: dict[str, str] = {}
    for dep in {d for d, _ in kind_of}:
        cands = [(sid, kind) for (d, sid), kind in kind_of.items() if d == dep]
        on_disk = [c for c in cands if (dep, c[0]) in bp_of]
        for sid, kind in sorted(on_disk or cands):
            if dep not in best or (kind == "addgene_verified_full",
                                   bp_of.get((dep, sid), 0)) > (
                    kind_of.get((dep, best[dep])) == "addgene_verified_full",
                    bp_of.get((dep, best[dep]), 0)):
                best[dep] = sid
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("data/route_b_raw"))
    ap.add_argument("--links", type=Path, default=Path("data/route_b_deposit_sequences.csv"))
    ap.add_argument("--selected", type=Path, default=Path("data/route_b_selected.csv"))
    ap.add_argument("--payloads", type=Path,
                    default=Path("results/route_b_homologs/reference_twilight.fasta"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/route_b_f"))
    ap.add_argument("--min-aa", type=int, default=MIN_AA)
    ap.add_argument("--validate", action="store_true",
                    help="score the same path against the 23 hand-audited constructs, where the "
                         "payload is the depositor's stated insert, and print the identity/coverage "
                         "distribution of known-true pairs (this project's working rules: validate a rule against the 23)")
    ap.add_argument("--hand-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--hand-manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--hand-payloads", type=Path,
                    default=Path("results/gate_all173/reference_twilight.fasta"))
    args = ap.parse_args(argv)

    if args.validate:
        return validate(args)

    for p in (args.raw_dir, args.links, args.selected, args.payloads):
        if not p.exists():
            print(f"missing input: {p}", file=sys.stderr)
            return 2

    sel = {r["uniprot_accession"]: r for r in csv.DictReader(
        open(args.selected, encoding="utf-8-sig"))}
    links = list(csv.DictReader(open(args.links, encoding="utf-8-sig")))

    # deposit -> the selected payloads it is an example of, and the deposit's title
    dep_accs: dict[str, list[str]] = {}
    dep_title: dict[str, str] = {}
    for r in links:
        d = r["addgene_id"]
        dep_title[d] = r["deposit_title"]
        accs = [a for a in r["payload_accessions"].split(";") if a]
        dep_accs.setdefault(d, [])
        for a in accs:
            if a not in dep_accs[d]:
                dep_accs[d].append(a)

    seqs = read_payload_seqs(args.payloads, set(sel))
    missing_seq = sorted(set(sel) - set(seqs))

    # One sequence record per deposit. Several deposits carry two records, and counting both would
    # double those constructs in every denominator: the kind of quiet n inflation this project's working rules's
    # "state the catalogue" rule exists to stop.
    #
    # Two rules, in order, and the second one was bought the hard way. First prefer the
    # Addgene-Verified record over a Depositor one, which is the stricter provenance. Then, among
    # records of the same kind, **prefer the longest**, because 236780 and 236781 each carry two
    # Addgene-Verified Full records that are not the same molecule: 19,523 bp against 20,956 bp and
    # 20,369 bp against 21,498 bp, 3 and 5 annotated CDS against 21 and 22. At most one record of a
    # deposit can be the complete plasmid, so taking whichever sorted first silently measured a
    # partial record and called T7 RNAP absent from it. Length is a property of the
    # record and not of the result, so this rule cannot select for the answer, which is the line
    # this project's working rules draws against tuning a manipulation until it matches its own validation target.
    kind_of = {(r["addgene_id"], r["gbk_url"].rsplit("-", 1)[-1].removesuffix(".gbk")):
               r["sequence_kind"] for r in links if r["gbk_url"]}

    files = sorted(args.raw_dir.glob("*.gbk"))
    bp_of: dict[tuple[str, str], int] = {}
    for fp in files:
        d = fp.name.split("-")[2] if fp.name.count("-") >= 2 else fp.stem
        try:
            pls = load_plasmids(fp)
        except Exception:
            continue
        if pls:
            bp_of[(d, fp.stem.rsplit("-", 1)[-1])] = len(pls[0].seq)

    best_seq_id = choose_records(kind_of, bp_of)
    print(f"route B f measurement")
    print(f"  GenBank files: {len(files)} in {args.raw_dir}")
    print(f"  payload sequences resolved: {len(seqs)} of {len(sel)} selected"
          + (f"  MISSING {missing_seq}" if missing_seq else ""))
    print(f"  min ORF length: {args.min_aa} aa\n")

    rows = []
    no_orf = []
    per_file_problems = []
    skipped_dup = []
    for fp in files:
        dep = fp.name.split("-")[2] if fp.name.count("-") >= 2 else fp.stem
        seq_id = fp.stem.rsplit("-", 1)[-1]
        if dep in best_seq_id and seq_id != best_seq_id[dep]:
            skipped_dup.append((dep, seq_id, best_seq_id[dep]))
            continue
        try:
            plasmids = load_plasmids(fp)
        except Exception as exc:  # untrusted input: parse failures are data, not crashes
            per_file_problems.append((fp.name, f"parse failed: {exc}"))
            continue
        if not plasmids:
            per_file_problems.append((fp.name, "no record in file"))
            continue
        pl = plasmids[0]
        orf_list = find_orfs(pl.seq, circular=True, min_aa=args.min_aa)
        orfs = {f"orf{i}": o.protein for i, o in enumerate(orf_list)}
        for acc in dep_accs.get(dep, []):
            pseq = seqs.get(acc)
            if not pseq:
                continue
            oid, span = pick_payload_orf(orfs, pseq, args.min_aa)
            if not oid or span is None:
                no_orf.append((dep, acc, fp.name))
                rows.append({
                    "addgene_id": dep, "sequence_id": seq_id, "file": fp.name,
                    "deposit_title": dep_title.get(dep, ""), "payload_accession": acc,
                    "payload_name": sel[acc]["protein_name"], "group": sel[acc]["group"],
                    "plasmid_bp": len(pl.seq), "n_orfs": len(orfs),
                    "orf_aa": "", "span_aa": "", "f": "",
                    "span_identity": "", "payload_coverage": "", "stratum": "unscorable",
                })
                continue
            orf_aa = len(orfs[oid])
            f = span.length / orf_aa
            rows.append({
                "addgene_id": dep, "sequence_id": seq_id, "file": fp.name,
                "deposit_title": dep_title.get(dep, ""), "payload_accession": acc,
                "payload_name": sel[acc]["protein_name"], "group": sel[acc]["group"],
                "plasmid_bp": len(pl.seq), "n_orfs": len(orfs),
                "orf_aa": orf_aa, "span_aa": span.length, "f": round(f, 4),
                "span_identity": round(span.identity, 4),
                "payload_coverage": round(span.payload_coverage, 4),
                "stratum": stratum(f),
                "payload_present": ("absent" if span.identity < ABSENT_BELOW_IDENTITY
                                    else "engineered_variant"
                                    if span.identity < KNOWN_TRUE_MIN_IDENTITY else "canonical"),
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0]) if rows else []
    with open(args.out_dir / "f_by_construct.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    aligned = [r for r in rows if r["f"] != ""]
    absent = [r for r in aligned if r["payload_present"] == "absent"]
    variant = [r for r in aligned if r["payload_present"] == "engineered_variant"]
    scored = [r for r in aligned if r["payload_present"] == "canonical"]
    fs = [r["f"] for r in scored]
    if skipped_dup:
        print(f"deduplicated: {len(skipped_dup)} second sequence record(s) skipped "
              f"{[(d, s) for d, s, _ in skipped_dup]}")
    print(f"BINDING, something aligns: {len(aligned)} of {len(rows)} (deposit, payload) pairs; "
          f"{len(no_orf)} have no ORF at all")
    print(f"  the identity gap this classification rests on: "
          f"{sorted(round(r['span_identity'], 3) for r in aligned)[:8]} ...")
    print(f"BINDING, payload ABSENT (identity < {ABSENT_BELOW_IDENTITY}): {len(absent)}")
    for r in sorted(absent, key=lambda r: r["span_identity"]):
        print(f"    ABSENT  id={r['span_identity']:.3f} cov={r['payload_coverage']:.3f}  "
              f"{r['payload_accession']}  {r['addgene_id']:>7}    {r['deposit_title'][:46]}")
    print(f"BINDING, ENGINEERED VARIANT ({ABSENT_BELOW_IDENTITY} <= identity < "
          f"{KNOWN_TRUE_MIN_IDENTITY}, the known-true floor over {KNOWN_TRUE_N} hand pairs): "
          f"{len(variant)}: excluded from the f figures below; whether these count as the payload "
          f"is my call")
    for r in sorted(variant, key=lambda r: r["span_identity"]):
        print(f"    VARIANT id={r['span_identity']:.3f} cov={r['payload_coverage']:.3f}  "
              f"{r['payload_accession']}  {r['addgene_id']:>7}    {r['deposit_title'][:46]}")
    for d, a, fn in no_orf:
        print(f"    unscorable: {d} / {a}    {dep_title.get(d,'')[:52]}")
    if per_file_problems:
        print(f"BINDING, files that would not parse: {len(per_file_problems)}")
        for n, why in per_file_problems:
            print(f"        {n}: {why}")

    if not scored:
        print("\nnothing scorable; stopping before the summary")
        return 1

    counts: dict[str, int] = defaultdict(int)
    for r in scored:
        counts[r["stratum"]] += 1
    print(f"\nf over {len(fs)} scorable pairs: "
          f"min {min(fs):.3f} / median {statistics.median(fs):.3f} / mean {statistics.fmean(fs):.3f}  "
          f"/ max {max(fs):.3f}")
    print(f"  below 0.70: {sum(1 for x in fs if x < 0.70)} of {len(fs)}  "
          f"({100*sum(1 for x in fs if x < 0.70)/len(fs):.1f}%)")
    for k in ([f"<{F_STRATA[0]:.2f}"]
              + [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(F_STRATA, F_STRATA[1:])]
              + [f">={F_STRATA[-1]:.2f}"]):
        print(f"        {k:<12}  {counts[k]:>3}")

    # Group A is a stated bet (48.3); report it as its own line so the bet is settled explicitly.
    print("\nby group (the bet in the working record is group A):")
    bygroup: dict[str, list[float]] = defaultdict(list)
    for r in scored:
        bygroup[r["group"]].append(r["f"])
    for g in sorted(bygroup):
        v = bygroup[g]
        print(f"    {g:<28} n={len(v):>3}  median f {statistics.median(v):.3f}    "
              f"below 0.70: {sum(1 for x in v if x < 0.70)}/{len(v)}")

    print("\nlowest 12 by f:")
    for r in sorted(scored, key=lambda r: r["f"])[:12]:
        print(f"  f={r['f']:.3f}    {r['payload_accession']}  orf {r['orf_aa']:>5} aa  "
              f"span {r['span_aa']:>4} aa  {r['addgene_id']:>7}    {r['deposit_title'][:46]}")

    summary = {
        "n_files": len(files),
        "n_pairs": len(rows),
        "n_scorable": len(scored),
        "n_unscorable": len(no_orf),
        "n_absent": len(absent),
        "n_engineered_variant": len(variant),
        "absent": [{"addgene_id": r["addgene_id"], "payload_accession": r["payload_accession"], "span_identity": r["span_identity"]} for r in absent],
        "engineered_variant": [{"addgene_id": r["addgene_id"], "payload_accession": r["payload_accession"], "span_identity": r["span_identity"]} for r in variant],
        "unscorable": [{"addgene_id": d, "payload_accession": a, "file": fn} for d, a, fn in no_orf],
        "parse_problems": [{"file": n, "why": w} for n, w in per_file_problems],
        "f_median": statistics.median(fs),
        "f_mean": statistics.fmean(fs),
        "f_min": min(fs),
        "f_max": max(fs),
        "n_below_070": sum(1 for x in fs if x < 0.70),
        "strata": dict(counts),
        "by_group": {g: {"n": len(v), "median_f": statistics.median(v),
                         "n_below_070": sum(1 for x in v if x < 0.70)}
                     for g, v in bygroup.items()},
        "payloads_without_sequence": missing_seq,
        "min_aa": args.min_aa,
    }
    (args.out_dir / "route_b_f.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out_dir / 'f_by_construct.csv'} and {args.out_dir / 'route_b_f.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
