#!/usr/bin/env python3
"""Measure reference-side mean-pool dilution across the catalogue (n = 1 -> a distribution).

    python scripts/40_catalogue_dilution.py --out-dir results/catalogue_dilution
    python scripts/40_catalogue_dilution.py --reference results/gate_all173/reference_twilight.fasta \
        --homologs results/gate_all173/homologs.csv --out-dir results/catalogue_dilution

**Why this exists.** `addgene_171782_pET28_MBP_sTEV` ranks **12,136 of 40,750** on the
`oracle` arm at the **100% holdout**: clean payload span as the query, answer present at 100%
identity, nothing masked. Its payload P04517 is a 3,054 aa polyprotein and the construct carries a
~240 aa domain of it. The thesis of this project is that mean-pooling dilutes the *query*; here it is
the *reference* that is diluted, and no decomposition of the query can repair that. 46.5 and 47.7 both
list the finding as not verified, because it has n = 1 and the catalogue had never been examined.

This script examines it. Four measurements, in the order the question needs them:

1. **The length census**, with counts rather than a plot: entries and residues above 1,022 aa (ESM-2's
   position budget at this project's truncation-row setting) and above 2,048 aa (the current
   `--max-length`, whose binding count is recorded nowhere). this project's working rules records 3,889 of 40,750 above
   1,022; this recomputes it rather than quoting it.
2. **The payloads that are themselves long**, since a payload is also a catalogue entry and is
   therefore susceptible as a *target*. Reconciled across `payload_lengths.csv`, both manifests and the
   catalogue's own headers, because the three disagree about which accessions are in play.
3. **A domain-count proxy**, because no domain annotation is available offline and nothing may be
   downloaded. `family_length_ratio` with `repeat_fraction` as a disqualifier; the module docstring
   states the blind spot and section 3 of `report.md` measures it.
4. **The question the benchmark actually asks**: of the entries a query must out-rank, how many are
   poorly served by one mean-pooled vector: as a fraction of the catalogue, and separately as a
   fraction of the *homolog targets*, which is what retrieval is scored against.

**Every limit gets its binding count printed.** That rule has found every real bug in this repo, and the
one it finds here is already known: the catalogue's reader drops entries outside `homology.MIN_LEN`,
and the single 41 aa payload Q9TT33 is why the residue total printed here is 20,806,772
against CONTEXT's 20,806,731.

Writes `catalogue_dilution.json`, `catalogue_entries.csv`, `payload_dilution.csv`,
`f_ref_by_payload.csv` and `catalogue_dilution.log` to a new `--out-dir`.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from pilot.catalogue_dilution import (F_STRATA, LENGTH_THRESHOLDS, REPEAT_DISQUALIFY,
                                      counter_summary, family_length_ratio,
                                      family_lengths_from_rows, length_census, macro_micro,
                                      reference_payload_fraction, repeat_fraction, stratify,
                                      summarise_repeats)
from pilot.homology import MAX_LEN, MIN_LEN
from pilot.reference import iter_fasta

HOMOLOG_TAG = "homolog_of:"


def read_csv(path: Path) -> list[dict]:
    """A project CSV. `utf-8-sig` is not optional: every output here is written for Excel."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path,
                    default=Path("results/gate_all173/reference_twilight.fasta"),
                    help="the catalogue under test. the working record is the register; name the row in any "
                         "number quoted from this run.")
    ap.add_argument("--homologs", type=Path,
                    default=Path("results/gate_all173/homologs.csv"))
    ap.add_argument("--payload-lengths", type=Path,
                    default=Path("results/gate_all173/payload_lengths.csv"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--manifest-auto", type=Path, default=Path("data/manifest_auto.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/catalogue_dilution"))
    ap.add_argument("--thresholds", type=int, nargs="+", default=list(LENGTH_THRESHOLDS),
                    help="length thresholds for the census, in aa")
    ap.add_argument("--repeat-k", type=int, default=6)
    ap.add_argument("--repeat-disqualify", type=float, default=REPEAT_DISQUALIFY)
    ap.add_argument("--ratio-flag", type=float, default=1.5,
                    help="family_length_ratio above which an entry is flagged as carrying material its "
                         "relatives do not. A flag on a proxy, not a gate on a measurement: the "
                         "distribution is written out whole so a different line can be drawn without "
                         "re-running.")
    args = ap.parse_args(argv)

    for p in (args.reference, args.homologs, args.payload_lengths, args.manifest,
              args.manifest_auto):
        if not p.is_file():
            print(f"not found: {p}", file=sys.stderr)
            return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = open(args.out_dir / "catalogue_dilution.log", "w", encoding="utf-8")

    def say(line: str = "") -> None:
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    out: dict = {"catalogue": str(args.reference), "homologs": str(args.homologs),
                 "thresholds": args.thresholds}

    # ---- 1. the length census -----------------------------------------------------------------
    entries = list(iter_fasta(args.reference))
    lengths = [len(e.seq) for e in entries]
    by_acc = {e.accession: e for e in entries}
    cen = length_census(lengths, tuple(args.thresholds), window=(MIN_LEN, MAX_LEN))

    say(f"catalogue: {args.reference}")
    say(f"    {cen.n_entries:,} entries, {len(by_acc):,} distinct accessions, "
        f"{cen.n_residues:,} residues")
    say(f"  length min {cen.min_len} / median {cen.median_len:.0f} / mean {cen.mean_len:.1f} / "
        f"max {cen.max_len}")
    say(f"  BINDING, reference window [{MIN_LEN}, {MAX_LEN}] aa: {cen.n_below_window} entries below, "
        f"{cen.n_above_window} above -> the reader sees "
        f"{cen.n_entries - cen.n_below_window - cen.n_above_window:,} entries and "
        f"{cen.residues_in_window:,} residues")
    if cen.n_below_window or cen.n_above_window:
        drop = [e.accession for e in entries if not (MIN_LEN <= len(e.seq) <= MAX_LEN)]
        say(f"    dropped: {', '.join(f'{a} ({len(by_acc[a].seq)} aa)' for a in drop)}")
        say("    this is this project's payload length-window bypass, and it is why the residue total "
            "above differs from CONTEXT's 20,806,731 for this catalogue.")
    for t in args.thresholds:
        n, res = cen.above[t]
        say(f"  BINDING, > {t:,} aa: {n:,} entries ({100 * cen.fraction_entries(t):.2f}% of entries), "
            f"{res:,} residues ({100 * cen.fraction_residues(t):.2f}% of residues)")

    out["length_census"] = {
        "n_entries": cen.n_entries, "n_distinct_accessions": len(by_acc),
        "n_residues": cen.n_residues, "min": cen.min_len, "max": cen.max_len,
        "median": cen.median_len, "mean": round(cen.mean_len, 2),
        "window": [MIN_LEN, MAX_LEN], "n_below_window": cen.n_below_window,
        "n_above_window": cen.n_above_window, "residues_in_window": cen.residues_in_window,
        "above": {str(t): {"entries": cen.above[t][0], "residues": cen.above[t][1],
                           "pct_entries": round(100 * cen.fraction_entries(t), 4),
                           "pct_residues": round(100 * cen.fraction_residues(t), 4)}
                  for t in args.thresholds}}

    # ---- 2. the payloads, which are catalogue entries too --------------------------------------
    pl_rows = read_csv(args.payload_lengths)
    payload_aa = {r["payload_accession"]: int(r["payload_aa"]) for r in pl_rows}
    man_hand = {(r.get("payload_accession") or "").strip()
                for r in read_csv(args.manifest)} - {""}
    man_auto = {(r.get("payload_accession") or "").strip()
                for r in read_csv(args.manifest_auto)} - {""}
    manifest_union = man_hand | man_auto

    say(f"\npayloads: {len(payload_aa)} in {args.payload_lengths.name}, "
        f"{len(man_hand)} in {args.manifest.name}, {len(man_auto)} in {args.manifest_auto.name}  "
        f"({len(manifest_union)} distinct across both manifests)")
    missing_from_cat = sorted(a for a in payload_aa if a not in by_acc)
    in_manifest_not_tested = sorted(manifest_union - set(payload_aa))
    tested_not_in_manifest = sorted(set(payload_aa) - manifest_union)
    say(f"  BINDING, reconciliation: {len(missing_from_cat)} tested payloads absent from the "
        f"catalogue; {len(in_manifest_not_tested)} manifest accessions not among the "
        f"{len(payload_aa)} tested; {len(tested_not_in_manifest)} tested payloads in neither manifest")
    for label, acc_list in (("absent from catalogue", missing_from_cat),
                            ("manifest-only", in_manifest_not_tested),
                            ("tested but not in a manifest", tested_not_in_manifest)):
        if acc_list:
            say(f"        {label}: {', '.join(acc_list[:20])}"
                + (f" ... (+{len(acc_list) - 20})" if len(acc_list) > 20 else ""))

    long_payloads = sorted(((a, L) for a, L in payload_aa.items() if L > min(args.thresholds)),
                           key=lambda x: -x[1])
    say(f"  payloads longer than {min(args.thresholds):,} aa: {len(long_payloads)} of "
        f"{len(payload_aa)}")
    for a, L in long_payloads:
        marks = " ".join(f">{t}" for t in sorted(args.thresholds) if L > t)
        desc = by_acc[a].description[:60] if a in by_acc else "(absent)"
        say(f"        {a}    {L:>5} aa  [{marks}]  {desc}")
    out["payloads"] = {
        "n_tested": len(payload_aa), "n_manifest_hand": len(man_hand),
        "n_manifest_auto": len(man_auto), "n_manifest_union": len(manifest_union),
        "missing_from_catalogue": missing_from_cat,
        "manifest_only": in_manifest_not_tested,
        "tested_not_in_manifest": tested_not_in_manifest,
        "above": {str(t): sorted((a for a, L in payload_aa.items() if L > t),
                                 key=lambda a: -payload_aa[a]) for t in args.thresholds}}

    # ---- 3. the domain-count proxy -------------------------------------------------------------
    hom_rows = read_csv(args.homologs)
    fam_lengths = family_lengths_from_rows(hom_rows, lambda r: int(r["length"]))
    fam_median = {p: statistics.median(v) for p, v in fam_lengths.items()}
    # an entry can be a homolog of several payloads; the proxy takes the family that makes it look
    # LEAST unusual, so a flag means every family it belongs to calls it long
    entry_families: dict[str, list[str]] = {}
    for r in hom_rows:
        entry_families.setdefault(r["homolog_accession"], []).append(r["payload_accession"])

    say(f"\nhomolog table: {len(hom_rows):,} rows, {len(fam_lengths)} payload families, "
        f"{len(entry_families):,} distinct subjects")
    no_family = [e.accession for e in entries if e.accession not in entry_families]
    singleton = [p for p, v in fam_lengths.items() if len(v) < 2]
    say(f"  BINDING, proxy coverage: {len(no_family):,} of {cen.n_entries:,} entries "
        f"({100 * len(no_family) / cen.n_entries:.2f}%) belong to no payload family, so the ratio is "
        f"undefined for them; {len(singleton)} families have a single member, where it is "
        f"uninformative by construction")

    header_fams = Counter()
    for e in entries:
        parts = e.header.split()
        if len(parts) > 1 and parts[1].startswith(HOMOLOG_TAG):
            header_fams[parts[1][len(HOMOLOG_TAG):]] += 1
    say(f"  BINDING, header tags: {sum(header_fams.values()):,} entries carry a "
        f"'{HOMOLOG_TAG}' tag over {len(header_fams)} families: fewer than the "
        f"{len(fam_lengths)} families the homolog table knows, because a payload's own entry is "
        f"written under its Swiss-Prot header. Families here come from the table, not the headers.")

    ent_rows, ratios, flagged = [], [], []
    for e in entries:
        L = len(e.seq)
        fams = entry_families.get(e.accession, [])
        cands = [family_length_ratio(L, fam_lengths[p]) for p in fams]
        cands = [c for c in cands if c is not None]
        ratio = min(cands) if cands else None
        rep = repeat_fraction(e.seq, args.repeat_k)
        if ratio is not None:
            ratios.append(ratio)
            if ratio >= args.ratio_flag and rep <= args.repeat_disqualify:
                flagged.append((e.accession, L, ratio, rep))
        ent_rows.append({"accession": e.accession, "length_aa": L,
                         "n_families": len(fams),
                         "family_length_ratio": "" if ratio is None else round(ratio, 4),
                         "repeat_fraction": round(rep, 5),
                         "is_payload": int(e.accession in payload_aa),
                         "is_homolog_target": int(bool(fams))})
    reps_all = [r["repeat_fraction"] for r in ent_rows]
    reps_long = [r["repeat_fraction"] for r in ent_rows
                 if r["length_aa"] > min(args.thresholds)]
    say(f"  family_length_ratio over {len(ratios):,} entries: median "
        f"{statistics.median(ratios):.3f}, p90 {sorted(ratios)[int(0.9 * (len(ratios) - 1))]:.3f}, "
        f"max {max(ratios):.2f}")
    say(f"  flagged at ratio >= {args.ratio_flag} and repeat_fraction <= {args.repeat_disqualify}: "
        f"{len(flagged):,} entries ({100 * len(flagged) / cen.n_entries:.2f}% of the catalogue)")
    rep_all = summarise_repeats(reps_all, args.repeat_disqualify)
    rep_long = summarise_repeats(reps_long, args.repeat_disqualify)
    say(f"  BINDING, repeat disqualifier > {args.repeat_disqualify}: "
        f"{rep_all['n_disqualified']} of {rep_all['n']:,} entries catalogue-wide, "
        f"{rep_long['n_disqualified']} of {rep_long['n']:,} among entries over "
        f"{min(args.thresholds):,} aa (median repeat fraction {rep_long['median']:.4f}), so length "
        f"in this catalogue is not a repeat artefact")

    # the proxy's own blind spot, measured rather than asserted
    blind = []
    for a in sorted(payload_aa, key=lambda x: -payload_aa[x])[:10]:
        if a in fam_lengths and len(fam_lengths[a]) >= 2:
            blind.append({"payload": a, "length_aa": payload_aa[a],
                          "family_n": len(fam_lengths[a]),
                          "family_median_aa": fam_median[a],
                          "ratio": round(payload_aa[a] / fam_median[a], 4)})
    say("  the proxy's blind spot, on the ten longest payloads (ratio ~ 1 means the whole family is "
        "long, so the proxy cannot see dilution that every member shares):")
    for b in blind:
        say(f"        {b['payload']:<8}  {b['length_aa']:>5} aa  family n={b['family_n']:<5}  "
            f"median {b['family_median_aa']:>7.1f} aa  ratio {b['ratio']:.3f}")
    out["proxy"] = {
        "n_with_ratio": len(ratios), "n_no_family": len(no_family),
        "n_singleton_families": len(singleton),
        "ratio_median": round(statistics.median(ratios), 4),
        "ratio_p90": round(sorted(ratios)[int(0.9 * (len(ratios) - 1))], 4),
        "ratio_max": round(max(ratios), 4),
        "ratio_flag": args.ratio_flag, "n_flagged": len(flagged),
        "pct_flagged": round(100 * len(flagged) / cen.n_entries, 4),
        "repeat_all": rep_all, "repeat_long": rep_long,
        "blind_spot_longest_payloads": blind,
        "header_tag_families": len(header_fams),
        "header_tag_entries": sum(header_fams.values())}

    # ---- 4. f_ref on the homolog targets -------------------------------------------------------
    fref_rows, bound_rows, per_row = [], 0, []
    missing_len = 0
    for r in hom_rows:
        p = r["payload_accession"]
        if p not in payload_aa:
            missing_len += 1
            continue
        L = int(r["length"])
        f_ref, is_bound = reference_payload_fraction(float(r["coverage"]), payload_aa[p], L)
        bound_rows += int(is_bound)
        fref_rows.append({"payload_accession": p, "homolog_accession": r["homolog_accession"],
                          "subject_aa": L, "payload_aa": payload_aa[p],
                          "identity": r["identity"], "coverage": r["coverage"],
                          "f_ref": round(f_ref, 4), "f_ref_is_lower_bound": int(is_bound)})
        per_row.append((p, f_ref))

    vals = [v for _, v in per_row]
    strata = stratify(vals, F_STRATA)
    say(f"\nreference-side payload fraction f_ref = aligned columns / subject length, over "
        f"{len(vals):,} homolog rows")
    if missing_len:
        say(f"  BINDING: {missing_len} rows dropped, payload length unknown")
    say(f"  BINDING, coverage cap at 1.0: {bound_rows:,} of {len(vals):,} rows "
        f"({100 * bound_rows / len(vals):.1f}%) have coverage == 1.0, so their f_ref is a LOWER BOUND "
        f"and the true f_ref is at least as high: every count below is therefore conservative")
    say(f"  deciles: " + ", ".join(
        f"p{int(100 * q)}={sorted(vals)[min(int(q * (len(vals) - 1)), len(vals) - 1)]:.3f}"
        for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)))
    say("  strata (the same boundaries the working record uses for the query side):")
    for k, n in strata.items():
        say(f"    f_ref {k:<12}  {n:>6,}  ({100 * n / len(vals):5.2f}%)")
    for t in (0.70, 0.50):
        micro, macro, ngrp = macro_micro(per_row, lambda v, t=t: v < t)
        say(f"  f_ref < {t:.2f}:  micro (over rows) {100 * micro:.2f}%  |  "
            f"macro (over {ngrp} payload families) {100 * macro:.2f}%")

    # per payload family, so the macro number can be read rather than trusted
    by_payload: dict[str, list[float]] = {}
    for p, v in per_row:
        by_payload.setdefault(p, []).append(v)
    fam_rows = []
    for p, vs in sorted(by_payload.items()):
        fam_rows.append({"payload_accession": p, "payload_aa": payload_aa[p], "n_targets": len(vs),
                         "f_ref_median": round(statistics.median(vs), 4),
                         "f_ref_mean": round(statistics.fmean(vs), 4),
                         "pct_below_0.70": round(100 * sum(1 for v in vs if v < 0.70) / len(vs), 2),
                         "pct_below_0.50": round(100 * sum(1 for v in vs if v < 0.50) / len(vs), 2)})
    worst = sorted(fam_rows, key=lambda r: r["f_ref_mean"])[:10]
    say("  the ten payload families whose targets are most diluted (lowest mean f_ref):")
    for r in worst:
        say(f"        {r['payload_accession']:<8}  {r['payload_aa']:>5} aa  n={r['n_targets']:<5}  "
            f"mean f_ref {r['f_ref_mean']:.3f}  median {r['f_ref_median']:.3f}    "
            f"{r['pct_below_0.70']:.0f}% below 0.70")

    # ---- 4b. separating 46.3's mechanism from the confound that looks like it --------------------
    # A low f_ref has two causes and only one of them is 46.3. Either the SUBJECT is a long
    # multi-domain protein of which the payload matches one piece, the mechanism, or the PAYLOAD
    # is short, so even a whole-payload match covers little of an ordinary-length subject. Both dilute
    # a mean-pooled reference vector, but only the first is the architecture claim, and pooling them
    # would be CLAIM-2-without-its-condition in a new costume. So the cross-tab is reported, not the
    # headline alone.
    bands = [("<=1022", lambda L: L <= 1022), ("1022-2048", lambda L: 1022 < L <= 2048),
             (">2048", lambda L: L > 2048)]
    cross: dict[str, dict[str, int]] = {}
    for bname, pred in bands:
        sub = [r["f_ref"] for r in fref_rows if pred(r["subject_aa"])]
        cross[bname] = stratify(sub, F_STRATA) | {"n": len(sub)}
    say("\n  cross-tab, f_ref by SUBJECT length band (a low f_ref on a short subject is a short "
        "payload, not a multi-domain reference):")
    for bname, row in cross.items():
        n = row["n"]
        if not n:
            say(f"    subject {bname:<10} n=     0  (no rows in this band)")
            continue
        say(f"    subject {bname:<10} n={n:>6,}    "
            + "  ".join(f"{k}  {100 * v / n:5.1f}%" for k, v in row.items() if k != "n"))

    long_sub_rows = [r for r in fref_rows if r["subject_aa"] > 1022]
    mech_rows = [r for r in long_sub_rows if r["f_ref"] < 0.70]
    mech_subjects = {r["homolog_accession"] for r in mech_rows}
    target_subjects = {r["homolog_accession"] for r in fref_rows}
    tgt_1022 = {a for a in target_subjects if a in by_acc and len(by_acc[a].seq) > 1022}
    tgt_2048 = {a for a in target_subjects if a in by_acc and len(by_acc[a].seq) > 2048}
    mech_micro, mech_macro, mech_ngrp = macro_micro(
        [(r["payload_accession"], 1.0 if (r["subject_aa"] > 1022 and r["f_ref"] < 0.70) else 0.0)
         for r in fref_rows], lambda v: v > 0.5)
    say(f"\n  46.3's mechanism proper: subject > 1,022 aa AND f_ref < 0.70: "
        f"{len(mech_rows):,} of {len(fref_rows):,} rows ({100 * len(mech_rows) / len(fref_rows):.2f}%), "
        f"{len(mech_subjects):,} distinct subjects")
    say(f"    micro {100 * mech_micro:.2f}% of rows | macro {100 * mech_macro:.2f}% over "
        f"{mech_ngrp} payload families")
    say(f"  homolog targets by length: {len(target_subjects):,} distinct subjects, of which "
        f"{len(tgt_1022):,} ({100 * len(tgt_1022) / len(target_subjects):.2f}%) exceed 1,022 aa and "
        f"{len(tgt_2048):,} ({100 * len(tgt_2048) / len(target_subjects):.2f}%) exceed 2,048 aa")
    out["mechanism"] = {
        "cross_tab_f_ref_by_subject_band": cross,
        "n_rows_long_subject": len(long_sub_rows),
        "n_rows_long_subject_and_f_ref_below_070": len(mech_rows),
        "pct_rows": round(100 * len(mech_rows) / len(fref_rows), 3),
        "micro_pct": round(100 * mech_micro, 3), "macro_pct": round(100 * mech_macro, 3),
        "n_distinct_subjects": len(mech_subjects),
        "n_target_subjects": len(target_subjects),
        "n_target_subjects_above_1022": len(tgt_1022),
        "n_target_subjects_above_2048": len(tgt_2048),
        "pct_target_subjects_above_1022": round(100 * len(tgt_1022) / len(target_subjects), 3),
        "pct_target_subjects_above_2048": round(100 * len(tgt_2048) / len(target_subjects), 3)}

    # the two numbers section 4 of the brief asks for, side by side
    cat_1022 = cen.fraction_entries(1022) if 1022 in cen.above else None
    cat_2048 = cen.fraction_entries(2048) if 2048 in cen.above else None
    m70, M70, _ = macro_micro(per_row, lambda v: v < 0.70)
    say("\nthe benchmark question, both denominators, "
        f"catalogue {args.reference}:")
    if cat_1022 is not None:
        say(f"  of the whole catalogue: {100 * cat_1022:.2f}% of entries exceed 1,022 aa "
            f"({100 * cen.fraction_residues(1022):.2f}% of residues)")
    if cat_2048 is not None:
        say(f"  of the whole catalogue: {100 * cat_2048:.2f}% of entries exceed 2,048 aa "
            f"({100 * cen.fraction_residues(2048):.2f}% of residues)")
    say(f"  of the homolog targets retrieval is scored against: {100 * m70:.2f}% of rows "
        f"(micro) and {100 * M70:.2f}% of payload families (macro) sit at f_ref < 0.70")
    say(f"  restricted to 46.3's mechanism (subject > 1,022 aa and f_ref < 0.70): "
        f"{100 * mech_micro:.2f}% of rows (micro), {100 * mech_macro:.2f}% of payload families "
        f"(macro), {len(mech_subjects):,} distinct catalogue entries "
        f"({100 * len(mech_subjects) / cen.n_entries:.2f}% of the catalogue)")

    out["f_ref"] = {
        "n_rows": len(vals), "n_dropped_unknown_payload_length": missing_len,
        "n_lower_bound_rows": bound_rows,
        "pct_lower_bound": round(100 * bound_rows / len(vals), 3),
        "strata": strata,
        "pct_strata": {k: round(100 * n / len(vals), 3) for k, n in strata.items()},
        "deciles": {f"p{int(100 * q)}":
                    round(sorted(vals)[min(int(q * (len(vals) - 1)), len(vals) - 1)], 4)
                    for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)},
        "below": {f"{t:.2f}": {"micro_pct": round(100 * macro_micro(per_row, lambda v, t=t: v < t)[0], 3),
                               "macro_pct": round(100 * macro_micro(per_row, lambda v, t=t: v < t)[1], 3)}
                  for t in (0.95, 0.70, 0.50, 0.30)},
        "worst_families": worst}
    out["header_family_counts"] = counter_summary(Counter({k: v for k, v in
                                                           header_fams.most_common(10)}))

    write_csv(args.out_dir / "catalogue_entries.csv", ent_rows,
              ["accession", "length_aa", "n_families", "family_length_ratio", "repeat_fraction",
               "is_payload", "is_homolog_target"])
    write_csv(args.out_dir / "f_ref_by_payload.csv", fam_rows,
              ["payload_accession", "payload_aa", "n_targets", "f_ref_median", "f_ref_mean",
               "pct_below_0.70", "pct_below_0.50"])
    write_csv(args.out_dir / "payload_dilution.csv",
              [{"payload_accession": a, "payload_aa": L,
                "exceeds_1022": int(L > 1022), "exceeds_2048": int(L > 2048),
                "in_catalogue": int(a in by_acc),
                "n_targets": len(by_payload.get(a, [])),
                "family_median_aa": fam_median.get(a, ""),
                "family_length_ratio": (round(L / fam_median[a], 4)
                                        if a in fam_median and fam_median[a] else "")}
               for a, L in sorted(payload_aa.items(), key=lambda kv: -kv[1])],
              ["payload_accession", "payload_aa", "exceeds_1022", "exceeds_2048", "in_catalogue",
               "n_targets", "family_median_aa", "family_length_ratio"])
    with open(args.out_dir / "catalogue_dilution.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    # f_ref per row is 51k rows; written last and separately so a reader can skip it
    write_csv(args.out_dir / "f_ref_rows.csv", fref_rows,
              ["payload_accession", "homolog_accession", "subject_aa", "payload_aa", "identity",
               "coverage", "f_ref", "f_ref_is_lower_bound"])

    say(f"\nwrote {args.out_dir}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
