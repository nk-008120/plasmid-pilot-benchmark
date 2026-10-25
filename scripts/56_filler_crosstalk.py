#!/usr/bin/env python3
"""Which mechanism makes random filler fatal to mean-pooled ESM-2? Cross-talk, or pooled displacement?

    python scripts/56_filler_crosstalk.py --embedder facebook/esm2_t12_35M_UR50D   # harness check
    python scripts/56_filler_crosstalk.py --embedder facebook/esm2_t33_650M_UR50D  # the measurement

this project measured the filler as an independent variable and found a **sign reversal on exactly one
scorer**. Same 24 payloads, same dilution ladder, same 40,750-protein catalogue, same holdout: swapping
SwissProt filler for random sequence makes exact Smith-Waterman *better* (`whole` becomes bit-identical
to `oracle` on all 336 constructs), makes a mean-pooled 3-mer arm *better*, and takes mean-pooled ESM-2
from `recall@10` 4.2% to **0.0% at a median rank of 19,170 against a chance median of 20,375**. The one
thing that separates ESM-2 from the other two is that it is **contextual**.

this project named two mechanisms that both predict that sign, and nothing in 60.2 distinguishes them:

* **Contextual cross-talk.** Random sequence is out of distribution for a protein language model, so
  flanking it corrupts the payload's *own* per-residue embeddings, before any pooling happens.
* **Pooled-vector displacement.** The payload's per-residue embeddings are essentially unchanged and only
  the pooled mean moves, into a region of embedding space that no entry in a catalogue of 40,750 real
  proteins occupies.

This script measures both, on one scale, from one forward pass per sequence.

**The measurement.** For each payload, ESM-2 per-residue embeddings three ways: (a) the payload span
queried alone: the `oracle` arm's own query; (b) the same payload residues read out of the
random-filler ORF; (c) the same residues read out of the SwissProt-filler ORF. Then, per construct:

1. `cos_res_*`: the per-residue cosine between (a) and (b), and between (a) and (c), summarised over
   the payload's residues. Cross-talk predicts the random-filler cosine is materially the lower of the
   two. Displacement predicts both are ~1.0.
2. `pooled_cos_payload_rows`: cosine between the mean of the payload's **raw** rows taken from inside
   the ORF and the mean of the oracle's own raw rows. This is the cross-talk effect expressed as a pooled
   cosine, so it is directly comparable with:
3. `pooled_cos_whole_res`: the same thing with the mean taken over the *whole* ORF: the total movement
   of the pooled vector, carrying both effects together.

`(1 - #2) / (1 - #3)` is then the **share of the whole-ORF cosine deficit that survives restricting the
mean to the payload's own residues**. It is a share of a deficit and not an orthogonal decomposition,
the two effects are not independent and nothing here claims they are, but it is on one scale, it is
bounded, and it separates the two branches of 60.3 in the direction each predicts.

Both sides of that ratio are **residue-only** means, because both sides of a ratio must be computed the
same way. `pooled_cos_whole_arm` is emitted beside it as the retrieval arm's own
quantity, the mask-mean that includes `<cls>` and `<eos>`, exactly what `pilot.embed.Esm2Embedder`
ranks on, so the link to 60.2's table is explicit and the size of the special-token difference is a
printed number rather than an assurance.

**The scale that makes "materially below 1.0" mean something.** A cosine of 0.95 is not interpretable
without knowing where unrelated sits. For this exact embedder this project measured the background
residue-pair cosine at **0.4131** (`pilot.cosine_floor.background_pair_stats`, 40,750-protein catalogue,
clustered SE 0.00325) and this project's working rules says not to re-measure it. So every cosine here is also reported as
`corruption = (1 - cos) / (1 - 0.4131)`: the fraction of the available range lost. That is this
repository's own rule about computing both sides of a ratio the same way, applied to the
one number that would otherwise be read against an imagined null.

**The verdict rule is pre-registered**, in `VERDICT_*` below and in `verdict()`, and pinned by
`tests/test_filler_crosstalk.py` before any model ran. It is stated so the label cannot be chosen after
seeing the numbers; the numbers are printed beside it either way, and a reader who disagrees with the
constants can re-derive the label from the CSV.

## Why these constructs, and the three confounds the choice removes

`--archetype motifless_fusion`, `--levels 0.8 0.6`, one construct per payload per level per filler: 24
payloads x 2 dilution levels x 2 fillers, plus 24 payload-alone queries. The most diluted level available
below f 0.50 is `dilution_target` 0.8 (f = 0.200); 0.6 (f = 0.400) is the one extra level, for a
dose-response. At f < 0.50 the tier offers three archetypes and `motifless_fusion` is the only one that
removes all three of the following:

1. **Position.** In `motifless_fusion` the payload occupies ORF positions 1..`payload_aa`: the *same
   absolute offsets* it has when queried alone. ESM-2 is position-sensitive, so in `tagged_fusion`, where
   the payload sits at offset 2,798 of a 3,525 aa ORF and at offset 0 when queried alone, a per-residue
   cosine drop would confound position with context and could not answer 60.3's question at all.
2. **Truncation.** `--max-length 2048` is in the embedding cache key and must not move.
   With the payload at the N-terminus it can never be truncated. In `tagged_fusion` at this level the
   payload is **fully** past 2,048 for 2 of 24 constructs and partly past it for 7 more, so 9 of 24 would
   be unmeasurable. The binding count of the limit is reported either way.
3. **Non-filler content.** `motifless_fusion` is `payload | fusion partner`, and the fusion partner *is*
   the filler. `tagged_fusion` adds 3xFLAG, a carrier, a TEV site, His6 and an NLS; `multi_gene_cassette`
   adds 2A peptides and passengers. Those are identical across the two tiers, so they do not bias the
   paired comparison, but they put sequence next to the payload that is neither payload nor the variable
   under test.

**And the falsifying case this choice cannot show** (this project's working rules, "name the falsifying case before trusting
a subset"): flanking here is **one-sided**, C-terminal only. ESM-2's attention is bidirectional, so
downstream context does reach the payload and a cross-talk effect can appear; but this is the *milder*
geometry. A material cross-talk result here is therefore strong, and a null here is weaker than a null
would be on a two-sided geometry. The two-sided test is `tagged_fusion` at `--levels 0.4` (f = 0.600,
where nothing truncates), and it needs a position control before its cosines mean anything. It is not run
here and no claim in the working record covers it.

## What the script refuses to do

`--tier` labels are free text and four of them name two tiers, so this script takes no
tier label at all. It identifies each `orfs.faa` by **md5** and exits 2 if either file is not the one
whose number it expects, and it verifies field by field that the two manifests carry the same
`construct_id`, `payload_accession`, `payload_aa` and `orf_aa` before comparing anything. `f` is
recomputed from its definition `payload_aa / orf_aa` and never read from the `payload_fraction` column,
which is a different quantity on this manifest.

Writes `crosstalk_<embedder>.csv` (one row per construct x filler), `crosstalk_<embedder>_residues.csv`
(the per-residue cosines, so the distribution is auditable rather than summarised away) and
`crosstalk_<embedder>_summary.json`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.arms import pick_payload_orf  # noqa: E402
from pilot.positional import get_positional_embedder  # noqa: E402
from pilot.reference import iter_fasta  # noqa: E402

#: The ESM-2 positional embedder's background residue-pair cosine, measured in the working record on the
#: 40,750-protein catalogue (clustered SE 0.00325). this project's working rules: do not re-measure it. It is the denominator
#: that turns a cosine into a fraction of the available range, and it is a property of the embedder, so
#: it is only the right denominator for an ESM-2 run, which `--embedder` is checked against.
ESM2_BACKGROUND_COSINE = 0.4131

#: md5 of the two `orfs.faa` files this measurement is defined on. The tier is established from these and
#: never from a directory name or a `--tier` label.
EXPECT_MD5 = {
    "random": "6f7acc26dcc85209f2a5d15e4c797ebe",
    "real": "62173fd47546186c7167b455015a6dc2",
}

# ---- the pre-registered verdict rule ----------------------------------------------------------------
#: The whole-ORF pooled cosine deficit is "material", there is something to explain at all, once it
#: costs this fraction of the 1.0 -> background range. It gates on the quantity being EXPLAINED, not on
#: one of the two candidate explanations; see `verdict` for the draft of this rule that got that wrong
#: and the fixture that refuted it.
VERDICT_MATERIAL = 0.10
#: Share of constructs that must agree in sign for the random-vs-real difference to be called directional.
VERDICT_PAIRED_MAJORITY = 0.80
#: Share of the whole-ORF cosine deficit that must survive restricting the mean to the payload's own
#: residues before cross-talk is called the dominant mechanism, and below which displacement is.
VERDICT_CROSSTALK_SHARE = 0.60
VERDICT_DISPLACEMENT_SHARE = 0.15

#: Residues this close to the payload/filler junction are reported separately from the rest. ESM-2's
#: attention is global, so cross-talk confined to the junction and cross-talk spread over the whole
#: payload are different findings with different consequences for decomposition: a junction-only effect
#: is a boundary artefact a segmenter trims away, and a payload-wide one is not. Nothing in the verdict
#: rule depends on this split, it is a description of where the effect sits.
JUNCTION_AA = 10


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_faa(path: Path) -> dict[str, dict[str, str]]:
    """plasmid id -> {orf id: protein}. A fourth copy of `10_twilight.py`'s five-line reader, for the
    reason that one gives: what must not drift is the ranking rule, not the FASTA parse."""
    out: dict[str, dict[str, str]] = {}
    for e in iter_fasta(path):
        oid = e.header.split()[0]
        out.setdefault(oid.split("|")[0], {})[oid] = e.seq
    return out


def load_manifest(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return {r["construct_id"]: r for r in csv.DictReader(fh)}


def payload_fraction(row: dict) -> float:
    """`f` from its definition, never from the `payload_fraction` column."""
    return int(row["payload_aa"]) / int(row["orf_aa"])


def check_manifests_match(man_a: dict[str, dict], man_b: dict[str, dict]) -> list[str]:
    """The shared-tier invariant, field by field. Returns the disagreements, empty when it holds.

    60.1's whole claim is that these two tiers are the same constructs with different filler. Every paired
    comparison below is void if that is false, so it is checked here rather than cited.
    """
    bad = []
    if set(man_a) != set(man_b):
        bad.append(f"construct id sets differ: {len(set(man_a) ^ set(man_b))} ids not in both")
    for cid in sorted(set(man_a) & set(man_b)):
        a, b = man_a[cid], man_b[cid]
        for field in ("payload_accession", "payload_aa", "orf_aa", "payload_start_aa", "payload_end_aa"):
            if a[field] != b[field]:
                bad.append(f"{cid}: {field}  {a[field]!r} vs {b[field]!r}")
    return bad


def select_constructs(man: dict[str, dict], archetype: str, levels: list[str],
                      max_payloads: int = 0) -> list[dict]:
    """One construct per (payload, dilution level) for `archetype`, with every exclusion countable.

    Payloads are taken in sorted accession order and `--max-payloads` truncates that list, so a reduced
    run is a prefix of the full one and not a different sample. A payload missing a level is dropped from
    *both* levels, so the dose-response stays paired rather than quietly changing n between rows.
    """
    by_payload: dict[str, dict[str, dict]] = {}
    for row in man.values():
        if row["archetype_manual"] != archetype:
            continue
        lvl = f"{float(row['dilution_target']):.1f}"
        if lvl not in levels:
            continue
        by_payload.setdefault(row["payload_accession"], {})[lvl] = row
    complete = sorted(acc for acc, d in by_payload.items() if set(d) == set(levels))
    if max_payloads:
        complete = complete[:max_payloads]
    out = []
    for acc in complete:
        for lvl in levels:
            r = by_payload[acc][lvl]
            out.append(dict(construct_id=r["construct_id"], payload_accession=acc, level=lvl,
                            payload_aa=int(r["payload_aa"]), orf_aa=int(r["orf_aa"]),
                            f=payload_fraction(r),
                            manifest_payload_start=int(r["payload_start_aa"]),
                            manifest_payload_end=int(r["payload_end_aa"])))
    return out


def residue_cosines(rows_a: np.ndarray, rows_b: np.ndarray) -> np.ndarray:
    """Per-residue cosine between two aligned stacks of L2-normalised rows.

    Both stacks come out of `embed_residues_pooled`, which normalises each row, so the cosine is a plain
    row-wise dot product. The two stacks are the SAME residues in two contexts, in the same order, so
    position `i` against position `i` is the comparison and no alignment is involved, which is the whole
    reason the archetype was chosen for identical absolute offsets.
    """
    if rows_a.shape != rows_b.shape:
        raise ValueError(f"row stacks differ in shape: {rows_a.shape} vs {rows_b.shape}")
    return np.clip(np.einsum("ij,ij->i", rows_a.astype(np.float64), rows_b.astype(np.float64)), -1.0, 1.0)


def pooled_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine between two pooled vectors, normalising defensively rather than assuming it."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.clip(np.dot(a.astype(np.float64), b.astype(np.float64)) / (na * nb), -1.0, 1.0))


def corruption(cos: float, background: float = ESM2_BACKGROUND_COSINE) -> float:
    """Fraction of the 1.0 -> background range a cosine has lost. 0.0 is identical, 1.0 is unrelated."""
    return (1.0 - cos) / (1.0 - background)


def l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _stats(x: np.ndarray) -> dict:
    return dict(
        mean=round(float(np.mean(x)), 6), median=round(float(np.median(x)), 6),
        p10=round(float(np.percentile(x, 10)), 6), min=round(float(np.min(x)), 6),
        sd=round(float(np.std(x, ddof=1)) if x.size > 1 else 0.0, 6),
    )


def summarise(rows: list[dict], levels: list[str]) -> list[dict]:
    """One row per (level, filler): the medians over constructs, each with its corruption.

    Medians over constructs rather than means, and the paired per-construct comparison reported beside
    them, because 24 constructs is a distribution and not a point: the same reason the working record refuses
    a bare mean for an embedding arm.
    """
    out = []
    for lvl in levels:
        for filler in ("random", "real"):
            rs = [r for r in rows if r["level"] == lvl and r["filler"] == filler and r["measurable"]]
            if not rs:
                continue
            def med(key, _rs=rs):
                return float(np.median([r[key] for r in _rs]))
            cres = med("cos_res_mean")
            cpay, cwhole = med("pooled_cos_payload_rows"), med("pooled_cos_whole_res")
            out.append(dict(
                level=lvl, filler=filler, n_constructs=len(rs),
                n_payloads=len({r["payload_accession"] for r in rs}),
                median_f=round(med("f"), 4),
                cos_res_mean=round(cres, 6), cos_res_corruption=round(corruption(cres), 4),
                cos_res_min=round(med("cos_res_min"), 6),
                cos_res_mean_junction=round(med("cos_res_mean_junction"), 6),
                cos_res_mean_core=(round(float(np.median([r["cos_res_mean_core"] for r in rs
                                                          if r["cos_res_mean_core"] != ""])), 6)
                                   if any(r["cos_res_mean_core"] != "" for r in rs) else ""),
                pooled_cos_payload_rows=round(cpay, 6),
                pooled_payload_rows_corruption=round(corruption(cpay), 4),
                pooled_cos_whole_res=round(cwhole, 6),
                pooled_cos_whole_arm=round(med("pooled_cos_whole_arm"), 6),
                pooled_whole_corruption=round(corruption(cwhole), 4),
                crosstalk_share_of_deficit=(
                    "" if abs(1.0 - cwhole) < 1e-9
                    else round((1.0 - cpay) / (1.0 - cwhole), 4)),
            ))
    return out


def paired(rows: list[dict], level: str, key: str) -> dict:
    """Random against real, construct by construct, at one level. A sign count needs no threshold."""
    by = {}
    for r in rows:
        if r["level"] == level and r["measurable"]:
            by.setdefault(r["payload_accession"], {})[r["filler"]] = r[key]
    both = {k: v for k, v in by.items() if {"random", "real"} <= set(v)}
    lower = sum(1 for v in both.values() if v["random"] < v["real"])
    higher = sum(1 for v in both.values() if v["random"] > v["real"])
    deltas = [v["random"] - v["real"] for v in both.values()]
    return dict(n=len(both), random_lower=lower, random_higher=higher,
                equal=len(both) - lower - higher,
                median_delta=round(float(np.median(deltas)), 6) if deltas else 0.0)


def verdict(summary: list[dict], paired_res: dict, level: str) -> dict:
    """The pre-registered call, at one level. Returns the label and every quantity it was made from.

    The rule, fixed before any model ran and pinned by `tests/test_filler_crosstalk.py`. It is **two
    questions in order**, which is what keeps a magnitude and a mechanism from being summed into one
    number: the error the working record and this project's "name the falsifying case" section both describe:

    1. *Did anything move?* `corruption(pooled_cos_whole_res)` under random filler must reach
       `VERDICT_MATERIAL`. If it does not, the label is **`neither`**: the pooled vector did not leave the
       payload's neighbourhood by a material fraction of the available range, so 60.2's collapse is not
       visible in cosine space at all and neither branch of 60.3 is supported.
    2. *What is it made of?* `crosstalk_share_of_deficit`, the share of that movement which survives
       restricting the mean to the payload's **own** residues, decides between the branches:
       at or above `VERDICT_CROSSTALK_SHARE` the payload's own embeddings carry most of it
       (**`crosstalk`**, 60.3's first branch); below `VERDICT_DISPLACEMENT_SHARE` they carry almost none
       of it (**`displacement`**, the second branch); in between, **`both`**: both effects are real and
       neither dominates, which this project would rather report than round to a story.

    **`attributable_to_random_filler` is reported beside the label and is deliberately not part of it.**
    Cross-talk that is equally material under *both* fillers is a real effect on per-residue embeddings
    and still does not explain a *difference between the fillers*, which is the thing 60.2 measured. The
    mechanism question and the attribution question have different answers and are reported separately;
    folding the second into the label would be testing a quantity against a condition derived for a
    different one, which is this repository's most expensive recurring mistake.

    An earlier draft of this rule gated step 1 on the **per-residue** corruption instead of the pooled
    one. It was refuted by its own test fixture before any model ran: a case whose pooled vector moved a
    material 0.12 of the range, 71% of it surviving a payload-only mean, cross-talk by any reading,
    came out `neither`, because the per-residue cosine of 0.960 sat below the magnitude gate. The gate
    belongs on the quantity being explained, not on one of the candidate explanations.
    """
    rnd = next((s for s in summary if s["level"] == level and s["filler"] == "random"), None)
    if rnd is None:
        return dict(level=level, label="unmeasured", reason="no random-filler rows at this level")
    share = rnd["crosstalk_share_of_deficit"]
    share = None if share == "" else float(share)
    total_corr = corruption(rnd["pooled_cos_whole_res"])
    directional = (paired_res["n"] > 0
                   and paired_res["random_lower"] / paired_res["n"] >= VERDICT_PAIRED_MAJORITY)
    material = total_corr >= VERDICT_MATERIAL
    if not material:
        label = "neither"
    elif share is None:
        label = "unmeasured"
    elif share >= VERDICT_CROSSTALK_SHARE:
        label = "crosstalk"
    elif share < VERDICT_DISPLACEMENT_SHARE:
        label = "displacement"
    else:
        label = "both"
    return dict(
        level=level, label=label,
        pooled_corruption_random=round(total_corr, 6),
        pooled_movement_material=bool(material),
        per_residue_corruption_random=rnd["cos_res_corruption"],
        crosstalk_share_of_deficit=share,
        attributable_to_random_filler=bool(directional),
        paired=paired_res,
        rule=dict(material=VERDICT_MATERIAL, paired_majority=VERDICT_PAIRED_MAJORITY,
                  crosstalk_share=VERDICT_CROSSTALK_SHARE,
                  displacement_share=VERDICT_DISPLACEMENT_SHARE,
                  background_cosine=ESM2_BACKGROUND_COSINE,
                  material_applies_to="corruption(pooled_cos_whole_res) under random filler"),
    )


def make_embedder(spec: str, batch_size: int, max_length: int):
    """Indirection so the tests can drive `main` with a deterministic stand-in for a 2.6 GB download."""
    return get_positional_embedder(spec, batch_size=batch_size, max_length=max_length)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-random", type=Path,
                    default=Path("data/synthetic/synthetic_manifest.csv"))
    ap.add_argument("--manifest-real", type=Path,
                    default=Path("data/synthetic_real/synthetic_manifest.csv"))
    ap.add_argument("--orfs-random", type=Path, default=Path("results/persegment_synthetic/orfs.faa"))
    ap.add_argument("--orfs-real", type=Path, default=Path("results/persegment_synthreal/orfs.faa"))
    ap.add_argument("--reference", type=Path,
                    default=Path("results/gate_all173/reference_twilight.fasta"),
                    help="the catalogue the payload sequences are read from, as the oracle arm does")
    ap.add_argument("--out-dir", type=Path, default=Path("results/filler_crosstalk"))
    ap.add_argument("--embedder", default="facebook/esm2_t33_650M_UR50D",
                    help="bare checkpoint name; 650M is what this project's runs used")
    ap.add_argument("--archetype", default="motifless_fusion",
                    help="see the module docstring for why this one and what it cannot show")
    ap.add_argument("--levels", nargs="+", default=["0.8", "0.6"],
                    help="dilution_target values, most diluted first")
    ap.add_argument("--max-payloads", type=int, default=0, help="0 means every payload")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="in the embedding cache key; do not change it")
    ap.add_argument("--max-length", type=int, default=2048,
                    help="in the embedding cache key; do not change it")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU threads; 0 leaves torch's own default. It changes the wall clock "
                         "and nothing else: it is not in the cache key and no cosine depends on it")
    ap.add_argument("--allow-md5-mismatch", action="store_true",
                    help="run anyway when an orfs.faa is not the file this measurement is defined on. "
                         "The md5 is the only thing that establishes which tier a file is ")
    args = ap.parse_args(argv)

    paths = {"random": (args.manifest_random, args.orfs_random),
             "real": (args.manifest_real, args.orfs_real)}
    for mp, op in paths.values():
        for p in (mp, op):
            if not p.is_file():
                print(f"not found: {p}", file=sys.stderr)
                return 2
    if not args.reference.is_file():
        print(f"not found: {args.reference}", file=sys.stderr)
        return 2

    # ---- the tier is established by md5, never by a label or a directory name ----
    seen_md5 = {k: md5(op) for k, (_, op) in paths.items()}
    for k, got in seen_md5.items():
        want = EXPECT_MD5[k]
        mark = "as expected" if got == want else f"EXPECTED {want}"
        print(f"orfs.faa [{k:6s}] md5 {got}    {mark}")
    bad_md5 = [k for k, v in seen_md5.items() if v != EXPECT_MD5[k]]
    if bad_md5 and not args.allow_md5_mismatch:
        print(f"refusing to run: {', '.join(bad_md5)} orfs.faa is not the file this measurement is "
              f"defined on. Pass --allow-md5-mismatch to override and say so in the write-up.",
              file=sys.stderr)
        return 2

    man = {k: load_manifest(mp) for k, (mp, _) in paths.items()}
    bad = check_manifests_match(man["random"], man["real"])
    if bad:
        print("the two manifests are NOT the same constructs; every paired comparison here would be "
              "void:", file=sys.stderr)
        for b in bad[:10]:
            print(f"    {b}", file=sys.stderr)
        return 2
    print(f"manifests agree field by field on {len(man['random'])} constructs "
          f"(construct_id, payload_accession, payload_aa, orf_aa, payload span)")

    levels = [f"{float(x):.1f}" for x in args.levels]
    chosen = select_constructs(man["random"], args.archetype, levels, args.max_payloads)
    if not chosen:
        print(f"no construct matches archetype {args.archetype!r} at levels {levels}", file=sys.stderr)
        return 2
    n_pay = len({c["payload_accession"] for c in chosen})
    print(f"selected {len(chosen)} constructs: {n_pay} payloads x {len(levels)} levels "
          f"({args.archetype}), f = "
          + ", ".join(f"{lvl}:{np.median([c['f'] for c in chosen if c['level'] == lvl]):.4f}"
                      for lvl in levels))

    orfs = {k: read_faa(op) for k, (_, op) in paths.items()}
    ref_by_acc = {e.accession: e.seq for e in iter_fasta(args.reference)}
    missing = sorted({c["payload_accession"] for c in chosen} - set(ref_by_acc))
    if missing:
        print(f"payloads absent from the catalogue, so they have no oracle query: {missing}",
              file=sys.stderr)
        return 2

    if args.threads:
        try:
            import torch  # noqa: PLC0415 (optional, and only for the wall clock)
            torch.set_num_threads(args.threads)
            print(f"torch CPU threads set to {torch.get_num_threads()}")
        except ImportError:
            print("torch is not installed; --threads ignored")

    emb = make_embedder(args.embedder, args.batch_size, args.max_length)
    is_esm2 = "esm2" in args.embedder.lower()
    if not is_esm2:
        print(f"WARNING: --embedder {args.embedder!r} is not an ESM-2 checkpoint, so "
              f"{ESM2_BACKGROUND_COSINE} is NOT its background cosine and every corruption figure and "
              f"the verdict label are uninterpretable. the working record / 39.6: the background is a property "
              f"of the embedder.", flush=True)
    print(f"embedder {getattr(emb, 'name', args.embedder)}  batch {args.batch_size}    "
          f"max_length {args.max_length}", flush=True)
    print(f"background residue-pair cosine {ESM2_BACKGROUND_COSINE} (not re-measured); "
          f"every cosine is also given as corruption = (1 - cos) / (1 - background)", flush=True)

    # ---- the three contexts, one forward pass each ---------------------------------------------
    rows: list[dict] = []
    residue_rows: list[dict] = []
    span_mismatch, truncation_hit, orf_missing = [], [], []
    t0 = time.time()
    total_res = sum(c["payload_aa"] for c in chosen) // len(levels) + sum(
        min(c["orf_aa"], args.max_length) * 2 for c in chosen)
    print(f"about {total_res:,} residues to embed across "
          f"{n_pay + 2 * len(chosen)} sequences", flush=True)

    # The oracle query, once per payload: the payload span as the oracle arm queries it. Taken from the
    # random-filler ORF and then checked against the real-filler ORF, so "the payload span is the same
    # sequence in both tiers" is a verified fact here and not an inherited assumption.
    done = 0
    for acc in sorted({c["payload_accession"] for c in chosen}):
        per_level: dict[str, dict] = {}
        for lvl in levels:
            c = next(c for c in chosen if c["payload_accession"] == acc and c["level"] == lvl)
            spans, seqs, ctx = {}, {}, {}
            skip = None
            for filler in ("random", "real"):
                group = orfs[filler].get(c["construct_id"], {})
                if not group:
                    orf_missing.append((c["construct_id"], filler))
                    skip = f"no ORF group in the {filler} orfs.faa"
                    break
                oid, span = pick_payload_orf(group, ref_by_acc[acc])
                if span is None:
                    orf_missing.append((c["construct_id"], filler))
                    skip = f"no ORF in the {filler} tier carries the payload"
                    break
                spans[filler], seqs[filler] = span, group[oid]
                # The manifest's own span, as a cross-check and not as the source. this project's working rules: recompute a
                # quantity from its definition and cross-check it against the column, never the reverse.
                if (span.start, span.end) != (c["manifest_payload_start"], c["manifest_payload_end"]):
                    span_mismatch.append(
                        dict(construct_id=c["construct_id"], filler=filler,
                             aligned=f"{span.start}-{span.end}",
                             manifest=f"{c['manifest_payload_start']}-{c['manifest_payload_end']}"))
            if skip:
                for filler in ("random", "real"):
                    rows.append(dict(**_blank_row(c, filler), measurable=False, skip_reason=skip))
                continue

            # One forward pass over: the oracle query, then both ORFs. The oracle query is the payload
            # span from the random tier; it is asserted equal to the real tier's before it is reused.
            if spans["random"].seq != spans["real"].seq:
                for filler in ("random", "real"):
                    rows.append(dict(**_blank_row(c, filler), measurable=False,
                                     skip_reason="the payload span differs between the two tiers"))
                continue
            oracle_seq = spans["random"].seq
            batch = [oracle_seq, seqs["random"], seqs["real"]]
            l2_rows, raw_rows, pooled = emb.embed_residues_pooled(batch)
            done += 1
            or_l2, or_pool = l2_rows[0], pooled[0]
            # The oracle's residue-only mean. Both sides of the share-of-deficit ratio are computed this
            # way; `or_pool` is kept for the arm's own quantity (compute both sides of a
            # ratio the same way, and say which quantity each number is).
            or_res_mean = l2(raw_rows[0].mean(axis=0))
            per_level[lvl] = dict(oracle_len=len(oracle_seq))

            for j, filler in enumerate(("random", "real"), start=1):
                span = spans[filler]
                lo, hi = span.start - 1, span.end  # OracleSpan is 1-based inclusive
                have = l2_rows[j].shape[0]
                if hi > have:
                    # The payload runs past what `--max-length` kept. Reported, never silently clipped:
                    # this is the limit this project's table says to count rather than assume.
                    truncation_hit.append(dict(construct_id=c["construct_id"], filler=filler,
                                               payload_end=span.end, residues_embedded=have))
                    rows.append(dict(**_blank_row(c, filler), measurable=False,
                                     skip_reason=f"payload ends at {span.end} but only {have} residues "
                                                 f"survived --max-length {args.max_length}"))
                    continue
                pay_l2 = l2_rows[j][lo:hi]
                pay_raw = raw_rows[j][lo:hi]
                if pay_l2.shape[0] != or_l2.shape[0]:
                    # The two stacks must be the same residues in the same order or the comparison is
                    # not the one this script claims to make. Reported as unmeasurable, never slice-fitted.
                    rows.append(dict(**_blank_row(c, filler), measurable=False,
                                     skip_reason=f"payload rows in the ORF ({pay_l2.shape[0]}) do not "
                                                 f"match the oracle's ({or_l2.shape[0]})"))
                    continue
                cos = residue_cosines(or_l2, pay_l2)
                st = _stats(cos)
                row = dict(
                    **_blank_row(c, filler), measurable=True, skip_reason="",
                    n_payload_res=int(cos.size),
                    orf_res_embedded=int(have),
                    orf_truncated=int(c["orf_aa"] > args.max_length),
                    cos_res_mean=st["mean"], cos_res_median=st["median"], cos_res_p10=st["p10"],
                    cos_res_min=st["min"], cos_res_sd=st["sd"],
                    cos_res_corruption=round(corruption(st["mean"]), 6),
                    # Where the effect sits, not how big it is. See JUNCTION_AA.
                    cos_res_mean_junction=round(float(np.mean(cos[-JUNCTION_AA:])), 6),
                    cos_res_mean_core=(round(float(np.mean(cos[:-JUNCTION_AA])), 6)
                                       if cos.size > JUNCTION_AA else ""),
                    pooled_cos_payload_rows=round(
                        pooled_cosine(l2(pay_raw.mean(axis=0)), or_res_mean), 6),
                    pooled_cos_whole_res=round(
                        pooled_cosine(l2(raw_rows[j].mean(axis=0)), or_res_mean), 6),
                    pooled_cos_whole_arm=round(pooled_cosine(pooled[j], or_pool), 6),
                )
                row["pooled_payload_rows_corruption"] = round(corruption(row["pooled_cos_payload_rows"]), 6)
                row["pooled_whole_corruption"] = round(corruption(row["pooled_cos_whole_res"]), 6)
                row["special_token_effect"] = round(
                    abs(row["pooled_cos_whole_arm"] - row["pooled_cos_whole_res"]), 6)
                rows.append(row)
                for i, v in enumerate(cos, start=1):
                    residue_rows.append(dict(construct_id=c["construct_id"], payload_accession=acc,
                                             level=lvl, filler=filler, residue_index=i,
                                             residue=oracle_seq[i - 1], cosine=round(float(v), 6)))
        if per_level:
            print(f"    {acc}    {done}/{len(chosen)} constructs, {time.time() - t0:.0f}s", flush=True)

    # ---- every limit's binding count --------------------------------------------
    print()
    print(f"truncation at --max-length {args.max_length}: "
          f"{sum(1 for r in rows if r.get('orf_truncated'))} of {len(rows)} ORF embeddings truncated; "
          f"{len(truncation_hit)} lost payload residues to it")
    print(f"aligned span against the manifest's own columns: {len(span_mismatch)} disagreements "
          f"of {2 * len(chosen)} checks")
    print(f"constructs with no payload-bearing ORF: {len(orf_missing)}")
    n_meas = sum(1 for r in rows if r["measurable"])
    print(f"measurable rows: {n_meas} of {len(rows)}")
    if getattr(emb, "n_truncated", 0):
        print(f"embedder reported {emb.n_truncated} sequences over max_length")

    # The rows are written whatever happens, including when none is measurable: the skip reasons ARE the
    # result in that case, and a run that produced no number should still leave the evidence of why.
    summary = summarise(rows, levels)
    cols = ["level", "filler", "n_constructs", "median_f", "cos_res_mean", "cos_res_corruption",
            "cos_res_min", "cos_res_mean_junction", "cos_res_mean_core", "pooled_cos_payload_rows",
            "pooled_cos_whole_res", "pooled_cos_whole_arm", "crosstalk_share_of_deficit"]
    print()
    print("medians over constructs (cos_res_* is the payload's OWN residues, alone vs in-ORF):")
    print("  ".join(f"{c:>26}" for c in cols))
    for s in summary:
        print("  ".join(f"{str(s.get(c, '')):>26}" for c in cols))
    ste = [r["special_token_effect"] for r in rows if r["measurable"]]
    if ste:
        print(f"\n<cls>/<eos> in the pooled mean move the whole-ORF cosine by a median of "
              f"{np.median(ste):.6f} (max {max(ste):.6f}): the arm's quantity and the residue-only one "
              f"agree to that much, which is why the ratio uses the residue-only pair")

    verdicts = []
    print()
    for lvl in levels:
        pr = paired(rows, lvl, "cos_res_mean")
        v = verdict(summary, pr, lvl)
        verdicts.append(v)
        if v["label"] == "unmeasured":
            print(f"level {lvl} -> **unmeasured**: {v['reason']}")
            continue
        print(f"level {lvl} (f median "
              f"{np.median([c['f'] for c in chosen if c['level'] == lvl]):.3f}) -> "
              f"**{v['label']}**")
        print(f"  pooled movement under random filler: corruption "
              f"{v['pooled_corruption_random']} (material at >= {VERDICT_MATERIAL}); "
              f"per-residue corruption {v['per_residue_corruption_random']}")
        print(f"  share of the whole-ORF cosine deficit surviving a payload-only mean: "
              f"{v['crosstalk_share_of_deficit']}  "
              f"(crosstalk at >= {VERDICT_CROSSTALK_SHARE}, displacement at < "
              f"{VERDICT_DISPLACEMENT_SHARE})")
        print(f"  attributable to the random filler: {v['attributable_to_random_filler']}: "
              f"random lower on {pr['random_lower']} of {pr['n']} payloads, higher on "
              f"{pr['random_higher']}, median delta {pr['median_delta']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(args.embedder).split("/")[-1].replace(":", "_")
    detail = args.out_dir / f"crosstalk_{tag}.csv"
    fields = sorted({k for r in rows for k in r})
    order = ["construct_id", "payload_accession", "level", "filler", "f", "payload_aa", "orf_aa",
             "measurable", "skip_reason"]
    fields = order + [f for f in fields if f not in order]
    with open(detail, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    res_path = args.out_dir / f"crosstalk_{tag}_residues.csv"
    with open(res_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["construct_id", "payload_accession", "level", "filler",
                                           "residue_index", "residue", "cosine"])
        w.writeheader()
        w.writerows(residue_rows)
    blob = dict(
        embedder=getattr(emb, "name", args.embedder), embedder_spec=args.embedder,
        is_esm2=is_esm2, background_cosine=ESM2_BACKGROUND_COSINE,
        batch_size=args.batch_size, max_length=args.max_length,
        archetype=args.archetype, levels=levels,
        orfs_md5=seen_md5, orfs_md5_expected=EXPECT_MD5,
        orfs_random=str(args.orfs_random), orfs_real=str(args.orfs_real),
        manifest_random=str(args.manifest_random), manifest_real=str(args.manifest_real),
        reference=str(args.reference),
        n_constructs=len(chosen), n_payloads=n_pay, n_rows=len(rows), n_measurable=n_meas,
        binding=dict(orf_embeddings_truncated=sum(1 for r in rows if r.get("orf_truncated")),
                     payload_residues_lost_to_truncation=truncation_hit,
                     span_disagreements_with_manifest=span_mismatch,
                     constructs_with_no_payload_orf=orf_missing),
        by_level_and_filler=summary, verdicts=verdicts,
        elapsed_s=round(time.time() - t0, 1),
        note=("cos_res_* compares the payload's own residues alone against the same residues inside the "
              "ORF, position for position: the archetype was chosen so those positions are identical. "
              "crosstalk_share_of_deficit is a share of a cosine deficit, NOT an orthogonal "
              "decomposition of two effects."),
    )
    with open(args.out_dir / f"crosstalk_{tag}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print(f"\nwrote {detail} ({len(rows)} rows), {res_path} ({len(residue_rows)} rows) and the summary "
          f"JSON in {time.time() - t0:.0f}s")
    if not n_meas:
        print("no row was measurable, so there is no verdict. The skip reasons in the CSV are the "
              "result: read them before changing anything.", file=sys.stderr)
    return 0


def _blank_row(c: dict, filler: str) -> dict:
    return dict(construct_id=c["construct_id"], payload_accession=c["payload_accession"],
                level=c["level"], filler=filler, f=round(c["f"], 6),
                payload_aa=c["payload_aa"], orf_aa=c["orf_aa"])


if __name__ == "__main__":
    raise SystemExit(main())
