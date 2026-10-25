#!/usr/bin/env python3
"""Derive the EBA score scale from the embedder's own cosine distribution instead of sweeping it blind.

    python scripts/20_eba_scale.py --embedder blosum
    python scripts/20_eba_scale.py --embedder facebook/esm2_t33_650M_UR50D   # on the GPU machine

**The problem this fixes.** `scripts/18_eba_sweep.py` swept `shift` over 0 to 0.6 and the best value was
0.6, the last point, still climbing. A sweep whose optimum sits on its own boundary has not found an
optimum, and the reason was that the range was picked by hand rather than from the embedding.

**The derivation.** Smith-Waterman only behaves like a *local* method if an unrelated pair of residues
scores at or below zero; otherwise every extension is profitable and the alignment runs to the full length
of both sequences, which is global alignment wearing a local alignment's clothes. So the substitution
score `(cos - shift) * scale` wants `shift` set where unrelated pairs sit:

    shift = mu            the mean cosine of residue pairs from unrelated proteins
    gap_open = 2 sigma    gaps priced in the natural units of the score, which is the spread of that
    gap_extend = 0.4 sigma    same distribution

Two embedders will not share a `mu` or a `sigma`, BLOSUM62 rows are 20-dimensional and strongly
correlated, ESM-2's are 1,280-dimensional, so carrying one scale to the other is exactly the mistake
that produced the boundary-hugging shift. This script is therefore run once per embedder.

**Unrelated means unrelated.** Pairs are drawn from proteins in different families, where family
membership comes from `homologs.csv`: two proteins that are homologs of the same payload are excluded, as
is a payload and its own homolog. Without that the catalogue's family structure would inflate `mu`, since
a twilight reference is deliberately dense in homologs.

Writes `eba_scale_<embedder>.json` and a histogram of the cosine distribution.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from pilot.positional import get_positional_embedder
from pilot.reference import iter_fasta


def _digest(path: Path) -> str:
    """SHA-256 of the reference file. The sampled proteins come from `rng.sample` over *file order*, so
    two runs are only comparable if this matches; without it, a different catalogue looks like a bug."""
    import hashlib  # noqa: PLC0415

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def families(homologs: Path) -> dict[str, set[str]]:
    """accession -> the payload families it belongs to, so same-family pairs can be excluded."""
    fam: dict[str, set[str]] = defaultdict(set)
    if not homologs.is_file():
        return fam
    with open(homologs, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            fam[r["homolog_accession"]].add(r["payload_accession"])
            fam[r["payload_accession"]].add(r["payload_accession"])
    return fam


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path,
                    default=Path("results/homologs_audited/reference_twilight.fasta"))
    ap.add_argument("--homologs", type=Path, default=Path("results/homologs_audited/homologs.csv"))
    ap.add_argument("--embedder", default="blosum")
    ap.add_argument("--n-proteins", type=int, default=400,
                    help="proteins embedded; residue pairs are then drawn across them, which keeps an "
                         "ESM-2 run affordable while still giving 10^5 pairs")
    ap.add_argument("--n-pairs", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--out-dir", type=Path, default=Path("results/baseline2"))
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args(argv)

    if not args.reference.is_file():
        print(f"not found: {args.reference}", file=sys.stderr)
        return 2
    t0 = time.time()
    refs = list(iter_fasta(args.reference))
    fam = families(args.homologs)
    print(f"{len(refs):,} proteins in {args.reference.name}; "
          f"{len(fam):,} of them carry a family label", flush=True)

    out_digest = _digest(args.reference)
    rng = random.Random(args.seed)
    picked = rng.sample(refs, min(args.n_proteins, len(refs)))
    kw = {} if args.embedder in ("blosum", "blosum62") or args.embedder.startswith("kmerpos") else {
        "batch_size": args.batch_size}
    emb = get_positional_embedder(args.embedder, **kw)
    print(f"embedding {len(picked)} proteins with {emb.name} ...", flush=True)
    mats = emb.embed_residues([e.seq for e in picked])
    print(f"  done in {time.time() - t0:.0f}s, dim {mats[0].shape[1]}", flush=True)

    usable = [i for i, m in enumerate(mats) if m.shape[0] > 0]
    cos: list[float] = []
    tries = 0
    while len(cos) < args.n_pairs and tries < args.n_pairs * 50:
        tries += 1
        a, b = rng.choice(usable), rng.choice(usable)
        if a == b:
            continue
        fa, fb = fam.get(picked[a].accession, set()), fam.get(picked[b].accession, set())
        if fa & fb:  # same payload family: not an unrelated pair
            continue
        ra = rng.randrange(mats[a].shape[0])
        rb = rng.randrange(mats[b].shape[0])
        cos.append(float(mats[a][ra] @ mats[b][rb]))
    c = np.asarray(cos, dtype=np.float64)
    mu, sigma = float(c.mean()), float(c.std(ddof=1))

    pct = {f"p{q}": float(np.percentile(c, q)) for q in (1, 5, 25, 50, 75, 95, 99)}
    # The pairs are NOT independent: all of them are drawn from `n_proteins` proteins, so mu is pinned
    # down by how many proteins were sampled, not by how many pairs. Quoting sigma/sqrt(n_pairs) makes
    # the derivation look about 16x more precise than it is and turns ordinary resampling noise into an
    # apparent cross-machine discrepancy. Measured: two seeds on one machine moved mu by 0.018 at 35M,
    # which is larger than the 0.012 between two machines at 650M.
    se_naive = sigma / np.sqrt(max(len(cos), 1))
    se_clustered = sigma / np.sqrt(len(picked))
    derived = {"shift": round(mu, 4), "gap_open": round(2 * sigma, 4), "gap_extend": round(0.4 * sigma, 4)}
    out = {"embedder": args.embedder, "embedder_name": emb.name, "reference": str(args.reference),
           "n_proteins": len(picked), "n_pairs": len(cos), "seed": args.seed,
           "mu": round(mu, 6), "sigma": round(sigma, 6), "min": float(c.min()), "max": float(c.max()),
           "percentiles": {k: round(v, 4) for k, v in pct.items()}, "derived": derived,
           "se_mu_clustered": round(float(se_clustered), 5),
           "se_mu_naive_do_not_use": round(float(se_naive), 6),
           "reference_sha256": _digest(args.reference),
           "sweep_range": {k: [round(v * 0.5, 4), round(v * 1.5, 4)] for k, v in derived.items()}}

    print(f"\ncosine of {len(cos):,} residue pairs from unrelated proteins, {emb.name}")
    print(f"  mean mu      {mu:+.4f}")
    print(f"  stdev sigma   {sigma:.4f}")
    print(f"  range        {c.min():+.4f} .. {c.max():+.4f}")
    print("  percentiles  " + "  ".join(f"{k}={v:+.3f}" for k, v in pct.items()))
    print(f"  SE(mu)       {se_clustered:.4f}   (sigma / sqrt({len(picked)} proteins); the pairs are "
          f"clustered within them, so sigma/sqrt(pairs) = {se_naive:.5f} is NOT the right scale)")
    print(f"  reference    sha256 {out_digest[:16]}...")
    print(f"\nderived scale:  shift = mu = {derived['shift']}      "
          f"gap_open = 2 sigma = {derived['gap_open']}   gap_extend = 0.4 sigma = {derived['gap_extend']}")
    print("sweep +/-50%:   " + "   ".join(f"{k}  {v[0]} .. {v[1]}" for k, v in out["sweep_range"].items()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.embedder.replace("/", "_")
    (args.out_dir / f"eba_scale_{stem}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        ax.hist(c, bins=120, color="#2f6fb3", alpha=0.85)
        ax.axvline(mu, color="#b3402f", lw=1.6, label=f"mu = {mu:+.3f}  (the derived shift)")
        for k in (-1, 1):
            ax.axvline(mu + k * sigma, color="#b3402f", lw=0.9, ls="--",
                       label="mu +/- sigma" if k == 1 else None)
        ax.set_xlabel(f"cosine between residues of unrelated proteins ({emb.name})")
        ax.set_ylabel("pairs")
        ax.set_title("Where 'unrelated' sits, and so where the substitution score should cross zero",
                     fontsize=10, loc="left")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25, lw=0.6)
        d = args.out_dir / "figures"
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(d / f"fig_d7_cosine_{stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {d}/fig_d7_cosine_{stem}.png / .pdf")
    except ImportError:
        print("matplotlib not installed; skipping the histogram")
    print(f"wrote eba_scale_{stem}.json   ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
