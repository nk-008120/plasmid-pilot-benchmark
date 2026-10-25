#!/usr/bin/env python3
"""Measure ESM-2 650M throughput and peak VRAM against batch size, and project the catalogue cost.

    python scripts/91_esm_throughput.py --swissprot %USERPROFILE%\\Downloads\\uniprot_sprot.fasta.gz

The sample is drawn through the project's own `pilot.homology.iter_swissprot`, so the 50-5,000 aa length
window matches every other run rather than being a second, quietly different filter. It is then sorted by
length and sampled evenly across the range, because throughput is dominated by sequence length and a
random sample would under-represent the long tail that sets the real cost.

The projection at the end is what the number is for: the audited twilight catalogue is 37,544 proteins,
and the per-residue embeddings for an EBA run are the expensive part. `--n-catalogue` sets the target.

Needs the Swiss-Prot FASTA, which is **not in the repository** (`.gitignore` excludes the large data
files). Copy it across or download it from UniProt before running this on a fresh clone.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from pilot.homology import iter_swissprot

MODEL = "facebook/esm2_t33_650M_UR50D"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swissprot", type=Path, default=Path.home() / "Downloads" / "uniprot_sprot.fasta.gz")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-length", type=int, default=1022)
    ap.add_argument("--n", type=int, default=200, help="sequences per batch-size trial")
    ap.add_argument("--pool", type=int, default=20_000, help="entries read before sampling")
    ap.add_argument("--n-catalogue", type=int, default=37_544,
                    help="catalogue size to project onto (the audited twilight reference)")
    ap.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--fp32", action="store_true", help="skip half precision")
    args = ap.parse_args(argv)

    if not args.swissprot.is_file():
        print(f"not found: {args.swissprot}\nThe FASTA is not in the repository; copy it across first.",
              file=sys.stderr)
        return 2

    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        print("no CUDA device visible; run scripts/90_gpu_check.py first", file=sys.stderr)
        return 1

    pool: list[str] = []
    for _, seq in iter_swissprot(args.swissprot):
        pool.append(seq)
        if len(pool) >= args.pool:
            break
    pool.sort(key=len)
    step = max(1, len(pool) // args.n)
    sample = pool[::step][: args.n]
    print(f"n={len(sample)}  length min/median/max: {len(sample[0])} / "
          f"{len(sample[len(sample) // 2])} / {len(sample[-1])}")
    print(f"device: {torch.cuda.get_device_properties(0).name}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval()
    model = (model if args.fp32 else model.half()).cuda()
    print(f"model: {args.model} ({'fp32' if args.fp32 else 'fp16'})\n")

    for bs in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.time()
            with torch.inference_mode():
                for i in range(0, len(sample), bs):
                    batch = [s[: args.max_length] for s in sample[i : i + bs]]
                    enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                              max_length=args.max_length)
                    model(**{k: v.cuda() for k, v in enc.items()})
            torch.cuda.synchronize()
            dt = time.time() - t0
            rate = len(sample) / dt
            print(f"batch {bs:>2}    {rate:>6.1f} seq/s  peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
                  f"  ({dt:.1f} s)  -> {args.n_catalogue:,} proteins in {args.n_catalogue / rate / 60:.0f} min")
        except torch.cuda.OutOfMemoryError:
            print(f"batch {bs:>2}  OOM")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
