#!/usr/bin/env python3
"""Confirm the GPU can actually execute a kernel, not merely that torch can see it.

    python scripts/90_gpu_check.py

**Why the matmul is the point.** On a Blackwell card (RTX 50-series, sm_120) with a wheel built for older
architectures, `torch.cuda.is_available()` returns True and `get_device_properties` reports the card
correctly. The failure only appears when a real kernel launches:

    RuntimeError: CUDA error: no kernel image is available for execution on the device

So a check that stops at `is_available()` passes on exactly the machine it was written to protect against.
This runs a 4096x4096 matmul and synchronises, because CUDA calls are asynchronous and an un-synchronised
failure surfaces later, somewhere unrelated.

Install torch from the cu128 index for Blackwell:

    pip install torch --index-url https://download.pytorch.org/whl/cu128
"""
from __future__ import annotations

import sys


def main() -> int:
    import torch

    print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
    if not torch.cuda.is_available():
        print("no CUDA device visible", file=sys.stderr)
        return 1

    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    print("device:", props.name)
    print("capability: sm_%d%d" % cap)
    print("VRAM total: %.2f GB" % (props.total_memory / 1e9))

    try:
        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        c = (a @ b).sum().item()
        torch.cuda.synchronize()
    except RuntimeError as exc:
        print(f"\nkernel launch FAILED: {exc}", file=sys.stderr)
        print("If this says 'no kernel image is available', the wheel predates this card. "
              "Reinstall from the cu128 index.", file=sys.stderr)
        return 2

    print("matmul executed, checksum", c)
    print("\nOK: the device runs real kernels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
