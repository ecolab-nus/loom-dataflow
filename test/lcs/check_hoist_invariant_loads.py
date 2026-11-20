#!/usr/bin/env python3
"""Hoisting keys on kernel-argument provenance, not on a memory named DRAM."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    loom_opt = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "build/tool/loom-opt/loom-opt"
    if not loom_opt.is_file():
        print(f"SKIP: loom-opt is unavailable at {loom_opt}")
        return 77
    fixture = Path(__file__).resolve().parent / "hoist_invariant_loads_input.mlir"
    out = subprocess.run(
        [str(loom_opt), "--loom-hoist-invariant-loads", str(fixture)],
        check=True, capture_output=True, text=True,
    ).stdout
    head, loop = out.split("scf.for", 1)
    body, tail = loop.split("} {loom.independent}", 1)
    hoisted = [l for l in head.splitlines() if "loom.copy" in l]
    assert len(hoisted) == 1 and "%arg0" in head and "@mem_HBM" in hoisted[0], head
    assert body.count("loom.copy") == 3, body  # on-chip load, load of written %out, store
    assert tail.count("loom.semaphore_give") == 1, tail
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
