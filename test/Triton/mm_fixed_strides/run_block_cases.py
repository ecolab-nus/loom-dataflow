#!/usr/bin/env python3
"""Run mm.py for a collection of block sizes and dump Triton IR per case."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[3]
MM_SCRIPT = SCRIPT_DIR / "mm.py"
RUNS_ROOT = SCRIPT_DIR / "runs"

DEFAULT_LLVM_BINARY_DIR = Path("/path/to/llvm-build/bin")
DEFAULT_TRITON_SHARED_OPT = Path(
    "/path/to/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"
)
DEFAULT_TRITON_CACHE_DIR = REPO_ROOT / ".triton_cache"
DEFAULT_PYTHON = Path(os.environ.get("TRITON_VENV_PYTHON", "/path/to/triton/.venv/bin/python"))

OUTPUT_FILES = ("tt.mlir", "ttshared.mlir", "ll.mlir", "ll.ir", "mlir_dump.txt")


@dataclass(frozen=True)
class BlockConfig:
    name: str
    block_m: int
    block_n: int
    block_k: int
    seed: int = 0
    description: str = ""


BLOCK_CONFIGS: tuple[BlockConfig, ...] = (
    BlockConfig(
        name="block_16x16x32",
        block_m=16,
        block_n=16,
        block_k=32,
        description="Smaller 16x16 tiles with BLOCK_K=32",
    ),
    BlockConfig(
        name="block_32x32x32",
        block_m=32,
        block_n=32,
        block_k=32,
        description="Baseline 32x32x32 blocking",
    ),
    BlockConfig(
        name="block_32x32x64",
        block_m=32,
        block_n=32,
        block_k=64,
        description="Wider block along K",
    ),
    BlockConfig(
        name="block_32x32x128",
        block_m=32,
        block_n=32,
        block_k=128,
        description="32x32x128 blocking",
    ),
    BlockConfig(
        name="block_64x64x32",
        block_m=64,
        block_n=64,
        block_k=32,
        description="64x64x32 blocking",
    ),
    BlockConfig(
        name="block_64x64x64",
        block_m=64,
        block_n=64,
        block_k=64,
        description="64x64x64 blocking",
    ),
    BlockConfig(
        name="block_64x64x128",
        block_m=64,
        block_n=64,
        block_k=128,
        description="64x64x128 blocking",
    ),
    BlockConfig(
        name="block_128x128x32",
        block_m=128,
        block_n=128,
        block_k=32,
        description="128x128x32 blocking",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Triton dumps for multiple block sizes.")
    parser.add_argument(
        "configs",
        nargs="*",
        help="Optional subset of configuration names (defaults to all).",
    )
    parser.add_argument(
        "--python",
        default=str(DEFAULT_PYTHON),
        help="Python interpreter to run mm.py (defaults to Triton venv).",
    )
    return parser.parse_args()


def select_configs(names: Iterable[str]) -> list[BlockConfig]:
    configs = list(BLOCK_CONFIGS)
    if not names:
        return configs
    lookup = {cfg.name: cfg for cfg in configs}
    missing = [name for name in names if name not in lookup]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Unknown configuration(s): {missing_list}")
    return [lookup[name] for name in names]


def prepare_environment(base_env: dict, dump_dir: Path) -> dict:
    env = base_env.copy()
    env.setdefault("LLVM_BINARY_DIR", str(DEFAULT_LLVM_BINARY_DIR))
    env.setdefault("TRITON_SHARED_OPT_PATH", str(DEFAULT_TRITON_SHARED_OPT))
    env.setdefault("TRITON_DISABLE_CACHE", "1")
    env["MLIR_ENABLE_DUMP"] = "1"
    env["LLVM_IR_ENABLE_DUMP"] = "1"

    cache_dir = Path(env.setdefault("TRITON_CACHE_DIR", str(DEFAULT_TRITON_CACHE_DIR)))
    cache_dir.mkdir(parents=True, exist_ok=True)

    dump_dir.mkdir(parents=True, exist_ok=True)
    env["TRITON_SHARED_DUMP_PATH"] = str(dump_dir)
    env["MLIR_DUMP_PATH"] = str(dump_dir / "mlir_dump.txt")
    return env


def clean_outputs(run_dir: Path) -> None:
    for filename in OUTPUT_FILES:
        candidate = run_dir / filename
        if candidate.exists():
            candidate.unlink()


def write_metadata(run_dir: Path, config: BlockConfig) -> None:
    metadata = {
        "BLOCK_M": config.block_m,
        "BLOCK_N": config.block_n,
        "BLOCK_K": config.block_k,
        "seed": config.seed,
        "description": config.description,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def run_config(python_bin: Path, config: BlockConfig) -> None:
    if not MM_SCRIPT.is_file():
        raise FileNotFoundError(f"Cannot find mm.py at {MM_SCRIPT}")

    run_dir = RUNS_ROOT / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    clean_outputs(run_dir)
    write_metadata(run_dir, config)

    env = prepare_environment(os.environ, run_dir)

    cmd = [
        str(python_bin),
        str(MM_SCRIPT),
        "--block-m",
        str(config.block_m),
        "--block-n",
        str(config.block_n),
        "--block-k",
        str(config.block_k),
        "--seed",
        str(config.seed),
    ]

    print(f"=== {config.name} ===")
    if config.description:
        print(config.description)
    print(
        f"Running BLOCK_M={config.block_m}, BLOCK_N={config.block_n}, BLOCK_K={config.block_k}, seed={config.seed}"
    )

    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    print(f"Wrote dumps under {run_dir}\n")


def main() -> int:
    args = parse_args()
    python_bin = Path(args.python)
    if not python_bin.is_file():
        print(f"Python interpreter not found: {python_bin}", file=sys.stderr)
        return 1

    try:
        configs = select_configs(args.configs)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    for config in configs:
        try:
            run_config(python_bin, config)
        except subprocess.CalledProcessError as exc:
            print(f"Configuration {config.name} failed with exit code {exc.returncode}", file=sys.stderr)
            return exc.returncode
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Configuration {config.name} failed: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
