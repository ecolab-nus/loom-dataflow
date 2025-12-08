#!/usr/bin/env python3
"""Run flashattn.py for a collection of shapes and block sizes."""

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
FLASHATTN_SCRIPT = SCRIPT_DIR / "flashattn.py"
RUNS_ROOT = SCRIPT_DIR / "runs"

DEFAULT_LLVM_BINARY_DIR = Path("/mnt/fast/llvm-build/bin")
DEFAULT_TRITON_SHARED_OPT = Path(
    "/home/zhenyu/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"
)
DEFAULT_TRITON_CACHE_DIR = REPO_ROOT / ".triton_cache"
DEFAULT_PYTHON = Path(os.environ.get("TRITON_VENV_PYTHON", "/home/zhenyu/triton/.venv/bin/python"))

OUTPUT_FILES = ("tt.mlir", "ttshared.mlir", "ll.mlir", "ll.ir", "mlir_dump.txt")


@dataclass(frozen=True)
class FlashAttnConfig:
    """!Container describing a FlashAttention problem size."""

    name: str
    seq_len: int
    head_dim: int
    block_q: int
    block_kv: int
    seed: int = 0
    description: str = ""


FLASHATTN_CONFIGS: tuple[FlashAttnConfig, ...] = (
    FlashAttnConfig(
        name="seq128_head32_block32",
        seq_len=128,
        head_dim=32,
        block_q=32,
        block_kv=32,
        description="Baseline 128-token head32 problem with 32x32 tiling.",
    ),
    FlashAttnConfig(
        name="seq256_head32_block64",
        seq_len=256,
        head_dim=32,
        block_q=32,
        block_kv=64,
        description="Longer sequence with wider streaming tile.",
    ),
    FlashAttnConfig(
        name="seq512_head64_block64",
        seq_len=512,
        head_dim=64,
        block_q=64,
        block_kv=64,
        description="Matches the default example but head_dim doubled.",
    ),
    FlashAttnConfig(
        name="seq512_head128_block64",
        seq_len=512,
        head_dim=128,
        block_q=64,
        block_kv=64,
        description="Stress head-dimension to 128 while keeping seq=512.",
    ),
    FlashAttnConfig(
        name="seq1024_head64_block128",
        seq_len=1024,
        head_dim=64,
        block_q=128,
        block_kv=128,
        description="1k sequence with large 128x128 blocks.",
    ),
)


def parse_args() -> argparse.Namespace:
    """!Parse CLI arguments for selecting configurations."""

    parser = argparse.ArgumentParser(description="Generate Triton dumps for FlashAttention block sweeps.")
    parser.add_argument(
        "configs",
        nargs="*",
        help="Optional subset of configuration names (defaults to all).",
    )
    parser.add_argument(
        "--python",
        default=str(DEFAULT_PYTHON),
        help="Python interpreter to run flashattn.py (defaults to Triton venv).",
    )
    return parser.parse_args()


def select_configs(names: Iterable[str]) -> list[FlashAttnConfig]:
    """!Return the requested configuration objects."""

    configs = list(FLASHATTN_CONFIGS)
    if not names:
        return configs
    lookup = {cfg.name: cfg for cfg in configs}
    missing = [name for name in names if name not in lookup]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Unknown configuration(s): {missing_list}")
    return [lookup[name] for name in names]


def prepare_environment(base_env: dict, dump_dir: Path) -> dict:
    """!Add Triton dump/misc variables to the environment."""

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
    """!Remove stale dump artifacts before generating a new run."""

    for filename in OUTPUT_FILES:
        candidate = run_dir / filename
        if candidate.exists():
            candidate.unlink()


def write_metadata(run_dir: Path, config: FlashAttnConfig) -> None:
    """!Dump the scalar configuration to config.json for traceability."""

    metadata = {
        "SEQ_LEN": config.seq_len,
        "HEAD_DIM": config.head_dim,
        "BLOCK_Q": config.block_q,
        "BLOCK_KV": config.block_kv,
        "seed": config.seed,
        "description": config.description,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def run_config(python_bin: Path, config: FlashAttnConfig) -> None:
    """!Execute flashattn.py for a single configuration."""

    if not FLASHATTN_SCRIPT.is_file():
        raise FileNotFoundError(f"Cannot find flashattn.py at {FLASHATTN_SCRIPT}")

    run_dir = RUNS_ROOT / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    clean_outputs(run_dir)
    write_metadata(run_dir, config)

    env = prepare_environment(os.environ, run_dir)

    cmd = [
        str(python_bin),
        str(FLASHATTN_SCRIPT),
        "--seq-len",
        str(config.seq_len),
        "--head-dim",
        str(config.head_dim),
        "--block-q",
        str(config.block_q),
        "--block-kv",
        str(config.block_kv),
        "--seed",
        str(config.seed),
    ]

    print(f"=== {config.name} ===")
    if config.description:
        print(config.description)
    print(
        "Running SEQ_LEN={seq}, HEAD_DIM={head}, BLOCK_Q={bq}, BLOCK_KV={bkv}, seed={seed}".format(
            seq=config.seq_len,
            head=config.head_dim,
            bq=config.block_q,
            bkv=config.block_kv,
            seed=config.seed,
        )
    )

    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    print(f"Wrote dumps under {run_dir}\n")


def main() -> int:
    """!Entry point for the FlashAttention block sweep runner."""

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

