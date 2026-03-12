"""End-to-end Loom pipeline orchestrator.

Runs the Python SMT solver to find optimal block sizes for every variant,
then calls the C++ MLIR pass pipeline (via libloom_pipeline.so) to materialize
those block sizes and produce the final bufferized MLIR.

Usage:
    python orchestrator.py \\
        --etg-json  path/to/staged_etg_dump.json \\
        --input-mlir path/to/05_after_enumerate_broadcast.mlir \\
        --output-mlir path/to/final.mlir \\
        [--njobs N] \\
        \\
        [--solver-log path/to/smt_solver.log] \\
        [--lib path/to/libloom_pipeline.so]

The --lib path defaults to:
    <repo_root>/build/lib/libloom_pipeline.so

Python environment: /opt/miniconda3/envs/loom-dev/bin/python
Required packages: z3-solver (already installed), ctypes (stdlib)
"""

import argparse
import ctypes
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate the SMT solver module (same directory as this script)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from main import run as smt_run  # noqa: E402


# ---------------------------------------------------------------------------
# Default path for the shared library
# ---------------------------------------------------------------------------
_REPO_ROOT = _HERE.parent.parent  # lib/smt/ → lib/ → repo root
_DEFAULT_LIB = _REPO_ROOT / "build" / "lib" / "libloom_pipeline.so"


# ---------------------------------------------------------------------------
# ctypes wrapper around libloom_pipeline.so
# ---------------------------------------------------------------------------

class LoomPipeline:
    """Thin Python wrapper around the libloom_pipeline C API."""

    def __init__(self, lib_path: str | Path):
        lib_path = str(lib_path)
        try:
            self._lib = ctypes.CDLL(lib_path)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to load libloom_pipeline.so from '{lib_path}'.\n"
                f"Make sure you have built the loom_pipeline CMake target:\n"
                f"  cd build && cmake .. && make loom_pipeline -j$(nproc)\n"
                f"Original error: {exc}"
            ) from exc

        # int loom_run_full_pipeline(const char*, const char*, const char*, char**)
        self._lib.loom_run_full_pipeline.restype = ctypes.c_int
        self._lib.loom_run_full_pipeline.argtypes = [
            ctypes.c_char_p,                   # input_mlir_path
            ctypes.c_char_p,                   # block_sizes_json
            ctypes.c_char_p,                   # output_mlir_path
            ctypes.POINTER(ctypes.c_char_p),   # error_msg (out)
        ]

        # void loom_free_string(char*)
        self._lib.loom_free_string.restype = None
        self._lib.loom_free_string.argtypes = [ctypes.c_char_p]

    def run(
        self,
        input_mlir: str | Path,
        block_sizes: dict[str, dict[str, int]],
        output_mlir: str | Path,
    ) -> None:
        """Run the full MLIR pipeline.

        Args:
            input_mlir:   Path to 05_after_enumerate_broadcast.mlir.
            block_sizes:  Dict mapping variant function names to their optimal
                          symbol assignments from the SMT solver.
            output_mlir:  Destination path for the final bufferized MLIR.

        Raises:
            RuntimeError: If the C++ pipeline fails.
        """
        error_ptr = ctypes.c_char_p(None)

        ret = self._lib.loom_run_full_pipeline(
            str(input_mlir).encode("utf-8"),
            json.dumps(block_sizes).encode("utf-8"),
            str(output_mlir).encode("utf-8"),
            ctypes.byref(error_ptr),
        )

        if ret != 0:
            msg = (
                error_ptr.value.decode("utf-8")
                if error_ptr.value
                else f"Unknown error (return code {ret})"
            )
            # Free the C-allocated string before raising
            if error_ptr.value:
                self._lib.loom_free_string(error_ptr)
            raise RuntimeError(f"loom_run_full_pipeline failed (code {ret}): {msg}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Loom end-to-end pipeline: SMT solve → Materialize → OSB → final MLIR."
    )
    parser.add_argument(
        "--etg-json",
        required=True,
        metavar="JSON",
        help="Path to staged_etg_dump.json produced by the staged_etg tool.",
    )
    parser.add_argument(
        "--input-mlir",
        required=True,
        metavar="MLIR",
        help="Path to 05_after_enumerate_broadcast.mlir (pipeline input).",
    )
    parser.add_argument(
        "--output-mlir",
        required=True,
        metavar="MLIR",
        help="Destination path for the final bufferized MLIR.",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel SMT solver worker processes (default: 1).",
    )
    parser.add_argument(
        "--solver-log",
        metavar="PATH",
        help="Optional path for per-variant SMT solver log.",
    )
    parser.add_argument(
        "--lib",
        metavar="PATH",
        default=str(_DEFAULT_LIB),
        help=f"Path to libloom_pipeline.so (default: {_DEFAULT_LIB}).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ---- Step 1: Run SMT solver ----
    print("=" * 72)
    print("STEP 1: SMT SOLVER")
    print("=" * 72)

    # Build a minimal Namespace to reuse main.py's run() function
    smt_args = argparse.Namespace(
        input=args.etg_json,
        njobs=args.njobs,
        output=args.solver_log,   # None → no log file
    )

    block_sizes = smt_run(smt_args)

    feasible_count = sum(1 for v in block_sizes.values() if v is not None)
    if feasible_count == 0:
        print("\nERROR: All variants UNSAT. No feasible block sizes. Aborting.")
        sys.exit(1)

    print(f"\nSolver found feasible block sizes for {feasible_count} variant(s).")

    # ---- Step 2: Run MLIR pipeline ----
    print()
    print("=" * 72)
    print("STEP 2: MLIR PIPELINE (Materialize → Canonicalize → OSB)")
    print("=" * 72)
    print(f"  Input  : {args.input_mlir}")
    print(f"  Output : {args.output_mlir}")
    print()

    pipeline = LoomPipeline(args.lib)
    pipeline.run(args.input_mlir, block_sizes, args.output_mlir)

    print(f"Pipeline complete. Final MLIR written to: {args.output_mlir}")


if __name__ == "__main__":
    main()
