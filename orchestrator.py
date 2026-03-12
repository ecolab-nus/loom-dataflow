"""End-to-end Loom pipeline orchestrator (pybind11 version).

Runs the full Loom compilation pipeline:
  1. Exploration pipeline (stages 0→5): C++ passes via pybind11
  2. SMT solver: finds optimal block sizes per variant
  3. Materialization pipeline (stages 5→7): C++ passes via pybind11

Usage:
    python orchestrator.py \\
        --input-mlir path/to/00_from_helion_frontend.mlir \\
        --df-mlir    path/to/2D_mesh.mlir \\
        --output-mlir path/to/final.mlir \\
        [--explored-mlir path/to/05_after_enumerate_broadcast.mlir] \\
        [--etg-json path/to/staged_etg_dump.json] \\
        [--njobs N] \\
        [--solver-log path/to/smt_solver.log] \\
        [--debug]

Python environment: /opt/miniconda3/envs/loom-dev/bin/python
Required packages: z3-solver, pybind11 (build-time)
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate the SMT solver module
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SMT_DIR = _HERE / "lib" / "smt"
if str(_SMT_DIR) not in sys.path:
    sys.path.insert(0, str(_SMT_DIR))

from main import run as smt_run  # noqa: E402

# Import the pybind11-based pipeline module (version-checked on import)
from loom_pipeline import run_exploration, run_materialization  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Loom end-to-end pipeline: "
            "Explore → SMT solve → Materialize → OSB → final MLIR."
        )
    )
    parser.add_argument(
        "--input-mlir",
        required=True,
        metavar="MLIR",
        help="Path to input MLIR (stage 00, from Helion frontend).",
    )
    parser.add_argument(
        "--df-mlir",
        required=True,
        metavar="MLIR",
        help="Path to DF hardware description MLIR (e.g. 2D_mesh.mlir).",
    )
    parser.add_argument(
        "--output-mlir",
        required=True,
        metavar="MLIR",
        help="Destination path for the final bufferized MLIR.",
    )
    parser.add_argument(
        "--explored-mlir",
        metavar="MLIR",
        help=(
            "Optional path to save the exploration output (stage 05). "
            "If not specified, a temporary file is used."
        ),
    )
    parser.add_argument(
        "--etg-json",
        metavar="JSON",
        help=(
            "Optional path for the staged ETG JSON. "
            "If not specified, a temporary file is used."
        ),
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
        "--debug",
        action="store_true",
        default=False,
        help="Enable detailed SMT analysis (active constraints, MUS).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Resolve paths for intermediate files
    explored_mlir = args.explored_mlir
    etg_json = args.etg_json

    # Use temp files if no explicit paths given
    tmp_files = []
    if not explored_mlir:
        f = tempfile.NamedTemporaryFile(suffix=".mlir", delete=False)
        explored_mlir = f.name
        f.close()
        tmp_files.append(explored_mlir)
    if not etg_json:
        f = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        etg_json = f.name
        f.close()
        tmp_files.append(etg_json)

    try:
        # ---- Step 1: Exploration pipeline (stages 0→5) ----
        print("=" * 72)
        print("STEP 1: EXPLORATION PIPELINE (stages 0→5)")
        print("=" * 72)
        print(f"  Input MLIR : {args.input_mlir}")
        print(f"  DF MLIR    : {args.df_mlir}")
        print(f"  Output     : {explored_mlir}")
        print(f"  ETG JSON   : {etg_json}")
        print()

        run_exploration(
            input_mlir=args.input_mlir,
            df_mlir=args.df_mlir,
            output_mlir=explored_mlir,
            # TODO: use true etg.json when hw analytical model is ready
            etg_json="test/Passes/mm_2Dmesh/constraint_space/temp.json",
        )

        print("Exploration pipeline complete.")

        # ---- Step 2: SMT solver ----
        print()
        print("=" * 72)
        print("STEP 2: SMT SOLVER")
        print("=" * 72)

        smt_args = argparse.Namespace(
            input=etg_json,
            njobs=args.njobs,
            output=args.solver_log,
            debug=args.debug,
        )

        block_sizes = smt_run(smt_args)

        feasible_count = sum(1 for v in block_sizes.values() if v is not None)
        if feasible_count == 0:
            print("\nERROR: All variants UNSAT. No feasible block sizes. Aborting.")
            sys.exit(1)

        print(f"\nSolver found feasible block sizes for {feasible_count} variant(s).")

        # ---- Step 3: Materialization pipeline (stages 5→7) ----
        print()
        print("=" * 72)
        print("STEP 3: MATERIALIZATION PIPELINE (Materialize → OSB)")
        print("=" * 72)
        print(f"  Input  : {explored_mlir}")
        print(f"  Output : {args.output_mlir}")
        print()

        run_materialization(
            input_mlir=explored_mlir,
            block_sizes_json=json.dumps(block_sizes),
            output_mlir=args.output_mlir,
        )

        print(f"Pipeline complete. Final MLIR written to: {args.output_mlir}")

    finally:
        # Clean up temp files
        import os
        for tf in tmp_files:
            try:
                os.unlink(tf)
            except OSError:
                pass


if __name__ == "__main__":
    main()
