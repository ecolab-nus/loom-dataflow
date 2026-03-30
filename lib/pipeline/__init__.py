"""Loom MLIR pipeline Python API (pybind11 bindings).

Provides safe, version-checked access to the two Loom C++ pipeline stages:
  - run_exploration():       stages 0-5 (tensor canonicalize → enumerate broadcast)
  - run_materialization():   stages 5-7 (materialize → OSB)

All MLIR data flows as in-memory strings — no intermediate file I/O.  The
caller decides when to persist strings to disk (e.g. for debugging).

The underlying C++ module (_loom_pipeline) is built via pybind11 and installed
alongside this file by scikit-build-core.  At import time we verify that the
module's embedded version matches the installed package version so that stale
.so files are caught immediately rather than causing silent corruption.

Usage::

    from loom_pipeline import run_exploration, run_materialization

    output_mlir, etg_json = run_exploration(
        input_mlir=mlir_text,
        df_mlir="test/Dialect/DataflowDialect/2D_mesh.mlir",
        hw_platform_file="../loom-mlar/tests/2d_mesh/2d_mesh_torus_ref.mlir",
    )

    final_mlir = run_materialization(
        input_mlir=output_mlir,
        block_sizes_json='{"variant": {"BM": 64, "BN": 64, "BK": 64}}',
    )
"""

from importlib.metadata import version as _pkg_version
from pathlib import Path

from . import _loom_pipeline

# ---------------------------------------------------------------------------
# Version check
# ---------------------------------------------------------------------------
_EXPECTED_VERSION = _pkg_version("loom-dataflow")

if _loom_pipeline.__version__ != _EXPECTED_VERSION:
    raise RuntimeError(
        f"Loom pipeline version mismatch: Python package expects "
        f"{_EXPECTED_VERSION}, but the C++ module reports "
        f"{_loom_pipeline.__version__}. Please rebuild the C++ module:\n"
        f"  pip install -e . -v --no-build-isolation"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_exploration(
    input_mlir: str,
    df_mlir: str | Path,
    hw_platform_file: str | Path,
    produce_etg: bool = True,
) -> tuple[str, str]:
    """Run the exploration pipeline (stages 0→5).

    Consolidates tensor_canonicalize, memory_binding, enumerate_hw_mapping,
    analyze_reuse, and enumerate_copy_broadcast into a single in-memory run.
    Optionally produces staged ETG JSON for the external block-size solver.

    Args:
        input_mlir:        Input MLIR text (stage 00, from Helion frontend).
        df_mlir:           Path to DF hardware description MLIR.
        hw_platform_file:  Path to hardware platform MLIR file containing
                           sub-modules for compute and data mover components.
        produce_etg:       Whether to generate ETG JSON (default True).

    Returns:
        Tuple of (output_mlir, etg_json).  etg_json is empty when
        produce_etg is False.

    Raises:
        RuntimeError: If the C++ pipeline fails.
    """
    err, output_mlir, etg_json = _loom_pipeline.run_exploration_pipeline(
        input_mlir, str(df_mlir), str(hw_platform_file), produce_etg
    )
    if err:
        raise RuntimeError(f"Exploration pipeline failed: {err}")
    return output_mlir, etg_json


def run_materialization(
    input_mlir: str,
    block_sizes_json: str,
) -> str:
    """Run the materialization pipeline (stages 5→7).

    Takes explored MLIR and block sizes from the external solver, materializes
    symbolic values, canonicalizes, and runs One-Shot Bufferization.

    Args:
        input_mlir:       Input MLIR text (stage 05).
        block_sizes_json: JSON string mapping variant names to block size
                          assignments, e.g.
                          ``{"func_name": {"BM": 64, "BN": 128}, ...}``.
                          Pass empty string to use placeholder solver.

    Returns:
        Output MLIR text (final bufferized MLIR).

    Raises:
        RuntimeError: If the C++ pipeline fails.
    """
    err, output_mlir = _loom_pipeline.run_materialization_pipeline(
        input_mlir, block_sizes_json
    )
    if err:
        raise RuntimeError(f"Materialization pipeline failed: {err}")
    return output_mlir
