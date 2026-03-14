"""Loom MLIR pipeline Python API (pybind11 bindings).

Provides safe, version-checked access to the two Loom C++ pipeline stages:
  - run_exploration():       stages 0-5 (tensor canonicalize → enumerate broadcast)
  - run_materialization():   stages 5-7 (materialize → OSB)

The underlying C++ module (_loom_pipeline) is built via pybind11 and installed
alongside this file by scikit-build-core.  At import time we verify that the
module's embedded version matches the installed package version so that stale
.so files are caught immediately rather than causing silent corruption.

Usage::

    from loom_pipeline import run_exploration, run_materialization

    run_exploration(
        input_mlir="test/Passes/mm_2Dmesh/IR/00_from_helion_frontend.mlir",
        df_mlir="test/Dialect/DataflowDialect/2D_mesh.mlir",
        output_mlir="test/Passes/mm_2Dmesh/IR/05_after_enumerate_broadcast.mlir",
        etg_json="test/Passes/mm_2Dmesh/constraint_space/staged_etg_dump.json",
    )

    run_materialization(
        input_mlir="test/Passes/mm_2Dmesh/IR/05_after_enumerate_broadcast.mlir",
        block_sizes_json='{"variant": {"BM": 64, "BN": 64, "BK": 64}}',
        output_mlir="test/Passes/mm_2Dmesh/IR/07_after_osb.mlir",
    )
"""

import json as _json
import sys as _sys
from importlib.metadata import Distribution as _Distribution
from importlib.metadata import version as _pkg_version
from pathlib import Path

from . import _loom_pipeline

# ---------------------------------------------------------------------------
# Version check
# ---------------------------------------------------------------------------
_EXPECTED_VERSION = _pkg_version("loom-dataflow")

# ---------------------------------------------------------------------------
# SMT solver: locate lib/smt/ via the package's editable-install metadata and
# expose run() so callers need no sys.path manipulation.
# ---------------------------------------------------------------------------
def _find_smt_dir() -> Path | None:
    try:
        raw = _Distribution.from_name("loom-dataflow").read_text("direct_url.json")
        url = _json.loads(raw or "{}").get("url", "")
        if url.startswith("file://"):
            return Path(url[len("file://"):]) / "lib" / "smt"
    except Exception:
        pass
    return None

_smt_dir = _find_smt_dir()
if _smt_dir and _smt_dir.is_dir() and str(_smt_dir) not in _sys.path:
    _sys.path.insert(0, str(_smt_dir))

try:
    from main import run as smt_run  # noqa: E402
except ImportError:
    smt_run = None  # type: ignore[assignment]

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
    input_mlir: str | Path,
    df_mlir: str | Path,
    output_mlir: str | Path,
    etg_json: str | Path = "",
) -> None:
    """Run the exploration pipeline (stages 0→5).

    Consolidates tensor_canonicalize, memory_binding, enumerate_hw_mapping,
    analyze_reuse, and enumerate_copy_broadcast into a single in-memory run.
    Optionally produces a staged ETG JSON file for the SMT solver.

    Args:
        input_mlir:  Path to input MLIR (stage 00, from Helion frontend).
        df_mlir:     Path to DF hardware description MLIR.
        output_mlir: Where to write the explored MLIR (stage 05).
        etg_json:    Where to write staged ETG JSON.  Empty string = skip.

    Raises:
        RuntimeError: If the C++ pipeline fails.
    """
    err = _loom_pipeline.run_exploration_pipeline(
        str(input_mlir), str(df_mlir), str(output_mlir), str(etg_json)
    )
    if err:
        raise RuntimeError(f"Exploration pipeline failed: {err}")


def run_materialization(
    input_mlir: str | Path,
    block_sizes_json: str,
    output_mlir: str | Path,
) -> None:
    """Run the materialization pipeline (stages 5→7).

    Takes explored MLIR and block sizes from the SMT solver, materializes
    symbolic values, canonicalizes, and runs One-Shot Bufferization.

    Args:
        input_mlir:       Path to input MLIR (stage 05).
        block_sizes_json: JSON string mapping variant names to block size
                          assignments, e.g.
                          ``{"func_name": {"BM": 64, "BN": 128}, ...}``.
                          Pass empty string to use placeholder solver.
        output_mlir:      Destination path for the final bufferized MLIR.

    Raises:
        RuntimeError: If the C++ pipeline fails.
    """
    err = _loom_pipeline.run_materialization_pipeline(
        str(input_mlir), block_sizes_json, str(output_mlir)
    )
    if err:
        raise RuntimeError(f"Materialization pipeline failed: {err}")
