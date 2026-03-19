/// pybind11 bindings for the Loom MLIR pipeline.
///
/// Exposes two pipeline functions and a version string:
///   - run_exploration_pipeline(...)   → stages 0-5
///   - run_materialization_pipeline(...) → stages 5-7
///   - __version__                     → compile-time version from CMake

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "loom_version.h"
#include "loom_exploration_pipeline.h"
#include "loom_materialization_pipeline.h"

namespace py = pybind11;

PYBIND11_MODULE(_loom_pipeline, m) {
  m.doc() = "Loom MLIR pipeline - pybind11 interface";
  m.attr("__version__") = LOOM_VERSION_STRING;

  m.def(
      "run_exploration_pipeline",
      &loom::pipeline::runExplorationPipeline,
      py::arg("input_mlir_text"),
      py::arg("df_mlir_path"),
      py::arg("hw_compute_dir"),
      py::arg("produce_etg") = true,
      R"doc(Run the exploration pipeline (stages 0-5).

      Consolidates tensor_canonicalize, memory_binding, enumerate_hw_mapping,
      analyze_reuse, and enumerate_copy_broadcast into a single in-memory run.

      Args:
          input_mlir_text: Input MLIR as a string (stage 00).
          df_mlir_path: Path to DF hardware description MLIR.
          hw_compute_dir: Path to directory containing hardware compute IR
              (.mlir) files for workload dispatch (e.g., matrix_lane.mlir).
          produce_etg: Whether to produce ETG JSON (default True).

      Returns:
          Tuple of (error, output_mlir, etg_json).
          error is empty on success.
      )doc",
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "run_materialization_pipeline",
      &loom::pipeline::runMaterializationPipeline,
      py::arg("input_mlir_text"),
      py::arg("block_sizes_json"),
      R"doc(Run the materialization pipeline (Materialize -> OSB).

      Takes explored MLIR and block sizes from the external solver, materializes
      symbolic values, canonicalizes, and runs One-Shot Bufferization.

      Args:
          input_mlir_text: Input MLIR as a string (stage 05).
          block_sizes_json: JSON string mapping variant names to block sizes.

      Returns:
          Tuple of (error, output_mlir).
          error is empty on success.
      )doc",
      py::call_guard<py::gil_scoped_release>());
}
