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
      py::arg("input_mlir_path"),
      py::arg("df_mlir_path"),
      py::arg("output_mlir_path"),
      py::arg("etg_json_path") = std::string(""),
      R"doc(Run the exploration pipeline (stages 0-5).

      Consolidates tensor_canonicalize, memory_binding, enumerate_hw_mapping,
      analyze_reuse, and enumerate_copy_broadcast into a single in-memory run.

      Args:
          input_mlir_path: Path to input MLIR file (stage 00).
          df_mlir_path: Path to DF hardware description MLIR.
          output_mlir_path: Where to write the explored MLIR (stage 05).
          etg_json_path: Where to write staged ETG JSON. Empty = skip.

      Returns:
          Empty string on success, error message on failure.
      )doc",
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "run_materialization_pipeline",
      &loom::pipeline::runMaterializationPipeline,
      py::arg("input_mlir_path"),
      py::arg("block_sizes_json"),
      py::arg("output_mlir_path"),
      R"doc(Run the materialization pipeline (Materialize -> OSB).

      Takes explored MLIR and block sizes from the SMT solver, materializes
      symbolic values, canonicalizes, and runs One-Shot Bufferization.

      Args:
          input_mlir_path: Path to input MLIR file (stage 05).
          block_sizes_json: JSON string mapping variant names to block sizes.
          output_mlir_path: Destination path for the final bufferized MLIR.

      Returns:
          Empty string on success, error message on failure.
      )doc",
      py::call_guard<py::gil_scoped_release>());
}
