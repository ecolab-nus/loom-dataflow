#ifndef LOOM_EXPLORATION_PIPELINE_H
#define LOOM_EXPLORATION_PIPELINE_H

#include <string>

namespace loom {
namespace pipeline {

/// Run the full exploration pipeline (stages 0-5) in memory.
///
/// Consolidates tensor_canonicalize, memory_binding, enumerate_hw_mapping,
/// analyze_reuse, and enumerate_copy_broadcast into a single in-memory run.
/// Optionally produces a staged ETG JSON file for the SMT solver.
///
/// @param input_mlir_path   Path to input MLIR file (stage 00).
/// @param df_mlir_path      Path to DF hardware description MLIR.
/// @param output_mlir_path  Where to write the explored MLIR (stage 05).
/// @param etg_json_path     Where to write the staged ETG JSON. Empty = skip.
/// @return empty string on success, error message on failure.
std::string runExplorationPipeline(const std::string &input_mlir_path,
                                   const std::string &df_mlir_path,
                                   const std::string &output_mlir_path,
                                   const std::string &etg_json_path);

} // namespace pipeline
} // namespace loom

#endif // LOOM_EXPLORATION_PIPELINE_H
