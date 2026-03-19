#ifndef LOOM_EXPLORATION_PIPELINE_H
#define LOOM_EXPLORATION_PIPELINE_H

#include <string>
#include <tuple>

namespace loom {
namespace pipeline {

/// Run the full exploration pipeline (stages 0-5) in memory.
///
/// Consolidates tensor_canonicalize, memory_binding, enumerate_hw_mapping,
/// analyze_reuse, and enumerate_copy_broadcast into a single in-memory run.
/// Optionally produces staged ETG JSON for the external block-size solver.
///
/// @param input_mlir_text   Input MLIR text (stage 00) as a string.
/// @param df_mlir_path      Path to DF hardware description MLIR file.
/// @param hw_compute_dir    Path to directory containing hardware compute IR
///                          (.mlir) files for the ComputeOpRegistry.
/// @param produce_etg       Whether to generate ETG JSON output.
/// @return tuple of (error, output_mlir, etg_json).
///         error is empty on success; etg_json is empty when produce_etg
///         is false.
std::tuple<std::string, std::string, std::string>
runExplorationPipeline(const std::string &input_mlir_text,
                       const std::string &df_mlir_path,
                       const std::string &hw_compute_dir,
                       bool produce_etg);

} // namespace pipeline
} // namespace loom

#endif // LOOM_EXPLORATION_PIPELINE_H
