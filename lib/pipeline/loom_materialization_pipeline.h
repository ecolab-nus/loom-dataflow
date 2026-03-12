#ifndef LOOM_MATERIALIZATION_PIPELINE_H
#define LOOM_MATERIALIZATION_PIPELINE_H

#include <string>

namespace loom {
namespace pipeline {

/// Run the materialization pipeline (Materialize -> OSB) in memory.
///
/// Takes the explored MLIR (stage 05) and block sizes from the SMT solver,
/// materializes symbolic block sizes, canonicalizes, and runs One-Shot
/// Bufferization to produce the final memref-level MLIR.
///
/// @param input_mlir_path   Path to input MLIR file (stage 05).
/// @param block_sizes_json  JSON string with block size assignments per variant.
///                          Format: {"func_name": {"SYM": value, ...}, ...}
///                          Pass empty string to use placeholder solver.
/// @param output_mlir_path  Destination path for the final bufferized MLIR.
/// @return empty string on success, error message on failure.
std::string runMaterializationPipeline(const std::string &input_mlir_path,
                                       const std::string &block_sizes_json,
                                       const std::string &output_mlir_path);

} // namespace pipeline
} // namespace loom

#endif // LOOM_MATERIALIZATION_PIPELINE_H
