#ifndef LOOM_MATERIALIZATION_PIPELINE_H
#define LOOM_MATERIALIZATION_PIPELINE_H

#include <string>
#include <utility>

namespace loom {
namespace pipeline {

/// Run the materialization pipeline (Materialize -> OSB) in memory.
///
/// Takes the explored MLIR (stage 05) and block sizes from the SMT solver,
/// materializes symbolic block sizes, canonicalizes, and runs One-Shot
/// Bufferization to produce the final memref-level MLIR.
///
/// @param input_mlir_text   Input MLIR text (stage 05) as a string.
/// @param block_sizes_json  JSON string with block size assignments per variant.
///                          Format: {"func_name": {"SYM": value, ...}, ...}
///                          Pass empty string to use placeholder solver.
/// @return pair of (error, output_mlir). error is empty on success.
std::pair<std::string, std::string>
runMaterializationPipeline(const std::string &input_mlir_text,
                           const std::string &block_sizes_json);

} // namespace pipeline
} // namespace loom

#endif // LOOM_MATERIALIZATION_PIPELINE_H
