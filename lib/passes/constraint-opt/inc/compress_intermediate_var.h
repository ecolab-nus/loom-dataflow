/**
 * @file intermediate_var_compression.h
 * @brief Header for intermediate variable compression pass.
 */

#ifndef LOOM_PASSES_CONSTRAINT_OPT_INTERMEDIATE_VAR_COMPRESSION_H
#define LOOM_PASSES_CONSTRAINT_OPT_INTERMEDIATE_VAR_COMPRESSION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace loom {
namespace constraint_opt {

/**
 * @brief Run the intermediate variable compression pass.
 *
 * This pass substitutes addition expressions directly into downstream
 * linear constraints, eliminating redundant intermediate variables.
 */
mlir::LogicalResult runIntermediateVarCompression(mlir::ModuleOp module);

} // namespace constraint_opt
} // namespace loom

#endif // LOOM_PASSES_CONSTRAINT_OPT_INTERMEDIATE_VAR_COMPRESSION_H
