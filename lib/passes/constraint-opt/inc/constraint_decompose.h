/**
 * @file constraint_decompose.h
 * @brief Header for polynomial constraint decomposition pass.
 */

#ifndef LOOM_PASSES_CONSTRAINT_OPT_CONSTRAINT_DECOMPOSE_H
#define LOOM_PASSES_CONSTRAINT_OPT_CONSTRAINT_DECOMPOSE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace loom {
namespace constraint_opt {

/**
 * @brief Run the polynomial constraint decomposition pass.
 *
 * This pass isolates non-linear terms in polynomial constraints by replacing
 * them with loom.expression proxy variables.
 */
mlir::LogicalResult runConstraintDecompose(mlir::ModuleOp module);

} // namespace constraint_opt
} // namespace loom

#endif // LOOM_PASSES_CONSTRAINT_OPT_CONSTRAINT_DECOMPOSE_H
