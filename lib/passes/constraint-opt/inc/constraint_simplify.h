/**
 * @file constraint_simplify.h
 * @brief Header for constraint simplification pass.
 */

#ifndef LOOM_PASSES_CONSTRAINT_OPT_CONSTRAINT_SIMPLIFY_H
#define LOOM_PASSES_CONSTRAINT_OPT_CONSTRAINT_SIMPLIFY_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace loom {
namespace constraint_opt {

/**
 * @brief Run the constraint simplification pass.
 *
 * This pass simplifies constraint spaces by removing redundant constraints
 * using the Presburger library.
 */
mlir::LogicalResult runConstraintSimplify(mlir::ModuleOp module);

} // namespace constraint_opt
} // namespace loom

#endif // LOOM_PASSES_CONSTRAINT_OPT_CONSTRAINT_SIMPLIFY_H
