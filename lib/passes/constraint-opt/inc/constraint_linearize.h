/**
 * @file constraint_linearize.h
 * @brief Header for polynomial constraint linearization pass.
 */

#ifndef LOOM_PASSES_CONSTRAINT_OPT_CONSTRAINT_LINEARIZE_H
#define LOOM_PASSES_CONSTRAINT_OPT_CONSTRAINT_LINEARIZE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace loom {
namespace constraint_opt {

/**
 * @brief Run the polynomial constraint linearization pass.
 *
 * This pass converts polynomial constraints into linear constraints using
 * McCormick relaxation for multiplication expressions.
 */
mlir::LogicalResult runConstraintLinearize(mlir::ModuleOp module);

} // namespace constraint_opt
} // namespace loom

#endif // LOOM_PASSES_CONSTRAINT_OPT_CONSTRAINT_LINEARIZE_H
