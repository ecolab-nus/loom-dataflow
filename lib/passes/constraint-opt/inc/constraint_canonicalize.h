/**
 * @file constraint_canonicalize.h
 * @brief Canonicalization pass for polynomial constraints.
 *
 * Performs mathematically equivalent transformations:
 * - GCD coefficient reduction
 * - Term variable sorting by SSA index
 * - Like-term combining
 */

#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace loom {
namespace constraint_opt {

/**
 * @brief Run canonicalization on all polynomial constraints in the module.
 *
 * Walks all loom.constraint_space ops and canonicalizes each
 * loom.polynomial_constraint by:
 * 1. Computing GCD of all coefficients and upper_bound, dividing through
 * 2. Sorting variable indices within each monomial
 * 3. Merging monomials with identical variable sets
 *
 * @param module The module to transform
 * @return success if canonicalization completed without errors
 */
mlir::LogicalResult runConstraintCanonicalize(mlir::ModuleOp module);

} // namespace constraint_opt
} // namespace loom
