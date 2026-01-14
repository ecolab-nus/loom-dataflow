/**
 * @file constraint_factorize.h
 * @brief Factorization pass for polynomial constraints.
 *
 * Extracts common factors from polynomial terms using greedy
 * frequency counting, reducing non-linear coupling.
 */

#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace loom {
namespace constraint_opt {

/**
 * @brief Run factorization on all polynomial constraints in the module.
 *
 * Walks all loom.constraint_space ops and factorizes each
 * loom.polynomial_constraint by:
 * 1. Counting variable frequency across non-linear monomials
 * 2. Selecting highest-scoring common factor (score = freq * degree)
 * 3. Extracting factor and injecting loom.expression ops
 * 4. Recursing until no further factorization is possible
 *
 * Example transformation:
 *   MK + NK + CK -> K * expression(M + N + C)
 *
 * @param module The module to transform
 * @return success if factorization completed without errors
 */
mlir::LogicalResult runConstraintFactorize(mlir::ModuleOp module);

} // namespace constraint_opt
} // namespace loom
