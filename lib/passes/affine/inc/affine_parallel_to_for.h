#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace tmd_affine {

/**
 * \brief Convert an outermost `affine.parallel` to a perfectly nested chain of
 * `affine.for` loops using the specified iterator order.
 *
 * Semantics:
 * - For a root `affine.parallel` with P iterators, this rewrites it into P
 *   nested `affine.for` loops. Each loop i iterates with the same lower/upper
 *   bound and step as the corresponding iterator in the original parallel op.
 * - The body of the original parallel loop is cloned into the innermost `for`.
 *   Each original induction variable is remapped to the induction variable of
 *   the corresponding `for` loop, independent of the nesting order.
 * - Reductions are not supported.
 *
 * Preconditions:
 * - `par` must be an outermost `affine.parallel` (i.e., not nested within
 *   another `affine.parallel`).
 * - `order` must be a permutation of [0..P-1].
 * - The operation must have no results (no reductions).
 *
 * \param par   The outermost parallel loop to convert.
 * \param order The nesting order as a permutation of iterator indices.
 * \return success on a successful conversion, failure otherwise.
 */
mlir::LogicalResult
ConvertParallelToNested(mlir::affine::AffineParallelOp par,
                                     llvm::ArrayRef<unsigned> order);

} // namespace tmd_affine
