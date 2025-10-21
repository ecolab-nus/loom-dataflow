/**
 * @file
 * @brief Conversion utilities from `mlir::affine::AffineParallelOp` to nested
 * `mlir::affine::AffineForOp`.
 *
 * @details This file provides a helper to convert an outermost
 * `affine.parallel` into a perfectly nested sequence of `affine.for` loops,
 * preserving per-dimension lower/upper bound maps and steps, and remapping the
 * original induction variables accordingly.
 *
 * Preconditions enforced by the conversion:
 * - The `affine.parallel` has no results (i.e., no reductions)
 * - It is the outermost parallel (no parent `affine.parallel`)
 * - The requested nesting `order` is a valid permutation of the iterator
 *   indices `0..P-1`, where `P` is the number of dimensions
 *
 * Algorithm overview:
 * - For each requested nesting depth, create an `affine.for` carrying the
 *   chosen iterator's lower/upper bound maps, operands, and step
 * - Ensure each created `affine.for` body has a terminator (`affine.yield`)
 * - Compute the mapping from the original parallel IVs to the corresponding
 *   `affine.for` IVs (based on the iterator's position in `order`)
 * - Clone the original parallel body (excluding the terminator) into the
 *   innermost `affine.for` and erase the source `affine.parallel`
 */
#include "affine_parallel_to_for.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace tmd_affine {
/// Affine structural transformations used within TMD passes.

/**
 * @brief Extract iterator bounds and step for a specific dimension of a
 *        `affine::AffineParallelOp`.
 *
 * @param par The source `AffineParallelOp`.
 * @param dim The zero-based iterator index to query.
 * @param[out] lbMap The lower bound affine map for the chosen iterator.
 * @param[out] lbOperands The operands used by the lower bound map.
 * @param[out] ubMap The upper bound affine map for the chosen iterator.
 * @param[out] ubOperands The operands used by the upper bound map.
 * @param[out] step The step of the chosen iterator.
 * @return `success()` if `dim` is valid and data is extracted; `failure()`
 *         otherwise.
 */
static LogicalResult getIteratorBoundsAndStep(affine::AffineParallelOp par,
                                              unsigned dim, AffineMap &lbMap,
                                              SmallVector<Value> &lbOperands,
                                              AffineMap &ubMap,
                                              SmallVector<Value> &ubOperands,
                                              int64_t &step) {
  if (dim >= par.getNumDims())
    return failure();
  lbMap = par.getLowerBoundMap(dim);
  ubMap = par.getUpperBoundMap(dim);
  auto allLbOperands = par.getLowerBoundsOperands();
  auto allUbOperands = par.getUpperBoundsOperands();
  lbOperands.assign(allLbOperands.begin(), allLbOperands.end());
  ubOperands.assign(allUbOperands.begin(), allUbOperands.end());
  auto steps = par.getSteps();
  if (dim >= steps.size())
    return failure();
  step = steps[dim];
  return success();
}

/**
 * @brief Convert an outermost `affine.parallel` into nested `affine.for` ops.
 *
 * @param par The outermost `AffineParallelOp` to convert.
 * @param order A permutation of iterator indices `0..P-1` that defines the
 *              loop nesting from outermost to innermost, where `P` equals
 *              `par.getNumDims()`.
 * @return `success()` if conversion succeeds; `failure()` if preconditions are
 *         not met or if any construction step fails.
 *
 * @details Preconditions:
 * - `par` has no results (no reductions)
 * - `par` does not have an ancestor `AffineParallelOp` (is outermost)
 * - `order` is a permutation of `0..P-1`, with `P = par.getNumDims()`
 *
 * Behavior:
 * - Creates `P` nested `affine.for` operations in the given order, each
 *   preserving the original per-dimension bound maps, operands, and step.
 * - Ensures each newly created `affine.for` body has a terminator.
 * - Remaps the original IV at iterator index `i` to the IV of the `affine.for`
 *   located at the position of `i` within `order`.
 * - Clones the original parallel body (excluding the terminator) into the
 *   innermost loop and erases the source parallel op.
 */
LogicalResult convertOutermostParallelToNestedFors(affine::AffineParallelOp par,
                                                   ArrayRef<unsigned> order) {
  // Preconditions: no reductions, is outermost, valid permutation.
  if (par.getNumResults() != 0)
    return failure();
  if (par->getParentOfType<affine::AffineParallelOp>())
    return failure();
  const unsigned P = par.getNumDims();
  if (order.size() != P)
    return failure();
  SmallVector<unsigned> seen(P, 0);
  for (unsigned v : order) {
    if (v >= P)
      return failure();
    if (seen[v])
      return failure();
    seen[v] = 1;
  }

  OpBuilder builder(par);
  Location loc = par.getLoc();

  // Build nested `affine.for` in the requested order.
  SmallVector<affine::AffineForOp> newFors;
  newFors.reserve(P);
  affine::AffineForOp innermostFor = nullptr;

  // Create the outermost for inserted before `par`.
  for (unsigned i = 0; i < P; ++i) {
    unsigned iterIdx = order[i];
    AffineMap lbMap, ubMap;
    SmallVector<Value> lbOperands, ubOperands;
    int64_t stepVal = 1;
    if (failed(getIteratorBoundsAndStep(par, iterIdx, lbMap, lbOperands, ubMap,
                                        ubOperands, stepVal)))
      return failure();

    if (i == 0)
      builder.setInsertionPoint(par);
    else
      builder.setInsertionPointToStart(newFors.back().getBody());

    affine::AffineForOp forOp = affine::AffineForOp::create(
        builder, loc, lbOperands, lbMap, ubOperands, ubMap, stepVal);
    // Ensure a terminator exists in the newly created body.
    if (forOp.getBody()->empty() ||
        !isa<affine::AffineYieldOp>(forOp.getBody()->back())) {
      OpBuilder termBuilder = OpBuilder::atBlockEnd(forOp.getBody());
      termBuilder.create<affine::AffineYieldOp>(loc);
    }
    newFors.push_back(forOp);
    innermostFor = forOp;
  }

  // Clone the body of the parallel into the innermost for, remapping IVs.
  IRMapping mapping;
  // Map IVs: original iterator `i` maps to the IV of the `affine.for` located
  // at the position of `i` within `order`.
  SmallVector<Value> forIVForIter(P);
  for (unsigned depth = 0; depth < P; ++depth) {
    unsigned iterIdx = order[depth];
    forIVForIter[iterIdx] = newFors[depth].getInductionVar();
  }
  for (unsigned iterIdx = 0; iterIdx < P; ++iterIdx)
    mapping.map(par.getIVs()[iterIdx], forIVForIter[iterIdx]);

  // Splice-clone operations from parallel body (excluding the terminator)
  // into the innermost for's body.
  Block &srcBody = par.getRegion().front();
  Block &dstBody = *innermostFor.getBody();
  builder.setInsertionPointToStart(&dstBody);
  for (Operation &op : llvm::make_early_inc_range(srcBody)) {
    if (isa<affine::AffineYieldOp>(op))
      continue;
    builder.clone(op, mapping);
  }

  // Erase the original parallel.
  par.erase();
  return success();
}

} // namespace tmd_affine
