#include "affine_parallel_to_for.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace tmd_affine {

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
  // These return all operands; we use the shared operand list of the op.
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

  // Build nested AffineForOps in the requested order. We create them empty
  // first, wiring bounds from the corresponding parallel iterator.
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
        builder, loc, lbOperands, lbMap, ubOperands, ubMap, stepVal,
        /*iterArgs=*/ValueRange(),
        [&](OpBuilder &b, Location bl, Value iv, ValueRange /*iters*/) {
          (void)b;
          (void)bl;
          (void)iv;
        });
    newFors.push_back(forOp);
    innermostFor = forOp;
  }

  // Clone the body of the parallel into the innermost for, remapping IVs.
  IRMapping mapping;
  // Map IVs: original order 0..P-1 maps to the corresponding for IVs located
  // at position of that iterator within `order`.
  SmallVector<Value> forIVForIter(P);
  for (unsigned depth = 0; depth < P; ++depth) {
    unsigned iterIdx = order[depth];
    forIVForIter[iterIdx] = newFors[depth].getInductionVar();
  }
  for (unsigned iterIdx = 0; iterIdx < P; ++iterIdx)
    mapping.map(par.getIVs()[iterIdx], forIVForIter[iterIdx]);

  // Splice-clone operations from parallel body (excluding terminator) into the
  // innermost for's body.
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
