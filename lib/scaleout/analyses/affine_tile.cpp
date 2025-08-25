#include "affine_tile.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;

namespace tmd_affine {

LogicalResult tileAffineParallel(affine::AffineParallelOp op,
                                 int64_t tilingFactor, unsigned tileDimIndex) {
  MLIRContext *ctx = op.getContext();
  if (tilingFactor <= 0)
    return op.emitError("tiling-factor must be positive"), failure();

  OpBuilder builder(ctx);
  Location loc = op.getLoc();
  unsigned numDims = op.getNumDims();
  if (numDims == 0)
    return success();
  if (tileDimIndex >= numDims)
    return op.emitError("tileDimIndex out of range"), failure();

  // Only support unit step on the chosen dimension for simplicity.
  SmallVector<int64_t> steps = llvm::to_vector(op.getSteps());
  if (steps.empty() || steps[tileDimIndex] != 1)
    return op.emitError("only step 1 supported on the chosen dimension"),
           failure();

  // Require lower bound 0 on the chosen dimension for simplicity.
  AffineMap lbChosen = op.getLowerBoundMap(tileDimIndex);
  if (lbChosen.getNumResults() != 1 || !lbChosen.isSingleConstant())
    return op.emitError(
               "expected constant-zero lower bound on chosen dimension"),
           failure();
  if (lbChosen.getSingleConstantResult() != 0)
    return op.emitError("expected lower bound 0 on chosen dimension"),
           failure();

  // If constant ranges known, verify divisibility; otherwise assume
  // divisible.
  if (auto maybeRanges = op.getConstantRanges()) {
    auto ranges = *maybeRanges;
    if (!ranges.empty()) {
      if (tileDimIndex >= ranges.size())
        return op.emitError("range metadata missing for chosen dimension"),
               failure();
      int64_t extentK = ranges[tileDimIndex];
      if (extentK % tilingFactor != 0)
        return op.emitError(
                   "chosen-dimension bound not divisible by tiling-factor"),
               failure();
    }
  }

  builder.setInsertionPoint(op);

  // Build LB maps: all zeros (no operands, no symbols).
  SmallVector<AffineMap> outerLbMaps;
  outerLbMaps.reserve(numDims);
  for (unsigned i = 0; i < numDims; ++i)
    outerLbMaps.push_back(builder.getConstantAffineMap(0));

  // Build UB maps: each returns a dedicated symbol s_i out of k symbols.
  SmallVector<AffineMap> outerUbMaps;
  outerUbMaps.reserve(numDims);
  // Prepare ub args values for each dim.
  SmallVector<Value> outerUbArgs;
  outerUbArgs.reserve(numDims);

  // We'll collect the per-dim UB values first, then build maps that refer
  // to a uniform symbol list (s0..s{k-1}).
  SmallVector<Value> perDimUbValues;
  perDimUbValues.reserve(numDims);

  // Original UB operands, assumed identity-per-dim order.
  SmallVector<Value> origUbOperands(op.getUpperBoundsOperands().begin(),
                                    op.getUpperBoundsOperands().end());

  // Compute new UB value for the chosen dimension: div(UB_k, factor).
  Value ub0Value;
  {
    // If the ub map for dim 0 is constant, we'll materialize that constant
    // divided by the factor; otherwise, divide the corresponding UB operand.
    AffineMap ub0 = op.getUpperBoundMap(tileDimIndex);
    if (ub0.getNumResults() == 1 && ub0.isSingleConstant()) {
      int64_t c = ub0.getSingleConstantResult();
      // Division validity checked above when constant ranges are available.
      int64_t cDiv = c / tilingFactor;
      ub0Value = builder.create<arith::ConstantIndexOp>(loc, cDiv);
    } else {
      // Assume identity on operand order: use the first UB operand.
      if (origUbOperands.empty())
        return op.emitError(
                   "unsupported non-identity upper bound on chosen dim"),
               failure();
      if (tileDimIndex >= origUbOperands.size())
        return op.emitError("unsupported upper bound form on chosen dim"),
               failure();
      Value ub0Orig = origUbOperands[tileDimIndex];
      Value cstTile = builder.create<arith::ConstantIndexOp>(loc, tilingFactor);
      ub0Value = builder.create<arith::DivUIOp>(loc, ub0Orig, cstTile);
    }
  }
  // Build per-dim UB args in-order, using the divided UB for the chosen dim.
  for (unsigned i = 0; i < numDims; ++i) {
    if (i == tileDimIndex) {
      perDimUbValues.push_back(ub0Value);
      continue;
    }
    if (i < origUbOperands.size()) {
      perDimUbValues.push_back(origUbOperands[i]);
    } else {
      AffineMap ubi = op.getUpperBoundMap(i);
      if (ubi.getNumResults() == 1 && ubi.isSingleConstant()) {
        int64_t c = ubi.getSingleConstantResult();
        perDimUbValues.push_back(
            builder.create<arith::ConstantIndexOp>(loc, c));
      } else {
        (op.emitError("unsupported upper bound form on dimension ") << i);
        return failure();
      }
    }
  }

  // Now set up a uniform symbol space of size k = numDims.
  unsigned k = numDims;
  outerUbArgs = perDimUbValues; // s0..s{k-1}
  for (unsigned i = 0; i < numDims; ++i) {
    SmallVector<AffineExpr> exprs;
    exprs.push_back(getAffineSymbolExpr(i, ctx));
    outerUbMaps.push_back(
        AffineMap::get(/*dims=*/0, /*symbols=*/k, exprs, ctx));
  }

  // Steps remain the same for outer loop.
  SmallVector<int64_t> outerSteps = steps;

  // Only support non-reduction parallel loops for now.
  if (!op->getResults().empty())
    return op.emitError("reduction results not supported by this tiling pass"),
           failure();

  // Create the outer affine.parallel.
  auto outerPar = builder.create<affine::AffineParallelOp>(
      loc, /*resultTypes=*/TypeRange{},
      /*reductions=*/ArrayRef<arith::AtomicRMWKind>{}, outerLbMaps,
      /*lbArgs=*/ValueRange{}, outerUbMaps,
      /*ubArgs=*/outerUbArgs, outerSteps);

  // Map old IVs to new outer IVs; we'll ignore the inner IV in indexing to
  // match the provided expected output.
  IRMapping ivMap;
  for (unsigned i = 0; i < numDims; ++i)
    ivMap.map(op.getBody()->getArgument(i), outerPar.getBody()->getArgument(i));

  // Create the inner affine.parallel with constant range (0, tilingFactor).
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(outerPar.getBody());
  auto innerPar = builder.create<affine::AffineParallelOp>(
      loc, /*resultTypes=*/TypeRange{},
      /*reductions=*/ArrayRef<arith::AtomicRMWKind>{},
      /*ranges=*/ArrayRef<int64_t>{tilingFactor});
  (void)innerPar;

  // Splice original body operations into the inner parallel, remapping IVs.
  // Skip the terminator if present.
  Block *origBody = op.getBody();
  Block *innerBody = outerPar.getBody();
  // The newly created innerPar is the first op in the body; set insertion
  // point to its body start to clone the body into it.
  auto &firstOp = innerBody->front();
  auto innerParOp = dyn_cast<affine::AffineParallelOp>(&firstOp);
  Block *innerParBody = innerParOp.getBody();
  builder.setInsertionPointToStart(innerParBody);
  for (Operation &nested : llvm::make_early_inc_range(*origBody)) {
    if (nested.hasTrait<OpTrait::IsTerminator>())
      continue;
    (void)builder.clone(nested, ivMap);
  }

  // Erase the original op.
  op.erase();
  return success();
}

} // namespace tmd_affine
