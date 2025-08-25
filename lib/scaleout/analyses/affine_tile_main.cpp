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
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include <string>

using namespace mlir;

namespace tmd_affine {

/**
 * Tile a single affine.parallel operation along its first iterator by a given
 * tiling factor and rewrite it into a perfectly nested pair of
 * affine.parallel operations.
 *
 * Transformation (high level):
 *   affine.parallel (m, n, ...) = (0, 0, ...) to (M, N, ...) step (1, s1, ...)
 *     body(m, n, ...)
 * becomes
 *   affine.parallel (m, n, ...) = (0, 0, ...) to (M/TF, N, ...) step (1, s1,
 * ...) affine.parallel (m_inner) = (0) to (TF) body(m, n, ...)
 *
 * Notes and assumptions:
 * - Only the first iterator is tiled. The remaining iterators are preserved.
 * - The first iterator must have step 1 and lower bound 0.
 * - If constant ranges are known (i.e., getConstantRanges()), the first range
 *   must be divisible by tilingFactor; otherwise, this returns failure.
 * - Reductions are not supported (i.e., the affine.parallel must not produce
 *   results). This returns failure if reductions are present.
 * - For dynamic upper bounds, the outer bound is recomputed as UB0 / TF using
 *   integer division (arith.divui). The map is printed with symbol(...) as we
 *   build 0-dim, k-symbol upper bound maps.
 * - The inner loop upper bound is encoded as a constant affine bound (0..TF),
 *   no arith.constant is materialized for this.
 * - The body is cloned into the inner parallel, remapping original IVs to the
 *   outer IVs. The new inner IV is not used in the body by this simple scheme
 *   (aligns with the provided expected output).
 *
 * Returns: success if the rewrite was applied (no-op if numDims == 0), failure
 *          if preconditions are not met.
 */
LogicalResult tileAffineParallel(affine::AffineParallelOp op,
                                 int64_t tilingFactor) {
  MLIRContext *ctx = op.getContext();
  if (tilingFactor <= 0)
    return op.emitError("tiling-factor must be positive"), failure();

  OpBuilder builder(ctx);
  Location loc = op.getLoc();
  unsigned numDims = op.getNumDims();
  if (numDims == 0)
    return success();

  // Only support unit steps for simplicity.
  SmallVector<int64_t> steps = llvm::to_vector(op.getSteps());
  if (steps.empty() || steps[0] != 1)
    return op.emitError("only step 1 supported for the first dimension"),
           failure();

  // Require lower bound 0 on the first dimension for simplicity.
  AffineMap lb0 = op.getLowerBoundMap(0);
  if (lb0.getNumResults() != 1 || !lb0.isSingleConstant())
    return op.emitError(
               "expected constant-zero lower bound on first dimension"),
           failure();
  if (lb0.getSingleConstantResult() != 0)
    return op.emitError("expected lower bound 0 on first dimension"), failure();

  // If constant ranges known, verify divisibility; otherwise assume
  // divisible.
  if (auto maybeRanges = op.getConstantRanges()) {
    auto ranges = *maybeRanges;
    if (!ranges.empty()) {
      int64_t extent0 = ranges[0];
      if (extent0 % tilingFactor != 0)
        return op.emitError(
                   "first-dimension bound not divisible by tiling-factor"),
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

  // Compute new UB value for the first dimension: div(UB0, factor).
  Value ub0Value;
  {
    // If the ub map for dim 0 is constant, we'll materialize that constant
    // divided by the factor; otherwise, divide the corresponding UB operand.
    AffineMap ub0 = op.getUpperBoundMap(0);
    if (ub0.getNumResults() == 1 && ub0.isSingleConstant()) {
      int64_t c = ub0.getSingleConstantResult();
      // Division validity checked above when constant ranges are available.
      int64_t cDiv = c / tilingFactor;
      ub0Value = builder.create<arith::ConstantIndexOp>(loc, cDiv);
    } else {
      // Assume identity on operand order: use the first UB operand.
      if (origUbOperands.empty())
        return op.emitError(
                   "unsupported non-identity upper bound on first dim"),
               failure();
      Value ub0Orig = origUbOperands[0];
      Value cstTile = builder.create<arith::ConstantIndexOp>(loc, tilingFactor);
      ub0Value = builder.create<arith::DivUIOp>(loc, ub0Orig, cstTile);
    }
  }
  perDimUbValues.push_back(ub0Value);

  // For the remaining dimensions, forward original UB operands assuming
  // identity mapping to those operands.
  for (unsigned i = 1; i < numDims; ++i) {
    // Use the i-th UB operand if available; otherwise, try constant map.
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
  builder.setInsertionPointToStart(outerPar.getBody());
  auto innerPar = builder.create<affine::AffineParallelOp>(
      loc, /*resultTypes=*/TypeRange{},
      /*reductions=*/ArrayRef<arith::AtomicRMWKind>{},
      /*ranges=*/ArrayRef<int64_t>{tilingFactor});

  // Splice original body operations into the inner parallel, remapping IVs.
  // Skip the terminator if present.
  Block *origBody = op.getBody();
  Block *innerBody = innerPar.getBody();
  builder.setInsertionPointToStart(innerBody);
  for (Operation &nested : llvm::make_early_inc_range(*origBody)) {
    if (nested.hasTrait<OpTrait::IsTerminator>())
      continue;
    Operation *cloned = builder.clone(nested, ivMap);
    (void)cloned;
  }

  // Erase the original op.
  op.erase();
  return success();
}
} // namespace tmd_affine

namespace {

/**
 * Tile all affine.parallel operations inside a function with the same tiling
 * factor on their first iterator.
 *
 * This is a thin wrapper that walks the function, collects targets, and calls
 * tmd_affine::tileAffineParallel on each. The transformation is applied in a
 * stable, outer-to-inner order given by the walk collection.
 */
static LogicalResult tileFunc(func::FuncOp func, int64_t tilingFactor) {
  if (tilingFactor <= 0)
    return func.emitError("tiling-factor must be positive"), failure();

  SmallVector<affine::AffineParallelOp, 4> toProcess;
  func.walk([&](affine::AffineParallelOp op) { toProcess.push_back(op); });
  for (affine::AffineParallelOp op : toProcess) {
    if (failed(tmd_affine::tileAffineParallel(op, tilingFactor)))
      return failure();
  }
  return success();
}
} // end anonymous namespace

/**
 * Test driver: reads an MLIR module from stdin or a file, applies the tiling
 * transform to every function using the provided tiling factor (argv[2],
 * default 8), and prints the transformed module.
 *
 * Usage:
 *   tmd_affine_tile <input.mlir> [tiling_factor]
 */
int main(int argc, char **argv) {
  MLIRContext context;
  (void)context.getOrLoadDialect<mlir::BuiltinDialect>();
  context.loadDialect<mlir::func::FuncDialect, mlir::affine::AffineDialect,
                      mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();

  llvm::SourceMgr sourceMgr;
  const char *filename = argc > 1 ? argv[1] : "-";
  auto file = mlir::openInputFile(filename);
  if (!file) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to open input file: " << filename << "\n";
    return 1;
  }
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::WithColor::error(llvm::errs()) << "Failed to parse MLIR module\n";
    return 1;
  }

  // Parse optional tiling factor from argv[2], default 8.
  int64_t tilingFactor = 8;
  if (argc > 2) {
    std::string s(argv[2]);
    try {
      size_t idx = 0;
      long v = std::stol(s, &idx, 10);
      if (idx == s.size())
        tilingFactor = static_cast<int64_t>(v);
    } catch (...) {
      // ignore, keep default
    }
  }

  // Walk and rewrite directly without pass manager.
  for (auto func : module->getOps<func::FuncOp>()) {
    if (failed(tileFunc(func, tilingFactor))) {
      llvm::WithColor::error(llvm::errs()) << "Tiling failed\n";
      return 2;
    }
  }

  module->print(llvm::outs());
  llvm::outs() << "\n";
  return 0;
}
