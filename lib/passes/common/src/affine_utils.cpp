#include "affine_utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace loom_affine {

// --- Tiling ---

LogicalResult tileAffineParallel(affine::AffineParallelOp original_parop,
                                 int64_t tilingFactor, unsigned tileDimIndex,
                                 TiledParallels &result) {
  MLIRContext *ctx = original_parop.getContext();
  if (tilingFactor <= 0)
    return original_parop.emitError("tiling-factor must be positive"),
           failure();

  OpBuilder builder(ctx);
  Location loc = original_parop.getLoc();
  unsigned numDims = original_parop.getNumDims();
  if (numDims == 0)
    return success();
  if (tileDimIndex >= numDims)
    return original_parop.emitError("tileDimIndex out of range"), failure();

  SmallVector<int64_t> steps = llvm::to_vector(original_parop.getSteps());
  if (steps.empty() || steps[tileDimIndex] != 1)
    return original_parop.emitError(
               "only step 1 supported on the chosen dimension"),
           failure();

  AffineMap lbChosen = original_parop.getLowerBoundMap(tileDimIndex);
  if (lbChosen.getNumResults() != 1 || !lbChosen.isSingleConstant())
    return original_parop.emitError(
               "expected constant-zero lower bound on chosen dimension"),
           failure();
  if (lbChosen.getSingleConstantResult() != 0)
    return original_parop.emitError(
               "expected lower bound 0 on chosen dimension"),
           failure();

  if (auto maybeRanges = original_parop.getConstantRanges()) {
    auto ranges = *maybeRanges;
    if (!ranges.empty()) {
      if (tileDimIndex >= ranges.size())
        return original_parop.emitError(
                   "range metadata missing for chosen dimension"),
               failure();
      int64_t extentK = ranges[tileDimIndex];
      if (extentK % tilingFactor != 0)
        return original_parop.emitError(
                   "chosen-dimension bound not divisible by tiling-factor"),
               failure();
    }
  }

  builder.setInsertionPoint(original_parop);
  IRMapping ivMap;

  SmallVector<AffineMap> tiled_new_lb_maps = {builder.getConstantAffineMap(0)};
  SmallVector<AffineMap> tiled_new_ub_maps = {
      builder.getConstantAffineMap(static_cast<int64_t>(tilingFactor))};
  SmallVector<int64_t> tiled_new_steps = {1};
  auto tiled_new = affine::AffineParallelOp::create(
      builder, loc, TypeRange{}, ArrayRef<arith::AtomicRMWKind>{},
      tiled_new_lb_maps, ValueRange{}, tiled_new_ub_maps, ValueRange{},
      tiled_new_steps);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(tiled_new.getBody());
  SmallVector<AffineMap> tiled_org_lb_maps;
  SmallVector<AffineMap> tiled_org_ub_maps;
  tiled_org_lb_maps.reserve(numDims);
  tiled_org_ub_maps.reserve(numDims);
  for (unsigned i = 0; i < numDims; ++i) {
    tiled_org_lb_maps.push_back(original_parop.getLowerBoundMap(i));
    AffineMap ubI = original_parop.getUpperBoundMap(i);
    if (i == tileDimIndex) {
      if (ubI.getNumResults() != 1)
        return original_parop.emitError("expected single-result UB map"),
               failure();
      AffineExpr e = ubI.getResult(0);
      AffineExpr transformed = e.ceilDiv(tilingFactor);
      tiled_org_ub_maps.push_back(
          AffineMap::get(ubI.getNumDims(), ubI.getNumSymbols(), transformed));
    } else {
      tiled_org_ub_maps.push_back(ubI);
    }
  }

  ValueRange org_lb_args = original_parop.getLowerBoundsOperands();
  ValueRange org_ub_args = original_parop.getUpperBoundsOperands();
  SmallVector<int64_t> org_steps = steps;

  auto tiled_org = affine::AffineParallelOp::create(
      builder, loc, TypeRange{}, ArrayRef<arith::AtomicRMWKind>{},
      tiled_org_lb_maps, org_lb_args, tiled_org_ub_maps, org_ub_args,
      org_steps);

  Block *org_body = original_parop.getBody();
  Block *tiled_org_body = tiled_org.getBody();

  for (unsigned i = 0; i < numDims; ++i) {
    org_body->getArgument(i).replaceAllUsesWith(tiled_org_body->getArgument(i));
  }

  tiled_org_body->getOperations().clear();
  tiled_org_body->getOperations().splice(tiled_org_body->begin(),
                                         org_body->getOperations());

  TilingMetadata metadata;
  metadata.originalDimIdx = tileDimIndex;
  metadata.tilingFactor = tilingFactor;
  metadata.outerIV = tiled_new.getBody()->getArgument(0);
  metadata.innerIV = tiled_org.getBody()->getArgument(tileDimIndex);

  result.tiled_org_ = tiled_org;
  result.tiled_new_ = tiled_new;
  result.tilingMetadata.push_back(metadata);

  original_parop.erase();
  return success();
}

// --- Parallel to Nested For ---

static AffineExpr remapDimsToSymbols(
    AffineExpr expr, const llvm::DenseMap<unsigned, unsigned> &dimToSymMap,
    const llvm::DenseMap<unsigned, unsigned> &dimRemapMap, MLIRContext *ctx) {
  if (!expr)
    return expr;

  std::function<AffineExpr(AffineExpr)> remap =
      [&](AffineExpr e) -> AffineExpr {
    switch (e.getKind()) {
    case AffineExprKind::DimId: {
      auto dim = llvm::cast<AffineDimExpr>(e);
      unsigned oldPos = dim.getPosition();
      auto symIt = dimToSymMap.find(oldPos);
      if (symIt != dimToSymMap.end()) {
        return getAffineSymbolExpr(symIt->second, ctx);
      }
      auto dimIt = dimRemapMap.find(oldPos);
      if (dimIt != dimRemapMap.end()) {
        return getAffineDimExpr(dimIt->second, ctx);
      }
      return e;
    }
    case AffineExprKind::SymbolId:
    case AffineExprKind::Constant:
      return e;
    case AffineExprKind::Add: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return remap(bin.getLHS()) + remap(bin.getRHS());
    }
    case AffineExprKind::Mul: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return remap(bin.getLHS()) * remap(bin.getRHS());
    }
    case AffineExprKind::Mod: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return remap(bin.getLHS()) % remap(bin.getRHS());
    }
    case AffineExprKind::FloorDiv: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return remap(bin.getLHS()).floorDiv(remap(bin.getRHS()));
    }
    case AffineExprKind::CeilDiv: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return remap(bin.getLHS()).ceilDiv(remap(bin.getRHS()));
    }
    }
    return e;
  };
  return remap(expr);
}

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

LogicalResult ConvertParallelToNested(affine::AffineParallelOp par,
                                      ArrayRef<unsigned> order) {
  if (par.getNumResults() != 0)
    return failure();
  const unsigned P = par.getNumDims();
  if (order.size() != P)
    return failure();
  SmallVector<unsigned> seen(P, 0);
  for (unsigned v : order) {
    if (v >= P || seen[v])
      return failure();
    seen[v] = 1;
  }

  OpBuilder builder(par);
  Location loc = par.getLoc();
  SmallVector<affine::AffineForOp> newFors;
  newFors.reserve(P);
  affine::AffineForOp innermostFor = nullptr;
  func::FuncOp parentFunc = par->getParentOfType<func::FuncOp>();

  for (unsigned i = 0; i < P; ++i) {
    unsigned iterIdx = order[i];
    AffineMap lbMap, ubMap;
    SmallVector<Value> lbOperands, ubOperands;
    int64_t stepVal = 1;
    if (failed(getIteratorBoundsAndStep(par, iterIdx, lbMap, lbOperands, ubMap,
                                        ubOperands, stepVal)))
      return failure();

    SmallVector<Value> ubDimOperands, ubSymOperands;
    llvm::DenseMap<unsigned, unsigned> dimToSymMap;
    llvm::DenseMap<unsigned, unsigned> dimRemapMap;

    if (parentFunc) {
      unsigned newDimIdx = 0;
      unsigned newSymIdx = 0;
      for (unsigned j = 0; j < ubOperands.size(); ++j) {
        Value operand = ubOperands[j];
        if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
          if (blockArg.getOwner() == &parentFunc.getBody().front()) {
            dimToSymMap[j] = newSymIdx++;
            ubSymOperands.push_back(operand);
            continue;
          }
        }
        dimRemapMap[j] = newDimIdx++;
        ubDimOperands.push_back(operand);
      }

      if (!ubSymOperands.empty()) {
        MLIRContext *ctx = ubMap.getContext();
        SmallVector<AffineExpr, 4> newResults;
        for (AffineExpr expr : ubMap.getResults()) {
          newResults.push_back(
              remapDimsToSymbols(expr, dimToSymMap, dimRemapMap, ctx));
        }
        ubMap = AffineMap::get(ubDimOperands.size(), ubSymOperands.size(),
                               newResults, ctx);
        ubOperands.clear();
        ubOperands.append(ubDimOperands.begin(), ubDimOperands.end());
        ubOperands.append(ubSymOperands.begin(), ubSymOperands.end());
      }
    }

    if (i == 0)
      builder.setInsertionPoint(par);
    else
      builder.setInsertionPointToStart(newFors.back().getBody());

    affine::AffineForOp forOp = affine::AffineForOp::create(
        builder, loc, lbOperands, lbMap, ubOperands, ubMap, stepVal);
    if (forOp.getBody()->empty() ||
        !isa<affine::AffineYieldOp>(forOp.getBody()->back())) {
      OpBuilder termBuilder = OpBuilder::atBlockEnd(forOp.getBody());
      affine::AffineYieldOp::create(termBuilder, loc);
    }
    newFors.push_back(forOp);
    innermostFor = forOp;
  }

  IRMapping mapping;
  SmallVector<Value> forIVForIter(P);
  for (unsigned depth = 0; depth < P; ++depth) {
    forIVForIter[order[depth]] = newFors[depth].getInductionVar();
  }
  for (unsigned iterIdx = 0; iterIdx < P; ++iterIdx)
    mapping.map(par.getIVs()[iterIdx], forIVForIter[iterIdx]);

  Block &srcBody = par.getRegion().front();
  builder.setInsertionPointToStart(innermostFor.getBody());
  for (Operation &op : srcBody) {
    if (!isa<affine::AffineYieldOp>(op))
      builder.clone(op, mapping);
  }

  par.erase();
  return success();
}

// --- Flatten nested ceildiv ---

AffineExpr flattenNestedCeilDiv(AffineExpr expr) {
  if (!expr)
    return expr;
  if (expr.getKind() != AffineExprKind::CeilDiv)
    return expr;

  auto bin = llvm::cast<AffineBinaryOpExpr>(expr);
  // Recursively flatten the numerator side first.
  AffineExpr lhs = flattenNestedCeilDiv(bin.getLHS());
  AffineExpr rhs = bin.getRHS();

  // If the (possibly-flattened) numerator is itself a CeilDiv, merge:
  //   (N ceildiv A) ceildiv B  -->  N ceildiv (A * B)
  // MLIR's AffineExpr multiplication folds constants automatically,
  // so  (s0 * 8) * 8  becomes  s0 * 64.
  if (lhs.getKind() == AffineExprKind::CeilDiv) {
    auto innerBin = llvm::cast<AffineBinaryOpExpr>(lhs);
    AffineExpr combinedDenom = innerBin.getRHS() * rhs;
    return innerBin.getLHS().ceilDiv(combinedDenom);
  }

  return lhs.ceilDiv(rhs);
}

void flattenCeilDivInForBounds(func::FuncOp func) {
  func.walk([](affine::AffineForOp forOp) {
    AffineMap ubMap = forOp.getUpperBoundMap();
    bool changed = false;
    SmallVector<AffineExpr> newExprs;
    newExprs.reserve(ubMap.getNumResults());
    for (AffineExpr expr : ubMap.getResults()) {
      AffineExpr flattened = flattenNestedCeilDiv(expr);
      newExprs.push_back(flattened);
      if (flattened != expr)
        changed = true;
    }
    if (changed) {
      AffineMap newMap =
          AffineMap::get(ubMap.getNumDims(), ubMap.getNumSymbols(), newExprs,
                         ubMap.getContext());
      forOp.setUpperBoundMap(newMap);
    }
  });
}

} // namespace loom_affine
