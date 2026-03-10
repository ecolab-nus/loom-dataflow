/**
 * @file lcs_utils.cpp
 * @brief Implementation of tracing utilities for LCS analysis.
 */

#include "lcs_utils.h"
#include "utils.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"
#include <cassert>

namespace loom {
namespace lcs {

// ==========================================
// Internal Helpers
// ==========================================

namespace {

/// Helper to trace BlockArgument to its init value.
/// Returns Expr::none() if tracing fails.
Expr traceBlockArgumentToInit(mlir::BlockArgument blockArg) {
  using namespace mlir;
  Block *block = blockArg.getOwner();
  Operation *parentOp = block->getParentOp();

  // Handle affine.for: block args are (induction_var, iter_args...)
  if (auto forOp = mlir::dyn_cast<mlir::affine::AffineForOp>(parentOp)) {
    unsigned argIdx = blockArg.getArgNumber();
    auto inits = forOp.getInits();
    // iter_args start after the induction variable (argIdx 0)
    if (argIdx > 0 && argIdx - 1 < inits.size()) {
      return traceAllocDimsFromTensor(inits[argIdx - 1]);
    }
  }
  // Handle affine.parallel: all block args are iter_args (no induction var)
  else if (auto parOp =
               mlir::dyn_cast<mlir::affine::AffineParallelOp>(parentOp)) {
    unsigned argIdx = blockArg.getArgNumber();
    auto inits = parOp.getInits();
    if (argIdx < inits.size()) {
      return traceAllocDimsFromTensor(inits[argIdx]);
    }
  }

  return Expr::none();
}

} // namespace

// ==========================================
// Tracing Implementations
// ==========================================

loom::AllocOp traceToAlloc(mlir::Value memrefVal) {
  if (!memrefVal)
    return nullptr;

  auto op = memrefVal.getDefiningOp();
  if (!op)
    return nullptr;

  // If it's already an AllocOp, return it
  if (auto allocOp = llvm::dyn_cast<loom::AllocOp>(op)) {
    return allocOp;
  }

  // If it's a SemaphoreTakeOp, follow its source
  if (auto semTake = llvm::dyn_cast<loom::SemaphoreTakeOp>(op)) {
    return traceToAlloc(semTake.getSource());
  }

  return nullptr;
}

Expr formatAllocDims(loom::AllocOp allocOp) {
  if (!allocOp)
    return Expr::none();

  // Iterate through mixed sizes (static + dynamic), accumulate product.
  auto staticSizes = allocOp.getStaticSizes();
  auto dynamicSizes = allocOp.getSizes();

  Expr result = Expr::none();
  unsigned dynamicIdx = 0;

  auto accumulate = [&](Expr dim) {
    result = result.isNone() ? dim : result * dim;
  };

  for (int64_t staticDim : staticSizes) {
    if (mlir::ShapedType::isDynamic(staticDim)) {
      if (dynamicIdx < dynamicSizes.size()) {
        llvm::StringRef symVar =
            loom::utils::traceToSymbolicVar(dynamicSizes[dynamicIdx]);
        accumulate(symVar.empty() ? Expr::sym("?") : Expr::sym(symVar.str()));
        dynamicIdx++;
      }
    } else {
      accumulate(Expr::con(staticDim));
    }
  }

  return result;
}

Expr traceAllocDimsFromTensor(mlir::Value tensorVal) {
  using namespace mlir;

  if (!tensorVal)
    return Expr::none();

  // Handle BlockArgument first (before getDefiningOp which returns nullptr)
  if (auto blockArg = dyn_cast<BlockArgument>(tensorVal)) {
    return traceBlockArgumentToInit(blockArg);
  }

  // Now handle OpResults (from operations)
  Operation *op = tensorVal.getDefiningOp();
  if (!op)
    return Expr::none();

  // Case 1: loom.copy_to_tensor
  if (auto copyOp = dyn_cast<loom::CopyToTensorOp>(op)) {
    if (auto allocOp = traceToAlloc(copyOp.getBuffer())) {
      return formatAllocDims(allocOp);
    }
    return Expr::none();
  }

  // Case 2: loom.init_tensor
  if (auto initTensor = dyn_cast<loom::InitTensorOp>(op)) {
    if (auto allocOp = traceToAlloc(initTensor.getBuffer())) {
      return formatAllocDims(allocOp);
    }
    return Expr::none();
  }

  // Case 3: linalg operation (fill, copy, generic, matmul, etc.)
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    auto inits = linalgOp.getDpsInits();
    if (!inits.empty()) {
      unsigned resultIdx = cast<OpResult>(tensorVal).getResultNumber();
      if (resultIdx < inits.size()) {
        return traceAllocDimsFromTensor(inits[resultIdx]);
      }
      // If resultIdx >= size, try tracing the first init as fallback
      return traceAllocDimsFromTensor(inits[0]);
    }
    return Expr::none();
  }

  // Case 4: affine.for result
  if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
    unsigned resultIdx = cast<OpResult>(tensorVal).getResultNumber();
    if (resultIdx < forOp.getInits().size()) {
      return traceAllocDimsFromTensor(forOp.getInits()[resultIdx]);
    }
    return Expr::none();
  }

  return Expr::none();
}

std::string formatElementType(mlir::Type elemType) {
  std::string typeStr;
  llvm::raw_string_ostream os(typeStr);
  elemType.print(os);
  os.flush();
  return typeStr;
}

Expr affineExprToExpr(mlir::AffineExpr expr,
                      const llvm::SmallVector<std::string> &symbolNames) {
  using namespace mlir;

  if (!expr)
    return Expr::none();

  switch (expr.getKind()) {
  case AffineExprKind::Constant: {
    auto constExpr = mlir::cast<mlir::AffineConstantExpr>(expr);
    return Expr::con(constExpr.getValue());
  }
  case AffineExprKind::SymbolId: {
    auto symExpr = mlir::cast<mlir::AffineSymbolExpr>(expr);
    unsigned symId = symExpr.getPosition();
    if (symId < symbolNames.size() && !symbolNames[symId].empty()) {
      return Expr::sym(symbolNames[symId]);
    }
    return Expr::sym("s" + std::to_string(symId));
  }
  case AffineExprKind::DimId: {
    auto dimExpr = mlir::cast<mlir::AffineDimExpr>(expr);
    unsigned dimId = dimExpr.getPosition();
    return Expr::sym("d" + std::to_string(dimId));
  }
  case AffineExprKind::Add: {
    auto binExpr = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
    return affineExprToExpr(binExpr.getLHS(), symbolNames) +
           affineExprToExpr(binExpr.getRHS(), symbolNames);
  }
  case AffineExprKind::Mul: {
    auto binExpr = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
    return affineExprToExpr(binExpr.getLHS(), symbolNames) *
           affineExprToExpr(binExpr.getRHS(), symbolNames);
  }
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv: {
    auto binExpr = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
    return affineExprToExpr(binExpr.getLHS(), symbolNames) /
           affineExprToExpr(binExpr.getRHS(), symbolNames);
  }
  case AffineExprKind::Mod:
    // AffineExprKind::Mod has no direct Expr equivalent; return none.
    return Expr::none();
  }
  return Expr::none();
}

/// Count the number of CeilDiv nodes anywhere in an AffineExpr tree.
static unsigned countCeilDivs(mlir::AffineExpr expr) {
  if (!expr)
    return 0;
  if (expr.getKind() == mlir::AffineExprKind::CeilDiv) {
    auto bin = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
    return 1 + countCeilDivs(bin.getLHS()) + countCeilDivs(bin.getRHS());
  }
  if (auto bin = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expr))
    return countCeilDivs(bin.getLHS()) + countCeilDivs(bin.getRHS());
  return 0;
}

Expr extractLoopTripCount(mlir::affine::AffineForOp forOp) {
  using namespace mlir;

  if (!forOp)
    return Expr::none();

  // Get the upper bound map
  auto upperBoundMap = forOp.getUpperBoundMap();
  if (upperBoundMap.getNumResults() != 1)
    return Expr::none();

  // Get upper bound operands
  auto upperBoundOperands = forOp.getUpperBoundOperands();

  // Build symbol names from operands using traceToSymbolicVar
  llvm::SmallVector<std::string> symbolNames;
  for (auto operand : upperBoundOperands) {
    llvm::StringRef symName = loom::utils::traceToSymbolicVar(operand);
    symbolNames.push_back(std::string(symName));
  }

  // Expect at most one ceildiv after the flattening pass.
  auto resultExpr = upperBoundMap.getResult(0);
  assert(countCeilDivs(resultExpr) <= 1 &&
         "UB affine expression must contain at most one ceildiv; "
         "run flattenCeilDivInForBounds first");

  return affineExprToExpr(resultExpr, symbolNames);
}

} // namespace lcs
} // namespace loom
