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
/// Returns empty string if tracing fails.
std::string traceBlockArgumentToInit(mlir::BlockArgument blockArg) {
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
  else if (auto parOp = mlir::dyn_cast<mlir::affine::AffineParallelOp>(parentOp)) {
    unsigned argIdx = blockArg.getArgNumber();
    auto inits = parOp.getInits();
    if (argIdx < inits.size()) {
      return traceAllocDimsFromTensor(inits[argIdx]);
    }
  }

  return "";
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

std::string formatAllocDims(loom::AllocOp allocOp) {
  if (!allocOp)
    return "";

  llvm::SmallVector<std::string> dimStrs;

  // Iterate through mixed sizes (static + dynamic)
  auto staticSizes = allocOp.getStaticSizes();
  auto dynamicSizes = allocOp.getSizes();

  unsigned dynamicIdx = 0;
  for (int64_t staticDim : staticSizes) {
    if (mlir::ShapedType::isDynamic(staticDim)) {
      if (dynamicIdx < dynamicSizes.size()) {
        llvm::StringRef symVar =
            loom::utils::traceToSymbolicVar(dynamicSizes[dynamicIdx]);
        if (!symVar.empty()) {
          dimStrs.push_back(std::string(symVar));
        } else {
          dimStrs.push_back("?");
        }
        dynamicIdx++;
      }
    } else {
      dimStrs.push_back(std::to_string(staticDim));
    }
  }

  // Join with " * "
  std::string result;
  for (size_t i = 0; i < dimStrs.size(); ++i) {
    if (i > 0)
      result += " * ";
    result += dimStrs[i];
  }
  return result;
}

std::string traceAllocDimsFromTensor(mlir::Value tensorVal) {
  using namespace mlir;

  if (!tensorVal)
    return "";

  // Handle BlockArgument first (before getDefiningOp which returns nullptr)
  if (auto blockArg = dyn_cast<BlockArgument>(tensorVal)) {
    return traceBlockArgumentToInit(blockArg);
  }

  // Now handle OpResults (from operations)
  Operation *op = tensorVal.getDefiningOp();
  if (!op)
    return "";

  // Case 1: loom.copy_to_tensor
  if (auto copyOp = dyn_cast<loom::CopyToTensorOp>(op)) {
    if (auto allocOp = traceToAlloc(copyOp.getBuffer())) {
      return formatAllocDims(allocOp);
    }
    return "";
  }

  // Case 2: loom.init_tensor
  if (auto initTensor = dyn_cast<loom::InitTensorOp>(op)) {
    if (auto allocOp = traceToAlloc(initTensor.getBuffer())) {
      return formatAllocDims(allocOp);
    }
    return "";
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
    return "";
  }

  // Case 4: affine.for result
  if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
    unsigned resultIdx = cast<OpResult>(tensorVal).getResultNumber();
    if (resultIdx < forOp.getInits().size()) {
      return traceAllocDimsFromTensor(forOp.getInits()[resultIdx]);
    }
    return "";
  }

  return "";
}

std::string formatElementType(mlir::Type elemType) {
  std::string typeStr;
  llvm::raw_string_ostream os(typeStr);
  elemType.print(os);
  os.flush();
  return typeStr;
}

std::string affineExprToSymbolicString(
    mlir::AffineExpr expr,
    const llvm::SmallVector<std::string> &symbolNames) {
  using namespace mlir;

  if (!expr)
    return "";

  switch (expr.getKind()) {
  case AffineExprKind::Constant: {
    auto constExpr = mlir::cast<mlir::AffineConstantExpr>(expr);
    return std::to_string(constExpr.getValue());
  }
  case AffineExprKind::SymbolId: {
    auto symExpr = mlir::cast<mlir::AffineSymbolExpr>(expr);
    unsigned symId = symExpr.getPosition();
    if (symId < symbolNames.size()) {
      return symbolNames[symId];
    }
    return "s" + std::to_string(symId);
  }
  case AffineExprKind::DimId: {
    auto dimExpr = mlir::cast<mlir::AffineDimExpr>(expr);
    unsigned dimId = dimExpr.getPosition();
    return "d" + std::to_string(dimId);
  }
  case AffineExprKind::Add: {
    auto binExpr = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
    auto lhs = affineExprToSymbolicString(binExpr.getLHS(), symbolNames);
    auto rhs = affineExprToSymbolicString(binExpr.getRHS(), symbolNames);
    return lhs + " + " + rhs;
  }
  case AffineExprKind::Mul: {
    auto binExpr = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
    auto lhs = affineExprToSymbolicString(binExpr.getLHS(), symbolNames);
    auto rhs = affineExprToSymbolicString(binExpr.getRHS(), symbolNames);
    return lhs + " * " + rhs;
  }
  case AffineExprKind::FloorDiv: {
    auto binExpr = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
    auto lhs = affineExprToSymbolicString(binExpr.getLHS(), symbolNames);
    auto rhs = affineExprToSymbolicString(binExpr.getRHS(), symbolNames);
    return lhs + " / " + rhs;
  }
  case AffineExprKind::CeilDiv: {
    auto binExpr = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
    auto lhs = affineExprToSymbolicString(binExpr.getLHS(), symbolNames);
    auto rhs = affineExprToSymbolicString(binExpr.getRHS(), symbolNames);
    // Wrap the denominator in parentheses when it is a product so that
    // e.g.  "32 ceildiv (s0 * 64)"  is unambiguous.
    if (binExpr.getRHS().getKind() == AffineExprKind::Mul)
      return lhs + " / (" + rhs + ")";
    return lhs + " / " + rhs;
  }
  case AffineExprKind::Mod: {
    auto binExpr = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
    auto lhs = affineExprToSymbolicString(binExpr.getLHS(), symbolNames);
    auto rhs = affineExprToSymbolicString(binExpr.getRHS(), symbolNames);
    return lhs + " % " + rhs;
  }
  }
  return "";
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

std::string extractLoopTripCount(mlir::affine::AffineForOp forOp) {
  using namespace mlir;

  if (!forOp)
    return "";

  // Get the upper bound map
  auto upperBoundMap = forOp.getUpperBoundMap();
  if (upperBoundMap.getNumResults() != 1)
    return "";

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

  // Convert the result expression to symbolic string
  return affineExprToSymbolicString(resultExpr, symbolNames);
}

} // namespace lcs
} // namespace loom
