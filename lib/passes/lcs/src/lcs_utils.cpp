/**
 * @file lcs_utils.cpp
 * @brief Implementation of tracing utilities for LCS analysis.
 */

#include "lcs_utils.h"
#include "utils.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <cassert>

namespace loom {
namespace lcs {

// ==========================================
// Internal Helpers
// ==========================================

namespace {

/// Helper to trace BlockArgument to its init value.
/// Returns empty vector if tracing fails.
std::vector<Expr> traceBlockArgumentToInit(mlir::BlockArgument blockArg) {
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
  // Handle scf.for: block args are (induction_var, iter_args...) — same layout
  else if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(parentOp)) {
    unsigned argIdx = blockArg.getArgNumber();
    auto initArgs = forOp.getInitArgs();
    if (argIdx > 0 && argIdx - 1 < initArgs.size()) {
      return traceAllocDimsFromTensor(initArgs[argIdx - 1]);
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

  return {};
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

std::vector<Expr> formatAllocDims(loom::AllocOp allocOp) {
  std::vector<Expr> dims;
  if (!allocOp)
    return dims;

  // Iterate through mixed sizes (static + dynamic), one Expr per dimension.
  auto staticSizes = allocOp.getStaticSizes();
  auto dynamicSizes = allocOp.getSizes();
  unsigned dynamicIdx = 0;

  for (int64_t staticDim : staticSizes) {
    if (mlir::ShapedType::isDynamic(staticDim)) {
      if (dynamicIdx < dynamicSizes.size()) {
        llvm::StringRef symVar =
            loom::utils::traceToSymbolicVar(dynamicSizes[dynamicIdx]);
        dims.push_back(symVar.empty() ? Expr::sym("?")
                                      : Expr::sym(symVar.str()));
        dynamicIdx++;
      }
    } else {
      dims.push_back(Expr::con(staticDim));
    }
  }

  return dims;
}

std::vector<Expr> traceAllocDimsFromTensor(mlir::Value tensorVal) {
  using namespace mlir;

  if (!tensorVal)
    return {};

  // Handle BlockArgument first (before getDefiningOp which returns nullptr)
  if (auto blockArg = dyn_cast<BlockArgument>(tensorVal)) {
    return traceBlockArgumentToInit(blockArg);
  }

  // Now handle OpResults (from operations)
  Operation *op = tensorVal.getDefiningOp();
  if (!op)
    return {};

  // Case 1: loom.copy_to_tensor
  if (auto copyOp = dyn_cast<loom::CopyToTensorOp>(op)) {
    if (auto allocOp = traceToAlloc(copyOp.getBuffer())) {
      return formatAllocDims(allocOp);
    }
    return {};
  }

  // Case 1b: loom.bufferize_to_tensor — reinterprets an L1 memref as a tensor
  if (auto bufOp = dyn_cast<loom::BufferizeToTensorOp>(op)) {
    if (auto allocOp = traceToAlloc(bufOp.getSource()))
      return formatAllocDims(allocOp);
    return {};
  }

  // Case 2: loom.init_tensor
  if (auto initTensor = dyn_cast<loom::InitTensorOp>(op)) {
    if (auto allocOp = traceToAlloc(initTensor.getBuffer())) {
      return formatAllocDims(allocOp);
    }
    return {};
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
    return {};
  }

  // Case 4: affine.for result
  if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
    unsigned resultIdx = cast<OpResult>(tensorVal).getResultNumber();
    if (resultIdx < forOp.getInits().size()) {
      return traceAllocDimsFromTensor(forOp.getInits()[resultIdx]);
    }
    return {};
  }

  // Case 4b: scf.for result — trace back to the corresponding init arg
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    unsigned resultIdx = cast<OpResult>(tensorVal).getResultNumber();
    auto initArgs = forOp.getInitArgs();
    if (resultIdx < initArgs.size()) {
      return traceAllocDimsFromTensor(initArgs[resultIdx]);
    }
    return {};
  }

  return {};
}

Expr productOfDims(const std::vector<Expr> &dims) {
  Expr result = Expr::none();
  for (const auto &d : dims) {
    result = result.isNone() ? d : result * d;
  }
  return result;
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

/// Recursively trace an SSA index value to a symbolic Expr.
/// Handles: loom.sym → Sym, arith.constant → Const,
///          arith.ceildivui/si → Div, arith.muli → Mul, arith.addi → Add.
static Expr traceIndexValueToExpr(mlir::Value val) {
  using namespace mlir;
  if (!val)
    return Expr::none();

  // First try to resolve directly to a named symbolic variable (loom.sym).
  llvm::StringRef symName = loom::utils::traceToSymbolicVar(val);
  if (!symName.empty())
    return Expr::sym(symName.str());

  Operation *op = val.getDefiningOp();
  if (!op)
    return Expr::none();

  // arith.constant
  if (auto constOp = dyn_cast<arith::ConstantOp>(op))
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return Expr::con(intAttr.getInt());

  // arith.ceildivui / arith.ceildivsi → Div (same semantics for trip counts)
  if (auto cdiv = dyn_cast<arith::CeilDivUIOp>(op))
    return traceIndexValueToExpr(cdiv.getLhs()) /
           traceIndexValueToExpr(cdiv.getRhs());
  if (auto cdiv = dyn_cast<arith::CeilDivSIOp>(op))
    return traceIndexValueToExpr(cdiv.getLhs()) /
           traceIndexValueToExpr(cdiv.getRhs());

  // arith.muli → Mul
  if (auto mul = dyn_cast<arith::MulIOp>(op))
    return traceIndexValueToExpr(mul.getLhs()) *
           traceIndexValueToExpr(mul.getRhs());

  // arith.addi → Add
  if (auto add = dyn_cast<arith::AddIOp>(op))
    return traceIndexValueToExpr(add.getLhs()) +
           traceIndexValueToExpr(add.getRhs());

  return Expr::none();
}

Expr extractLoopTripCount(mlir::scf::ForOp forOp) {
  if (!forOp)
    return Expr::none();
  // Assumes lb=0 and step=1, so trip count == upper bound.
  return traceIndexValueToExpr(forOp.getUpperBound());
}

// ==========================================
// Generic Op Classification & Shape Analysis
// ==========================================

GenericClass classifyIteratorTypes(
    llvm::ArrayRef<mlir::utils::IteratorType> iteratorTypes) {
  bool hasPar = false, hasRed = false;
  for (auto it : iteratorTypes) {
    if (it == mlir::utils::IteratorType::parallel)
      hasPar = true;
    else if (it == mlir::utils::IteratorType::reduction)
      hasRed = true;
  }
  if (hasPar && hasRed)
    return GenericClass::Mixed;
  if (hasRed)
    return GenericClass::Reduction;
  return GenericClass::Parallel;
}

GenericDimAnalysis analyzeGenericDims(mlir::linalg::LinalgOp genericOp) {
  using namespace mlir;

  // 1. Read iterator_types → classify
  auto iteratorTypes = genericOp.getIteratorTypesArray();
  GenericClass cls = classifyIteratorTypes(iteratorTypes);

  // 2. Collect indexing maps and trace all operand dims
  auto indexingMaps = genericOp.getIndexingMapsArray();
  SmallVector<Value> allOperands;
  for (auto v : genericOp.getDpsInputs())
    allOperands.push_back(v);
  for (auto v : genericOp.getDpsInits())
    allOperands.push_back(v);
  assert(indexingMaps.size() == allOperands.size() &&
         "indexing maps count must match operand count");

  // Pre-trace dims for each operand
  SmallVector<std::vector<Expr>> operandDims;
  for (auto val : allOperands)
    operandDims.push_back(traceAllocDimsFromTensor(val));

  // 3. For each loop dim, find symbolic Expr via indexing maps
  unsigned numLoopDims = iteratorTypes.size();
  Expr parallelProduct = Expr::none();
  Expr reductionProduct = Expr::none();

  for (unsigned di = 0; di < numLoopDims; ++di) {
    Expr dimExpr = Expr::none();
    bool found = false;

    // Try each operand's indexing map
    for (unsigned opIdx = 0; opIdx < indexingMaps.size(); ++opIdx) {
      AffineMap map = indexingMaps[opIdx];
      const auto &tracedDims = operandDims[opIdx];
      if (tracedDims.empty())
        continue;

      // Scan map results for a simple AffineDimExpr matching d_i
      for (unsigned r = 0; r < map.getNumResults(); ++r) {
        AffineExpr resultExpr = map.getResult(r);
        auto affineDim = dyn_cast<AffineDimExpr>(resultExpr);
        assert((!resultExpr || affineDim ||
                isa<AffineConstantExpr>(resultExpr)) &&
               "indexing map result must be a simple AffineDimExpr or constant");
        if (affineDim && affineDim.getPosition() == di) {
          if (r < tracedDims.size() && !tracedDims[r].isNone()) {
            dimExpr = tracedDims[r];
            found = true;
            break;
          }
        }
      }
      if (found)
        break;
    }
    assert(found && "could not resolve loop dim from any operand's indexing map");

    // 4. Fold into appropriate product
    if (iteratorTypes[di] == mlir::utils::IteratorType::parallel) {
      parallelProduct =
          parallelProduct.isNone() ? dimExpr : parallelProduct * dimExpr;
    } else {
      reductionProduct =
          reductionProduct.isNone() ? dimExpr : reductionProduct * dimExpr;
    }
  }

  return GenericDimAnalysis{cls, parallelProduct, reductionProduct};
}

} // namespace lcs
} // namespace loom
