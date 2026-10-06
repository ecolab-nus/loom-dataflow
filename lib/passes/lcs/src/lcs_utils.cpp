/**
 * @file lcs_utils.cpp
 * @brief Implementation of tracing utilities for LCS analysis.
 */

#include "lcs_utils.h"
#include "ssa_utils.h"
#include "utils.h"
#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

namespace loom {
namespace lcs {

/// Optional per-IV values used instead of the default IV upper bound.
using IVSubstitution = llvm::DenseMap<mlir::Value, Expr>;

static Expr traceIndexValueToExpr(mlir::Value val,
                                  const IVSubstitution *subst = nullptr);

// ==========================================
// Tracing — thin wrappers over loom::utils canonical SSA walk
// ==========================================

loom::AllocOp traceToAlloc(mlir::Value memrefVal) {
  loom::AllocOp alloc = loom::utils::traceToRootAllocOp(memrefVal);
  if (!alloc) {
    if (memrefVal)
      if (mlir::Operation *op = memrefVal.getDefiningOp())
        op->emitError() << "lcs::traceToAlloc: cannot resolve to loom.alloc";
    llvm::report_fatal_error("lcs::traceToAlloc failed");
  }
  return alloc;
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
        dims.push_back(traceIndexValueToExpr(dynamicSizes[dynamicIdx]));
        dynamicIdx++;
      }
    } else {
      dims.push_back(Expr::con(staticDim));
    }
  }

  return dims;
}

std::vector<Expr> traceAllocDimsFromTensor(mlir::Value tensorVal) {
  return formatAllocDims(traceToAlloc(tensorVal));
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

/// Trace an affine expression whose dims/symbols are bound to `operands`
/// (dims first, then symbols) into a symbolic Expr.
static Expr traceAffineExprToExpr(mlir::AffineExpr affineExpr,
                                  mlir::ValueRange operands, unsigned numDims,
                                  const IVSubstitution *subst) {
  if (auto dim = llvm::dyn_cast<mlir::AffineDimExpr>(affineExpr)) {
    unsigned pos = dim.getPosition();
    if (pos >= numDims || pos >= operands.size())
      return Expr::none();
    return traceIndexValueToExpr(operands[pos], subst);
  }

  if (auto sym = llvm::dyn_cast<mlir::AffineSymbolExpr>(affineExpr)) {
    unsigned pos = numDims + sym.getPosition();
    if (pos >= operands.size())
      return Expr::none();
    return traceIndexValueToExpr(operands[pos], subst);
  }

  if (auto cst = llvm::dyn_cast<mlir::AffineConstantExpr>(affineExpr))
    return Expr::con(cst.getValue());

  if (auto bin = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(affineExpr)) {
    Expr lhs = traceAffineExprToExpr(bin.getLHS(), operands, numDims, subst);
    Expr rhs = traceAffineExprToExpr(bin.getRHS(), operands, numDims, subst);
    switch (bin.getKind()) {
    case mlir::AffineExprKind::Add:
      return lhs + rhs;
    case mlir::AffineExprKind::Mul:
      return lhs * rhs;
    case mlir::AffineExprKind::FloorDiv:
    case mlir::AffineExprKind::CeilDiv:
      return lhs / rhs;
    default:
      return Expr::none();
    }
  }

  return Expr::none();
}

/// Upper-bound (worst-case) value of a loop induction variable, assuming
/// lb = 0 and step = 1: max(iv) = ub - 1 (emitted as ub + (-1): the Python
/// solver AST has no Sub node). Used for loops whose trip count
/// depends on an enclosing IV (e.g. causal/triangular loops), so the ETG
/// models the largest iteration.
static Expr traceInductionVarUpperBound(mlir::BlockArgument arg,
                                        const IVSubstitution *subst) {
  using namespace mlir;
  Operation *owner = arg.getOwner()->getParentOp();

  if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
    if (arg != forOp.getInductionVar())
      return Expr::none();
    return traceIndexValueToExpr(forOp.getUpperBound(), subst) +
           Expr::con(-1);
  }

  if (auto parOp = dyn_cast<affine::AffineParallelOp>(owner)) {
    unsigned pos = arg.getArgNumber();
    AffineMap ubMap = parOp.getUpperBoundMap(pos);
    if (ubMap.getNumResults() != 1)
      return Expr::none();
    Expr ub = traceAffineExprToExpr(ubMap.getResult(0),
                                    parOp.getUpperBoundsOperands(),
                                    ubMap.getNumDims(), subst);
    return ub + Expr::con(-1);
  }

  return Expr::none();
}

/// Recursively trace an SSA index value to a symbolic Expr.
/// Handles: loom.sym → Sym, arith.constant → Const,
///          arith.ceildivui/si → Div, arith.muli → Mul, arith.addi → Add,
///          affine.apply → its affine expression,
///          scf.for / affine.parallel IV → `subst` value if given, else its
///          upper bound (ub - 1).
static Expr traceIndexValueToExpr(mlir::Value val,
                                  const IVSubstitution *subst) {
  using namespace mlir;
  if (!val)
    return Expr::none();

  // First try to resolve directly to a named symbolic variable (loom.sym).
  llvm::StringRef symName = loom::utils::traceToSymbolicVar(val);
  if (!symName.empty())
    return Expr::sym(symName.str());

  if (auto arg = dyn_cast<BlockArgument>(val)) {
    if (subst) {
      auto it = subst->find(val);
      if (it != subst->end())
        return it->second;
    }
    return traceInductionVarUpperBound(arg, subst);
  }

  Operation *op = val.getDefiningOp();
  if (!op)
    return Expr::none();

  // affine.apply → trace its (single-result) affine map
  if (auto apply = dyn_cast<affine::AffineApplyOp>(op))
    return traceAffineExprToExpr(apply.getAffineMap().getResult(0),
                                 apply.getMapOperands(),
                                 apply.getAffineMap().getNumDims(), subst);

  // arith.constant
  if (auto constOp = dyn_cast<arith::ConstantOp>(op))
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return Expr::con(intAttr.getInt());

  // arith.ceildivui / arith.ceildivsi → Div (same semantics for trip counts)
  if (auto cdiv = dyn_cast<arith::CeilDivUIOp>(op))
    return traceIndexValueToExpr(cdiv.getLhs(), subst) /
           traceIndexValueToExpr(cdiv.getRhs(), subst);
  if (auto cdiv = dyn_cast<arith::CeilDivSIOp>(op))
    return traceIndexValueToExpr(cdiv.getLhs(), subst) /
           traceIndexValueToExpr(cdiv.getRhs(), subst);

  // arith.muli → Mul
  if (auto mul = dyn_cast<arith::MulIOp>(op))
    return traceIndexValueToExpr(mul.getLhs(), subst) *
           traceIndexValueToExpr(mul.getRhs(), subst);

  // arith.addi → Add
  if (auto add = dyn_cast<arith::AddIOp>(op))
    return traceIndexValueToExpr(add.getLhs(), subst) +
           traceIndexValueToExpr(add.getRhs(), subst);

  // Type-conversion ops in the trip-count def-use chain are explicitly
  // unsupported.  Canonicalize/CSE should have removed them before ETG
  // extraction; if any survive, fail loudly rather than silently producing
  // an empty expression.
  if (llvm::isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp,
                arith::ExtUIOp, arith::TruncIOp>(op)) {
    op->emitError() << "traceIndexValueToExpr: type-conversion op in the "
                       "trip-count def-use chain is unsupported. Run "
                       "--canonicalize/--cse first, or keep index-typed "
                       "values throughout the bound chain.";
    llvm::report_fatal_error("unsupported type-cast in trip-count chain");
  }

  // Any other op in the def-use chain is unexpected; refuse to silently
  // drop the expression.
  {
    std::string msg;
    llvm::raw_string_ostream os(msg);
    os << "traceIndexValueToExpr: unhandled op '" << op->getName()
       << "' in trip-count def-use chain";
    op->emitError() << msg;
    llvm::report_fatal_error("unhandled op in trip-count chain");
  }
}

Expr extractLoopTripCount(mlir::scf::ForOp forOp) {
  if (!forOp)
    return Expr::none();
  // Assumes lb=0 and step=1, so trip count == upper bound.
  return traceIndexValueToExpr(forOp.getUpperBound());
}

/// Collect the loop induction variables (scf.for / affine.parallel) that `val`
/// depends on through its index def-use chain.
static void collectInductionVars(mlir::Value val,
                                 llvm::SetVector<mlir::Value> &ivs) {
  using namespace mlir;
  if (!val || !loom::utils::traceToSymbolicVar(val).empty())
    return;
  if (auto arg = dyn_cast<BlockArgument>(val)) {
    Operation *owner = arg.getOwner()->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
      if (arg == forOp.getInductionVar())
        ivs.insert(val);
    } else if (isa<affine::AffineParallelOp>(owner)) {
      ivs.insert(val);
    }
    return;
  }
  if (Operation *op = val.getDefiningOp())
    for (Value operand : op->getOperands())
      collectInductionVars(operand, ivs);
}

std::vector<ConstraintExpr>
ivDependentTripCountDivisibility(mlir::scf::ForOp forOp) {
  using namespace mlir;
  std::vector<ConstraintExpr> constraints;
  Value num, den;
  Operation *ubDef = forOp.getUpperBound().getDefiningOp();
  if (auto cdiv = dyn_cast_or_null<arith::CeilDivUIOp>(ubDef)) {
    num = cdiv.getLhs();
    den = cdiv.getRhs();
  } else if (auto cdiv = dyn_cast_or_null<arith::CeilDivSIOp>(ubDef)) {
    num = cdiv.getLhs();
    den = cdiv.getRhs();
  } else {
    return constraints;
  }

  llvm::SetVector<Value> ivs;
  collectInductionVars(num, ivs);
  if (ivs.empty())
    return constraints;
  Expr denominator = traceIndexValueToExpr(den);
  if (denominator.isNone())
    return constraints;

  // For a numerator affine in the IVs, divisibility at all-zero and at each
  // unit vector implies divisibility at every iteration.
  IVSubstitution subst;
  for (Value iv : ivs)
    subst[iv] = Expr::con(0);
  auto pushSample = [&]() {
    Expr numerator = traceIndexValueToExpr(num, &subst);
    if (!numerator.isNone())
      constraints.push_back(ConstraintExpr::divisible(numerator, denominator));
  };
  pushSample();
  for (Value iv : ivs) {
    subst[iv] = Expr::con(1);
    pushSample();
    subst[iv] = Expr::con(0);
  }
  return constraints;
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
