//===- analysis_engine.cpp - Loom Constraint Analysis Engine -------------===//
//
// Implementation of the AnalysisEngine class that converts Loom constraint
// operations into mathematical ConstraintSet objects.
//
//===----------------------------------------------------------------------===//

#include "analysis_engine.h"
#include "constraint_set.h"

// MLIR core headers must be included before generated dialect headers
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

// Include the generated Loom dialect headers
#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#define DEBUG_TYPE "loom-analysis-engine"

namespace loom {
namespace lcs {

using namespace mlir;

//===----------------------------------------------------------------------===//
// Coefficient Extraction Helper
//===----------------------------------------------------------------------===//

namespace {

/// @brief Visitor to extract linear coefficients from an AffineExpr.
///
/// This visitor walks an AffineExpr and extracts the coefficient for each
/// dimension variable and the constant term. It only supports affine
/// expressions (linear combinations of dimensions with integer coefficients).
class CoefficientExtractor : public AffineExprVisitor<CoefficientExtractor> {
public:
  CoefficientExtractor(unsigned numDims, llvm::SmallVectorImpl<int64_t> &coeffs,
                       int64_t &constant)
      : numDims_(numDims), coeffs_(coeffs), constant_(constant), multiplier_(1),
        success_(true) {
    coeffs_.assign(numDims, 0);
    constant_ = 0;
  }

  void visitDimExpr(AffineDimExpr expr) {
    unsigned pos = expr.getPosition();
    if (pos < numDims_) {
      coeffs_[pos] += multiplier_;
    } else {
      success_ = false;
    }
  }

  void visitSymbolExpr(AffineSymbolExpr /*expr*/) {
    // Symbols are not supported in LinearConstraintOp
    // All variables should be dimensions
    LLVM_DEBUG(llvm::dbgs()
               << "Warning: Symbol encountered in constraint expression\n");
    success_ = false;
  }

  void visitConstantExpr(AffineConstantExpr expr) {
    constant_ += multiplier_ * expr.getValue();
  }

  void visitAddExpr(AffineBinaryOpExpr expr) {
    visit(expr.getLHS());
    visit(expr.getRHS());
  }

  void visitMulExpr(AffineBinaryOpExpr expr) {
    // For multiply, one side must be a constant
    auto lhsConst = dyn_cast<AffineConstantExpr>(expr.getLHS());
    auto rhsConst = dyn_cast<AffineConstantExpr>(expr.getRHS());

    if (lhsConst) {
      int64_t savedMult = multiplier_;
      multiplier_ *= lhsConst.getValue();
      visit(expr.getRHS());
      multiplier_ = savedMult;
    } else if (rhsConst) {
      int64_t savedMult = multiplier_;
      multiplier_ *= rhsConst.getValue();
      visit(expr.getLHS());
      multiplier_ = savedMult;
    } else {
      // Non-linear: product of two non-constants
      success_ = false;
    }
  }

  void visitModExpr(AffineBinaryOpExpr /*expr*/) {
    // Modulo is not linear
    success_ = false;
  }

  void visitFloorDivExpr(AffineBinaryOpExpr /*expr*/) {
    // Floor division is not linear in general
    success_ = false;
  }

  void visitCeilDivExpr(AffineBinaryOpExpr /*expr*/) {
    // Ceiling division is not linear in general
    success_ = false;
  }

  bool succeeded() const { return success_; }

private:
  unsigned numDims_;
  llvm::SmallVectorImpl<int64_t> &coeffs_;
  int64_t &constant_;
  int64_t multiplier_;
  bool success_;
};

} // namespace

//===----------------------------------------------------------------------===//
// ValueTracker Implementation
//===----------------------------------------------------------------------===//

unsigned ValueTracker::trackDimension(mlir::Value val,
                                      llvm::StringRef /*name*/) {
  unsigned idx = valueToDimIndex_.size();
  valueToDimIndex_[val] = idx;
  return idx;
}

unsigned ValueTracker::trackLocalId(mlir::Value val) {
  unsigned idx = valueToLocalIndex_.size();
  valueToLocalIndex_[val] = idx;
  return idx;
}

std::optional<unsigned> ValueTracker::getColumnIndex(mlir::Value val) const {
  auto it = valueToDimIndex_.find(val);
  if (it != valueToDimIndex_.end()) {
    return it->second;
  }
  auto itLocal = valueToLocalIndex_.find(val);
  if (itLocal != valueToLocalIndex_.end()) {
    // Local IDs start after dimensions
    return valueToDimIndex_.size() + itLocal->second;
  }
  return std::nullopt;
}

bool ValueTracker::isDimension(mlir::Value val) const {
  return valueToDimIndex_.count(val) > 0;
}

//===----------------------------------------------------------------------===//
// BoundInferenceService Implementation
//===----------------------------------------------------------------------===//

void BoundInferenceService::initialize() {
  csOp_->walk([&](RangeOp rangeOp) {
    boundsTable_[rangeOp.getVariable()] = {
        static_cast<int64_t>(rangeOp.getLowerBound()),
        static_cast<int64_t>(rangeOp.getUpperBound())};
  });
}

Interval BoundInferenceService::getRange(mlir::Value val) {
  if (boundsTable_.count(val)) {
    return boundsTable_[val];
  }

  // If it's an expression, compute its bounds
  if (auto exprOp = val.getDefiningOp<ExpressionOp>()) {
    return computeExpressionBounds(exprOp);
  }

  return Interval::unbounded();
}

void BoundInferenceService::setRange(mlir::Value val, Interval range) {
  boundsTable_[val] = range;
}

Interval BoundInferenceService::add(Interval a, Interval b) {
  return {a.lower + b.lower, a.upper + b.upper};
}

Interval BoundInferenceService::multiply(Interval a, Interval b) {
  int64_t v1 = a.lower * b.lower;
  int64_t v2 = a.lower * b.upper;
  int64_t v3 = a.upper * b.lower;
  int64_t v4 = a.upper * b.upper;

  return {std::min({v1, v2, v3, v4}), std::max({v1, v2, v3, v4})};
}

Interval BoundInferenceService::scalarMultiply(int64_t scalar, Interval a) {
  int64_t v1 = scalar * a.lower;
  int64_t v2 = scalar * a.upper;
  return {std::min(v1, v2), std::max(v1, v2)};
}

Interval BoundInferenceService::computeExpressionBounds(ExpressionOp op) {
  llvm::StringRef logic = op.getLogic();
  Interval result = {0, 0};

  if (logic == "mul") {
    auto operands = op.getOperands();
    assert(operands.size() == 2 &&
           "Mul expression must have exactly 2 operands");
    result = multiply(getRange(operands[0]), getRange(operands[1]));
  } else if (logic == "add") {
    auto operands = op.getOperands();
    auto coeffs = op.getCoeffs();
    for (unsigned i = 0; i < operands.size(); ++i) {
      int64_t coeff = cast<IntegerAttr>(coeffs[i]).getInt();
      result = add(result, scalarMultiply(coeff, getRange(operands[i])));
    }
  } else {
    assert(false && "Unsupported logic type");
  }

  boundsTable_[op.getResult()] = result;
  return result;
}

BoundInferenceService::BoundInferenceService(loom::ConstraintSpaceOp csOp)
    : csOp_(csOp.getOperation()) {}

ConstraintSet AnalysisEngine::buildConstraintSet(ConstraintSpaceOp csOp) {
  AnalysisEngine engine;
  engine.processConstraintSpace(csOp);
  return std::move(engine.constraintSet_);
}

void AnalysisEngine::processConstraintSpace(ConstraintSpaceOp csOp) {
  LLVM_DEBUG(llvm::dbgs() << "Processing constraint space: "
                          << csOp.getSymName() << "\n");

  // Initialize bound service
  boundService_ = std::make_unique<BoundInferenceService>(csOp);
  boundService_->initialize();

  // First pass: register all symbolic variables to establish dimension indices
  for (Operation &op : csOp.getBodyBlock()->getOperations()) {
    if (auto symVar = dyn_cast<SymbolicVarOp>(&op)) {
      visitSymbolicVar(symVar);
    }
  }

  // Second pass: process all constraint operations
  processOps(csOp.getBodyBlock()->getOperations());

  LLVM_DEBUG({
    llvm::dbgs() << "Finished processing constraint space. Result:\n";
    constraintSet_.dump();
  });
}

void AnalysisEngine::processOps(llvm::iterator_range<Block::iterator> ops) {
  for (Operation &op : ops) {
    llvm::TypeSwitch<Operation *>(&op)
        .Case<RangeOp>([this](RangeOp rangeOp) { visitRange(rangeOp); })
        .Case<AlignOp>([this](AlignOp alignOp) { visitAlign(alignOp); })
        .Case<LinearConstraintOp>(
            [this](LinearConstraintOp lcOp) { visitLinearConstraint(lcOp); })
        .Case<PolynomialConstraintOp>([this](PolynomialConstraintOp pcOp) {
          visitPolynomialConstraint(pcOp);
        })
        .Case<ExpressionOp>(
            [this](ExpressionOp exprOp) { visitExpression(exprOp); })
        .Case<loom::IntermediateVarOp>([this](loom::IntermediateVarOp ivOp) {
          visitIntermediateVar(ivOp);
        })
        .Case<SymbolicVarOp>([](SymbolicVarOp) {
          // Already processed in first pass
        })
        .Default([](Operation *op) {
          LLVM_DEBUG(llvm::dbgs() << "Skipping unknown operation: "
                                  << op->getName() << "\n");
        });
  }
}

void AnalysisEngine::visitSymbolicVar(SymbolicVarOp op) {
  llvm::StringRef varName = op.getName();
  unsigned dimIdx = constraintSet_.registerVariable(varName);

  // Map the SSA result value to this dimension index
  valueTracker_.trackDimension(op.getResult(), varName);

  LLVM_DEBUG(llvm::dbgs() << "  Registered symbolic var '" << varName
                          << "' -> dim " << dimIdx << "\n");
}

void AnalysisEngine::visitIntermediateVar(loom::IntermediateVarOp op) {
  unsigned localIdx = constraintSet_.registerLocalVariable();

  // Map the SSA result value to this local index
  valueTracker_.trackLocalId(op.getResult());

  LLVM_DEBUG(llvm::dbgs() << "  Registered intermediate var -> local "
                          << localIdx << "\n");
}

void AnalysisEngine::visitRange(RangeOp op) {
  Value var = op.getVariable();
  auto dimIdxOpt = resolveDimIndex(var);

  if (!dimIdxOpt) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "Warning: Could not resolve dimension for range constraint\n");
    return;
  }

  unsigned dimIdx = *dimIdxOpt;
  int64_t lb = op.getLowerBound();
  int64_t ub = op.getUpperBound();

  constraintSet_.addRange(dimIdx, lb, ub);

  LLVM_DEBUG(llvm::dbgs() << "  Added range constraint: dim" << dimIdx
                          << " in [" << lb << ", " << ub << "]\n");
}

void AnalysisEngine::visitAlign(AlignOp op) {
  Value var = op.getVariable();
  auto dimIdxOpt = resolveDimIndex(var);

  if (!dimIdxOpt) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "Warning: Could not resolve dimension for align constraint\n");
    return;
  }

  unsigned dimIdx = *dimIdxOpt;
  int64_t alignment = op.getAlignment();

  constraintSet_.addAlignment(dimIdx, alignment);

  LLVM_DEBUG(llvm::dbgs() << "  Added alignment constraint: dim" << dimIdx
                          << " ≡ 0 (mod " << alignment << ")\n");
}

void AnalysisEngine::visitLinearConstraint(LinearConstraintOp op) {
  AffineMap map = op.getMap();
  OperandRange operands = op.getOperands();

  unsigned numDims = constraintSet_.getNumDims();

  // Build a mapping from AffineMap dimensions to ConstraintSet dimensions
  llvm::SmallVector<unsigned, 8> dimMapping;
  for (Value operand : operands) {
    auto dimIdxOpt = resolveDimIndex(operand);
    if (!dimIdxOpt) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Warning: Could not resolve operand in linear constraint\n");
      return;
    }
    dimMapping.push_back(*dimIdxOpt);
  }

  // Process each result of the AffineMap as a separate constraint
  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    AffineExpr expr = map.getResult(i);

    // Extract coefficients from the expression (in terms of map dimensions)
    llvm::SmallVector<int64_t, 8> mapCoeffs;
    int64_t constant = 0;

    CoefficientExtractor extractor(map.getNumDims(), mapCoeffs, constant);
    extractor.visit(expr);

    if (!extractor.succeeded()) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Warning: Could not extract coefficients from expression\n");
      continue;
    }

    // Map coefficients from AffineMap dimensions to ConstraintSet dimensions
    llvm::SmallVector<int64_t, 8> csCoeffs(numDims, 0);
    for (unsigned d = 0; d < dimMapping.size(); ++d) {
      csCoeffs[dimMapping[d]] += mapCoeffs[d];
    }

    // Add the constraint
    if (op.getIsEquality()) {
      // sum(coeffs * vars) + constant == 0
      constraintSet_.addEquality(csCoeffs, constant);
    } else {
      // sum(coeffs * vars) + constant >= 0
      constraintSet_.addInequality(csCoeffs, constant);
    }

    LLVM_DEBUG({
      llvm::dbgs() << "  Added linear constraint from map result " << i << ": ";
      expr.dump();
      llvm::dbgs() << (op.getIsEquality() ? " == 0\n" : " >= 0\n");
    });
  }
}

void AnalysisEngine::visitPolynomialConstraint(PolynomialConstraintOp op) {
  LLVM_DEBUG({
    llvm::dbgs() << "  Encountered polynomial constraint: ";
    op->print(llvm::dbgs());
    llvm::dbgs() << "\n  (Polynomial constraints are currently treated as data "
                    "structure only and not checked for feasibility)\n";
  });
}

void AnalysisEngine::visitExpression(ExpressionOp op) {
  unsigned localIdx = valueTracker_.trackLocalId(op.getResult());
  LLVM_DEBUG(llvm::dbgs() << "  Registered expression as local " << localIdx
                          << "\n");
}

std::optional<unsigned> AnalysisEngine::resolveDimIndex(Value val) const {
  return valueTracker_.getColumnIndex(val);
}

bool AnalysisEngine::extractCoefficients(AffineExpr expr, unsigned numDims,
                                         llvm::SmallVectorImpl<int64_t> &coeffs,
                                         int64_t &constant) const {
  CoefficientExtractor extractor(numDims, coeffs, constant);
  extractor.visit(expr);
  return extractor.succeeded();
}

} // namespace lcs
} // namespace loom
