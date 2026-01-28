#include "Passes.h"
#include "analysis_engine.h"
#include "constraint_exporter.h"
#include "constraint_space_utils.h"

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/Support/Debug.h"

#include "LoomDialect.h.inc"
#include "mlir/Interfaces/ViewLikeInterface.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#define DEBUG_TYPE "loom-constraint-simplify"

namespace loom {
namespace constraint_opt {

#define GEN_PASS_DEF_LOOMCONSTRAINTSIMPLIFY
#include "Passes.h.inc"

using namespace mlir;
using namespace loom::lcs;
using namespace mlir::presburger;

namespace {

struct SimplifyState {
  ConstraintSpaceOp csOp;
  ValueTracker tracker;
  std::unique_ptr<IntegerPolyhedron> fullPoly;
  std::unique_ptr<IntegerPolyhedron> contextPoly;

  SimplifyState(ConstraintSpaceOp op) : csOp(op) {}

  LogicalResult Initialize() {
    // 1. Register variables
    // Symbolic vars go to Dimensions
    for (auto symVar : csOp.getOps<SymbolicVarOp>()) {
      tracker.trackDimension(symVar.getResult(), symVar.getName());
    }

    // Intermediate vars go to Local IDs
    for (auto ivOp : csOp.getOps<loom::IntermediateVarOp>()) {
      tracker.trackLocalId(ivOp.getResult());
    }

    unsigned numDims = tracker.getNumDims();
    unsigned numLocals = tracker.getNumLocals();

    // 2. Initialize polyhedra
    PresburgerSpace space = PresburgerSpace::getSetSpace(numDims);
    fullPoly = std::make_unique<IntegerPolyhedron>(space);
    fullPoly->appendVar(VarKind::Local, numLocals);

    contextPoly = std::make_unique<IntegerPolyhedron>(space);
    contextPoly->appendVar(VarKind::Local, numLocals);

    // 3. Inject Context (Ranges and Aligns)
    for (auto rangeOp : csOp.getOps<RangeOp>()) {
      auto colIdx = tracker.getColumnIndex(rangeOp.getVariable());
      if (!colIdx)
        continue;

      fullPoly->addBound(BoundType::LB, *colIdx, rangeOp.getLowerBound());
      fullPoly->addBound(BoundType::UB, *colIdx, rangeOp.getUpperBound());

      contextPoly->addBound(BoundType::LB, *colIdx, rangeOp.getLowerBound());
      contextPoly->addBound(BoundType::UB, *colIdx, rangeOp.getUpperBound());
    }

    for (auto alignOp : csOp.getOps<AlignOp>()) {
      auto colIdx = tracker.getColumnIndex(alignOp.getVariable());
      if (!colIdx)
        continue;

      int64_t alignment = alignOp.getAlignment();
      if (alignment <= 1)
        continue;

      // Add local ID for alignment
      unsigned qIdx = fullPoly->appendVar(VarKind::Local, 1);
      contextPoly->appendVar(VarKind::Local, 1);

      // dim - alignment * q = 0
      SmallVector<int64_t, 8> eq(fullPoly->getNumCols(), 0);
      eq[*colIdx] = 1;
      eq[qIdx] = -alignment;

      fullPoly->addEquality(eq);
      contextPoly->addEquality(eq);
    }

    // 4. Inject Subject Constraints (Linear constraints)
    for (auto lcOp : csOp.getOps<LinearConstraintOp>()) {
      AffineMap map = lcOp.getMap();
      auto operands = lcOp.getOperands();

      for (unsigned i = 0; i < map.getNumResults(); ++i) {
        AffineExpr expr = map.getResult(i);
        SmallVector<int64_t, 8> coeffs;
        int64_t constant = 0;

        // We need to map results back to our fullPoly columns
        // This is tricky because CoefficientExtractor doesn't know about our
        // locals Let's use a custom approach or helper

        // Re-implementing parts of extractCoefficients to handle locals
        SmallVector<int64_t, 8> mapCoeffs(map.getNumDims(), 0);
        if (failed(extractMapCoefficients(expr, map.getNumDims(), mapCoeffs,
                                          constant))) {
          continue;
        }

        SmallVector<int64_t, 8> fullCoeffs(fullPoly->getNumCols(), 0);
        for (unsigned d = 0; d < operands.size(); ++d) {
          auto colIdx = tracker.getColumnIndex(operands[d]);
          if (colIdx) {
            fullCoeffs[*colIdx] += mapCoeffs[d];
          }
        }
        fullCoeffs.back() = constant;

        if (lcOp.getIsEquality()) {
          fullPoly->addEquality(fullCoeffs);
        } else {
          fullPoly->addInequality(fullCoeffs);
        }
      }
    }

    return success();
  }

  // Simplified version of the extractor in analysis_engine.cpp
  LogicalResult extractMapCoefficients(AffineExpr expr, unsigned numDims,
                                       SmallVectorImpl<int64_t> &coeffs,
                                       int64_t &constant) {
    auto visit = [&](AffineExpr e, int64_t mult,
                     auto &self_ref) -> LogicalResult {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(e)) {
        if (dimExpr.getPosition() < numDims) {
          coeffs[dimExpr.getPosition()] += mult;
          return success();
        }
        return failure();
      }
      if (auto constExpr = dyn_cast<AffineConstantExpr>(e)) {
        constant += mult * constExpr.getValue();
        return success();
      }
      if (auto binaryOp = dyn_cast<AffineBinaryOpExpr>(e)) {
        if (binaryOp.getKind() == AffineExprKind::Add) {
          if (failed(self_ref(binaryOp.getLHS(), mult, self_ref)))
            return failure();
          if (failed(self_ref(binaryOp.getRHS(), mult, self_ref)))
            return failure();
          return success();
        }
        if (binaryOp.getKind() == AffineExprKind::Mul) {
          auto lhsConst = dyn_cast<AffineConstantExpr>(binaryOp.getLHS());
          auto rhsConst = dyn_cast<AffineConstantExpr>(binaryOp.getRHS());
          if (lhsConst) {
            return self_ref(binaryOp.getRHS(), mult * lhsConst.getValue(),
                            self_ref);
          }
          if (rhsConst) {
            return self_ref(binaryOp.getLHS(), mult * rhsConst.getValue(),
                            self_ref);
          }
          return failure(); // Non-linear
        }
      }
      return failure();
    };
    return visit(expr, 1, visit);
  }

  void Simplify() {
    // Stage 3: Core Simplification
    fullPoly->simplify();

    LLVM_DEBUG({
      llvm::dbgs() << "Simplified Polyhedron:\n";
      fullPoly->dump();
    });
  }

  void Reconstruct() {
    // Stage 4: Gist-based reconstruction
    // 1. Project out anonymous local variables (those added for align)
    // We only want to keep variables that have a corresponding MLIR Value.
    // Dimensions are always kept. Locals from IntermediateVarOp are kept.
    // The ones added for alignment are added at the end of VarKind::Local.
    unsigned numTrackedLocals = tracker.getNumLocals();
    if (fullPoly->getNumVarKind(VarKind::Local) > numTrackedLocals) {
      unsigned pos =
          fullPoly->getVarKindOffset(VarKind::Local) + numTrackedLocals;
      unsigned num = fullPoly->getNumVarKind(VarKind::Local) - numTrackedLocals;
      fullPoly->projectOut(pos, num);
      contextPoly->projectOut(pos, num);
    }

    // 2. Remove old constraints (except ranges/aligns/vars)
    SmallVector<Operation *> toErase;
    for (auto &op : csOp.getBodyBlock()->getOperations()) {
      if (isa<LinearConstraintOp>(&op)) {
        toErase.push_back(&op);
      }
    }
    for (auto op : toErase)
      op->erase();

    // Note: IntermediateVarOp might be removed later if unused

    OpBuilder builder(csOp.getBodyBlock(), csOp.getBodyBlock()->end());
    MLIRContext *ctx = csOp.getContext();

    // 3. Write back essential equalities
    for (unsigned i = 0; i < fullPoly->getNumEqualities(); ++i) {
      ArrayRef<DynamicAPInt> eq = fullPoly->getEquality(i);
      if (isRedundant(eq, true))
        continue;
      emitConstraint(eq, true, builder, ctx);
    }

    // 3. Write back essential inequalities
    for (unsigned i = 0; i < fullPoly->getNumInequalities(); ++i) {
      ArrayRef<DynamicAPInt> ineq = fullPoly->getInequality(i);
      if (isRedundant(ineq, false))
        continue;
      emitConstraint(ineq, false, builder, ctx);
    }

    // 4. Cleanup unused intermediate vars
    toErase.clear();
    for (auto ivOp : csOp.getOps<loom::IntermediateVarOp>()) {
      if (ivOp.getResult().use_empty()) {
        toErase.push_back(ivOp);
      }
    }
    for (auto op : toErase)
      op->erase();
  }

  bool isRedundant(ArrayRef<DynamicAPInt> coeffs, bool isEquality) {
    // Move into contextPoly and check if its empty with the negation
    // Inequality: C >= 0. Redundant if Context && C < 0 is empty.
    // Equality: C == 0. Redundant if Context && (C > 0 || C < 0) is empty.

    if (isEquality) {
      // Check C > 0
      {
        IntegerPolyhedron test = *contextPoly;
        SmallVector<int64_t, 8> ineq;
        for (auto c : coeffs)
          ineq.push_back(int64_t(c));
        ineq.back() -= 1; // C >= 1  => C > 0
        test.addInequality(ineq);
        if (!test.isEmpty())
          return false;
      }
      // Check C < 0
      {
        IntegerPolyhedron test = *contextPoly;
        SmallVector<int64_t, 8> ineq;
        for (auto c : coeffs)
          ineq.push_back(-int64_t(c));
        ineq.back() -= 1; // -C >= 1 => C <= -1 => C < 0
        test.addInequality(ineq);
        if (!test.isEmpty())
          return false;
      }
      return true;
    } else {
      IntegerPolyhedron test = *contextPoly;
      SmallVector<int64_t, 8> ineq;
      for (auto c : coeffs)
        ineq.push_back(-int64_t(c));
      ineq.back() -= 1; // -C >= 1 => C <= -1 => C < 0
      test.addInequality(ineq);
      return test.isEmpty();
    }
  }

  void emitConstraint(ArrayRef<DynamicAPInt> coeffs, bool isEquality,
                      OpBuilder &builder, MLIRContext *ctx) {
    SmallVector<Value> allValues;
    // Dimensions
    SmallVector<Value> dims(tracker.getNumDims());
    for (auto it : tracker.getDimensions()) {
      dims[it.second] = it.first;
    }
    allValues.append(dims.begin(), dims.end());

    // Locals (IntermediateVars)
    SmallVector<Value> locals(tracker.getNumLocals());
    for (auto it : tracker.getLocalIds()) {
      locals[it.second] = it.first;
    }
    allValues.append(locals.begin(), locals.end());

    AffineExpr expr = builder.getAffineConstantExpr(int64_t(coeffs.back()));
    for (unsigned i = 0; i < allValues.size(); ++i) {
      int64_t c = int64_t(coeffs[i]);
      if (c != 0) {
        expr = expr + c * builder.getAffineDimExpr(i);
      }
    }

    auto map = AffineMap::get(allValues.size(), 0, {expr}, ctx);
    builder.create<LinearConstraintOp>(
        csOp.getLoc(), allValues, AffineMapAttr::get(map),
        isEquality ? builder.getBoolAttr(true) : nullptr);
  }
};

struct LoomConstraintSimplify
    : public impl::LoomConstraintSimplifyBase<LoomConstraintSimplify> {
  using LoomConstraintSimplifyBase::LoomConstraintSimplifyBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Export simplified constraints to JSON
    llvm::SmallVector<std::string, 8> allJsonStrings;
    module.walk([&](ConstraintSpaceOp csOp) {
      SimplifyState state(csOp);
      if (failed(state.Initialize()))
        return;
      state.Simplify();
      state.Reconstruct();

      allJsonStrings.push_back(
          loom::lcs::exportConstraintSpaceToJson(csOp, "ConstraintSimplify"));
    });

    if (!allJsonStrings.empty()) {
      llvm::errs() << "[\n";
      for (size_t i = 0; i < allJsonStrings.size(); ++i) {
        // Indent the internal JSON
        std::string indented;
        llvm::raw_string_ostream os(indented);
        bool firstLine = true;
        for (char c : allJsonStrings[i]) {
          if (firstLine) {
            os << "  ";
            firstLine = false;
          }
          os << c;
          if (c == '\n')
            os << "  ";
        }
        llvm::errs() << os.str();
        if (i < allJsonStrings.size() - 1)
          llvm::errs() << ",";
        llvm::errs() << "\n";
      }
      llvm::errs() << "]\n";
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLoomConstraintSimplifyPass() {
  return std::make_unique<LoomConstraintSimplify>();
}

} // namespace constraint_opt
} // namespace loom
