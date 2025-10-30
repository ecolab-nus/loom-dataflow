//===- const_dedup_cleanup.cpp -----------------------------*- C++ -*-===//
// Deduplicate constants, erase the unused ones, and fold constants into
// affine.apply where possible.
//===------------------------------------------------------------------===//

#include "const_dedup_cleanup.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace tmd {
namespace passes {

namespace {

struct ConstKey {
  Attribute value;
  Type type;
  bool operator==(const ConstKey &other) const {
    return value == other.value && type == other.type;
  }
};

struct ConstKeyInfo : public llvm::DenseMapInfo<ConstKey> {
  static inline ConstKey getEmptyKey() {
    return {llvm::DenseMapInfo<Attribute>::getEmptyKey(),
            llvm::DenseMapInfo<Type>::getEmptyKey()};
  }
  static inline ConstKey getTombstoneKey() {
    return {llvm::DenseMapInfo<Attribute>::getTombstoneKey(),
            llvm::DenseMapInfo<Type>::getTombstoneKey()};
  }
  static unsigned getHashValue(const ConstKey &k) {
    return llvm::DenseMapInfo<Attribute>::getHashValue(k.value) ^
           (llvm::DenseMapInfo<Type>::getHashValue(k.type) << 1);
  }
  static bool isEqual(const ConstKey &a, const ConstKey &b) { return a == b; }
};

class ConstDedupCleanupPass
    : public PassWrapper<ConstDedupCleanupPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstDedupCleanupPass)

  StringRef getArgument() const override { return "tmd-const-cleanup"; }
  StringRef getDescription() const override {
    return "Deduplicate and remove unused arith.constant and fold constants "
           "into affine.apply";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Per-function constant deduping and cleanup.
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      denseDedupInFunc(func);
      foldConstantsIntoAffine(func);
      eraseTriviallyDeadConstants(func);
    }
  }

private:
  void denseDedupInFunc(func::FuncOp func) {
    // Map constant (value,type) -> canonical Value
    llvm::DenseMap<ConstKey, Value, ConstKeyInfo> canonical;
    SmallVector<Operation *, 16> toErase;

    func.walk([&](Operation *op) {
      if (auto cstIdx = dyn_cast<arith::ConstantIndexOp>(op)) {
        ConstKey key{cstIdx.getValueAttr(), cstIdx.getType()};
        auto it = canonical.find(key);
        if (it == canonical.end()) {
          canonical.insert({key, cstIdx.getResult()});
        } else {
          cstIdx.replaceAllUsesWith(it->second);
          toErase.push_back(op);
        }
        return;
      }
      if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
        ConstKey key{cst.getValue(), cst.getType()};
        auto it = canonical.find(key);
        if (it == canonical.end()) {
          canonical.insert({key, cst.getResult()});
        } else {
          cst.replaceAllUsesWith(it->second);
          toErase.push_back(op);
        }
        return;
      }
    });

    for (Operation *op : toErase) {
      bool allResultsDead = true;
      for (Value res : op->getResults()) {
        if (!res.use_empty()) {
          allResultsDead = false;
          break;
        }
      }
      if (allResultsDead)
        op->erase();
    }
  }

  void eraseTriviallyDeadConstants(func::FuncOp func) {
    SmallVector<Operation *, 16> toErase;
    func.walk([&](arith::ConstantIndexOp c) {
      if (c.getResult().use_empty())
        toErase.push_back(c);
    });
    func.walk([&](arith::ConstantOp c) {
      if (c.getResult().use_empty())
        toErase.push_back(c);
    });
    for (Operation *op : toErase) {
      bool allResultsDead = true;
      for (Value res : op->getResults()) {
        if (!res.use_empty()) {
          allResultsDead = false;
          break;
        }
      }
      if (allResultsDead)
        op->erase();
    }
  }

  void foldConstantsIntoAffine(func::FuncOp func) {
    SmallVector<affine::AffineApplyOp, 16> applies;
    func.walk([&](affine::AffineApplyOp a) { applies.push_back(a); });

    for (affine::AffineApplyOp a : applies) {
      AffineMap map = a.getAffineMap();
      unsigned oldNumDims = map.getNumDims();
      unsigned oldNumSyms = map.getNumSymbols();
      ValueRange operands = a.getMapOperands();
      if (operands.size() != oldNumDims + oldNumSyms)
        continue;

      SmallVector<AffineExpr, 8> dimRepls(oldNumDims);
      SmallVector<AffineExpr, 8> symRepls(oldNumSyms);
      SmallVector<Value, 8> newDimOperands;
      SmallVector<Value, 8> newSymOperands;

      unsigned newDimIdx = 0, newSymIdx = 0;

      // Build replacements for dims
      for (unsigned i = 0; i < oldNumDims; ++i) {
        Value v = operands[i];
        if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>()) {
          dimRepls[i] = getAffineConstantExpr(cst.value(), a.getContext());
        } else {
          dimRepls[i] = getAffineDimExpr(newDimIdx++, a.getContext());
          newDimOperands.push_back(v);
        }
      }

      // And for symbols (if any)
      for (unsigned s = 0; s < oldNumSyms; ++s) {
        Value v = operands[oldNumDims + s];
        if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>()) {
          symRepls[s] = getAffineConstantExpr(cst.value(), a.getContext());
        } else {
          symRepls[s] = getAffineSymbolExpr(newSymIdx++, a.getContext());
          newSymOperands.push_back(v);
        }
      }

      // If nothing is constant, skip
      if (newDimIdx == oldNumDims && newSymIdx == oldNumSyms)
        continue;

      AffineMap newMap = map.replaceDimsAndSymbols(dimRepls, symRepls,
                                                   /*newNumDims=*/newDimIdx,
                                                   /*newNumSymbols=*/newSymIdx);

      // Rebuild apply with pruned operands (dims first, then syms)
      SmallVector<Value, 8> newOperands;
      newOperands.append(newDimOperands.begin(), newDimOperands.end());
      newOperands.append(newSymOperands.begin(), newSymOperands.end());

      OpBuilder b(a);
      auto newApply =
          b.create<affine::AffineApplyOp>(a.getLoc(), newMap, newOperands);
      a.replaceAllUsesWith(newApply.getResult());
      // Erase only if truly dead to avoid accidental erasure when IR printing
      // or instrumentation holds transient references.
      if (a->use_empty())
        a.erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConstDedupCleanupPass() {
  return std::make_unique<ConstDedupCleanupPass>();
}

} // namespace passes
} // namespace tmd
