#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;
using namespace loom;

namespace {

struct FuseZeroFillMatmulPass
    : public PassWrapper<FuseZeroFillMatmulPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseZeroFillMatmulPass)

  StringRef getArgument() const override { return "tt-fuse-zero-fill-matmul"; }

  StringRef getDescription() const override {
    return "Fuse linalg.fill(0) and linalg.matmul into loom.matmul";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp funcOp) {
      SmallVector<linalg::MatmulOp> matmuls;
      funcOp.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });

      SmallPtrSet<Operation *, 4> fillsToErase;

      for (auto matmulOp : matmuls) {
        if (matmulOp.getNumDpsInits() != 1)
          continue;

        Value outs = matmulOp.getDpsInits()[0];

        // Find if there's a linalg.fill(0) that writes to this memref
        linalg::FillOp fillOp = nullptr;
        for (Operation *user : outs.getUsers()) {
          if (auto candidateFill = dyn_cast<linalg::FillOp>(user)) {
            if (candidateFill.getOutputs().size() == 1 &&
                candidateFill.getOutputs()[0] == outs) {
              // Check if it's a zero fill
              if (!candidateFill.getInputs().empty()) {
                Value fillVal = candidateFill.getInputs()[0];
                if (matchPattern(fillVal, m_AnyZeroFloat()) ||
                    matchPattern(fillVal, m_Zero())) {
                  fillOp = candidateFill;
                  break;
                }
              }
            }
          }
        }

        if (!fillOp)
          continue;

        // Pattern matched: linalg.fill(0, outs) exists and matmul uses outs.
        // We convert the matmul to loom.matmul.
        OpBuilder builder(matmulOp);

        builder.create<loom::MatmulOp>(matmulOp.getLoc(),
                                       matmulOp.getInputs()[0],
                                       matmulOp.getInputs()[1], outs);

        // Erase matmul
        matmulOp.erase();

        // Mark fill for erasure
        fillsToErase.insert(fillOp);
      }

      for (auto *op : fillsToErase) {
        op->erase();
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createFuseZeroFillMatmulPass() {
  return std::make_unique<FuseZeroFillMatmulPass>();
}
