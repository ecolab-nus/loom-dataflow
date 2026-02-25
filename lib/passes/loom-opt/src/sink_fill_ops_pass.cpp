#include "Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/**
 * SinkFillOpsPass implements a "De-CSE" transformation for linalg.fill
 * operations.
 *
 * Frontends often share the results of linalg.fill (e.g., initializing multiple
 * tensors with -inf). When this sharing crosses loop boundaries, it creates
 * false "Eternal" virtual buffers in memory analysis.
 *
 * This pass clones and sinks each linalg.fill to its specific use site,
 * ensuring that:
 * 1. Each consumer gets a unique, locally-defined source tensor.
 * 2. cross-scope sharing is eliminated, enabling better Phi-Fusion and memory
 * reuse.
 */
struct SinkFillOpsPass
    : public PassWrapper<SinkFillOpsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SinkFillOpsPass)

  StringRef getArgument() const override { return "loom-sink-fill-ops"; }

  StringRef getDescription() const override {
    return "Clone and sink linalg.fill ops to their use sites to prevent "
           "cross-scope sharing";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp funcOp) {
      // 1. Collect all linalg.FillOp in the function.
      SmallVector<linalg::FillOp> fillOps;
      funcOp.walk([&](linalg::FillOp fillOp) { fillOps.push_back(fillOp); });

      // 2. Process in reverse order to stay safe if we had nested fills (rare
      // for tensors).
      for (auto fillOp : llvm::reverse(fillOps)) {
        Value result = fillOp.getResult(0);
        if (result.use_empty()) {
          fillOp.erase();
          continue;
        }

        // Snapshot uses
        SmallVector<OpOperand *> uses;
        for (OpOperand &use : result.getUses()) {
          uses.push_back(&use);
        }

        // For each use, clone the fill and sink it.
        for (OpOperand *use : uses) {
          Operation *user = use->getOwner();
          OpBuilder builder(user);

          // Re-use the original inputs (scalar and outs operand).
          // In tensor semantics, sharing the outs operand (usually
          // tensor.empty) is valid as each fill produces a new tensor value.
          auto clonedFill =
              cast<linalg::FillOp>(builder.clone(*fillOp.getOperation()));

          // Replace this specific use with the cloned result.
          use->set(clonedFill.getResult(0));
        }

        // The original fill is now unused.
        fillOp.erase();
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createSinkFillOpsPass() {
  return std::make_unique<SinkFillOpsPass>();
}
