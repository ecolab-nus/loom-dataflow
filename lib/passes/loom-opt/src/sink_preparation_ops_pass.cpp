#include "Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// Loom dialect ops.
#include "LoomDialect.h.inc"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace {

/**
 * SinkPreparationOpsPass implements a "De-CSE" transformation for tensor
 * preparation operations.
 *
 * Frontends often share the results of operations that prepare tensors for
 * downstream compute, such as linalg.fill and loom.broadcast. When this sharing
 * crosses loop boundaries, it creates false "Eternal" virtual buffers in memory
 * analysis and can hide producer/consumer patterns from backend fusion.
 *
 * This pass clones and sinks each preparation op to its specific use site,
 * ensuring that:
 * 1. Each consumer gets a unique, locally-defined source tensor.
 * 2. cross-scope sharing is eliminated, enabling better Phi-Fusion and memory
 * reuse.
 * 3. preparation ops stay adjacent to consumers for backend recognition.
 */
struct SinkPreparationOpsPass
    : public PassWrapper<SinkPreparationOpsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SinkPreparationOpsPass)

  StringRef getArgument() const override {
    return "loom-sink-preparation-ops";
  }

  StringRef getDescription() const override {
    return "Clone and sink tensor preparation ops to their use sites to "
           "prevent cross-scope sharing";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    loom::LoomDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp funcOp) {
      // 1. Collect all preparation ops in the function.
      SmallVector<Operation *> preparationOps;
      funcOp.walk([&](Operation *op) {
        if (isa<linalg::FillOp, loom::BroadcastOp>(op))
          preparationOps.push_back(op);
      });

      // 2. Process in reverse order to stay safe when preparation ops consume
      // results of other preparation ops.
      for (Operation *preparationOp : llvm::reverse(preparationOps)) {
        if (preparationOp->getNumResults() != 1)
          continue;

        Value result = preparationOp->getResult(0);
        if (result.use_empty()) {
          preparationOp->erase();
          continue;
        }

        // Snapshot uses
        SmallVector<OpOperand *> uses;
        for (OpOperand &use : result.getUses()) {
          uses.push_back(&use);
        }

        // For each use, clone the preparation op and sink it.
        for (OpOperand *use : uses) {
          Operation *user = use->getOwner();
          OpBuilder builder(user);

          // Re-use the original inputs. In tensor semantics, sharing the outs
          // operand (usually tensor.empty) is valid as each cloned op produces a
          // new tensor value.
          Operation *clonedOp = builder.clone(*preparationOp);

          // Replace this specific use with the cloned result.
          use->set(clonedOp->getResult(0));
        }

        // The original preparation op is now unused.
        preparationOp->erase();
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createSinkPreparationOpsPass() {
  return std::make_unique<SinkPreparationOpsPass>();
}
