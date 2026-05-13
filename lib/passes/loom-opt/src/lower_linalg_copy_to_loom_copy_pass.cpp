#include "Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"

#include "LoomDialect.h.inc"
#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace loom {
namespace passes {

#define GEN_PASS_DEF_LOWERLINALGCOPYTOLOOMCOPYPASS
#include "Passes.h.inc"

namespace {

static bool hasPriorOutsUse(Value destination, Operation *copyOp,
                            DominanceInfo &dominance) {
  for (OpOperand &use : destination.getUses()) {
    Operation *owner = use.getOwner();
    if (owner == copyOp)
      continue;

    auto dpsOp = dyn_cast<DestinationStyleOpInterface>(owner);
    if (!dpsOp || !dpsOp.isDpsInit(&use))
      continue;

    if (dominance.properlyDominates(owner, copyOp))
      return true;
  }
  return false;
}

struct LowerLinalgCopyToLoomCopyPass
    : public impl::LowerLinalgCopyToLoomCopyPassBase<
          LowerLinalgCopyToLoomCopyPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, loom::LoomDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    DominanceInfo dominance(module);
    SmallVector<linalg::CopyOp> copyOps;
    module.walk([&](linalg::CopyOp copyOp) { copyOps.push_back(copyOp); });

    for (linalg::CopyOp copyOp : copyOps) {
      auto linalgOp = cast<linalg::LinalgOp>(copyOp.getOperation());
      ValueRange inputs = linalgOp.getDpsInputs();
      ValueRange inits = linalgOp.getDpsInits();
      if (inputs.size() != 1 || inits.size() != 1)
        continue;

      Value source = inputs.front();
      Value destination = inits.front();
      if (!isa<MemRefType>(source.getType()) ||
          !isa<MemRefType>(destination.getType()))
        continue;

      OpBuilder builder(copyOp);
      auto l1Symbol = SymbolRefAttr::get(copyOp.getContext(), "mem_L1");
      auto reclaimAttr = builder.getBoolAttr(
          hasPriorOutsUse(destination, copyOp.getOperation(), dominance));

      loom::CopyOp::create(builder, copyOp.getLoc(), source, destination,
                           l1Symbol, l1Symbol, ValueRange{},
                           builder.getDenseI64ArrayAttr({1, 1}), Value{},
                           Value{}, Value{}, Value{}, reclaimAttr);
      copyOp.erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerLinalgCopyToLoomCopyPass() {
  return std::make_unique<LowerLinalgCopyToLoomCopyPass>();
}

} // namespace passes
} // namespace loom
