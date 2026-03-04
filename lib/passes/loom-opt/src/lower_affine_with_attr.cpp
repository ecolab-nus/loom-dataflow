#include "Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace loom {
namespace passes {

#define GEN_PASS_DEF_LOWERAFFINEWITHATTR
#include "Passes.h.inc"

namespace {
struct LowerAffineWithAttrPass
    : public impl::LowerAffineWithAttrBase<LowerAffineWithAttrPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](func::FuncOp func) { runOnFunction(func); });
  }

  void runOnFunction(func::FuncOp func) {
    MLIRContext *ctx = func.getContext();

    // 1. Process affine.parallel nests and convert to scf.parallel with
    // attributes We collect roots first because we'll be mutating the IR.
    SmallVector<affine::AffineParallelOp> rootOps;
    func.walk([&](affine::AffineParallelOp op) {
      if (!op->getParentOp() ||
          !isa<affine::AffineParallelOp>(op->getParentOp())) {
        rootOps.push_back(op);
      }
    });

    for (auto rootOp : rootOps) {
      if (failed(lowerParallelNest(rootOp)))
        return signalPassFailure();
    }

    // 2. Run standard lowering for the rest of affine operations (for, if,
    // load, store, apply)
    ConversionTarget target(*ctx);
    target.addIllegalDialect<affine::AffineDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<func::FuncDialect>();

    RewritePatternSet patterns(ctx);
    populateAffineToStdConversionPatterns(patterns);

    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      return signalPassFailure();
  }

  LogicalResult lowerParallelNest(affine::AffineParallelOp rootOp) {
    SmallVector<affine::AffineParallelOp> nest;
    affine::AffineParallelOp current = rootOp;
    while (current) {
      nest.push_back(current);
      // Fusable if body has 1 child op and 1 terminator, and child is
      // AffineParallelOp
      if (current.getBody()->getOperations().size() == 2 &&
          isa<affine::AffineParallelOp>(current.getBody()->front())) {
        current = cast<affine::AffineParallelOp>(current.getBody()->front());
      } else {
        break;
      }
    }

    OpBuilder builder(rootOp);
    Location loc = rootOp.getLoc();
    IRMapping mapping;

    SmallVector<Value> lowerBounds, upperBounds, steps;
    SmallVector<Attribute> mappedToDims;
    SmallVector<Attribute> iterTypes;
    bool hasAnyAttr = false;

    for (auto parOp : nest) {
      auto mt = parOp->getAttr("loom.mapped_to");
      auto it = parOp->getAttr("loom.iter_type");

      for (unsigned i = 0; i < parOp.getNumDims(); ++i) {
        // Lower bounds using affine.apply (temporary, will be lowered later)
        lowerBounds.push_back(builder.create<affine::AffineApplyOp>(
            loc, parOp.getLowerBoundMap(i), parOp.getLowerBoundsOperands()));
        upperBounds.push_back(builder.create<affine::AffineApplyOp>(
            loc, parOp.getUpperBoundMap(i), parOp.getUpperBoundsOperands()));
        steps.push_back(
            builder.create<arith::ConstantIndexOp>(loc, parOp.getSteps()[i]));

        mappedToDims.push_back(mt ? mt : builder.getUnitAttr());
        iterTypes.push_back(it ? it : builder.getUnitAttr());
      }
      if (mt || it)
        hasAnyAttr = true;
    }

    auto scfPar = builder.create<scf::ParallelOp>(
        loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &nestedBuilder, Location /*nestedLoc*/, ValueRange ivs) {
          // Map IVs
          unsigned scfIvIdx = 0;
          for (auto parOp : nest) {
            for (unsigned i = 0; i < parOp.getNumDims(); ++i) {
              mapping.map(parOp.getBody()->getArgument(i), ivs[scfIvIdx++]);
            }
          }

          // Clone body of innermost loop
          Block *innermostBody = nest.back().getBody();
          for (auto &innerOp : innermostBody->without_terminator()) {
            nestedBuilder.clone(innerOp, mapping);
          }
        });

    // Set mapped attributes if any
    if (hasAnyAttr) {
      scfPar->setAttr("loom.mapped_to_dims",
                      builder.getArrayAttr(mappedToDims));
      scfPar->setAttr("loom.iter_types", builder.getArrayAttr(iterTypes));
    }

    // Erase the old nest
    rootOp.erase();
    return success();
  }
};
} // namespace

std::unique_ptr<Pass> createLowerAffineWithAttrPass() {
  return std::make_unique<LowerAffineWithAttrPass>();
}

} // namespace passes
} // namespace loom
