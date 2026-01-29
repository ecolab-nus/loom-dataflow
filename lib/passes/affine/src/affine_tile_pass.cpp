#include "affine_tile.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace loom_affine {

namespace {
struct AffineTilePass
    : public PassWrapper<AffineTilePass, OperationPass<func::FuncOp>> {
  AffineTilePass() = default;
  AffineTilePass(int64_t factor, unsigned dim) {
    tilingFactor = factor;
    tileDimIndex = dim;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect>();
  }

  StringRef getArgument() const override { return "loom-affine-tile"; }
  StringRef getDescription() const override {
    return "Tile the first affine.parallel in the function by a fixed factor";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    affine::AffineParallelOp firstPar = nullptr;
    func.walk([&](affine::AffineParallelOp op) {
      // Only consider outermost affine.parallel (no affine.parallel ancestor).
      if (op->getParentOfType<affine::AffineParallelOp>())
        return;
      if (!firstPar)
        firstPar = op;
    });
    if (!firstPar) {
      func.emitError()
          << "loom-affine-tile: no outermost affine.parallel op found to tile";
      signalPassFailure();
      return;
    }

    if (tilingFactor <= 0) {
      func.emitError()
          << "loom-affine-tile: tiling-factor must be positive, got "
          << tilingFactor;
      signalPassFailure();
      return;
    }

    unsigned numDims = firstPar.getNumDims();
    if (tileDimIndex >= numDims) {
      firstPar.emitOpError()
          << "loom-affine-tile: tile-dim index out of range: " << tileDimIndex
          << " >= " << numDims;
      signalPassFailure();
      return;
    }

    if (failed(tileAffineParallel(firstPar, tilingFactor, tileDimIndex))) {
      firstPar.emitOpError() << "loom-affine-tile: tiling failed with factor "
                             << tilingFactor << " on dimension " << tileDimIndex
                             << "; see previous diagnostics for details";
      signalPassFailure();
    }
  }

  int64_t tilingFactor;
  unsigned tileDimIndex;
};
} // namespace

std::unique_ptr<mlir::Pass> createAffineTilePass(int64_t tilingFactor,
                                                 unsigned tileDimIndex) {
  return std::make_unique<AffineTilePass>(tilingFactor, tileDimIndex);
}

} // namespace loom_affine
