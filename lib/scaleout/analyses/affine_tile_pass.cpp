#include "affine_tile.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace tmd_affine {

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

  StringRef getArgument() const override { return "tmd-affine-tile"; }
  StringRef getDescription() const override {
    return "Tile the first affine.parallel in the function by a fixed factor";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    affine::AffineParallelOp firstPar = nullptr;
    func.walk([&](affine::AffineParallelOp op) {
      if (!firstPar)
        firstPar = op;
    });
    if (!firstPar)
      return; // nothing to do
    if (failed(tileAffineParallel(firstPar, tilingFactor, tileDimIndex))) {
      signalPassFailure();
    }
  }

  int64_t tilingFactor = 8;
  unsigned tileDimIndex = 0;
};
} // namespace

std::unique_ptr<mlir::Pass> createAffineTilePass(int64_t tilingFactor,
                                                 unsigned tileDimIndex) {
  return std::make_unique<AffineTilePass>(tilingFactor, tileDimIndex);
}

} // namespace tmd_affine
