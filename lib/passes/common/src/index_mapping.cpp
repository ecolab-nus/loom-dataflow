#include "index_mapping.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

using namespace mlir;

namespace loom {

Value emitGlobalIndex2d(OpBuilder &b, Location loc, Value coreI, Value coreJ,
                        unsigned meshWidth, Value waveIV, unsigned tileWidth,
                        unsigned totalCores) {
  MLIRContext *ctx = b.getContext();

  // d0=coreI, d1=coreJ, d2=waveIV
  auto d0 = b.getAffineDimExpr(0);
  auto d1 = b.getAffineDimExpr(1);
  auto d2 = b.getAffineDimExpr(2);

  // 1. Linearize 2D mesh: linearIdx = coreI * meshWidth + coreJ
  auto linearIdx = d0 * meshWidth + d1;

  // 2. Compute offset: offset = linearIdx / tileWidth
  // This calculates how many "wave steps" we are into the logical dimension.
  auto offsetExpr = linearIdx.floorDiv(tileWidth);

  // 3. Combine with wave base to get the absolute index in the original
  // dimension. The waveIV covers blocks of size 'totalCores'. globalExpr =
  // offset + d2 * (totalCores / tileWidth) Wait, if waveIV is the residual loop
  // [0, ceilDiv(originalUB, totalCores)), then globalIdx is: (waveIV *
  // (totalCores / tileWidth)) + offset E.g., if totalCores=64, tileWidth=1,
  // meshWidth=8: linearIdx = [0, 64) offset = [0, 64) globalIdx = waveIV * 64 +
  // offset

  auto globalExpr = offsetExpr + d2 * (totalCores / tileWidth);

  auto map = AffineMap::get(3, 0, globalExpr, ctx);
  return affine::AffineApplyOp::create(b, loc, map,
                                       ValueRange{coreI, coreJ, waveIV});
}

} // namespace loom
