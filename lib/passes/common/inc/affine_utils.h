#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace loom_affine {

/**
 * Metadata about a single tiling operation for a dimension
 */
struct TilingMetadata {
  unsigned originalDimIdx; // Which dimension was tiled
  int64_t tilingFactor;    // The tiling factor used
  mlir::Value outerIV;     // IV of the outer loop created
  mlir::Value innerIV;     // IV of the inner loop created
};

/**
 * Extended result struct with tiling metadata
 */
struct TiledParallels {
  mlir::affine::AffineParallelOp tiled_org_;
  mlir::affine::AffineParallelOp tiled_new_;
  llvm::SmallVector<TilingMetadata> tilingMetadata;
};

/**
 * Tiles the given parallel loop and returns the newly created outer and inner
 * `affine.parallel` operations via `result` on success.
 */
mlir::LogicalResult tileAffineParallel(mlir::affine::AffineParallelOp op,
                                       int64_t tilingFactor,
                                       unsigned tileDimIndex,
                                       TiledParallels &result);

/**
 * Convert an outermost `affine.parallel` to a perfectly nested chain of
 * `affine.for` loops using the specified iterator order.
 */
mlir::LogicalResult ConvertParallelToNested(mlir::affine::AffineParallelOp par,
                                            llvm::ArrayRef<unsigned> order);

/**
 * Flatten consecutive ceildiv chains in an AffineExpr:
 *   (N ceildiv A) ceildiv B  -->  N ceildiv (A * B)
 * Constant factors in the combined denominator are folded together.
 * The transformation is mathematically valid for positive integer operands.
 */
mlir::AffineExpr flattenNestedCeilDiv(mlir::AffineExpr expr);

/**
 * Walk all affine.for upper-bound maps in `func` and apply
 * flattenNestedCeilDiv to every result expression.
 */
void flattenCeilDivInForBounds(mlir::func::FuncOp func);

} // namespace loom_affine
