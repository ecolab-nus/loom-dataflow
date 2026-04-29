#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

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
 * `scf.for` loops using the specified iterator order.  The lower/upper
 * bound affine maps are materialized as plain index-typed arith ops so the
 * produced loops carry SSA bounds directly consumable by downstream
 * trip-count tracing.
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
 * Walk the def-use chain of \p v through arith binary ops (ceildivui, divui,
 * remui, muli, addi) to find a loom.sym op. Returns its symbol_ref
 * SymbolRefAttr, or nullopt if not found. Matching is by op-name string to
 * avoid pulling in loom dialect headers.
 * The caller guarantees at most one loom.sym is reachable from \p v.
 */
std::optional<mlir::SymbolRefAttr> traceToLoomSymRef(mlir::Value v);

} // namespace loom_affine

namespace loom {
namespace utils {

/**
 * @brief Compose and canonicalize all affine.apply operations in a function.
 * @details Walks every `affine.apply`, fully composes its map+operands and
 * canonicalizes them; rebuilds the op in place when the result differs, then
 * sweeps trivially-dead ops produced by the rewrite.
 */
void composeAndCanonicalizeAffineApplies(mlir::func::FuncOp func);

} // namespace utils
} // namespace loom
