#pragma once

#include "affine_tile.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <string>

namespace tmd_affine {

/**
 * Information about a single spatial dimension declared in the DF module.
 * If the size is not statically known (dynamic), `size` is set to
 * `std::nullopt` to model an effectively unbounded dimension.
 */
struct SpatialDimInfo {
  std::string name;
  std::optional<int64_t> size;
};

/**
 * Parse `df.spatial_dim` ops in `dfModule` and collect them in program order.
 *
 * Returns failure if none were found.
 */
mlir::LogicalResult
collectSpatialDims(mlir::ModuleOp dfModule,
                   llvm::SmallVectorImpl<SpatialDimInfo> &out);

/**
 * Greedily maps available spatial dimensions to affine.parallel loops in the
 * given module by repeatedly applying tiling to map one spatial dimension at a
 * time.
 *
 * - One spatial dimension can map only one affine.parallel (global constraint).
 * - One affine.parallel may be mapped by multiple spatial dimensions, provided
 *   the loop extent is large enough (or dynamic, treated as infinite).
 * - The inner affine.parallel created by tiling is annotated with
 *   `tmd.mapped_to` = string attribute naming the spatial dimension.
 *
 * The mapping proceeds in function order and considers outermost
 * `affine.parallel` first. At each step, it attempts to tile the selected
 * iterator dimension (`tileDimIndex`, default 0) with factor equal to the
 * spatial dimension size (or a default factor of 1 for dynamic sizes), and
 * marks the inner tiling loop. It continues until all spatial dimensions are
 * consumed or no further mapping is possible.
 */
mlir::LogicalResult mapSpatialDimsToAffine(mlir::ModuleOp affineModule,
                                           llvm::ArrayRef<SpatialDimInfo> dims,
                                           unsigned tileDimIndex);

/**
 * Enumerate all possible mappings of spatial dimensions to the iterators of
 * each outermost `affine.parallel`, cloning the original function for each
 * mapping and returning a new module containing all clones. Each clone is
 * named with a suffix encoding its mapping.
 *
 * - For each outermost affine.parallel with P iterators, and for a prefix of
 *   K spatial dims (K <= P), every assignment of those K dims to distinct
 *   iterators is considered, with tiling factors derived from dim sizes.
 * - Dynamic dim sizes still map, using factor 1. The inner loops are annotated
 *   with `tmd.mapped_to`.
 */
mlir::OwningOpRef<mlir::ModuleOp>
enumerateSpatialMappings(mlir::ModuleOp affineModule,
                         llvm::ArrayRef<SpatialDimInfo> dims);

} // namespace tmd_affine
