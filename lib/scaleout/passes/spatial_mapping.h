#pragma once

#include "affine_tile.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <string>

namespace tmd_affine {

/**
 * \brief Hardware spatial dimension description parsed from the DF module.
 *
 * A spatial dimension is declared by `df.spatial_dim` in the dataflow (DF)
 * module. This structure captures a display name for diagnostics and an
 * optional static size.
 *
 * - When the dimension size is not statically known (dynamic), `size` is
 *   set to `std::nullopt` to approximate an unbounded capacity.
 */
struct SpatialDimInfo {
  std::string name;
  std::optional<int64_t> size;
};

/**
 * \brief Collect spatial dimensions from a DF module.
 *
 * Scans `dfModule` for `df.spatial_dim` operations (in program order) and
 * appends each discovered dimension as a `SpatialDimInfo` into `out`.
 *
 * \param dfModule DF module containing hardware declarations.
 * \param out      Output vector to append discovered dimensions to.
 * \return success if at least one dimension was found; failure otherwise.
 */
mlir::LogicalResult
collectSpatialDims(mlir::ModuleOp dfModule,
                   llvm::SmallVectorImpl<SpatialDimInfo> &out);

/**
 * \brief Greedily map spatial dimensions to `affine.parallel` loops by tiling.
 *
 * Applies repeated `tileAffineParallel` to the outermost `affine.parallel`
 * operations in `affineModule`, consuming spatial dimensions in order. The
 * inner loop created at each step is annotated with `tmd.mapped_to` to
 * indicate the mapped dimension.
 *
 * Semantics and constraints:
 * - One spatial dimension is used at most once globally (greedy consumption).
 * - A single `affine.parallel` may be mapped by multiple dimensions if the
 *   iterator extent allows (dynamic extents are treated as unbounded).
 * - The tiling factor equals the dimension size when static, or 1 if dynamic.
 *
 * \param affineModule Module containing the Affine program to transform.
 * \param dims         Spatial dimensions to map in order of preference.
 * \param tileDimIndex Iterator index within each `affine.parallel` to tile.
 * \return success if at least one mapping was applied; failure otherwise.
 */
mlir::LogicalResult mapSpatialDimsToAffine(mlir::ModuleOp affineModule,
                                           llvm::ArrayRef<SpatialDimInfo> dims,
                                           unsigned tileDimIndex);

/**
 * \brief Enumerate all unique mappings and emit one function clone per mapping.
 *
 * For each function, find its first outermost `affine.parallel` with `P`
 * iterators and compute the set of all unique mappings from `D` spatial
 * dimensions to those iterators using the two-step algorithm:
 *
 * 1) Generate all partitions of the `D` dims into `P` ordered buckets
 *    (iterator groups). This removes redundancy from pure assignment order.
 * 2) For each partition, iterate iterator buckets `0..P-1`; within each
 *    bucket, apply `tileAffineParallel` once per dimension in the chosen
 *    order (all permutations of that bucket) with factor equal to the static
 *    size or 1 if dynamic.
 *
 * Each mapping yields a clone of the original function in a new output module;
 * every created inner loop is annotated with `tmd.mapped_to`, and the function
 * name is suffixed to encode the mapping.
 *
 * \param affineModule Source Affine module to enumerate mappings for.
 * \param dims         Spatial dimensions defining the mapping space.
 * \return A new module containing one clone per mapping (original functions
 *         are preserved in the input module).
 */
mlir::OwningOpRef<mlir::ModuleOp>
enumerateSpatialMappings(mlir::ModuleOp affineModule,
                         llvm::ArrayRef<SpatialDimInfo> dims);

} // namespace tmd_affine
