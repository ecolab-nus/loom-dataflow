/**
 * @file spatial_mapping.h
 * @brief Query and enumerate mappings between hardware spatial dims and IR.
 * @details
 * This utility layer provides:
 * - Discovery of hardware spatial dimensions from the DF module.
 * - Greedy mapping of discovered dims to `affine.parallel` loops by tiling.
 * - Exhaustive enumeration of unique mappings, cloning functions per mapping
 *   and annotating loops with `tmd.mapped_to`.
 * - A Triton-shared specific enumerator that maps program grid dims {x,y,z} to
 *   hardware spatial dims and records the association as function attributes.
 *
 * Intended usage
 * - After `tmd-triton-shared-grid-to-parallel` introduced an outer
 *   `affine.parallel`, use these APIs to explore valid hardware placements and
 *   downstream loop-linearization options.
 */

#pragma once

#include "mlir/IR/BuiltinOps.h"
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

struct HardwareInfo {
  llvm::SmallVector<tmd_affine::SpatialDimInfo> spatialDimInfoVec;
  bool hasBidirInterconnect = false;

  bool skipPermutation() const {
    return spatialDimInfoVec.size() == 2 && hasBidirInterconnect;
  }
};

typedef llvm::SmallVector<llvm::SmallVector<unsigned>> DimBuckets;

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
GetHardwareInfoForExploration(mlir::ModuleOp dfModule, 
  HardwareInfo &hardwareInfo);


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
                         const HardwareInfo& hardwareInfo);

/**
 * \brief Enumerate spatial mappings and also convert the remaining
 * outermost `affine.parallel` into nested `affine.for` loops in all possible
 * iterator orders.
 *
 * This is identical to `enumerateSpatialMappings`, but after mapping spatial
 * dims (tiling), it replaces the surviving top-level `affine.parallel` with
 * `affine.for` nests. If the surviving parallel has `P` iterators, the
 * exploration clones the function `P!` times to cover all permutations of the
 * iterator ordering.
 */
mlir::OwningOpRef<mlir::ModuleOp>
enumerateSpatialMappingsWithOuterFors(mlir::ModuleOp affineModule,
                                      const HardwareInfo& hardwareInfo);

/**
 * \brief Enumerate mappings from Triton-shared grid dims to hardware spatial
 * dims.
 *
 * Given a module containing Triton-shared style kernels (with ABI providing
 * grid sizes and program_id.{x,y,z}), enumerate all unique assignments of the
 * grid dimensions \{x,y,z\} to the hardware spatial dimensions discovered in a
 * DF module. For each mapping, clone the function into a new output module and
 * annotate the clone with attributes encoding the mapping and tile factors.
 *
 * Notes:
 * - This pass does not rewrite the kernel body; it only records mapping
 *   metadata on the cloned functions for subsequent lowering passes that will
 *   rewrite index math or introduce explicit loops.
 * - The number of grid dimensions considered defaults to 3 (x,y,z). If fewer
 *   are desired, set \p numGridDims accordingly (1..3).
 * - Spatial dimension sizes, when static, are recorded as tile factors. When
 *   dynamic, a value of -1 is recorded.
 *
 * Function attributes set on each clone:
 * - `tmd.spatial_dim_names`: ArrayAttr<StringAttr> of spatial dim names.
 * - `tmd.spatial_dim_sizes`: ArrayAttr<IntegerAttr i64> (size or -1 if
 * dynamic).
 * - `tmd.grid_to_spatial_buckets`: ArrayAttr of ArrayAttr<IntegerAttr> where
 *    the outer index is the spatial dimension index, and the inner array lists
 *    the grid dim indices (0:x, 1:y, 2:z) assigned to that spatial dimension
 *    in order.
 *
 * Each clone's function name is suffixed to encode the mapping as
 * `__g<g>d<s>_...` tokens.
 */
mlir::OwningOpRef<mlir::ModuleOp>
enumerateTritonSharedSpatialMappings(mlir::ModuleOp module,
                                     const HardwareInfo& hardwareInfo,
                                     unsigned numGridDims = 3);

} // namespace tmd_affine
