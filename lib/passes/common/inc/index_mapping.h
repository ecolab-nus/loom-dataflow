#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"

namespace loom {

/**
 * @brief Description of how a hardware dimension maps to a logical iteration
 * dimension.
 */
struct HWDimMapping {
  unsigned hwDimIdx; // Which hardware dimension (0=x, 1=y)
  int64_t hwDimSize; // Size of that dimension (e.g., 8)
  mlir::Value iv;    // The IV for this hardware dimension
};

/**
 * @brief Emit affine.apply to compute global index from 2D hardware mesh IVs.
 *
 * For 8x8 mesh with tileWidth=1 (64 cores map to one dimension):
 *   globalIdx = (coreI * 8 + coreJ) / 1 + waveIV * 64
 *             = coreI * 8 + coreJ + waveIV * 64
 *
 * @param b          OpBuilder for IR creation
 * @param loc        Location for new ops
 * @param coreI      Outer mesh IV (row index in 2D mesh, e.g., d0 of 8x8)
 * @param coreJ      Inner mesh IV (col index in 2D mesh, e.g., d1 of 8x8)
 * @param meshWidth  Width of the mesh (e.g., 8 for 8x8)
 * @param waveIV     Wave iteration IV (residual after tiling)
 * @param tileWidth  Logical tile width for offset calculation (e.g., 1 if fully
 * mapped)
 * @param totalCores Product of all hw dim sizes (e.g., 64 for 8x8)
 * @return The computed global index Value
 *
 * TODO: Implement emitGlobalIndex1d for single-dimension hardware mapping
 */
mlir::Value emitGlobalIndex2d(mlir::OpBuilder &b, mlir::Location loc,
                              mlir::Value outerIV, mlir::Value innerIV,
                              unsigned innerDimSize, mlir::Value waveIV,
                              unsigned tileWidth, unsigned totalCores);

} // namespace loom
