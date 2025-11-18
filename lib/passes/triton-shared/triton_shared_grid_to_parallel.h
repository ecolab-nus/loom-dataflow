/**
 * @file triton_shared_grid_to_parallel.h
 * @brief Replace grid index ABI with a 3-D `affine.parallel` loop nest.
 * @details
 * Converts the Triton-shared grid calling convention
 * `(sizeX, sizeY, sizeZ, idxX, idxY, idxZ)` into a single 3-D
 * `affine.parallel` whose IVs replace the index arguments, and erases the
 * index arguments from the function signature.
 *
 * Usage
 * - Register: `tmd::passes::registerTritonSharedGridToParallelPass()`
 * - CLI: `--tmd-triton-shared-grid-to-parallel`
 * - Run after `--tmd-triton-shared-affinize`.
 *
 * Constraints
 * - Function must have at least 6 arguments; the last six are interpreted as
 *   `(sizeX, sizeY, sizeZ, idxX, idxY, idxZ)`.
 * - Sizes are used as dynamic upper bounds; indices are fully replaced.
 *
 * Notes
 * - The resulting parallel region becomes the anchor for spatial mapping and
 *   may be further tiled/annotated by subsequent passes.
 */

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace tmd {
namespace passes {
class ScfForWrapperProcessor {
public:
    static void processParallelBody(mlir::affine::AffineParallelOp& parallelOp) {
    mlir::scf::ForOp scfFor = findScfForInParallelBody(parallelOp);
    if (!scfFor) return;
    
    auto [blockm, blockn] = extractBlockSizes(scfFor);
    auto [mLoop, nLoop] = createTwoLevelLoops(parallelOp, blockm, blockn);
    moveOperationsToNestedLoops(parallelOp, scfFor, nLoop);
    updateIndexCalculations(nLoop, parallelOp.getIVs()[0], 
                            parallelOp.getIVs()[1], blockm, blockn);
    }

private:
    static mlir::scf::ForOp findScfForInParallelBody(mlir::affine::AffineParallelOp& parallelOp);
    static std::pair<int64_t, int64_t> extractBlockSizes(mlir::scf::ForOp& scfFor);
    static std::pair<mlir::affine::AffineForOp, mlir::affine::AffineForOp> 
    createTwoLevelLoops(mlir::affine::AffineParallelOp& parallelOp, int64_t blockm, int64_t blockn);
    static void moveOperationsToNestedLoops(mlir::affine::AffineParallelOp parallelOp,
                                            mlir::scf::ForOp scfFor, mlir::affine::AffineForOp nLoop);
    static void updateIndexCalculations(mlir::affine::AffineForOp nLoop, 
                                        mlir::Value ivX, mlir::Value ivY, int64_t blockm, int64_t blockn);
};


class AffineParallelConverter {
public:
    static void convertFunction(mlir::func::FuncOp func) {
    if (!hasValidSignature(func)) return;
    
    auto [sizeX, sizeY, sizeZ, idxX, idxY, idxZ] = extractArgs(func);
    auto parallelOp = createAffineParallel(func, sizeX, sizeY, sizeZ);
    moveOriginalBody(parallelOp, func);
    replaceIndexUses(parallelOp, idxX, idxY, idxZ);
    cleanupFunctionSignature(func, idxX, idxY, idxZ);
    optimizeParallelOp(parallelOp);
    }

private:
    static bool hasValidSignature(mlir::func::FuncOp func);
    static std::tuple<mlir::Value, mlir::Value, mlir::Value, mlir::Value, mlir::Value, mlir::Value> 
    extractArgs(mlir::func::FuncOp func);
    static mlir::affine::AffineParallelOp 
    createAffineParallel(mlir::func::FuncOp func, mlir::Value sizeX, mlir::Value sizeY, mlir::Value sizeZ);
    static void moveOriginalBody(mlir::affine::AffineParallelOp parallelOp, mlir::func::FuncOp func);
    static void replaceIndexUses(mlir::affine::AffineParallelOp parallelOp, 
                                mlir::Value idxX, mlir::Value idxY, mlir::Value idxZ);
    static void cleanupFunctionSignature(mlir::func::FuncOp func, mlir::Value idxX, mlir::Value idxY, mlir::Value idxZ);
    static void optimizeParallelOp(mlir::affine::AffineParallelOp parallelOp);
};


/**
 * \brief Create a pass that wraps function bodies with a 3-D affine.parallel
 *        and removes grid index arguments.
 *
 * Expected calling convention (from Triton-shared after affinization):
 * - Last 6 function arguments are: sizeX, sizeY, sizeZ, idxX, idxY, idxZ.
 * - The pass inserts a single three-dimensional `affine.parallel` with upper
 *   bounds (sizeX, sizeY, sizeZ) and step (1, 1, 1).
 * - All uses of (idxX, idxY, idxZ) inside the function body are replaced by the
 *   induction variables of the new parallel loop.
 * - The last three arguments (idxX, idxY, idxZ) are erased from the function
 *   signature.
 */
std::unique_ptr<mlir::Pass> createTritonSharedGridToParallelPass();

/**
 * \brief Register the grid-to-parallel pass for textual pipelines.
 */
void registerTritonSharedGridToParallelPass();

} // namespace passes
} // namespace tmd
