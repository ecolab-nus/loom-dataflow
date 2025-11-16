/**
 * @file tile_scf_for_to_l1.h
 * @brief Tiling of scf.for loops to fit within the single df.memory (L1).
 * @details
 * The pass verifies that the module declares exactly one `df.memory` and that
 * all `memref.alloc` operations are annotated with `tmd.alloc = { local=true,
 * memory_name = <that memory's label> }`. For each `scf.for` loop, it estimates
 * the per-iteration memory footprint as the sum of byte sizes of all static
 * `memref.alloc` inside the loop body. It then picks the largest power-of-two
 * tile factor such that `factor * perIterationBytes <= L1SizeBytes` and checks
 * the loop trip count is a multiple of this factor. If any requirement cannot
 * be proven, the pass fails.
 *
 * Transformation:
 * - Rewrites each eligible `scf.for` into an outer `affine.for` over tiles and
 *   an inner `scf.for` that iterates within the tile with the original step.
 * - Keeps the inner loop as `scf.for` and materializes the outer loop as
 *   `affine.for`.
 *
 * Tiling semantics used by this pass (conceptual):
 * - Given an original loop `scf.for %i = lb to ub step s`, and a tiling factor
 *   `f`, the inner loop performs exactly `f` iterations with step `s`, i.e.
 *   it ranges over an extent of `f * s` in the induction variable space.
 * - The outer loop enumerates tiles; conceptually it runs from `0` to
 *   `ceildiv(N, f)` where `N = (ub - lb) / s` is the trip count.
 * - The original loop index at iteration (tile t, inner k) is computed as
 *   `i = lb + (t * f + k) * s`.
 * - Implementation note: this pass currently requires that `N` is provably
 *   divisible by `f` and fails otherwise; under that assumption,
 *   `ceildiv(N, f)` becomes a plain integer division.
 */

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"

namespace tmd {
namespace passes {

/**
 * \brief Create a pass that tiles `scf.for` loops to fit the single df.memory.
 */
std::unique_ptr<mlir::Pass> createTileScfForToL1Pass();

/**
 * \brief Register the tiling pass for textual pipelines.
 */
void registerTileScfForToL1Pass();

/**
 * @brief LoopDescriptor is a unified representation of different types of loops.
 */
struct LoopDescriptor {
    enum Type { AFFINE_FOR, SCF_FOR };

    Type type;
    mlir::Operation* op;           // affine::AffineForOp* or scf::ForOp*
    mlir::Value inductionVar;
    int64_t tripCount;
    uint64_t tileFactor;

    // For affine.for
    mlir::AffineMap lowerBoundMap;
    mlir::AffineMap upperBoundMap;
    mlir::SmallVector<mlir::Value, 4> lbOperands;
    mlir::SmallVector<mlir::Value, 4> ubOperands;
    int64_t step;

    // For scf.for
    mlir::Value lowerBound;
    mlir::Value upperBound;
    mlir::Value stepValue;

    /**
    * @brief Create LoopDescriptor from affine.for
    * @param op affine.for operation
    * @param tileFactor tile factor
    * @return LoopDescriptor or failure
    */
    static mlir::FailureOr<LoopDescriptor> FromAffineFor(mlir::affine::AffineForOp op, uint64_t tileFactor);

    /**
    * @brief Create LoopDescriptor from scf.for
    * @param op scf.for operation
    * @param tileFactor tile factor
    * @return LoopDescriptor or failure
    */
    static mlir::FailureOr<LoopDescriptor> FromScfFor(mlir::scf::ForOp op, uint64_t tileFactor);
};


/**
 * @brief Func level multi-level nested loop Tiling Manager
 * @details Supports simultaneous tiling of multi-level nested loops (affine.for and scf.for),
 *          maintains global IRMapping to correctly handle affine_map remapping
 */
class TillingManager {
public:
    /**
    * @brief Constructor: func level tiling manager
    * @param func target function
    */
    TillingManager(mlir::func::FuncOp func) 
    : func_(func), op_builder_(func.getContext()), loc_(func.getLoc()) {}

    /**
    * @brief Configure tiling scheme (tile factors from outer to inner)
    * @param tileFactors tile factors for each loop, in order from outer to inner
    * @return success on success, failure otherwise
    */
    mlir::LogicalResult SetupTiling(mlir::ArrayRef<uint64_t> tileFactors);

    /**
    * @brief Execute overall transformation
    * @return success on success, failure otherwise
    */
    mlir::LogicalResult Transform();

private:
    mlir::func::FuncOp func_;
    mlir::OpBuilder op_builder_;
    mlir::Location loc_;

    mlir::SmallVector<LoopDescriptor> loopDescriptors_;  // loop descriptors from outer to inner
    mlir::SmallVector<uint64_t> tileFactors_;            // Tile factors

    /**
    * @brief Automatically discover nested loop structure (two affine.for + one scf.for)
    * @return loop array {affine.for1, affine.for2, scf.for} or failure
    */
    mlir::FailureOr<mlir::SmallVector<mlir::Operation*>> DiscoverNestedLoops();

    /**
    * @brief Validate discovered loop count and type
    * @param loops discovered loops
    * @param tileFactors tile factors
    * @return success on success, failure otherwise
    */
    mlir::LogicalResult ValidateLoopStructure(mlir::ArrayRef<mlir::Operation*> loops, mlir::ArrayRef<uint64_t> tileFactors);

    /**
    * @brief Use MLIR built-in API to tile two affine.for loops simultaneously
    * @param outerDesc outer loop descriptor
    * @param innerDesc inner loop descriptor
    * @param outerTileSize outer loop tile size
    * @param innerTileSize inner loop tile size
    * @return tiled loop nested structure (from outer to inner: outerTile, outerLocal, innerTile, innerLocal)
    *         or failure
    */
    mlir::FailureOr<mlir::SmallVector<mlir::affine::AffineForOp, 4>>
    TileTwoAffineFors(const LoopDescriptor& outerDesc, const LoopDescriptor& innerDesc,
                        unsigned outerTileSize, unsigned innerTileSize);
};
} // namespace passes
} // namespace tmd
