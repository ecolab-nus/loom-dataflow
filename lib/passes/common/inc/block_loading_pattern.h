#pragma once

/**
 * @file block_loading_pattern.h
 * @brief Block loading pattern detection and hoisting for loom dialect.
 * @details
 * This module provides functionality to:
 * 1. Identify loom.copy_to_tensor operations within affine.for loops
 * 2. Collect backward slice dependencies of copy operations
 * 3. Hoist copy operations and their dependencies outside the loop
 */

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

// Forward declaration for loom ops
namespace loom {
class PackToTensorOp;
class AlloccOp;
class ViewOp;
} // namespace loom

namespace loom::affine {

/**
 * @brief Represents a loading block pattern centered around a loom.allocc
 * operation.
 * @details
 * A LoadingBlock captures:
 * - The loom.allocc operation that anchors this block
 * - The loom.copy_to_tensor operation that uses the allocc
 * - All operations in the backward slice (dependencies)
 * - The containing affine.for loop information
 * - Operations needed for hoisting transformation
 */
class LoadingBlock {
private:
  /// The outer for loop containing this loading block
  mlir::affine::AffineForOp outer_for_op_;

  /// The loom.allocc operation that anchors this loading block
  mlir::Operation *allocc_op_;

  /// The loom.copy_to_tensor operation that uses the allocc
  mlir::Operation *copy_to_tensor_op_;

  /// All operations in the backward slice (dependencies of copy)
  llvm::SetVector<mlir::Operation *> backward_slice_;

  /// Operations created for replacement after hoisting
  llvm::SmallVector<mlir::Operation *, 4> replacement_block_;

  /// Loop induction variable
  mlir::Value loop_iv_;

  /// Whether this block has been successfully hoisted
  bool is_valid_;

private:
  /**
   * @brief Set loop attributes from the outer_for_op_.
   * @details Extracts the induction variable and upper bound from the loop.
   */
  void SetLoopAttr();

  /**
   * @brief Clear all loop-related attributes.
   */
  void ClearLoopAttr();

  /**
   * @brief Get or reify the loop upper bound.
   * @param builder The OpBuilder to use.
   * @return The upper bound value.
   */
  mlir::Value getOrReifyLoopUB(mlir::OpBuilder &builder);

  /**
   * @brief Find the outer affine for loop that contains the current
   * outer_for_op_.
   * @return The outer affine for loop operation, or nullptr if not found.
   */
  mlir::affine::AffineForOp FindOuterAffineFor();

  /**
   * @brief Check if a value depends on the loop induction variable.
   * @param value The value to check.
   * @return true if value depends on loop_iv_, false otherwise.
   */
  bool DependsOnLoopIV(mlir::Value value);

  /**
   * @brief Collect backward slice of the loading chain.
   * @details Traverses the def-use chain backward to collect all operations
   * that the loading chain depends on, excluding operations defined outside the
   * loop.
   */
  void CollectBackwardSlice();

  /**
   * @brief Find the loom.allocc in the backward slice or as anchor.
   * @return The allocc operation, or nullptr if not found.
   */
  loom::AlloccOp FindAllocc();

  /**
   * @brief Find the loom.view consumed by the copy_to_tensor.
   * @return The view operation, or nullptr if not found.
   */
  loom::ViewOp FindView();

  /**
   * @brief Determine which dimension of the view depends on the loop IV.
   * @return The dimension index (0 or 1), or -1 if neither depends on IV.
   */
  int getMovingDimension();

  /**
   * @brief Compute the expanded shape for hoisted view.
   * @details Replaces the loop-varying dimension size with (loopUB *
   * blockSize).
   * @param builder The OpBuilder to use.
   * @return Vector of values representing the new shape.
   */
  llvm::SmallVector<mlir::Value, 2>
  inferHoistedViewShape(mlir::OpBuilder &builder);

  /**
   * @brief Generate outer_dims_perm for pack_to_tensor based on moving
   * dimension.
   * @return Permutation vector (e.g., [1, 0] or [0, 1]).
   */
  llvm::SmallVector<int64_t, 2> computePackPermutation();

  /**
   * @brief Get the inner tile size for pack_to_tensor.
   * @details This corresponds to the block size of the moving dimension.
   * @return The tile size value.
   */
  mlir::Value getInnerTileSize();

  /**
   * @brief Create hoisted operations before the loop.
   * @param builder The OpBuilder to use.
   */
  void CreateHoistedOps(mlir::OpBuilder &builder);

  /**
   * @brief Create replacement operations at the original location.
   * @details Creates slice operations to access hoisted data using loop IV.
   */
  void SetReplacementBlock();

public:
  /**
   * @brief Construct a LoadingBlock from a loom.allocc operation.
   * @param allocc_op The loom.allocc operation.
   * @param copy_op The loom.copy_to_tensor operation.
   * @param for_op The containing affine.for loop.
   */
  LoadingBlock(mlir::Operation *allocc_op, mlir::Operation *copy_op,
               mlir::affine::AffineForOp for_op);

  /**
   * @brief Recursively hoist the loading block to outer loops.
   * @param new_outer_for The new outer affine for loop to hoist to.
   */
  void HoistRec(mlir::affine::AffineForOp new_outer_for);

  /**
   * @brief Start the hoisting process.
   * @details Entry point for hoisting the loading block.
   */
  void Hoist();

  /**
   * @brief Perform a single hoist step.
   * @details Hoists operations from current loop to its parent loop.
   */
  void HoistLoadingBlock();

  /**
   * @brief Check if this block is valid (has been successfully hoisted).
   * @return true if the block is valid, false otherwise.
   */
  bool IsValid() const { return is_valid_; }

  /**
   * @brief Get the backward slice of this loading block.
   * @return Reference to the backward slice set.
   */
  const llvm::SetVector<mlir::Operation *> &GetBackwardSlice() const {
    return backward_slice_;
  }

  /**
   * @brief Get the copy operation.
   * @return The copy operation.
   */
  mlir::Operation *GetCopyOp() const { return copy_to_tensor_op_; }
};

/**
 * @brief Build loading blocks by finding loom.allocc operations in an affine
 * for loop.
 * @param inner_most_for_op The innermost affine.for loop operation.
 * @param block_vec Output vector to store the found loading blocks.
 * @return LogicalResult indicating success or failure.
 */
mlir::LogicalResult
BuildLoadingBlocks(mlir::affine::AffineForOp inner_most_for_op,
                   llvm::SmallVector<LoadingBlock, 2> &block_vec);

/**
 * @brief Check if an operation is a loop operation (AffineForOp).
 * @param op The operation to check.
 * @return true if the operation is a loop operation.
 */
static inline bool isInWhitelist(mlir::Operation *op) {
  return mlir::isa<mlir::affine::AffineForOp>(op);
}

/**
 * @brief Check if an operation is in the blacklist (affine.parallel).
 * @param op The operation to check.
 * @return true if the operation is in the blacklist.
 */
static inline bool isInBlacklist(mlir::Operation *op) {
  return mlir::isa<mlir::affine::AffineParallelOp>(op);
}

/**
 * @brief Hoist a single loading block at the specified index.
 * @param inner_most_loop The innermost affine.for loop operation.
 * @param block_index The index of the block to hoist.
 * @return LogicalResult indicating success or failure.
 */
mlir::LogicalResult HoistSingleBlock(mlir::affine::AffineForOp inner_most_loop,
                                     size_t block_index);

} // namespace loom::affine
