#pragma once

/**
 * @file block_loading_pattern.h
 * @brief Block loading pattern detection and hoisting for loom dialect.
 * @details
 * This module provides functionality to:
 * 1. Identify loom.copy operations within scf.for loops
 * 2. Collect backward slice dependencies of copy operations
 * 3. Hoist copy operations and their dependencies outside the loop
 */

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include <optional>
#include <variant>

// Forward declaration for loom::CopyOp
namespace loom {
class CopyOp;
class ReinterpretCastOp;
} // namespace loom

namespace loom::affine {

/**
 * @brief Represents a loading block pattern centered around a loom.copy operation.
 * @details
 * A LoadingBlock captures:
 * - The loom.copy operation itself
 * - All operations in the backward slice (dependencies)
 * - The containing scf.for loop information
 * - Operations needed for hoisting transformation
 */
class LoadingBlock {
private:
    /// The outer for loop containing this loading block
    std::variant<mlir::scf::ForOp, mlir::affine::AffineForOp> outer_for_op_;
    
    /// The loom.copy operation that anchors this loading block
    mlir::Operation* copy_op_;
    
    /// All operations in the backward slice (dependencies of copy)
    llvm::SetVector<mlir::Operation*> backward_slice_;
    
    /// Operations created for replacement after hoisting
    llvm::SmallVector<mlir::Operation*, 4> replacement_block_;
    
    /// The bufferization.to_tensor operation following the copy
    mlir::Operation* to_tensor_op_;
    
    /// Loop induction variable
    mlir::Value loop_iv_;
    
    /// Loop upper bound
    mlir::Value loop_ub_;
    
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
     * @brief Reset loop attributes to a new outer affine for loop.
     * @param new_outer_for The new outer affine for loop operation.
     */
    void ResetLoopAttr(mlir::affine::AffineForOp new_outer_for);

    /**
     * @brief Find the outer affine for loop that contains the current outer_for_op_.
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
     * @brief Collect backward slice of a copy operation.
     * @details Traverses the def-use chain backward to collect all operations
     * that the copy depends on, excluding operations defined outside the loop.
     */
    void CollectBackwardSlice();

    /**
     * @brief Find the loom.reinterpret_cast in the backward slice.
     * @return The reinterpret_cast operation, or nullptr if not found.
     */
    mlir::Operation* FindReinterpretCast();

    /**
     * @brief Find the memref.alloc in the backward slice or as dst of copy.
     * @return The alloc operation, or nullptr if not found.
     */
    mlir::memref::AllocOp FindAlloc();

    /**
     * @brief Create hoisted operations before the loop.
     * @param builder The OpBuilder to use.
     */
    void CreateHoistedOps(mlir::OpBuilder& builder);

    /**
     * @brief Create replacement operations at the original location.
     * @details Creates subview operations to access hoisted data using loop IV.
     */
    void SetReplacementBlock();

public:
    /**
     * @brief Construct a LoadingBlock from a loom.copy operation.
     * @param copy_op The loom.copy operation.
     * @param for_op The containing scf.for loop.
     * @param to_tensor_op Optional bufferization.to_tensor operation.
     */
    LoadingBlock(mlir::Operation* copy_op, mlir::scf::ForOp for_op, 
                 mlir::Operation* to_tensor_op = nullptr);

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
    const llvm::SetVector<mlir::Operation*>& GetBackwardSlice() const {
        return backward_slice_;
    }

    /**
     * @brief Get the copy operation.
     * @return The copy operation.
     */
    mlir::Operation* GetCopyOp() const { return copy_op_; }
};

/**
 * @brief Build loading blocks by finding loom.copy operations in a for loop.
 * @param inner_most_for_op The innermost scf.for loop operation.
 * @param block_vec Output vector to store the found loading blocks.
 * @return LogicalResult indicating success or failure.
 */
mlir::LogicalResult BuildLoadingBlocks(mlir::scf::ForOp inner_most_for_op,
                                       llvm::SmallVector<LoadingBlock, 2>& block_vec);

/**
 * @brief Check if an operation is a loop operation (AffineForOp or scf::ForOp).
 * @param op The operation to check.
 * @return true if the operation is a loop operation.
 */
static inline bool isInWhitelist(mlir::Operation* op) {
    return mlir::isa<mlir::affine::AffineForOp, mlir::scf::ForOp>(op);
}

/**
 * @brief Check if an operation is in the blacklist (affine.parallel).
 * @param op The operation to check.
 * @return true if the operation is in the blacklist.
 */
static inline bool isInBlacklist(mlir::Operation* op) {
    return mlir::isa<mlir::affine::AffineParallelOp>(op);
}

/**
 * @brief Hoist a single loading block at the specified index.
 * @param inner_most_loop The innermost scf.for loop operation.
 * @param block_index The index of the block to hoist.
 * @return LogicalResult indicating success or failure.
 */
mlir::LogicalResult HoistSingleBlock(mlir::scf::ForOp inner_most_loop, size_t block_index);

} // namespace loom::affine
