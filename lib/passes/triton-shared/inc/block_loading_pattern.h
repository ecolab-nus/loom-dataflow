#pragma once


#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <variant>


namespace tmd::affine {

enum LoopType {
    UNKNOWN = 0,
    AFFINE,
    SCF
};

class LoadingBlock {
private:
    std::variant<mlir::scf::ForOp, mlir::affine::AffineForOp> outer_for_op_;
    llvm::SmallVector<mlir::Operation *, 4> op_block_;
    llvm::SmallVector<mlir::Operation *, 2> replacement_block_;
    mlir::Operation* org_to_tensor_op_;

    mlir::Value loop_iv_;
    mlir::Value loop_ub_;

    std::optional<int64_t> mem_req_bytes_as_const_;
    std::optional<int64_t> loop_ub_as_const_;
    std::optional<std::vector<int64_t>> block_size_;
    int64_t coeff_loop_iv_;
    bool is_valid_;  // Whether this block has been hoisted using CreateHoistedOpsSimple at least once 


private:
    int64_t GetAllocSizeAsConst(mlir::memref::AllocOp alloc_op);
    std::optional<std::vector<int64_t>> GetBlockSize(mlir::memref::ReinterpretCastOp reinterpret_op);
    std::optional<int64_t> GetConstantIntValue(mlir::Value v);
    int64_t ExtractDimCoefficientRec(mlir::AffineExpr expr, int64_t tar_dim_pos);
    void SetLoopAttr();
    void ClearLoopAttr();
    void ResetLoopAttr(mlir::affine::AffineForOp new_outer_for);
    mlir::affine::AffineForOp FindOuterAffineFor();
    std::optional<unsigned> GetIvIndex(mlir::OperandRange& original_operands);


    void SetReplacementBlock();
    LoopType GetLoopType();
    bool IsIndependent();
    void ReplaceOpBlock(
        mlir::affine::AffineApplyOp new_apply,
        mlir::memref::ReinterpretCastOp new_reinterpret,
        mlir::memref::AllocOp new_alloc,
        mlir::memref::CopyOp new_copy);
    mlir::affine::AffineApplyOp CreateHoistedApply(
        mlir::OpBuilder& builder,
        mlir::affine::AffineApplyOp original_apply,
        std::optional<unsigned> index_to_remove);
    void HoistLoadingBlock();
    void CreateHoistedOpsWithReshape(
        mlir::OpBuilder& builder,
        mlir::affine::AffineApplyOp new_apply,
        mlir::memref::ReinterpretCastOp org_reinterpret,
        mlir::memref::AllocOp org_alloc,
        mlir::memref::ReinterpretCastOp& new_reinterpret,
        mlir::memref::AllocOp& new_alloc);
    void CreateHoistedOpsSimple(
        mlir::OpBuilder& builder,
        mlir::affine::AffineApplyOp new_apply,
        mlir::memref::ReinterpretCastOp org_reinterpret,
        mlir::memref::AllocOp org_alloc,
        mlir::memref::ReinterpretCastOp& new_reinterpret,
        mlir::memref::AllocOp& new_alloc);

public:
    LoadingBlock(llvm::SmallVector<mlir::Operation *> op_block, mlir::scf::ForOp for_op);
    void HoistRec(mlir::affine::AffineForOp new_outer_for);
    void Hoist();
    
    /// @brief Check if this block is valid (has been hoisted using CreateHoistedOpsSimple at least once).
    /// @return true if the block is valid, false otherwise.
    bool IsValid() const { return is_valid_; }

};

/// @brief Build loading blocks for the given forOp
/// @param forOp The innermost for loop operation (can be AffineForOp or scf::ForOp)
/// @param block_vec Output vector to store the found loading blocks
/// @return LogicalResult indicating success or failure
mlir::LogicalResult BuildLoadingBlocks(mlir::scf::ForOp inner_most_for_op, llvm::SmallVector<LoadingBlock, 2>& block_vec);

/// @brief Check if an operation is in the whitelist (AffineForOp or scf::ForOp)
/// @param op The operation to check
/// @return true if the operation is in the whitelist
static inline bool isInWhitelist(mlir::Operation *op) {
    return mlir::isa<mlir::affine::AffineForOp, mlir::scf::ForOp>(op);
}

/// @brief Check if an operation is in the blacklist (affine.parallel)
/// @param op The operation to check
/// @return true if the operation is in the blacklist
static inline bool isInBlacklist(mlir::Operation *op) {
    return mlir::isa<mlir::affine::AffineParallelOp>(op);
}


mlir::LogicalResult MatchAndHoist(mlir::scf::ForOp inner_most_loop);

/// @brief Hoist a single loading block at the specified index for the innermost loop.
/// @param inner_most_loop The innermost scf.for loop operation.
/// @param block_index The index of the block to hoist.
/// @return LogicalResult indicating success or failure.
mlir::LogicalResult HoistSingleBlock(mlir::scf::ForOp inner_most_loop, size_t block_index);
} // namespace tmd::affine