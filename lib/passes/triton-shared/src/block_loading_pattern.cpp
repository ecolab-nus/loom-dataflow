#include "block_loading_pattern.h"
#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

namespace tmd::affine {

/**
 * @brief Get the allocation size as a constant from the alloc operation.
 * @param alloc_op The memref allocation operation.
 * @return The size in bytes as a constant, or 0 if not found.
 */
int64_t LoadingBlock::GetAllocSizeAsConst(mlir::memref::AllocOp alloc_op) {
    if (auto dict = alloc_op->getAttrOfType<mlir::DictionaryAttr>("tmd.alloc")) {
        if (auto sizeAttr = dict.get("size")) {
            if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(sizeAttr)) {
                return static_cast<int64_t>(intAttr.getInt());
            }
        }
    }
    return 0;
}

/**
 * @brief Get the block sizes as constants from the reinterpret_cast operation.
 * @param reinterpret_op The memref reinterpret_cast operation.
 * @return The block sizes as a vector of constants, or std::nullopt if not all sizes are constants.
 */
std::optional<std::vector<int64_t>> LoadingBlock::GetBlockSize(mlir::memref::ReinterpretCastOp reinterpret_op) {
    std::vector<int64_t> block_sizes;
    
    // Try to get from static sizes attribute first
    if (auto static_sizes_attr = reinterpret_op.getStaticSizesAttr()) {
        auto static_sizes = static_sizes_attr.asArrayRef();
        bool all_static = true;
        for (int64_t size : static_sizes) {
            if (size == mlir::ShapedType::kDynamic) {
                all_static = false;
                break;
            }
            block_sizes.push_back(size);
        }
        if (all_static && !block_sizes.empty()) {
            return block_sizes;
        }
    }
    
    // If not found in static sizes, try from result type shape
    auto result_type = llvm::cast<mlir::MemRefType>(reinterpret_op.getResult().getType());
    auto shape = result_type.getShape();
    bool all_static = true;
    block_sizes.clear();
    for (int64_t dim : shape) {
        if (dim == mlir::ShapedType::kDynamic) {
            all_static = false;
            break;
        }
        block_sizes.push_back(dim);
    }
    if (all_static && !block_sizes.empty()) {
        return block_sizes;
    }
    
    // Last resort: try from dynamic sizes
    auto sizes = reinterpret_op.getSizes();
    if (!sizes.empty()) {
        block_sizes.clear();
        bool all_const = true;
        for (auto size_val : sizes) {
            if (auto const_val = GetConstantIntValue(size_val)) {
                block_sizes.push_back(const_val.value());
            } else {
                all_const = false;
                break;
            }
        }
        if (all_const && !block_sizes.empty()) {
            return block_sizes;
        }
    }
    
    return std::nullopt;
}

/**
 * @brief Extract a constant integer value from an MLIR value.
 * @param v The MLIR value to extract the constant from.
 * @return The constant integer value if the value is a constant, std::nullopt otherwise.
 */
std::optional<int64_t> LoadingBlock::GetConstantIntValue(mlir::Value v) {
    if (!v) return std::nullopt;
    if (auto constIndex = v.getDefiningOp<mlir::arith::ConstantIndexOp>())
        return constIndex.value();
    if (auto constInt = v.getDefiningOp<mlir::arith::ConstantIntOp>())
        return constInt.value();
    return std::nullopt;
}

int64_t LoadingBlock::ExtractDimCoefficientRec(mlir::AffineExpr expr, int64_t tar_dim_pos) {
    if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
        return dimExpr.getPosition() == tar_dim_pos ? 1 : 0;
    }
    if (expr.getKind() == mlir::AffineExprKind::Add) {
        auto binary = llvm::cast<mlir::AffineBinaryOpExpr>(expr);
        return ExtractDimCoefficientRec(binary.getLHS(), tar_dim_pos) +
               ExtractDimCoefficientRec(binary.getRHS(), tar_dim_pos);
    }
    if (expr.getKind() == mlir::AffineExprKind::Mul) {
        auto binary = llvm::cast<mlir::AffineBinaryOpExpr>(expr);
        auto lhs = binary.getLHS();
        auto rhs = binary.getRHS();
        if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(lhs)) {
            if (dimExpr.getPosition() == tar_dim_pos) {
                if (auto constExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs))
                    return constExpr.getValue();
            }
        }
        if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(rhs)) {
            if (dimExpr.getPosition() == tar_dim_pos) {
                if (auto constExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(lhs))
                    return constExpr.getValue();
            }
        }
    }
    return 0;
}

/**
 * @brief Set loop attributes from the outer_for_op_ variant.
 *
 * @details Extracts the induction variable and upper bound from the current outer loop operation.
 */
void LoadingBlock::SetLoopAttr() {
    loop_iv_ = std::visit([](auto& op) -> mlir::Value {
        return op.getInductionVar();
    }, outer_for_op_);

    loop_ub_ = std::visit([](auto& op) -> mlir::Value {
        if constexpr (std::is_same_v<std::decay_t<decltype(op)>, mlir::scf::ForOp>) {
            return op.getUpperBound();
        } else {
            return mlir::Value{};
        }
    }, outer_for_op_);

    loop_ub_as_const_ = GetConstantIntValue(loop_ub_);
}

/**
 * @brief Clear all loop-related attributes.
 *
 * @details Resets the induction variable, upper bound, and constant upper bound to null/empty.
 */
void LoadingBlock::ClearLoopAttr() {
    loop_iv_ = nullptr;
    loop_ub_ = nullptr;
    loop_ub_as_const_ = std::nullopt;
    coeff_loop_iv_ = 0;
}

/**
 * @brief Reset loop attributes to a new outer affine for loop.
 * @param new_outer_for The new outer affine for loop operation.
 */
void LoadingBlock::ResetLoopAttr(mlir::affine::AffineForOp new_outer_for) {
    ClearLoopAttr();
    if (new_outer_for) {
        outer_for_op_ = new_outer_for;
        SetLoopAttr();
    }
}

/**
 * @brief Find the outer affine for loop that contains the current outer_for_op_.
 * @return The outer affine for loop operation, or nullptr if not found.
 */
mlir::affine::AffineForOp LoadingBlock::FindOuterAffineFor() {
    mlir::Operation* current_op = std::visit([](auto&& forOp) -> mlir::Operation* {
        return forOp.getOperation();
    }, outer_for_op_);
    
    if (!current_op) {
        return mlir::affine::AffineForOp(nullptr);
    }
    
    for (mlir::Operation* parent = current_op->getParentOp(); parent; parent = parent->getParentOp()) {
        if (auto affine_for = mlir::dyn_cast<mlir::affine::AffineForOp>(parent)) {
            return affine_for;
        }
        if (mlir::isa<mlir::func::FuncOp>(parent)) {
            break;
        }
    }
    
    return mlir::affine::AffineForOp(nullptr);
}

/**
 * @brief Find the index of the loop induction variable in the operand range.
 * @param original_operands The operand range to search in.
 * @return The index of the loop induction variable, or std::nullopt if not found.
 */
std::optional<unsigned> LoadingBlock::GetIvIndex(mlir::OperandRange& original_operands) {
    unsigned loop_iv_index = 0;
    bool found = false;
    for (unsigned i = 0; i < original_operands.size(); ++i) {
        if (original_operands[i] == loop_iv_) {
            loop_iv_index = i;
            found = true;
            break;
        }
    }
    
    if (!found) {
        return std::nullopt;
    }

    return loop_iv_index;
}

/**
 * @brief Create a replacement block for accessing hoisted L1 data at the original location.
 *
 * @details This creates new operations that use the loop induction variable to access the hoisted data.
 */
void LoadingBlock::SetReplacementBlock() {
    // Get original operations
    auto new_reinterpret = llvm::dyn_cast<mlir::memref::ReinterpretCastOp>(op_block_[1]);
    auto new_alloc = llvm::dyn_cast<mlir::memref::AllocOp>(op_block_[2]);
    auto original_to_tensor = llvm::dyn_cast<mlir::bufferization::ToTensorOp>(org_to_tensor_op_);
    
    // Set insertion point after the last operation in op_block_
    // mlir::OpBuilder builder(op_block_.back()->getBlock(),
    //                 ++mlir::Block::iterator(op_block_.back()));
    mlir::OpBuilder builder(org_to_tensor_op_);
    
    // Validate required data
    if (!block_size_.has_value() || block_size_.value().size() < 2) {
        return;
    }
    
    // 1. Prepare offsets: [loop_iv_, 0, 0]
    llvm::SmallVector<mlir::OpFoldResult> offsets;
    offsets.push_back(loop_iv_);  // Dynamic value for first dimension
    offsets.push_back(mlir::IntegerAttr::get(
        mlir::IndexType::get(builder.getContext()), 0));  // Static constant 0
    offsets.push_back(mlir::IntegerAttr::get(
        mlir::IndexType::get(builder.getContext()), 0));  // Static constant 0
    
    // 2. Prepare sizes: [1, block_size_[0], block_size_[1]]
    llvm::SmallVector<mlir::OpFoldResult> sizes;
    sizes.push_back(mlir::IntegerAttr::get(
        mlir::IndexType::get(builder.getContext()), 1));  // Static constant 1
    sizes.push_back(mlir::IntegerAttr::get(
        mlir::IndexType::get(builder.getContext()), block_size_.value()[0]));
    sizes.push_back(mlir::IntegerAttr::get(
        mlir::IndexType::get(builder.getContext()), block_size_.value()[1]));
    
    // 3. Prepare strides: [1, 1, 1]
    llvm::SmallVector<mlir::OpFoldResult> strides;
    strides.push_back(mlir::IntegerAttr::get(
        mlir::IndexType::get(builder.getContext()), 1));
    strides.push_back(mlir::IntegerAttr::get(
        mlir::IndexType::get(builder.getContext()), 1));
    strides.push_back(mlir::IntegerAttr::get(
        mlir::IndexType::get(builder.getContext()), 1));
    
    // 4. Compute rank-reduced result type for subview
    // The subview will have shape [1, block_size_[0], block_size_[1]]
    // but we want to reduce it to [block_size_[0], block_size_[1]] to match
    // the original tensor type (2D layout)
    auto source_memref_type = llvm::cast<mlir::MemRefType>(new_alloc.getResult().getType());
    llvm::SmallVector<int64_t> reduced_shape;
    reduced_shape.push_back(block_size_.value()[0]);
    reduced_shape.push_back(block_size_.value()[1]);
    
    auto reduced_result_type = mlir::memref::SubViewOp::inferRankReducedResultType(
        reduced_shape,
        source_memref_type,
        offsets,
        sizes,
        strides);
    
    // 5. Create subview operation with rank-reduced result type (2D)
    auto new_subview = builder.create<mlir::memref::SubViewOp>(
        new_reinterpret.getLoc(),
        reduced_result_type,  // Explicitly specify rank-reduced 2D type
        new_alloc.getResult(),
        offsets,
        sizes,
        strides);
    
    replacement_block_.push_back(new_subview);
    
    // 6. Update the input operand of org_to_tensor_op_
    if (original_to_tensor) {
        builder.setInsertionPointAfter(original_to_tensor);
        auto new_to_tensor = builder.create<mlir::bufferization::ToTensorOp>(
            original_to_tensor.getLoc(),
            original_to_tensor.getType(),
            new_subview.getResult(),
            original_to_tensor.getRestrict(),
            original_to_tensor.getWritable());
        
        original_to_tensor.getResult().replaceAllUsesWith(new_to_tensor.getResult());
        original_to_tensor.erase();
        org_to_tensor_op_ = new_to_tensor;
    }
}

/**
 * @brief Get the type of the outer loop.
 * @return The loop type (AFFINE, SCF, or UNKNOW).
 */
LoopType LoadingBlock::GetLoopType() {
    if (std::holds_alternative<mlir::affine::AffineForOp>(outer_for_op_)) return LoopType::AFFINE;
    if (std::holds_alternative<mlir::scf::ForOp>(outer_for_op_)) return LoopType::SCF;
    return LoopType::UNKNOWN;
}

/**
 * @brief Check if the affine.apply operation is independent of the loop induction variable.
 * @return true if the operation does not depend on loop_iv_, false otherwise.
 */
bool LoadingBlock::IsIndependent() {
    if (auto apply_op = llvm::dyn_cast<mlir::affine::AffineApplyOp>(op_block_[0])) {
        mlir::ValueRange operands = apply_op->getOperands();
        if (llvm::is_contained(operands, loop_iv_)) return false;
        else return true;
    }
    return false;
}

/**
 * @brief Replace the original operations with new hoisted operations and erase the old ones.
 * @param new_apply The new affine.apply operation.
 * @param new_reinterpret The new memref.reinterpret_cast operation.
 * @param new_alloc The new memref.alloc operation.
 * @param new_copy The new memref.copy operation.
 */
void LoadingBlock::ReplaceOpBlock(
    mlir::affine::AffineApplyOp new_apply,
    mlir::memref::ReinterpretCastOp new_reinterpret,
    mlir::memref::AllocOp new_alloc,
    mlir::memref::CopyOp new_copy) {
    
    auto org_apply = llvm::dyn_cast<mlir::affine::AffineApplyOp>(op_block_[0]);
    auto org_reinterpret = llvm::dyn_cast<mlir::memref::ReinterpretCastOp>(op_block_[1]);
    auto org_alloc = llvm::dyn_cast<mlir::memref::AllocOp>(op_block_[2]);
    
    org_apply->getResult(0).replaceAllUsesWith(new_apply.getResult());
    org_reinterpret->getResult(0).replaceAllUsesWith(new_reinterpret.getResult());
    org_alloc->getResult(0).replaceAllUsesWith(new_alloc.getResult());
    
    for (int i = op_block_.size() - 1; i >= 0; --i) {
        if (op_block_[i] && op_block_[i]->getBlock()) {
            bool hasUses = false;
            for (auto result : op_block_[i]->getResults()) {
                if (!result.use_empty()) {
                    hasUses = true;
                    break;
                }
            }
            if (!hasUses) {
                op_block_[i]->erase();
            }
        }
    }
    
    op_block_[0] = new_apply.getOperation();
    op_block_[1] = new_reinterpret.getOperation();
    op_block_[2] = new_alloc.getOperation();
    op_block_[3] = new_copy.getOperation();
}

/**
 * @brief Create a hoisted affine.apply operation by removing specified operands.
 * @param builder The OpBuilder to create operations.
 * @param original_apply The original affine.apply operation.
 * @param index_to_remove The index of the operand to remove, or std::nullopt if none.
 * @return The newly created affine.apply operation.
 */
mlir::affine::AffineApplyOp LoadingBlock::CreateHoistedApply(
    mlir::OpBuilder& builder,
    mlir::affine::AffineApplyOp original_apply,
    std::optional<unsigned> index_to_remove) {
    
    auto org_operands = original_apply->getOperands();
    mlir::AffineMap original_map = original_apply.getAffineMap();
    
    // If no operand needs to be removed, directly copy
    if (!index_to_remove.has_value()) {
        return builder.create<mlir::affine::AffineApplyOp>(
            original_apply.getLoc(), original_map, org_operands);
    }
    
    unsigned remove_index = index_to_remove.value();
    
    // Build a new operand list (remove the specified index)
    llvm::SmallVector<mlir::Value> new_operands;
    for (unsigned i = 0; i < org_operands.size(); ++i) {
        if (i != remove_index) {
            new_operands.push_back(org_operands[i]);
        }
    }
    
    // Modify affine map: set the dimension to remove as constant 0, remap other dimensions
    mlir::AffineExpr original_expr = original_map.getResult(0);
    llvm::SmallVector<mlir::AffineExpr> dim_replacements;
    
    for (unsigned i = 0; i < original_map.getNumDims(); ++i) {
        if (i == remove_index) {
            dim_replacements.push_back(
                mlir::getAffineConstantExpr(0, builder.getContext()));
        } else if (i < remove_index) {
            dim_replacements.push_back(
                mlir::getAffineDimExpr(i, builder.getContext()));
        } else {
            dim_replacements.push_back(
                mlir::getAffineDimExpr(i - 1, builder.getContext()));
        }
    }
    
    // Symbols remain unchanged
    llvm::SmallVector<mlir::AffineExpr> sym_replacements(original_map.getNumSymbols());
    for (unsigned i = 0; i < original_map.getNumSymbols(); ++i) {
        sym_replacements[i] = mlir::getAffineSymbolExpr(i, builder.getContext());
    }
    
    // Replace dimension references in the expression
    mlir::AffineExpr new_expr = original_expr.replaceDimsAndSymbols(
        dim_replacements, sym_replacements);
    
    // Create a new affine map (number of dimensions reduced by 1)
    unsigned new_num_dims = original_map.getNumDims() - 1;
    mlir::AffineMap new_map = mlir::AffineMap::get(
        new_num_dims, original_map.getNumSymbols(), {new_expr}, builder.getContext());
    
    return builder.create<mlir::affine::AffineApplyOp>(
        original_apply.getLoc(), new_map, new_operands);
}

/**
 * @brief Create hoisted operations with reshaped memory access pattern.
 * 
 * @details This method creates new reinterpret_cast and alloc operations with modified
 * sizes and strides to access multiple consecutive memory blocks at once.
 */
void LoadingBlock::CreateHoistedOpsWithReshape(
    mlir::OpBuilder& builder,
    mlir::affine::AffineApplyOp new_apply,
    mlir::memref::ReinterpretCastOp org_reinterpret,
    mlir::memref::AllocOp org_alloc,
    mlir::memref::ReinterpretCastOp& new_reinterpret,
    mlir::memref::AllocOp& new_alloc) {

    // 2. Extract original static sizes/strides - more reliable to extract from result type
    auto org_result_type = llvm::cast<mlir::MemRefType>(org_reinterpret.getResult().getType());
    auto org_shape = org_result_type.getShape();  // This is [64, 64]
    
    // Extract original strides from layout
    auto org_layout = org_result_type.getLayout();
    mlir::SmallVector<int64_t> org_static_strides;
    if (auto strided_layout = llvm::dyn_cast<mlir::StridedLayoutAttr>(org_layout)) {
        auto strides_ref = strided_layout.getStrides();
        org_static_strides.append(strides_ref.begin(), strides_ref.end());
    }
    
    // Build new sizes: [loop_ub, 64, 64] - all are static constants
    mlir::SmallVector<int64_t> new_static_sizes;
    new_static_sizes.push_back(loop_ub_as_const_.value());
    for (int64_t dim : org_shape) {
        new_static_sizes.push_back(dim);
    }
    
    // Build new strides: [block_size_[1], 512, 1] - all are static constants
    mlir::SmallVector<int64_t> new_static_strides;
    new_static_strides.push_back(coeff_loop_iv_);
    new_static_strides.append(org_static_strides.begin(), org_static_strides.end());
    
    // 3. Create new static attributes
    auto new_sizes_attr = mlir::DenseI64ArrayAttr::get(builder.getContext(), new_static_sizes);
    auto new_strides_attr = mlir::DenseI64ArrayAttr::get(builder.getContext(), new_static_strides);
    
    // 4. Create new result type - 3D memref: [loop_ub, 64, 64]
    mlir::SmallVector<int64_t> new_result_shape;
    new_result_shape.push_back(loop_ub_as_const_.value());
    for (int64_t dim : org_shape) {
        new_result_shape.push_back(dim);
    }
    
    auto new_result_type = mlir::MemRefType::get(
        new_result_shape,
        org_result_type.getElementType(),
        mlir::StridedLayoutAttr::get(builder.getContext(), mlir::ShapedType::kDynamic, new_static_strides)
    );
    
    llvm::SmallVector<mlir::Value> new_offsets = {new_apply.getResult()};
    new_reinterpret = builder.create<mlir::memref::ReinterpretCastOp>(
        org_reinterpret.getLoc(),
        new_result_type,
        org_reinterpret.getSource(),
        mlir::ValueRange(new_offsets),
        mlir::ValueRange(),
        mlir::ValueRange(),
        mlir::DenseI64ArrayAttr::get(builder.getContext(), llvm::SmallVector<int64_t>{mlir::ShapedType::kDynamic}),
        new_sizes_attr,
        new_strides_attr
    );
    
    if (auto tmd_reuse_attr = org_reinterpret->getAttrOfType<mlir::DictionaryAttr>("tmd.reuse")) {
        mlir::NamedAttrList new_attrs;
        new_attrs.append("tmd.reuse", tmd_reuse_attr);
        new_reinterpret->setAttrs(mlir::DictionaryAttr::get(builder.getContext(), new_attrs));
    }
    
    // Create new alloc type: memref<loop_ubx64x64xf32>
    auto org_alloc_type = llvm::cast<mlir::MemRefType>(org_alloc.getResult().getType());
    llvm::SmallVector<int64_t> new_alloc_shape;
    new_alloc_shape.push_back(loop_ub_as_const_.value());
    for (int64_t dim : org_alloc_type.getShape()) {
        new_alloc_shape.push_back(dim);
    }
    
    // Create alloc type with default layout (no layout specified)
    auto new_alloc_type = mlir::MemRefType::get(
        new_alloc_shape,
        org_alloc_type.getElementType());
    
    // Create alloc
    new_alloc = builder.create<mlir::memref::AllocOp>(
        org_alloc.getLoc(), new_alloc_type);
    
    // Update alloc annotation with new size
    if (!org_alloc->getAttrs().empty()) {
        auto org_attrs = org_alloc->getAttrs();
        mlir::NamedAttrList new_attrs;
        
        // Copy all attributes except tmd.alloc
        for (auto attr : org_attrs) {
            if (attr.getName() != "tmd.alloc") {
                new_attrs.append(attr);
            }
        }
        
        // Update tmd.alloc with new size
        if (auto org_dict = org_alloc->getAttrOfType<mlir::DictionaryAttr>("tmd.alloc")) {
            mlir::NamedAttrList alloc_dict_attrs;
            for (auto attr : org_dict) {
                if (attr.getName() == "size") {
                    // Update size
                    if (mem_req_bytes_as_const_.has_value()) {
                        alloc_dict_attrs.append("size", 
                            mlir::IntegerAttr::get(
                                mlir::IntegerType::get(builder.getContext(), 64),
                                mem_req_bytes_as_const_.value()));
                    }
                } else {
                    alloc_dict_attrs.append(attr);
                }
            }
            if (mem_req_bytes_as_const_.has_value() && 
                !alloc_dict_attrs.get("size")) {
                // Add size if it wasn't there before
                alloc_dict_attrs.append("size",
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(builder.getContext(), 64),
                        mem_req_bytes_as_const_.value()));
            }
            new_attrs.append("tmd.alloc", 
                mlir::DictionaryAttr::get(builder.getContext(), alloc_dict_attrs));
        }
        
        new_alloc->setAttrs(mlir::DictionaryAttr::get(builder.getContext(), new_attrs));
    } else if (mem_req_bytes_as_const_.has_value()) {
        // Create new annotation if none existed
        mlir::NamedAttrList alloc_dict_attrs;
        alloc_dict_attrs.append("size",
            mlir::IntegerAttr::get(
                mlir::IntegerType::get(builder.getContext(), 64),
                mem_req_bytes_as_const_.value()));
        
        mlir::NamedAttrList new_attrs;
        new_attrs.append("tmd.alloc",
            mlir::DictionaryAttr::get(builder.getContext(), alloc_dict_attrs));
        new_alloc->setAttrs(mlir::DictionaryAttr::get(builder.getContext(), new_attrs));
    }
}

/**
 * @brief Create hoisted operations with simple copy pattern.
 * 
 * @details This method creates new reinterpret_cast and alloc operations by
 * simply copying the original operations with updated offsets.
 */
void LoadingBlock::CreateHoistedOpsSimple(
    mlir::OpBuilder& builder,
    mlir::affine::AffineApplyOp new_apply,
    mlir::memref::ReinterpretCastOp org_reinterpret,
    mlir::memref::AllocOp org_alloc,
    mlir::memref::ReinterpretCastOp& new_reinterpret,
    mlir::memref::AllocOp& new_alloc) {
    
    // Create reinterpret_cast
    llvm::SmallVector<mlir::Value> new_offsets = {new_apply.getResult()};
    new_reinterpret = builder.create<mlir::memref::ReinterpretCastOp>(
        org_reinterpret.getLoc(),
        org_reinterpret.getResult().getType(),
        org_reinterpret.getSource(),
        mlir::ValueRange(new_offsets),
        org_reinterpret.getSizes(),
        org_reinterpret.getStrides(),
        org_reinterpret.getStaticOffsetsAttr(),
        org_reinterpret.getStaticSizesAttr(),
        org_reinterpret.getStaticStridesAttr());
    
    if (!org_reinterpret->getAttrs().empty()) {
        new_reinterpret->setAttrs(org_reinterpret->getAttrs());
    }

    // Create alloc
    new_alloc = builder.create<mlir::memref::AllocOp>(
        org_alloc.getLoc(), org_alloc.getResult().getType());
    if (!org_alloc->getAttrs().empty()) {
        new_alloc->setAttrs(org_alloc->getAttrs());
    }
    
    // Mark this block as valid since we used CreateHoistedOpsSimple
    is_valid_ = true;
}

/**
 * @brief Hoist the loading block operations to the outer loop.
 *
 * @details Creates new operations before the outer_for_op_ and replaces the original ones.
 */
void LoadingBlock::HoistLoadingBlock() {
    mlir::OpBuilder builder = std::visit([](auto&& forOp) -> mlir::OpBuilder {
        return mlir::OpBuilder(forOp);
    }, outer_for_op_);

    auto org_apply = llvm::dyn_cast<mlir::affine::AffineApplyOp>(op_block_[0]);
    auto org_operands = org_apply->getOperands();
    auto org_affine_map = org_apply.getAffineMap();
    auto org_affine_expr = org_affine_map.getResult(0);
    auto loop_iv_index_opt = GetIvIndex(org_operands);

    // Get original operations
    auto org_reinterpret = llvm::dyn_cast<mlir::memref::ReinterpretCastOp>(op_block_[1]);
    auto org_alloc = llvm::dyn_cast<mlir::memref::AllocOp>(op_block_[2]);
    auto org_copy = llvm::dyn_cast<mlir::memref::CopyOp>(op_block_[3]);

    // Check if we need to modify memory access pattern
    bool need_reshape = loop_iv_index_opt.has_value() && 
                       loop_ub_as_const_.has_value() && 
                       block_size_.has_value() && block_size_.value().size() > 1;

    mlir::memref::ReinterpretCastOp new_reinterpret;
    mlir::memref::AllocOp new_alloc;

    auto new_apply = CreateHoistedApply(builder, org_apply, loop_iv_index_opt);
    if (need_reshape) {
        coeff_loop_iv_ = ExtractDimCoefficientRec(org_affine_expr,loop_iv_index_opt.value());
        CreateHoistedOpsWithReshape(builder, new_apply, org_reinterpret, org_alloc,
                                   new_reinterpret, new_alloc);
    } else {
        CreateHoistedOpsSimple(builder, new_apply, org_reinterpret, org_alloc,
                              new_reinterpret, new_alloc);
    }

    // Create memref.copy
    auto new_copy = builder.create<mlir::memref::CopyOp>(
        org_copy.getLoc(),
        new_reinterpret.getResult(),
        new_alloc.getResult());
    if (!org_copy->getAttrs().empty()) {
        new_copy->setAttrs(org_copy->getAttrs());
    }

    // Replace and update
    ReplaceOpBlock(new_apply, new_reinterpret, new_alloc, new_copy);
}

/**
 * @brief Construct a LoadingBlock from a sequence of operations.
 * @param op_block The sequence of operations: [affine.apply, reinterpret_cast, alloc, copy, to_tensor].
 * @param for_op The innermost scf.for loop operation.
 */
LoadingBlock::LoadingBlock(llvm::SmallVector<mlir::Operation *> op_block, mlir::scf::ForOp for_op) : 
    outer_for_op_(for_op), op_block_({op_block.begin(), op_block.begin() + 4UL}), is_valid_(false) {
    if(auto alloc_op = llvm::dyn_cast<mlir::memref::AllocOp>(op_block[2])) {
        mem_req_bytes_as_const_ = GetAllocSizeAsConst(alloc_op);
    }
    
    // Extract block_size from reinterpret_cast
    if (auto reinterpret_op = llvm::dyn_cast<mlir::memref::ReinterpretCastOp>(op_block[1])) {
        block_size_ = GetBlockSize(reinterpret_op);
    }
    
    org_to_tensor_op_ = op_block[4];
    SetLoopAttr();
}

/**
 * @brief Recursively hoist the loading block to outer loops.
 * @param new_outer_for The new outer affine for loop to hoist to.
 *
 * @details This method recursively hoists the block until the termination condition is met.
 */
void LoadingBlock::HoistRec(mlir::affine::AffineForOp new_outer_for) {
    if (!loop_ub_as_const_.has_value() && !IsIndependent()) {
        return;
    }

    if (!IsIndependent()) {
        if (loop_ub_as_const_.has_value() && mem_req_bytes_as_const_.has_value()) {
            auto mem = mem_req_bytes_as_const_.value();
            auto ub = loop_ub_as_const_.value();
            mem_req_bytes_as_const_ = mem * ub;
        } else {
            return;
        }
    }

    HoistLoadingBlock();

    if (replacement_block_.empty()) {
        SetReplacementBlock();
    }

    ResetLoopAttr(new_outer_for);
    
    mlir::affine::AffineForOp next_outer_for = FindOuterAffineFor();
    if (next_outer_for) {
        HoistRec(next_outer_for);
    }
}

/**
 * @brief Start the hoisting process by finding the first outer affine for loop.
 *
 * @details This is the entry point for hoisting the loading block.
 */
void LoadingBlock::Hoist() {
    mlir::affine::AffineForOp next_outer_for = FindOuterAffineFor();
    if (next_outer_for) {
        HoistRec(next_outer_for);
    }
}

/**
 * @brief Build loading blocks for the given innermost for loop.
 * @param inner_most_for_op The innermost scf.for loop operation.
 * @param block_vec Output vector to store the found loading blocks.
 * @return LogicalResult indicating success or failure.
 */
mlir::LogicalResult BuildLoadingBlocks(mlir::scf::ForOp inner_most_for_op,
                                       llvm::SmallVector<LoadingBlock, 2> &block_vec) {
    mlir::Block *body = inner_most_for_op.getBody();
    if (!body)
        return mlir::failure();

    bool foundAny = false;

    for (auto it = body->begin(); it != body->end(); ++it) {
        mlir::Operation *op = &*it;

        if (auto apply_op = llvm::dyn_cast<mlir::affine::AffineApplyOp>(op)) {
            auto next_it = std::next(it);
            if (next_it == body->end()) break;

            auto reinterpret_op =
                llvm::dyn_cast<mlir::memref::ReinterpretCastOp>(&*next_it);
            if (!reinterpret_op) continue;

            next_it = std::next(next_it);
            if (next_it == body->end()) break;
            auto alloc_op = llvm::dyn_cast<mlir::memref::AllocOp>(&*next_it);
            if (!alloc_op) continue;

            next_it = std::next(next_it);
            if (next_it == body->end()) break;
            auto copy_op = llvm::dyn_cast<mlir::memref::CopyOp>(&*next_it);
            if (!copy_op) continue;

            next_it = std::next(next_it);
            if (next_it == body->end()) break;
            auto to_tensor_op =
                llvm::dyn_cast<mlir::bufferization::ToTensorOp>(&*next_it);
            if (!to_tensor_op) continue;

            llvm::SmallVector<mlir::Operation *> op_block;
            op_block.push_back(apply_op);
            op_block.push_back(reinterpret_op);
            op_block.push_back(alloc_op);
            op_block.push_back(copy_op);
            op_block.push_back(to_tensor_op);

            LoadingBlock loading_block(std::move(op_block), inner_most_for_op);
            block_vec.push_back(std::move(loading_block));
            foundAny = true;
        }
    }

    return foundAny ? mlir::success() : mlir::failure();
}

/**
 * @brief Match and hoist loading blocks for the innermost loop.
 * @param inner_most_loop The innermost scf.for loop operation.
 * @return LogicalResult indicating success or failure.
 */
mlir::LogicalResult MatchAndHoist(mlir::scf::ForOp inner_most_loop) {
    llvm::SmallVector<LoadingBlock, 2> loading_blocks;
    BuildLoadingBlocks(inner_most_loop, loading_blocks);
    auto& block_A = loading_blocks[0];
    auto& block_B = loading_blocks[1];

    block_B.Hoist();

    return mlir::success();
}

/**
 * @brief Hoist a single loading block at the specified index for the innermost loop.
 * @param inner_most_loop The innermost scf.for loop operation.
 * @param block_index The index of the block to hoist.
 * @return LogicalResult indicating success or failure.
 */
mlir::LogicalResult HoistSingleBlock(mlir::scf::ForOp inner_most_loop, size_t block_index) {
    llvm::SmallVector<LoadingBlock, 2> loading_blocks;
    if (failed(BuildLoadingBlocks(inner_most_loop, loading_blocks))) {
        return mlir::failure();
    }
    
    if (block_index >= loading_blocks.size()) {
        return mlir::failure();
    }
    
    loading_blocks[block_index].Hoist();
    
    // Check if the block is valid (has been hoisted using CreateHoistedOpsSimple at least once)
    if (!loading_blocks[block_index].IsValid()) {
        return mlir::failure();
    }
    
    return mlir::success();
}

} // namespace tmd::affine