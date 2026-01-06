/**
 * @file block_loading_pattern.cpp
 * @brief Implementation of block loading pattern detection and hoisting for loom dialect.
 */

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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

// Include Loom dialect headers for CopyOp and ReinterpretCastOp
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#define DEBUG_TYPE "hoist-block-loading"

namespace loom::affine {

/**
 * @brief Set loop attributes from the outer_for_op_ variant.
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
}

/**
 * @brief Clear all loop-related attributes.
 */
void LoadingBlock::ClearLoopAttr() {
    loop_iv_ = nullptr;
    loop_ub_ = nullptr;
}

/**
 * @brief Reset loop attributes to a new outer affine for loop.
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
 * @brief Check if a value depends (transitively) on the loop induction variable.
 */
bool LoadingBlock::DependsOnLoopIV(mlir::Value value) {
    if (!value || !loop_iv_)
        return false;
    
    if (value == loop_iv_)
        return true;
    
    llvm::SmallPtrSet<mlir::Value, 16> visited;
    llvm::SmallVector<mlir::Value, 16> worklist = {value};
    
    while (!worklist.empty()) {
        mlir::Value current = worklist.pop_back_val();
        
        if (current == loop_iv_)
            return true;
        
        if (!visited.insert(current).second)
            continue;
        
        // If it's a block argument (other than loop IV), stop traversing this path
        if (mlir::isa<mlir::BlockArgument>(current))
            continue;
        
        // Get the defining operation and add its operands to the worklist
        if (mlir::Operation* defOp = current.getDefiningOp()) {
            for (mlir::Value operand : defOp->getOperands()) {
                if (!visited.contains(operand)) {
                    worklist.push_back(operand);
                }
            }
        }
    }
    
    return false;
}

/**
 * @brief Collect backward slice of the copy operation.
 * @details Collects all operations that the copy operation depends on,
 * stopping at block arguments or operations defined outside the loop.
 */
void LoadingBlock::CollectBackwardSlice() {
    backward_slice_.clear();
    
    if (!copy_op_)
        return;
    
    // Get the parent block of the loop
    mlir::Operation* loop_op = std::visit([](auto&& forOp) -> mlir::Operation* {
        return forOp.getOperation();
    }, outer_for_op_);
    
    mlir::Block* loop_body = std::visit([](auto&& forOp) -> mlir::Block* {
        return forOp.getBody();
    }, outer_for_op_);
    
    if (!loop_body)
        return;
    
    llvm::SmallVector<mlir::Operation*, 16> worklist;
    llvm::SmallPtrSet<mlir::Operation*, 16> visited;
    
    // Start from the copy operation's operands
    for (mlir::Value operand : copy_op_->getOperands()) {
        if (mlir::Operation* defOp = operand.getDefiningOp()) {
            worklist.push_back(defOp);
        }
    }
    
    while (!worklist.empty()) {
        mlir::Operation* op = worklist.pop_back_val();
        
        if (!op || visited.contains(op))
            continue;
        
        visited.insert(op);
        
        // Only include operations that are inside the loop body
        if (op->getBlock() != loop_body)
            continue;
        
        // Don't include the copy operation itself in the backward slice
        if (op == copy_op_)
            continue;
        
        backward_slice_.insert(op);
        
        // Add operands' defining operations to the worklist
        for (mlir::Value operand : op->getOperands()) {
            if (mlir::Operation* defOp = operand.getDefiningOp()) {
                if (!visited.contains(defOp)) {
                    worklist.push_back(defOp);
                }
            }
        }
    }
    
    LLVM_DEBUG({
        llvm::dbgs() << "Backward slice for copy operation:\n";
        for (mlir::Operation* op : backward_slice_) {
            llvm::dbgs() << "  - " << op->getName() << "\n";
        }
    });
}

/**
 * @brief Find the loom.reinterpret_cast in the backward slice.
 */
mlir::Operation* LoadingBlock::FindReinterpretCast() {
    for (mlir::Operation* op : backward_slice_) {
        if (mlir::isa<loom::ReinterpretCastOp>(op)) {
            return op;
        }
    }
    return nullptr;
}

/**
 * @brief Find the memref.alloc operation.
 */
mlir::memref::AllocOp LoadingBlock::FindAlloc() {
    // First check in backward slice
    for (mlir::Operation* op : backward_slice_) {
        if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
            return alloc;
        }
    }
    
    // Check the dst operand of the copy operation
    if (auto loomCopy = mlir::dyn_cast<loom::CopyOp>(copy_op_)) {
        if (auto alloc = loomCopy.getDst().getDefiningOp<mlir::memref::AllocOp>()) {
            return alloc;
        }
    }
    
    return nullptr;
}

/**
 * @brief Check if a value is defined in a block that dominates the target block.
 * @param value The value to check.
 * @param targetBlock The target block where we want to use the value.
 * @return true if value can be used in targetBlock, false otherwise.
 */
static bool isVisibleIn(mlir::Value value, mlir::Block* targetBlock) {
    if (!value || !targetBlock)
        return false;
    
    // Block arguments are visible if they belong to the target block or an ancestor
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
        mlir::Block* defBlock = blockArg.getOwner();
        // Check if defBlock is an ancestor of targetBlock
        mlir::Block* curr = targetBlock;
        while (curr) {
            if (curr == defBlock)
                return true;
            if (mlir::Operation* parentOp = curr->getParentOp()) {
                curr = parentOp->getBlock();
            } else {
                break;
            }
        }
        return false;
    }
    
    // For values defined by operations
    if (mlir::Operation* defOp = value.getDefiningOp()) {
        mlir::Block* defBlock = defOp->getBlock();
        // Check if defBlock is the same as or an ancestor of targetBlock
        mlir::Block* curr = targetBlock;
        while (curr) {
            if (curr == defBlock)
                return true;
            if (mlir::Operation* parentOp = curr->getParentOp()) {
                curr = parentOp->getBlock();
            } else {
                break;
            }
        }
        return false;
    }
    
    return false;
}

/**
 * @brief Clone an operation and all its dependencies to a target block.
 * @param builder OpBuilder positioned at the target insertion point.
 * @param op The operation to clone.
 * @param targetBlock The block where the operation will be inserted.
 * @param mapping IRMapping to track cloned values.
 * @param cloned Set of already cloned operations.
 * @return The cloned operation, or nullptr if cloning failed.
 */
static mlir::Operation* cloneWithDependencies(
    mlir::OpBuilder& builder,
    mlir::Operation* op,
    mlir::Block* targetBlock,
    mlir::IRMapping& mapping,
    llvm::DenseSet<mlir::Operation*>& cloned) {
    
    if (!op || cloned.contains(op))
        return nullptr;
    
    // First, clone all dependencies
    for (mlir::Value operand : op->getOperands()) {
        if (mapping.contains(operand))
            continue;
        
        if (isVisibleIn(operand, targetBlock)) {
            // Value is already visible, no need to clone
            continue;
        }
        
        if (mlir::Operation* defOp = operand.getDefiningOp()) {
            // Recursively clone the defining operation
            if (!cloned.contains(defOp)) {
                cloneWithDependencies(builder, defOp, targetBlock, mapping, cloned);
            }
        }
    }
    
    // Now clone the operation itself
    mlir::Operation* clonedOp = builder.clone(*op, mapping);
    cloned.insert(op);
    
    return clonedOp;
}

/**
 * @brief Create hoisted operations before the loop.
 */
void LoadingBlock::CreateHoistedOps(mlir::OpBuilder& builder) {
    auto reinterpret_op = mlir::dyn_cast_or_null<loom::ReinterpretCastOp>(FindReinterpretCast());
    auto alloc_op = FindAlloc();
    auto loom_copy = mlir::dyn_cast<loom::CopyOp>(copy_op_);
    
    if (!reinterpret_op || !alloc_op || !loom_copy) {
        LLVM_DEBUG(llvm::dbgs() << "Missing required operations for hoisting\n");
        return;
    }
    
    // Get the target block (where we're inserting)
    mlir::Block* targetBlock = builder.getInsertionBlock();
    
    // Get loop upper bound
    mlir::Value ub = loop_ub_;
    if (!ub) {
        LLVM_DEBUG(llvm::dbgs() << "No loop upper bound available\n");
        return;
    }
    
    // Check if upper bound is visible in target block
    if (!isVisibleIn(ub, targetBlock)) {
        LLVM_DEBUG(llvm::dbgs() << "Loop upper bound is not visible in target block\n");
        return;
    }
    
    // Collect values that need to be available
    mlir::IRMapping mapping;
    llvm::DenseSet<mlir::Operation*> clonedOps;
    
    // Get original sizes and strides from reinterpret_cast
    auto orig_sizes = reinterpret_op.getSizes();
    auto orig_strides = reinterpret_op.getStrides();
    auto orig_offsets = reinterpret_op.getOffsets();
    
    if (orig_sizes.size() < 2) {
        LLVM_DEBUG(llvm::dbgs() << "Reinterpret cast has less than 2 dimensions\n");
        return;
    }
    
    // Clone sizes if needed
    llvm::SmallVector<mlir::Value> new_sizes;
    new_sizes.push_back(ub);
    for (mlir::Value size : orig_sizes) {
        if (isVisibleIn(size, targetBlock)) {
            new_sizes.push_back(size);
        } else if (mlir::Operation* defOp = size.getDefiningOp()) {
            cloneWithDependencies(builder, defOp, targetBlock, mapping, clonedOps);
            new_sizes.push_back(mapping.lookupOrDefault(size));
        } else {
            LLVM_DEBUG(llvm::dbgs() << "Size value not visible and cannot be cloned\n");
            return;
        }
    }
    
    // Clone strides if needed
    llvm::SmallVector<mlir::Value> cloned_strides;
    for (mlir::Value stride : orig_strides) {
        if (isVisibleIn(stride, targetBlock)) {
            cloned_strides.push_back(stride);
        } else if (mlir::Operation* defOp = stride.getDefiningOp()) {
            cloneWithDependencies(builder, defOp, targetBlock, mapping, clonedOps);
            cloned_strides.push_back(mapping.lookupOrDefault(stride));
        } else {
            LLVM_DEBUG(llvm::dbgs() << "Stride value not visible and cannot be cloned\n");
            return;
        }
    }
    
    // Build new strides: [orig_sizes[0] * orig_strides[0], orig_strides[0], orig_strides[1]]
    llvm::SmallVector<mlir::Value> new_strides;
    
    // Calculate stride for new dimension: size[0] * stride[0]
    mlir::Value size0 = new_sizes[1]; // First original size (after ub)
    mlir::Value stride0 = cloned_strides[0];
    mlir::Value new_dim_stride = builder.create<mlir::arith::MulIOp>(
        reinterpret_op.getLoc(), size0, stride0);
    new_strides.push_back(new_dim_stride);
    
    for (mlir::Value stride : cloned_strides) {
        new_strides.push_back(stride);
    }
    
    // Calculate new offset: remove dependency on loop IV
    llvm::SmallVector<mlir::Value> new_offsets;
    for (mlir::Value offset : orig_offsets) {
        if (DependsOnLoopIV(offset)) {
            // For offsets that depend on loop IV, we need to compute the base offset
            // by substituting loop IV = 0
            mlir::IRMapping offsetMapping = mapping;
            offsetMapping.map(loop_iv_, builder.create<mlir::arith::ConstantIndexOp>(
                reinterpret_op.getLoc(), 0));
            
            if (mlir::Operation* defOp = offset.getDefiningOp()) {
                // Clone the entire computation with loop_iv = 0
                llvm::DenseSet<mlir::Operation*> offsetCloned = clonedOps;
                cloneWithDependencies(builder, defOp, targetBlock, offsetMapping, offsetCloned);
                new_offsets.push_back(offsetMapping.lookupOrDefault(offset));
            } else {
                new_offsets.push_back(offset);
            }
        } else if (isVisibleIn(offset, targetBlock)) {
            new_offsets.push_back(offset);
        } else if (mlir::Operation* defOp = offset.getDefiningOp()) {
            cloneWithDependencies(builder, defOp, targetBlock, mapping, clonedOps);
            new_offsets.push_back(mapping.lookupOrDefault(offset));
        } else {
            new_offsets.push_back(offset);
        }
    }
    
    // Create new result type with additional dimension
    auto orig_result_type = mlir::cast<mlir::MemRefType>(reinterpret_op.getResult().getType());
    llvm::SmallVector<int64_t> new_shape;
    new_shape.push_back(mlir::ShapedType::kDynamic); // loop_ub dimension
    for (int64_t dim : orig_result_type.getShape()) {
        new_shape.push_back(dim);
    }
    
    // Create strided layout for new type
    llvm::SmallVector<int64_t> new_static_strides(new_shape.size(), mlir::ShapedType::kDynamic);
    auto new_layout = mlir::StridedLayoutAttr::get(
        builder.getContext(), mlir::ShapedType::kDynamic, new_static_strides);
    
    auto new_result_type = mlir::MemRefType::get(
        new_shape, orig_result_type.getElementType(), new_layout);
    
    // Clone source if needed
    mlir::Value source = reinterpret_op.getSource();
    if (!isVisibleIn(source, targetBlock)) {
        LLVM_DEBUG(llvm::dbgs() << "Source value not visible in target block\n");
        return;
    }
    
    // Create new reinterpret_cast
    auto new_reinterpret = builder.create<loom::ReinterpretCastOp>(
        reinterpret_op.getLoc(),
        new_result_type,
        source,
        new_offsets,
        new_sizes,
        new_strides,
        reinterpret_op.getSequentialReuse(),
        reinterpret_op.getSpatialReuse(),
        reinterpret_op.getTemporalReuse());
    
    // Create new alloc with additional dimension
    auto orig_alloc_type = alloc_op.getType();
    llvm::SmallVector<int64_t> new_alloc_shape;
    new_alloc_shape.push_back(mlir::ShapedType::kDynamic); // loop_ub dimension
    for (int64_t dim : orig_alloc_type.getShape()) {
        new_alloc_shape.push_back(dim);
    }
    
    auto new_alloc_type = mlir::MemRefType::get(
        new_alloc_shape, orig_alloc_type.getElementType());
    
    // Clone alloc dynamic sizes
    llvm::SmallVector<mlir::Value> alloc_dyn_sizes;
    alloc_dyn_sizes.push_back(ub);
    for (mlir::Value size : alloc_op.getDynamicSizes()) {
        if (isVisibleIn(size, targetBlock)) {
            alloc_dyn_sizes.push_back(size);
        } else if (mlir::Operation* defOp = size.getDefiningOp()) {
            cloneWithDependencies(builder, defOp, targetBlock, mapping, clonedOps);
            alloc_dyn_sizes.push_back(mapping.lookupOrDefault(size));
        } else {
            LLVM_DEBUG(llvm::dbgs() << "Alloc size value not visible and cannot be cloned\n");
            return;
        }
    }
    
    auto new_alloc = builder.create<mlir::memref::AllocOp>(
        alloc_op.getLoc(), new_alloc_type, alloc_dyn_sizes);
    
    // Create new loom.copy
    auto new_copy = builder.create<loom::CopyOp>(
        loom_copy.getLoc(),
        new_reinterpret.getResult(),
        new_alloc.getResult(),
        loom_copy.getProvenance(),
        loom_copy.getInterconnectAttr(),
        loom_copy.getBroadcastAttr());
    
    // Mark as valid
    is_valid_ = true;
    
    // Store for replacement block creation
    replacement_block_.clear();
    replacement_block_.push_back(new_reinterpret);
    replacement_block_.push_back(new_alloc);
    replacement_block_.push_back(new_copy);
    
    LLVM_DEBUG({
        llvm::dbgs() << "Created hoisted operations:\n";
        llvm::dbgs() << "  reinterpret_cast: " << new_reinterpret << "\n";
        llvm::dbgs() << "  alloc: " << new_alloc << "\n";
        llvm::dbgs() << "  copy: " << new_copy << "\n";
    });
}

/**
 * @brief Create replacement operations at the original location.
 */
void LoadingBlock::SetReplacementBlock() {
    if (replacement_block_.size() < 3)
        return;
    
    auto new_alloc = mlir::dyn_cast<mlir::memref::AllocOp>(replacement_block_[1]);
    auto to_tensor = mlir::dyn_cast_or_null<mlir::bufferization::ToTensorOp>(to_tensor_op_);
    auto old_alloc = FindAlloc();
    auto old_reinterpret = FindReinterpretCast();
    
    if (!new_alloc || !copy_op_)
        return;
    
    mlir::OpBuilder builder(copy_op_);
    
    // Create subview to access the slice for current loop iteration
    // subview %alloc[%loop_iv, 0, 0][1, dim0, dim1][1, 1, 1]
    
    auto alloc_type = new_alloc.getType();
    auto rank = alloc_type.getRank();
    
    if (rank < 2) {
        LLVM_DEBUG(llvm::dbgs() << "Alloc has less than 2 dimensions\n");
        return;
    }
    
    // Offsets: [loop_iv, 0, 0, ...]
    llvm::SmallVector<mlir::OpFoldResult> offsets;
    offsets.push_back(loop_iv_);
    for (unsigned i = 1; i < rank; ++i) {
        offsets.push_back(builder.getIndexAttr(0));
    }
    
    // Get dynamic sizes from the NEW alloc
    auto dyn_sizes = new_alloc.getDynamicSizes();
    unsigned dyn_idx = 0;
    
    // Sizes: [1, original_sizes...]
    llvm::SmallVector<mlir::OpFoldResult> sizes;
    sizes.push_back(builder.getIndexAttr(1));
    
    // Skip first dynamic dimension (loop_ub)
    if (alloc_type.isDynamicDim(0)) {
        dyn_idx++;
    }
    
    for (unsigned i = 1; i < rank; ++i) {
        if (alloc_type.isDynamicDim(i)) {
            if (dyn_idx < dyn_sizes.size()) {
                sizes.push_back(dyn_sizes[dyn_idx++]);
            } else {
                // Fallback: use old alloc's dynamic sizes
                if (old_alloc && i - 1 < old_alloc.getDynamicSizes().size()) {
                    sizes.push_back(old_alloc.getDynamicSizes()[i - 1]);
                } else {
                    sizes.push_back(builder.getIndexAttr(mlir::ShapedType::kDynamic));
                }
            }
        } else {
            sizes.push_back(builder.getIndexAttr(alloc_type.getDimSize(i)));
        }
    }
    
    // Strides: all 1s
    llvm::SmallVector<mlir::OpFoldResult> strides(rank, builder.getIndexAttr(1));
    
    // Compute reduced shape (drop the first dimension of size 1)
    // Note: We use kDynamic for all dimensions, but must use inferRankReducedResultType
    // to get a layout compatible with the source memref (MLIR validation requirement)
    llvm::SmallVector<int64_t> reduced_shape;
    for (unsigned i = 1; i < rank; ++i) {
        reduced_shape.push_back(mlir::ShapedType::kDynamic);
    }
    
    auto reduced_type = mlir::memref::SubViewOp::inferRankReducedResultType(
        reduced_shape, alloc_type, offsets, sizes, strides);
    
    auto subview = builder.create<mlir::memref::SubViewOp>(
        copy_op_->getLoc(),
        reduced_type,
        new_alloc.getResult(),
        offsets,
        sizes,
        strides);
    
    LLVM_DEBUG({
        llvm::dbgs() << "Created subview: " << subview << "\n";
    });
    
    // Update to_tensor if present
    if (to_tensor) {
        builder.setInsertionPointAfter(to_tensor);
        auto new_to_tensor = builder.create<mlir::bufferization::ToTensorOp>(
            to_tensor.getLoc(),
            to_tensor.getType(),
            subview.getResult(),
            to_tensor.getRestrict(),
            to_tensor.getWritable());
        
        to_tensor.getResult().replaceAllUsesWith(new_to_tensor.getResult());
        to_tensor.erase();
        to_tensor_op_ = new_to_tensor;
        
        LLVM_DEBUG({
            llvm::dbgs() << "Created new to_tensor: " << new_to_tensor << "\n";
        });
    }
    
    // Erase original operations
    // First, erase the copy
    if (copy_op_) {
        copy_op_->erase();
        copy_op_ = nullptr;
    }
    
    // Then erase alloc if it has no other uses
    if (old_alloc && old_alloc->use_empty()) {
        old_alloc->erase();
    }
    
    // Then erase reinterpret_cast if it has no other uses
    if (old_reinterpret && old_reinterpret->use_empty()) {
        old_reinterpret->erase();
    }
    
    // Erase backward slice operations that have no uses
    for (auto it = backward_slice_.rbegin(); it != backward_slice_.rend(); ++it) {
        mlir::Operation* op = *it;
        if (op && op->use_empty() && op->getBlock()) {
            op->erase();
        }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "SetReplacementBlock completed\n");
}

/**
 * @brief Hoist the loading block operations to the outer loop.
 */
void LoadingBlock::HoistLoadingBlock() {
    mlir::OpBuilder builder = std::visit([](auto&& forOp) -> mlir::OpBuilder {
        return mlir::OpBuilder(forOp);
    }, outer_for_op_);
    
    // Collect backward slice first
    CollectBackwardSlice();
    
    // Create hoisted operations
    CreateHoistedOps(builder);
}

/**
 * @brief Construct a LoadingBlock from a loom.copy operation.
 */
LoadingBlock::LoadingBlock(mlir::Operation* copy_op, mlir::scf::ForOp for_op,
                           mlir::Operation* to_tensor_op)
    : outer_for_op_(for_op), copy_op_(copy_op), to_tensor_op_(to_tensor_op),
      loop_iv_(nullptr), loop_ub_(nullptr), is_valid_(false) {
    SetLoopAttr();
    CollectBackwardSlice();
    
    LLVM_DEBUG({
        llvm::dbgs() << "Created LoadingBlock for copy: " << *copy_op << "\n";
        llvm::dbgs() << "  Backward slice size: " << backward_slice_.size() << "\n";
    });
}

/**
 * @brief Recursively hoist the loading block to outer loops.
 */
void LoadingBlock::HoistRec(mlir::affine::AffineForOp new_outer_for) {
    HoistLoadingBlock();

    // After hoisting, create replacement operations at the original location
    // Only do this once (on the first hoist step)
    if (is_valid_ && !replacement_block_.empty()) {
        SetReplacementBlock();
    }

    // For now, don't recursively hoist to outer loops
    // This would require more complex handling of values visibility
    // ResetLoopAttr(new_outer_for);
    // if (auto next_outer_for = FindOuterAffineFor()) {
    //     HoistRec(next_outer_for);
    // }
}

/**
 * @brief Start the hoisting process.
 */
void LoadingBlock::Hoist() {
    mlir::affine::AffineForOp next_outer_for = FindOuterAffineFor();
    if (next_outer_for) {
        HoistRec(next_outer_for);
    }
}

/**
 * @brief Build loading blocks by finding loom.copy operations in a for loop.
 */
mlir::LogicalResult BuildLoadingBlocks(mlir::scf::ForOp inner_most_for_op,
                                       llvm::SmallVector<LoadingBlock, 2>& block_vec) {
    mlir::Block* body = inner_most_for_op.getBody();
    if (!body)
        return mlir::failure();

    // Find all loom.copy operations in the loop body
    for (mlir::Operation& op : *body) {
        if (auto copy_op = mlir::dyn_cast<loom::CopyOp>(&op)) {
            // Look for a following bufferization.to_tensor operation
            mlir::Operation* to_tensor_op = nullptr;
            
            // Check if dst has a to_tensor user
            for (mlir::Operation* user : copy_op.getDst().getUsers()) {
                if (mlir::isa<mlir::bufferization::ToTensorOp>(user)) {
                    to_tensor_op = user;
                    break;
                }
            }
            
            block_vec.emplace_back(copy_op.getOperation(), inner_most_for_op, to_tensor_op);
            
            LLVM_DEBUG({
                llvm::dbgs() << "Found loom.copy operation: " << *copy_op << "\n";
                if (to_tensor_op) {
                    llvm::dbgs() << "  with to_tensor: " << *to_tensor_op << "\n";
                }
            });
        }
    }

    return block_vec.empty() ? mlir::failure() : mlir::success();
}

/**
 * @brief Hoist a single loading block at the specified index.
 */
mlir::LogicalResult HoistSingleBlock(mlir::scf::ForOp inner_most_loop, size_t block_index) {
    llvm::SmallVector<LoadingBlock, 2> loading_blocks;
    if (failed(BuildLoadingBlocks(inner_most_loop, loading_blocks)) ||
        block_index >= loading_blocks.size()) {
        return mlir::failure();
    }
    
    loading_blocks[block_index].Hoist();
    
    return loading_blocks[block_index].IsValid() ? mlir::success() : mlir::failure();
}

} // namespace loom::affine
