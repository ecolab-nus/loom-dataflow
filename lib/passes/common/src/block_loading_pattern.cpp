/**
 * @file block_loading_pattern.cpp
 * @brief Implementation of block loading pattern detection and hoisting for
 * loom dialect.
 */

#include "block_loading_pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

// Include Loom dialect headers
#include "mlir/Interfaces/ViewLikeInterface.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#define DEBUG_TYPE "hoist-block-loading"

namespace loom::affine {

/**
 * @brief Set loop attributes from the outer_for_op_.
 */
void LoadingBlock::SetLoopAttr() { loop_iv_ = outer_for_op_.getInductionVar(); }

/**
 * @brief Get or reify the loop upper bound.
 */
mlir::Value LoadingBlock::getOrReifyLoopUB(mlir::OpBuilder &builder) {
  if (outer_for_op_.hasConstantUpperBound()) {
    return builder.create<mlir::arith::ConstantIndexOp>(
        outer_for_op_.getLoc(), outer_for_op_.getConstantUpperBound());
  }
  return builder.create<mlir::affine::AffineApplyOp>(
      outer_for_op_.getLoc(), outer_for_op_.getUpperBoundMap(),
      outer_for_op_.getUpperBoundOperands());
}

/**
 * @brief Clear all loop-related attributes.
 */
void LoadingBlock::ClearLoopAttr() { loop_iv_ = nullptr; }

/**
 * @brief Find the outer affine for loop that contains the current
 * outer_for_op_.
 */
mlir::affine::AffineForOp LoadingBlock::FindOuterAffineFor() {
  mlir::Operation *current_op = outer_for_op_.getOperation();

  if (!current_op) {
    return mlir::affine::AffineForOp(nullptr);
  }

  for (mlir::Operation *parent = current_op->getParentOp(); parent;
       parent = parent->getParentOp()) {
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
 * @brief Check if a value depends (transitively) on the loop induction
 * variable.
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
    if (mlir::Operation *defOp = current.getDefiningOp()) {
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
 * @brief Collect backward slice of the loading chain.
 * @details Collects all operations that the loading chain depends on,
 * stopping at block arguments or operations defined outside the loop.
 */
void LoadingBlock::CollectBackwardSlice() {
  backward_slice_.clear();

  if (!allocc_op_ || !copy_to_tensor_op_)
    return;

  mlir::Block *loop_body = outer_for_op_.getBody();
  if (!loop_body)
    return;

  llvm::SmallVector<mlir::Operation *, 16> worklist;
  llvm::SmallPtrSet<mlir::Operation *, 16> visited;

  // Start from allocc_op_ and copy_to_tensor_op_ operands
  for (mlir::Value operand : allocc_op_->getOperands()) {
    if (mlir::Operation *defOp = operand.getDefiningOp()) {
      worklist.push_back(defOp);
    }
  }
  for (mlir::Value operand : copy_to_tensor_op_->getOperands()) {
    if (mlir::Operation *defOp = operand.getDefiningOp()) {
      worklist.push_back(defOp);
    }
  }

  while (!worklist.empty()) {
    mlir::Operation *op = worklist.pop_back_val();

    if (!op || visited.contains(op))
      continue;

    visited.insert(op);

    // Only include operations that are inside the loop body
    if (op->getBlock() != loop_body)
      continue;

    // Don't include the anchor operations themselves in the backward slice
    if (op == allocc_op_ || op == copy_to_tensor_op_)
      continue;

    backward_slice_.insert(op);

    // Add operands' defining operations to the worklist
    for (mlir::Value operand : op->getOperands()) {
      if (mlir::Operation *defOp = operand.getDefiningOp()) {
        if (!visited.contains(defOp)) {
          worklist.push_back(defOp);
        }
      }
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Backward slice for loading block:\n";
    for (mlir::Operation *op : backward_slice_) {
      llvm::dbgs() << "  - " << op->getName() << "\n";
    }
  });
}

/**
 * @brief Find the loom.allocc operation.
 */
loom::AlloccOp LoadingBlock::FindAllocc() {
  return mlir::dyn_cast_or_null<loom::AlloccOp>(allocc_op_);
}

/**
 * @brief Find the loom.view consumed by copy_to_tensor.
 */
loom::ViewOp LoadingBlock::FindView() {
  if (auto copy_op =
          mlir::dyn_cast_or_null<loom::CopyToTensorOp>(copy_to_tensor_op_)) {
    return copy_op.getSourceView().getDefiningOp<loom::ViewOp>();
  }
  return nullptr;
}

/**
 * @brief Check if a value is defined in a block that dominates the target
 * block.
 */
static bool isVisibleIn(mlir::Value value, mlir::Block *targetBlock) {
  if (!value || !targetBlock)
    return false;

  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *defBlock = blockArg.getOwner();
    mlir::Block *curr = targetBlock;
    while (curr) {
      if (curr == defBlock)
        return true;
      if (mlir::Operation *parentOp = curr->getParentOp()) {
        curr = parentOp->getBlock();
      } else {
        break;
      }
    }
    return false;
  }

  if (mlir::Operation *defOp = value.getDefiningOp()) {
    mlir::Block *defBlock = defOp->getBlock();
    mlir::Block *curr = targetBlock;
    while (curr) {
      if (curr == defBlock)
        return true;
      if (mlir::Operation *parentOp = curr->getParentOp()) {
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
 */
static mlir::Operation *
cloneWithDependencies(mlir::OpBuilder &builder, mlir::Operation *op,
                      mlir::Block *targetBlock, mlir::IRMapping &mapping,
                      llvm::DenseSet<mlir::Operation *> &cloned) {

  if (!op || cloned.contains(op))
    return nullptr;

  for (mlir::Value operand : op->getOperands()) {
    if (mapping.contains(operand))
      continue;

    if (isVisibleIn(operand, targetBlock))
      continue;

    if (mlir::Operation *defOp = operand.getDefiningOp()) {
      cloneWithDependencies(builder, defOp, targetBlock, mapping, cloned);
    }
  }

  mlir::Operation *clonedOp = builder.clone(*op, mapping);
  cloned.insert(op);

  return clonedOp;
}

/**
 * @brief Create hoisted operations before the loop.
 */
void LoadingBlock::CreateHoistedOps(mlir::OpBuilder &builder) {
  auto allocc = FindAllocc();
  auto copy_to_tensor =
      mlir::dyn_cast_or_null<loom::CopyToTensorOp>(copy_to_tensor_op_);
  auto view = FindView();

  if (!allocc || !copy_to_tensor || !view) {
    LLVM_DEBUG(llvm::dbgs() << "Missing required operations for hoisting\n");
    return;
  }

  mlir::Block *targetBlock = builder.getInsertionBlock();
  mlir::Value ub = getOrReifyLoopUB(builder);

  if (!ub || !isVisibleIn(ub, targetBlock)) {
    LLVM_DEBUG(llvm::dbgs() << "Loop upper bound not visible\n");
    return;
  }

  mlir::IRMapping mapping;
  llvm::DenseSet<mlir::Operation *> clonedOps;

  // 1. Hoist view index calculations and view itself
  auto orig_view_offsets = view.getOffsets();
  auto orig_view_sizes = view.getSizes();
  auto orig_view_strides = view.getStrides();

  llvm::SmallVector<mlir::Value> new_view_offsets;
  for (mlir::Value offset : orig_view_offsets) {
    if (DependsOnLoopIV(offset)) {
      // Substitute loop IV = 0 for base offset computation
      mlir::IRMapping offsetMapping = mapping;
      offsetMapping.map(loop_iv_, builder.create<mlir::arith::ConstantIndexOp>(
                                      view.getLoc(), 0));
      if (mlir::Operation *defOp = offset.getDefiningOp()) {
        cloneWithDependencies(builder, defOp, targetBlock, offsetMapping,
                              clonedOps);
        new_view_offsets.push_back(offsetMapping.lookupOrDefault(offset));
      } else {
        new_view_offsets.push_back(offset);
      }
    } else if (isVisibleIn(offset, targetBlock)) {
      new_view_offsets.push_back(offset);
    } else if (mlir::Operation *defOp = offset.getDefiningOp()) {
      cloneWithDependencies(builder, defOp, targetBlock, mapping, clonedOps);
      new_view_offsets.push_back(mapping.lookupOrDefault(offset));
    } else {
      new_view_offsets.push_back(offset);
    }
  }

  llvm::SmallVector<mlir::Value> new_view_sizes;
  new_view_sizes.push_back(ub); // New outer dimension
  for (mlir::Value size : orig_view_sizes) {
    if (isVisibleIn(size, targetBlock)) {
      new_view_sizes.push_back(size);
    } else if (mlir::Operation *defOp = size.getDefiningOp()) {
      cloneWithDependencies(builder, defOp, targetBlock, mapping, clonedOps);
      new_view_sizes.push_back(mapping.lookupOrDefault(size));
    } else {
      new_view_sizes.push_back(size);
    }
  }

  // Strides for the new view.
  // Typically we want the new dimension to be contiguous with the original
  // ones.
  llvm::SmallVector<mlir::Value> new_view_strides;
  for (mlir::Value stride : orig_view_strides) {
    if (isVisibleIn(stride, targetBlock)) {
      new_view_strides.push_back(stride);
    } else if (mlir::Operation *defOp = stride.getDefiningOp()) {
      cloneWithDependencies(builder, defOp, targetBlock, mapping, clonedOps);
      new_view_strides.push_back(mapping.lookupOrDefault(stride));
    } else {
      new_view_strides.push_back(stride);
    }
  }

  // Clone source memref
  if (!isVisibleIn(view.getSource(), targetBlock))
    return;

  // Prepend prefix to static attributes for the new dimension
  auto updateStaticAttr = [&](llvm::ArrayRef<int64_t> attr, int64_t prefix) {
    llvm::SmallVector<int64_t> vals;
    vals.push_back(prefix);
    for (int64_t v : attr)
      vals.push_back(v);
    return builder.getDenseI64ArrayAttr(vals);
  };

  auto new_static_offsets = updateStaticAttr(view.getStaticOffsets(), 0);
  auto new_static_sizes =
      updateStaticAttr(view.getStaticSizes(), mlir::ShapedType::kDynamic);
  auto new_static_strides = updateStaticAttr(view.getStaticStrides(), 1);

  auto new_view = builder.create<loom::ViewOp>(
      view.getLoc(), view.getResult().getType(), view.getSource(),
      new_view_offsets, new_view_sizes, new_view_strides, new_static_offsets,
      new_static_sizes, new_static_strides, view.getSequentialReuse(),
      view.getSpatialReuse(), view.getTemporalReuse());

  // 2. Hoist AlloccOp
  llvm::SmallVector<mlir::Value> new_allocc_sizes;
  new_allocc_sizes.push_back(ub);
  for (mlir::Value size : allocc.getDynamicSizes()) {
    if (isVisibleIn(size, targetBlock)) {
      new_allocc_sizes.push_back(size);
    } else if (mlir::Operation *defOp = size.getDefiningOp()) {
      cloneWithDependencies(builder, defOp, targetBlock, mapping, clonedOps);
      new_allocc_sizes.push_back(mapping.lookupOrDefault(size));
    } else {
      new_allocc_sizes.push_back(size);
    }
  }

  auto new_allocc = builder.create<loom::AlloccOp>(
      allocc.getLoc(), allocc.getResult().getType(), new_allocc_sizes,
      allocc.getMemoryAttr(), allocc.getAlignmentAttr(),
      allocc.getBufferCountAttr());

  // 3. Hoist CopyToTensorOp
  // The result type needs to be updated (add dimension)
  auto orig_tensor_type =
      mlir::cast<mlir::RankedTensorType>(copy_to_tensor.getResult().getType());
  llvm::SmallVector<int64_t> new_tensor_shape;
  new_tensor_shape.push_back(mlir::ShapedType::kDynamic);
  for (int64_t dim : orig_tensor_type.getShape()) {
    new_tensor_shape.push_back(dim);
  }
  auto new_tensor_type = mlir::RankedTensorType::get(
      new_tensor_shape, orig_tensor_type.getElementType());

  auto new_copy = builder.create<loom::CopyToTensorOp>(
      copy_to_tensor.getLoc(), new_tensor_type, new_view.getResult(),
      new_allocc.getResult(), copy_to_tensor.getProvenance(),
      copy_to_tensor.getInterconnect(), copy_to_tensor.getBroadcast());

  is_valid_ = true;
  replacement_block_.push_back(new_view);
  replacement_block_.push_back(new_allocc);
  replacement_block_.push_back(new_copy);
}

/**
 * @brief Create replacement operations at the original location.
 */
void LoadingBlock::SetReplacementBlock() {
  if (replacement_block_.size() < 3)
    return;

  auto hoisted_copy =
      mlir::dyn_cast<loom::CopyToTensorOp>(replacement_block_[2]);
  if (!hoisted_copy)
    return;

  mlir::OpBuilder builder(copy_to_tensor_op_);
  auto loc = copy_to_tensor_op_->getLoc();

  // Create tensor.extract_slice inside the loop
  auto hoisted_tensor = hoisted_copy.getResult();
  auto hoisted_type =
      mlir::cast<mlir::RankedTensorType>(hoisted_tensor.getType());
  auto rank = hoisted_type.getRank();

  llvm::SmallVector<mlir::OpFoldResult> offsets, sizes, strides;
  offsets.push_back(loop_iv_);
  sizes.push_back(builder.getIndexAttr(1));
  strides.push_back(builder.getIndexAttr(1));

  for (int i = 1; i < rank; ++i) {
    offsets.push_back(builder.getIndexAttr(0));
    if (hoisted_type.isDynamicDim(i)) {
      // Use original dynamic sizes from the old copy op's result type if
      // available Or just re-retrieve from the hoisted tensor
      sizes.push_back(
          builder.create<mlir::tensor::DimOp>(loc, hoisted_tensor, i)
              .getResult());
    } else {
      sizes.push_back(builder.getIndexAttr(hoisted_type.getDimSize(i)));
    }
    strides.push_back(builder.getIndexAttr(1));
  }

  llvm::SmallVector<int64_t> reduced_shape;
  for (int i = 1; i < rank; ++i) {
    reduced_shape.push_back(hoisted_type.getDimSize(i));
  }
  auto reduced_type =
      mlir::RankedTensorType::get(reduced_shape, hoisted_type.getElementType());

  auto slice = builder.create<mlir::tensor::ExtractSliceOp>(
      loc, reduced_type, hoisted_tensor, offsets, sizes, strides);

  copy_to_tensor_op_->getResult(0).replaceAllUsesWith(slice.getResult());

  // Erase original operations
  copy_to_tensor_op_->erase();
  allocc_op_->erase();

  for (auto it = backward_slice_.rbegin(); it != backward_slice_.rend(); ++it) {
    mlir::Operation *op = *it;
    if (op && op->use_empty()) {
      op->erase();
    }
  }
}

/**
 * @brief Hoist the loading block operations to the outer loop.
 */
void LoadingBlock::HoistLoadingBlock() {
  mlir::OpBuilder builder(outer_for_op_);
  CollectBackwardSlice();
  CreateHoistedOps(builder);
}

/**
 * @brief Construct a LoadingBlock from a loom.allocc operation.
 */
LoadingBlock::LoadingBlock(mlir::Operation *allocc_op, mlir::Operation *copy_op,
                           mlir::affine::AffineForOp for_op)
    : outer_for_op_(for_op), allocc_op_(allocc_op), copy_to_tensor_op_(copy_op),
      loop_iv_(nullptr), is_valid_(false) {
  SetLoopAttr();
  CollectBackwardSlice();
}

/**
 * @brief Recursively hoist the loading block to outer loops.
 */
void LoadingBlock::HoistRec(mlir::affine::AffineForOp new_outer_for) {
  // Single step hoisting for now
  HoistLoadingBlock();
  if (is_valid_) {
    SetReplacementBlock();
  }
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
 * @brief Build loading blocks in an affine for loop.
 */
mlir::LogicalResult
BuildLoadingBlocks(mlir::affine::AffineForOp inner_most_for_op,
                   llvm::SmallVector<LoadingBlock, 2> &block_vec) {
  mlir::Block *body = inner_most_for_op.getBody();
  if (!body)
    return mlir::failure();

  for (mlir::Operation &op : *body) {
    if (auto allocc = mlir::dyn_cast<loom::AlloccOp>(&op)) {
      // Find its CopyToTensorOp user
      mlir::Operation *copy_op = nullptr;
      for (auto &user : allocc.getResult().getUses()) {
        if (auto copy = mlir::dyn_cast<loom::CopyToTensorOp>(user.getOwner())) {
          copy_op = copy.getOperation();
          break;
        }
      }

      if (copy_op) {
        block_vec.emplace_back(allocc.getOperation(), copy_op,
                               inner_most_for_op);
      }
    }
  }

  return block_vec.empty() ? mlir::failure() : mlir::success();
}

/**
 * @brief Hoist a single loading block at the specified index.
 */
mlir::LogicalResult HoistSingleBlock(mlir::affine::AffineForOp inner_most_loop,
                                     size_t block_index) {
  llvm::SmallVector<LoadingBlock, 2> loading_blocks;
  if (failed(BuildLoadingBlocks(inner_most_loop, loading_blocks)) ||
      block_index >= loading_blocks.size()) {
    return mlir::failure();
  }

  loading_blocks[block_index].Hoist();

  return loading_blocks[block_index].IsValid() ? mlir::success()
                                               : mlir::failure();
}

} // namespace loom::affine
