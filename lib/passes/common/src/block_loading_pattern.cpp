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

  if (!alloc_op_ || !copy_to_tensor_op_)
    return;

  mlir::Block *loop_body = outer_for_op_.getBody();
  if (!loop_body)
    return;

  llvm::SmallVector<mlir::Operation *, 16> worklist;
  llvm::SmallPtrSet<mlir::Operation *, 16> visited;

  // Start from alloc_op_ and copy_to_tensor_op_ operands
  for (mlir::Value operand : alloc_op_->getOperands()) {
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
    if (op == alloc_op_ || op == copy_to_tensor_op_)
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
 * @brief Find the loom.alloc operation.
 */
loom::AllocOp LoadingBlock::FindAlloc() {
  return mlir::dyn_cast_or_null<loom::AllocOp>(alloc_op_);
}

/**
 * @brief Find the loom.view consumed by copy_to_tensor (or pack_to_tensor in
 * future).
 */
loom::ViewOp LoadingBlock::FindView() {
  if (auto copy_op =
          mlir::dyn_cast_or_null<loom::CopyToTensorOp>(copy_to_tensor_op_)) {
    return copy_op.getSourceView().getDefiningOp<loom::ViewOp>();
  }
  return nullptr;
}

/**
 * @brief Determine which dimension of the view depends on the loop IV.
 */
int LoadingBlock::getMovingDimension() {
  auto view = FindView();
  if (!view)
    return -1;

  auto offsets = view.getOffsets();
  for (int i = 0; i < offsets.size(); ++i) {
    if (DependsOnLoopIV(offsets[i])) {
      return i;
    }
  }
  return -1;
}

/**
 * @brief Compute the expanded shape for hoisted view.
 */
/**
 * @brief Compute the expanded shape for hoisted view.
 */
llvm::SmallVector<mlir::Value, 2>
LoadingBlock::inferHoistedViewShape(mlir::OpBuilder &builder) {
  auto view = FindView();
  auto origSizes = view.getSizes();
  llvm::SmallVector<mlir::Value, 2> newSizes;

  int moveDim = getMovingDimension();
  if (moveDim < 0)
    return {};

  mlir::Value blockSize = origSizes[moveDim];

  // Lazy creation of loopUB to avoid redundant operations if optimization
  // applies
  mlir::Value loopUB = nullptr;
  auto ensureLoopUB = [&]() {
    if (!loopUB)
      loopUB = getOrReifyLoopUB(builder);
    return loopUB;
  };

  for (int i = 0; i < origSizes.size(); ++i) {
    if (i == moveDim) {
      // Optimization: If the source memref dimension is static and we are
      // covering it, use the constant size. We assume direct 1:1 mapping for
      // now (strides are usually [1, 1]). Check if source memref has static
      // shape.
      auto srcType = mlir::cast<mlir::MemRefType>(view.getSource().getType());
      if (srcType.hasStaticShape()) {
        // Assuming view dimensions map 1:1 to source (no transpose in view
        // itself). If offsets/strides are simple.
        if (i < srcType.getRank()) {
          int64_t dimSize = srcType.getDimSize(i);
          if (dimSize != mlir::ShapedType::kDynamic) {
            // Return constant op
            newSizes.push_back(builder.create<mlir::arith::ConstantIndexOp>(
                view.getLoc(), dimSize));
            continue;
          }
        }
      }

      // Fallback: Expand moving dimension: K = loopUB * blockSize
      mlir::Value expanded = builder.create<mlir::arith::MulIOp>(
          view.getLoc(), ensureLoopUB(), blockSize);
      newSizes.push_back(expanded);
    } else {
      newSizes.push_back(origSizes[i]);
    }
  }
  return newSizes;
}

/**
 * @brief Generate outer_dims_perm for pack_to_tensor.
 */
llvm::SmallVector<int64_t, 2> LoadingBlock::computePackPermutation() {
  int moveDim = getMovingDimension();
  // We want the resulting tensor to be [T_wave, M, N_tile] or similar 3D.
  // The input 2D view is [Dim0, Dim1].
  // If moving dim is 1 (Dim1), then it's split into T_wave * Tile.
  // If moving dim is 0 (Dim0), then it's split into T_wave * Tile.
  //
  // pack_to_tensor logic:
  // "It performs tiling on the second dimension of the input view and permutes
  // the outer dimensions." Wait, if it *always* tiles the second dimension (dim
  // 1), then we MUST ensure the moving dimension IS the second dimension?
  //
  // Let's re-read LoomOps.td:
  // "It performs tiling on the second dimension of the input view"
  // If this is strict, then if moving dim is 0, we can't use it directly?
  // User didn't say we should change pack_to_tensor semantics.
  //
  // If moving dim is 0 (Row moves), we effectively have [T_wave * BM, BK].
  // We want to slice [iv, :, :].
  //
  // Actually, let's look at `mm_2Dmesh`.
  // Matrix A: `loom.view %arg0[%23, %24] ...` where %23 (dim 0) varies with
  // outer loop, %24 (dim 1) varies with inner loop K. Wait, the hoisting target
  // is `affine.for %arg7 ...` (K loop). Inside this loop:
  // `%23 = arith.muli %15, %12` -> Depends on %15 which is loop invariant
  // relative to %arg7?
  // `%24 = arith.muli %arg7, %14` -> Depends on K loop IV.
  // So for Matrix A, Dim 1 (Columns) is moving. -> matches "tiles second
  // dimension".
  //
  // Matrix B: `loom.view %arg1[%24, %28] ...`
  // %24 (Dim 0) is K loop IV.
  // So for Matrix B, Dim 0 (Rows) is moving.
  //
  // If PackToTensor *only* tiles 2nd dimension, we have a problem for Matrix B.
  // BUT the description says:
  // "Example: %packed_A = loom.pack_to_tensor %view ... outer_dims_perm = [1,
  // 0]"
  //
  // If the op implies semantics tailored for these cases, maybe it tiles "the
  // dimension specified by inner_tiles"? "It performs tiling on the second
  // dimension of the input view" -> This text might be specific to the example
  // or default?
  //
  // However, I see "outer_dims_perm".
  // Let's assume the user wants me to implement `computePackPermutation` to
  // handle this. If the op is rigid to 2nd dim tiling, then for Matrix B (row
  // moving), we might need `transpose` view first? OR, maybe the description
  // text is just one example.
  //
  // Given the "Directly modify code" instruction, if PackToTensor handles
  // generic tiling, it should be fine. But if the C++ implementation of
  // PackToTensor (which I can't see, only TD) has logic, I might break it.
  //
  // Let's follow the user's hint:
  // "computePermutation(moveDim): 根据移动维度生成 T_{wave} 始终位于第 0
  // 维的排列向量"
  //
  // If moveDim == 1 (Cols), we want [T_wave, Rows, Cols_Tile].
  // Original 2D: [Rows, Cols].
  // Split Cols -> [Rows, T_wave, Cols_Tile].
  // Permute -> [1, 0] -> [T_wave, Rows, Cols_Tile].
  //
  // If moveDim == 0 (Rows), we want [T_wave, Rows_Tile, Cols].
  // Original 2D: [Rows, Cols].
  // Split Rows -> [T_wave, Rows_Tile, Cols].
  // Permute -> [0, 1] -> [T_wave, Rows_Tile, Cols].
  //
  // So:
  if (moveDim == 1)
    return {1, 0};
  return {0, 1};
}

/**
 * @brief Get the inner tile size.
 */
mlir::Value LoadingBlock::getInnerTileSize() {
  auto view = FindView();
  int moveDim = getMovingDimension();
  if (moveDim < 0)
    return nullptr;
  return view.getSizes()[moveDim];
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
  auto alloc = FindAlloc();
  auto copy_to_tensor =
      mlir::dyn_cast_or_null<loom::CopyToTensorOp>(copy_to_tensor_op_);
  auto view = FindView();

  if (!alloc || !copy_to_tensor || !view) {
    LLVM_DEBUG(llvm::dbgs() << "Missing required operations for hoisting\n");
    return;
  }

  mlir::Block *targetBlock = builder.getInsertionBlock();
  int moveDim = getMovingDimension();

  if (moveDim < 0) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot identify moving dimension\n");
    return;
  }

  mlir::IRMapping mapping;
  llvm::DenseSet<mlir::Operation *> clonedOps;

  // 1. Create 2D view with expanded shape
  // Hoist logical view definition
  auto orig_view_offsets = view.getOffsets();
  auto orig_view_sizes = view.getSizes();
  auto orig_view_strides = view.getStrides();

  // New Offsets:
  // If dimension moves (depends on IV), set offset to 0.
  // Otherwise, clone the offset calculation.
  llvm::SmallVector<mlir::Value> new_view_offsets;
  for (int i = 0; i < orig_view_offsets.size(); ++i) {
    mlir::Value offset = orig_view_offsets[i];
    if (i == moveDim) {
      new_view_offsets.push_back(
          builder.create<mlir::arith::ConstantIndexOp>(view.getLoc(), 0));
    } else {
      if (isVisibleIn(offset, targetBlock)) {
        new_view_offsets.push_back(offset);
      } else if (mlir::Operation *defOp = offset.getDefiningOp()) {
        cloneWithDependencies(builder, defOp, targetBlock, mapping, clonedOps);
        new_view_offsets.push_back(mapping.lookupOrDefault(offset));
      } else {
        new_view_offsets.push_back(offset);
      }
    }
  }

  // New Sizes:
  // If dimension moves, size = ub * blockSize.
  // Otherwise, clone size.
  llvm::SmallVector<mlir::Value> new_view_sizes =
      inferHoistedViewShape(builder);
  // Need to map non-moving sizes if they depend on internal computations
  // (usually they don't or they are hoisted)
  for (int i = 0; i < new_view_sizes.size(); ++i) {
    if (i != moveDim && !isVisibleIn(new_view_sizes[i], targetBlock)) {
      // If inferHoistedViewShape used original values that need cloning...
      // Actually inferHoistedViewShape reuses origSizes values.
      // We should probably safeguard this.
      mlir::Value s = new_view_sizes[i];
      if (mlir::Operation *defOp = s.getDefiningOp()) {
        if (!clonedOps.contains(defOp)) // if not already cloned
          cloneWithDependencies(builder, defOp, targetBlock, mapping,
                                clonedOps);
        new_view_sizes[i] = mapping.lookupOrDefault(s);
      }
    }
  }

  // New Strides: clone existing
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

  // Clone source memref if needed
  if (!isVisibleIn(view.getSource(), targetBlock))
    return;

  auto new_view = builder.create<loom::ViewOp>(
      view.getLoc(), view.getResult().getType(), view.getSource(),
      new_view_offsets, new_view_sizes, new_view_strides,
      view.getStaticOffsets(), view.getStaticSizes(), view.getStaticStrides(),
      view.getSequentialReuse(), view.getSpatialReuse(),
      view.getTemporalReuse());

  // 2. Hoist AllocOp (2D with expanded size)
  auto new_alloc = builder.create<loom::AllocOp>(
      alloc.getLoc(), alloc.getResult().getType(), new_view_sizes,
      alloc.getMemoryAttr(), alloc.getAlignmentAttr(),
      alloc.getBufferCountAttr());

  // 3. Create PackToTensorOp
  // Use tracked innerTile (block size) as operand.
  auto perm = computePackPermutation();

  // Pass ALL view sizes as inner_tiles
  llvm::SmallVector<mlir::Value> inner_tile_sizes_operands;
  auto original_view_sizes = view.getSizes();
  for (mlir::Value s : original_view_sizes) {
    // Need to ensure visibility, though usually symbolic constants.
    if (isVisibleIn(s, targetBlock)) {
      inner_tile_sizes_operands.push_back(s);
    } else if (mlir::Operation *defOp = s.getDefiningOp()) {
      cloneWithDependencies(builder, defOp, targetBlock, mapping, clonedOps);
      inner_tile_sizes_operands.push_back(mapping.lookupOrDefault(s));
    } else {
      // Should be visible or block arg
      inner_tile_sizes_operands.push_back(s);
    }
  }

  // Construct 3D tensor result type
  auto orig_tensor_type =
      mlir::cast<mlir::RankedTensorType>(copy_to_tensor.getResult().getType());
  llvm::SmallVector<int64_t> packed_shape;
  packed_shape.push_back(mlir::ShapedType::kDynamic); // T_wave
  packed_shape.push_back(mlir::ShapedType::kDynamic); // Outer
  packed_shape.push_back(mlir::ShapedType::kDynamic); // Inner tile

  auto packed_type = mlir::RankedTensorType::get(
      packed_shape, orig_tensor_type.getElementType());

  auto packed = builder.create<loom::PackToTensorOp>(
      copy_to_tensor.getLoc(), packed_type, new_view, new_alloc,
      inner_tile_sizes_operands, builder.getDenseI64ArrayAttr(perm));

  is_valid_ = true;
  replacement_block_.push_back(new_view);
  replacement_block_.push_back(new_alloc);
  replacement_block_.push_back(packed);
}

/**
 * @brief Create replacement operations at the original location.
 */
void LoadingBlock::SetReplacementBlock() {
  if (replacement_block_.size() < 3)
    return;

  auto hoisted_pack =
      mlir::dyn_cast<loom::PackToTensorOp>(replacement_block_[2]);
  if (!hoisted_pack)
    return;

  mlir::OpBuilder builder(copy_to_tensor_op_);
  auto loc = copy_to_tensor_op_->getLoc();

  // Create tensor.extract_slice inside the loop
  auto hoisted_tensor = hoisted_pack.getResult();
  auto hoisted_type =
      mlir::cast<mlir::RankedTensorType>(hoisted_tensor.getType());
  // Expected hoisted type is 3D: [T_wave, M_outer, N_inner] or similar

  llvm::SmallVector<mlir::OpFoldResult> offsets, sizes, strides;
  // Offset: [iv, 0, 0]
  offsets.push_back(loop_iv_);
  offsets.push_back(builder.getIndexAttr(0));
  offsets.push_back(builder.getIndexAttr(0));

  // Size: [1, dim1, dim2]
  // Size: [1, dim1, dim2]
  // Dimensions 1 and 2 correspond to M and N_tile (or similar).
  // One of them is the tiled dimension size = blockSize.
  // The other is from view size (which might be the other dimension).
  // But wait, the shape of hoisted tensor matches `[T_wave, Outer, Inner]`.
  // Permutation [1, 0] (Column move): [T_wave, M, N_tile].
  // Permutation [0, 1] (Row move): [T_wave, M_tile, N].
  //
  // We can just use `getInnerTileSize()` (SSA value) for the tile dimension!
  // And the other dimension size?
  // `view` has sizes [D0, D1].
  // If Col move: View [D0, D1_full]. Result [T_wave, D0, D1_tile].
  // D0 size is `view.getSizes()[0]`. D1_tile size is `getInnerTileSize()`.
  //
  // If Row move: View [D0_full, D1]. Result [T_wave, D0_tile, D1].
  // D0_tile is `getInnerTileSize()`. D1 is `view.getSizes()[1]`.
  //
  // So we can reconstruct the sizes SSA values instead of `tensor.dim`.

  sizes.push_back(builder.getIndexAttr(1)); // T_wave slice size is 1

  int moveDim = getMovingDimension();
  auto view = FindView();
  auto viewSizes = view.getSizes();

  if (moveDim == 1) { // Cols move, perm [1, 0] -> [T_wave, D0, D1_tile]
    sizes.push_back(viewSizes[0]);       // D0
    sizes.push_back(getInnerTileSize()); // D1_tile (blockSize)
  } else { // Rows move, perm [0, 1] -> [T_wave, D0_tile, D1]
    sizes.push_back(getInnerTileSize()); // D0_tile
    sizes.push_back(viewSizes[1]);       // D1
  }

  // Stride: [1, 1, 1]
  strides.append(3, builder.getIndexAttr(1));

  llvm::SmallVector<int64_t> slice_shape;
  slice_shape.push_back(1);
  for (int i = 1; i < hoisted_type.getRank(); ++i) {
    slice_shape.push_back(hoisted_type.getDimSize(i));
  }
  auto slice_type =
      mlir::RankedTensorType::get(slice_shape, hoisted_type.getElementType());

  auto slice = builder.create<mlir::tensor::ExtractSliceOp>(
      loc, slice_type, hoisted_tensor, offsets, sizes, strides);

  // Collapse shape: [1, BM, BK] -> [BM, BK]
  // Used reassociation: [[0, 1], [2]] for perm=[1, 0] case?
  // Wait, if result is [1, BM, BK], we want [BM, BK].
  // Reassociation indices map dimensions of source to dimensions of result.
  // Actually collapse_shape maps result dims to source dims.
  // source: [1, BM, BK] (rank 3)
  // target: [BM, BK] (rank 2)
  // assoc: [[0, 1], [2]] means target[0] covers source[0,1], target[1] covers
  // source[2]. This effectively merges dim 0 and 1.

  // We need to know which dimension we expanded.
  // But wait, the pack op output depends on permutation.
  // If perm=[1, 0], then output is [T_wave, M, N_tile].
  // Slice is [1, M, N_tile].
  // We want [M, N_tile].
  // So we merge 0 and 1: [[0, 1], [2]]

  llvm::SmallVector<mlir::ReassociationIndices> reassociation;
  reassociation.push_back({0, 1});
  reassociation.push_back({2});

  auto orig_type = mlir::cast<mlir::RankedTensorType>(
      copy_to_tensor_op_->getResult(0).getType());

  auto collapsed = builder.create<mlir::tensor::CollapseShapeOp>(
      loc, orig_type, slice, reassociation);

  copy_to_tensor_op_->getResult(0).replaceAllUsesWith(collapsed.getResult());

  // Erase original operations
  copy_to_tensor_op_->erase();
  alloc_op_->erase();

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
 * @brief Construct a LoadingBlock from a loom.alloc operation.
 */
LoadingBlock::LoadingBlock(mlir::Operation *alloc_op, mlir::Operation *copy_op,
                           mlir::affine::AffineForOp for_op)
    : outer_for_op_(for_op), alloc_op_(alloc_op), copy_to_tensor_op_(copy_op),
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
    if (auto alloc = mlir::dyn_cast<loom::AllocOp>(&op)) {
      // Find its CopyToTensorOp or PackToTensorOp user
      mlir::Operation *copy_op = nullptr;
      for (auto &user : alloc.getResult().getUses()) {
        if (auto copy = mlir::dyn_cast<loom::CopyToTensorOp>(user.getOwner())) {
          copy_op = copy.getOperation();
          break;
        }
        // TODO: Handle existing PackToTensorOp if needed
      }

      if (copy_op) {
        block_vec.emplace_back(alloc.getOperation(), copy_op,
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
