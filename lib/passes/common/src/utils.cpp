/**
 * @file utils.cpp
 * @brief Implementation of common utilities for function cloning.
 */

#include "utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

#include "hardware_info.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

// Include the generated Loom dialect headers
#include "LoomDialect.h.inc"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace loom {
namespace utils {

ModuleOp getParentModule(func::FuncOp func) {
  Operation *parent = func->getParentOp();
  if (auto module = dyn_cast_or_null<ModuleOp>(parent)) {
    return module;
  }
  return nullptr;
}

namespace {

/// Private helper to clone a function and optionally apply a modifier.
/// Handles insertion point management and failure cleanup.
func::FuncOp cloneFunctionImpl(
    OpBuilder &builder, func::FuncOp originalFunc, llvm::StringRef newName,
    Operation *insertAfter, DictionaryAttr moduleAttrs,
    std::function<LogicalResult(func::FuncOp)> modifier = nullptr) {

  // Set insertion point
  if (insertAfter) {
    builder.setInsertionPointAfter(insertAfter);
  }

  // Create wrapper module if needed
  ModuleOp wrapperModule = nullptr;
  OpBuilder effectiveBuilder = builder;
  if (moduleAttrs) {
    wrapperModule = ModuleOp::create(originalFunc.getLoc());
    wrapperModule->setAttrs(moduleAttrs);
    builder.insert(wrapperModule);
    effectiveBuilder = OpBuilder(wrapperModule.getBodyRegion());
  }

  // Clone the function
  IRMapping mapping;
  auto clonedFunc =
      cast<func::FuncOp>(effectiveBuilder.clone(*originalFunc, mapping));
  clonedFunc.setName(newName);

  // Apply modifier if provided
  if (modifier) {
    if (failed(modifier(clonedFunc))) {
      if (wrapperModule) {
        wrapperModule.erase();
      } else {
        clonedFunc.erase();
      }
      return nullptr;
    }
  }

  // Update builder's insertion point to after the inserted operation
  builder.setInsertionPointAfter(wrapperModule ? (Operation *)wrapperModule
                                               : (Operation *)clonedFunc);

  return clonedFunc;
}

} // namespace

func::FuncOp cloneAndInsertFunction(OpBuilder &builder,
                                    func::FuncOp originalFunc,
                                    llvm::StringRef newName,
                                    Operation *insertAfter) {
  return cloneFunctionImpl(builder, originalFunc, newName, insertAfter,
                           nullptr);
}

func::FuncOp cloneAndInsertFunctionWithModuleWrapper(OpBuilder &builder,
                                                     func::FuncOp originalFunc,
                                                     llvm::StringRef newName,
                                                     DictionaryAttr moduleAttrs,
                                                     Operation *insertAfter) {
  return cloneFunctionImpl(builder, originalFunc, newName, insertAfter,
                           moduleAttrs);
}

func::FuncOp cloneModifyAndInsertFunction(
    OpBuilder &builder, func::FuncOp originalFunc, llvm::StringRef newName,
    std::function<LogicalResult(func::FuncOp)> modifier,
    Operation *insertAfter) {
  return cloneFunctionImpl(builder, originalFunc, newName, insertAfter, nullptr,
                           modifier);
}

func::FuncOp cloneModifyAndInsertFunctionWithModuleWrapper(
    OpBuilder &builder, func::FuncOp originalFunc, llvm::StringRef newName,
    DictionaryAttr moduleAttrs,
    std::function<LogicalResult(func::FuncOp)> modifier,
    Operation *insertAfter) {
  return cloneFunctionImpl(builder, originalFunc, newName, insertAfter,
                           moduleAttrs, modifier);
}

llvm::SmallVector<func::FuncOp> collectFunctions(ModuleOp module) {
  llvm::SmallVector<func::FuncOp> funcs;

  // Recursively collect functions from nested modules
  module.walk([&](func::FuncOp func) { funcs.push_back(func); });

  return funcs;
}

func::FuncOp
cloneFuncWithConstraints(OpBuilder &builder, func::FuncOp originalFunc,
                         llvm::StringRef newName, DictionaryAttr moduleAttrs,
                         llvm::StringRef /*passName*/,
                         std::function<LogicalResult(func::FuncOp)> modifier,
                         Operation *insertAfter) {
  return cloneFunctionImpl(builder, originalFunc, newName, insertAfter,
                           moduleAttrs, modifier);
}

StringRef traceToSymbolicVar(Value val) {
  if (!val)
    return "";

  // Handle direct loom.sym
  if (auto getSym = val.getDefiningOp<loom::SymOp>()) {
    return getSym.getSymbolRef().getLeafReference().getValue();
  }

  // Handle arith.muli/addi/etc. if needed, but for now we follow the user's
  // sketch where block sizes are directly used from
  // loom.get_symbolic_block_size.

  return "";
}

llvm::SmallVector<AllocInfo> collectL1AllocInfos(func::FuncOp func) {
  llvm::SmallVector<AllocInfo> allocInfos;

  func.walk([&](loom::AllocOp alloc) {
    // 1. Only care about allocations on @L1
    if (alloc.getMemory().getLeafReference() != "L1")
      return;

    // 2. Find element type from users (loom.init_tensor or loom.copy_to_tensor)
    Type elementType;
    for (auto user : alloc.getResult().getUsers()) {
      if (auto initTensor = dyn_cast<loom::InitTensorOp>(user)) {
        elementType = cast<RankedTensorType>(initTensor.getResult().getType())
                          .getElementType();
        break;
      }
      if (auto copyToTensor = dyn_cast<loom::CopyToTensorOp>(user)) {
        elementType = cast<RankedTensorType>(copyToTensor.getResult().getType())
                          .getElementType();
        break;
      }
    }

    if (!elementType)
      return;

    // Get base element size (Bytes)
    int64_t baseElemSize = elementType.getIntOrFloatBitWidth() / 8;

    AllocInfo info;
    // 3. Consider buffer_count (multi-buffering takes more space)
    info.elemSize = baseElemSize * alloc.getBufferCount();

    // 4. Trace dynamic dimensions
    auto dynamicOperands = alloc.getSizes();
    llvm::SmallVector<StringRef> symbolicVars;
    llvm::SmallVector<std::pair<int64_t, StringRef>> ceildivs;

    for (Value val : dynamicOperands) {
      if (auto ceildiv = val.getDefiningOp<arith::CeilDivSIOp>()) {
        // ... (existing arith.ceildivsi handling) ...
        int64_t numerator = -1;
        if (auto constOp =
                ceildiv.getLhs().getDefiningOp<arith::ConstantIndexOp>()) {
          numerator = constOp.value();
        } else if (auto constIntOp =
                       ceildiv.getLhs().getDefiningOp<arith::ConstantIntOp>()) {
          numerator = constIntOp.value();
        }

        if (numerator > 0) {
          StringRef denominator = traceToSymbolicVar(ceildiv.getRhs());
          if (!denominator.empty()) {
            ceildivs.push_back({numerator, denominator});
            continue;
          }
        }
      } else if (auto apply =
                     val.getDefiningOp<mlir::affine::AffineApplyOp>()) {
        // Handle affine.apply affine_map<()[s0] -> (Constant ceildiv
        // s0)>()[%sym]
        auto map = apply.getAffineMap();
        if (map.getNumResults() == 1) {
          auto expr = map.getResult(0);
          if (expr.getKind() == mlir::AffineExprKind::CeilDiv) {
            auto binary = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
            auto lhs = binary.getLHS();
            auto rhs = binary.getRHS();
            if (lhs.getKind() == mlir::AffineExprKind::Constant &&
                rhs.getKind() == mlir::AffineExprKind::SymbolId) {
              int64_t numerator =
                  mlir::cast<mlir::AffineConstantExpr>(lhs).getValue();
              unsigned symIdx =
                  mlir::cast<mlir::AffineSymbolExpr>(rhs).getPosition();
              if (symIdx < apply.getNumOperands()) {
                StringRef denominator =
                    traceToSymbolicVar(apply.getOperand(symIdx));
                if (!denominator.empty()) {
                  ceildivs.push_back({numerator, denominator});
                  continue;
                }
              }
            }
          }
        }
      }

      // Normal symbolic variable tracing
      StringRef symName = traceToSymbolicVar(val);
      if (!symName.empty()) {
        symbolicVars.push_back(symName);
      }
    }

    // 5. Cancellation logic for hoisted blocks: block_K * (K_total / block_K)
    // -> K_total
    for (auto &pair : ceildivs) {
      int64_t numerator = pair.first;
      StringRef denominator = pair.second;
      for (auto it = symbolicVars.begin(); it != symbolicVars.end(); ++it) {
        if (*it == denominator) {
          symbolicVars.erase(it);
          info.elemSize *= numerator; // Merge total size multiplication
          break;
        }
      }
    }

    info.dims = symbolicVars;

    // 6. Record if it has symbolic dims or total size exceeds single element
    // (fixed-size multi-buffer or hoisted buffer)
    if (!info.dims.empty() || info.elemSize > baseElemSize) {
      allocInfos.push_back(std::move(info));
    }
  });

  return allocInfos;
}

} // namespace utils

void utils::composeAndCanonicalizeAffineApplies(func::FuncOp func) {
  SmallVector<affine::AffineApplyOp> applies;
  func.walk([&](affine::AffineApplyOp op) { applies.push_back(op); });
  for (affine::AffineApplyOp op : applies) {
    OpBuilder b(op);
    AffineMap map = op.getAffineMap();
    SmallVector<Value> operands(op.getOperands().begin(),
                                op.getOperands().end());
    affine::fullyComposeAffineMapAndOperands(&map, &operands);
    affine::canonicalizeMapAndOperands(&map, &operands);
    bool sameMap = (map == op.getAffineMap());
    bool sameOperands =
        operands.size() == op.getNumOperands() &&
        std::equal(operands.begin(), operands.end(), op.getOperands().begin());
    if (sameMap && sameOperands)
      continue;
    auto newOp = affine::AffineApplyOp::create(b, op.getLoc(), map, operands);
    op.replaceAllUsesWith(newOp.getResult());
    op.erase();
  }

  SmallVector<Operation *> toErase;
  func.walk([&](Operation *op) {
    if (mlir::isOpTriviallyDead(op))
      toErase.push_back(op);
  });
  for (Operation *op : toErase)
    op->erase();
} // namespace utils

namespace utils {

static OpFoldResult canonicalizeOFR(OpFoldResult ofr, MLIRContext *ctx) {
  if (mlir::isa<Value>(ofr))
    return ofr;
  if (auto attr = mlir::dyn_cast<Attribute>(ofr)) {
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
      return Builder(ctx).getIndexAttr(intAttr.getInt());
    }
  }
  return ofr;
}

static SmallVector<SymbolicDim, 4> getMixedSizesFromType(Type type) {
  SmallVector<SymbolicDim, 4> result;
  if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type)) {
    auto shape = tensorType.getShape();
    Builder b(type.getContext());
    for (int64_t dim : shape) {
      if (ShapedType::isDynamic(dim)) {
        result.push_back(b.getIndexAttr(-1));
      } else {
        result.push_back(b.getIndexAttr(dim));
      }
    }
  } else if (auto memrefType = mlir::dyn_cast<MemRefType>(type)) {
    auto shape = memrefType.getShape();
    Builder b(type.getContext());
    for (int64_t dim : shape) {
      if (ShapedType::isDynamic(dim)) {
        result.push_back(b.getIndexAttr(-1));
      } else {
        result.push_back(b.getIndexAttr(dim));
      }
    }
  }
  return result;
}

SmallVector<SymbolicDim, 4> traceShape(Value v) {
  if (!v)
    return {};

  // Case 1: BlockArgument (affine.for / scf.for iter_args)
  if (auto arg = mlir::dyn_cast<BlockArgument>(v)) {
    if (auto forOp = mlir::dyn_cast<affine::AffineForOp>(
            arg.getOwner()->getParentOp())) {
      unsigned argIdx = arg.getArgNumber() - 1;
      return traceShape(forOp.getInits()[argIdx]);
    }
    if (auto forOp = mlir::dyn_cast<scf::ForOp>(
            arg.getOwner()->getParentOp())) {
      unsigned argIdx = arg.getArgNumber() - 1; // 1 IV in scf.for
      return traceShape(forOp.getInits()[argIdx]);
    }
  }

  Operation *op = v.getDefiningOp();
  if (!op)
    return {};

  SmallVector<SymbolicDim, 4> rawDims;

  // Case A: tensor.empty
  if (auto emptyOp = mlir::dyn_cast<tensor::EmptyOp>(op)) {
    auto type = emptyOp.getType();
    auto shape = type.getShape();
    auto dynamicSizes = emptyOp.getDynamicSizes();
    unsigned dynamicIdx = 0;
    Builder b(op->getContext());
    for (int64_t dim : shape) {
      if (ShapedType::isDynamic(dim)) {
        rawDims.push_back(dynamicSizes[dynamicIdx++]);
      } else {
        rawDims.push_back(b.getIndexAttr(dim));
      }
    }
  }
  // Case B: bufferization.to_tensor + memref.subview
  else if (auto toTensor = mlir::dyn_cast<bufferization::ToTensorOp>(op)) {
    Value memref = toTensor.getOperand();
    if (auto subview = memref.getDefiningOp<memref::SubViewOp>()) {
      rawDims = subview.getMixedSizes();
    } else {
      rawDims = getMixedSizesFromType(memref.getType());
    }
  }
  // Case C: linalg.op (DPS)
  else if (auto linalgOp = mlir::dyn_cast<linalg::LinalgOp>(op)) {
    unsigned resultIdx = mlir::cast<OpResult>(v).getResultNumber();
    Value init = linalgOp.getDpsInits()[resultIdx];
    return traceShape(init);
  }
  // Case D: affine.for / scf.for (Results)
  else if (auto forOp = mlir::dyn_cast<affine::AffineForOp>(op)) {
    unsigned resultIdx = mlir::cast<OpResult>(v).getResultNumber();
    return traceShape(forOp.getInits()[resultIdx]);
  }
  else if (auto forOp = mlir::dyn_cast<scf::ForOp>(op)) {
    unsigned resultIdx = mlir::cast<OpResult>(v).getResultNumber();
    return traceShape(forOp.getInits()[resultIdx]);
  }
  // Case E: tensor.extract_slice
  else if (auto extractSlice = mlir::dyn_cast<tensor::ExtractSliceOp>(op)) {
    rawDims = extractSlice.getMixedSizes();
  }
  // Case F: loom.reduce_sum (output has same shape as input)
  else if (auto reduceSumOp = mlir::dyn_cast<loom::ReduceSumOp>(op)) {
    return traceShape(reduceSumOp.getInput());
  }
  // Fallback
  else {
    rawDims = getMixedSizesFromType(v.getType());
  }

  // Canonicalize all dimensions to ensure consistent Attribute types
  SmallVector<SymbolicDim, 4> canonicalDims;
  for (auto dim : rawDims) {
    canonicalDims.push_back(canonicalizeOFR(dim, v.getContext()));
  }
  return canonicalDims;
}

} // namespace utils
} // namespace loom
