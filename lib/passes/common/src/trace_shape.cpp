/**
 * @file trace_shape.cpp
 * @brief Implementation of symbolic-shape tracing for tensor/memref values.
 */

#include "trace_shape.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace loom {
namespace utils {

namespace {

OpFoldResult canonicalizeOFR(OpFoldResult ofr, MLIRContext *ctx) {
  if (mlir::isa<Value>(ofr))
    return ofr;
  if (auto attr = mlir::dyn_cast<Attribute>(ofr)) {
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
      return Builder(ctx).getIndexAttr(intAttr.getInt());
    }
  }
  return ofr;
}

SmallVector<SymbolicDim, 4> getMixedSizesFromType(Type type) {
  SmallVector<SymbolicDim, 4> result;
  ArrayRef<int64_t> shape;
  if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type))
    shape = tensorType.getShape();
  else if (auto memrefType = mlir::dyn_cast<MemRefType>(type))
    shape = memrefType.getShape();
  else
    return result;

  Builder b(type.getContext());
  for (int64_t dim : shape)
    result.push_back(b.getIndexAttr(ShapedType::isDynamic(dim) ? -1 : dim));
  return result;
}

} // namespace

SmallVector<SymbolicDim, 4> traceShape(Value v) {
  if (!v)
    return {};

  // BlockArgument (affine.for / scf.for iter_args)
  if (auto arg = mlir::dyn_cast<BlockArgument>(v)) {
    if (auto forOp = mlir::dyn_cast<affine::AffineForOp>(
            arg.getOwner()->getParentOp())) {
      unsigned argIdx = arg.getArgNumber() - 1;
      return traceShape(forOp.getInits()[argIdx]);
    }
    if (auto forOp =
            mlir::dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())) {
      unsigned argIdx = arg.getArgNumber() - 1; // 1 IV in scf.for
      return traceShape(forOp.getInits()[argIdx]);
    }
  }

  Operation *op = v.getDefiningOp();
  if (!op) {
    if (auto tensorType = mlir::dyn_cast<RankedTensorType>(v.getType());
        tensorType && tensorType.getRank() == 0)
      return {};
    assert(false && "traceShape cannot trace block argument shape");
    llvm::report_fatal_error("traceShape cannot trace block argument shape");
    return {};
  }

  SmallVector<SymbolicDim, 4> rawDims;

  // tensor.empty
  if (auto emptyOp = mlir::dyn_cast<tensor::EmptyOp>(op)) {
    auto type = emptyOp.getType();
    auto shape = type.getShape();
    auto dynamicSizes = emptyOp.getDynamicSizes();
    unsigned dynamicIdx = 0;
    Builder b(op->getContext());
    for (int64_t dim : shape) {
      if (ShapedType::isDynamic(dim))
        rawDims.push_back(dynamicSizes[dynamicIdx++]);
      else
        rawDims.push_back(b.getIndexAttr(dim));
    }
  }
  // bufferization.to_tensor + memref.subview
  else if (auto toTensor = mlir::dyn_cast<bufferization::ToTensorOp>(op)) {
    Value memref = toTensor.getOperand();
    if (auto subview = memref.getDefiningOp<memref::SubViewOp>()) {
      // Drop unit dims for rank-reducing subviews so the dim count matches
      // the result tensor rank.
      auto allSizes = subview.getMixedSizes();
      ArrayRef<int64_t> allStaticSizes = subview.getStaticSizes();
      for (size_t i = 0; i < allSizes.size(); ++i) {
        int64_t s = allStaticSizes[i];
        if (s == ShapedType::kDynamic || s != 1)
          rawDims.push_back(allSizes[i]);
      }
    } else {
      rawDims = getMixedSizesFromType(memref.getType());
    }
  }
  // loom.init_tensor — explicit mixed sizes.
  else if (auto initTensor = mlir::dyn_cast<loom::InitTensorOp>(op)) {
    rawDims = initTensor.getMixedSizes();
  }
  // loom.bufferize_to_tensor — explicit mixed sizes.
  else if (auto toTensor = mlir::dyn_cast<loom::BufferizeToTensorOp>(op)) {
    rawDims = toTensor.getMixedSizes();
  }
  // linalg DPS — result tensor shape == corresponding init.
  else if (auto linalgOp = mlir::dyn_cast<linalg::LinalgOp>(op)) {
    unsigned resultIdx = mlir::cast<OpResult>(v).getResultNumber();
    return traceShape(linalgOp.getDpsInits()[resultIdx]);
  }
  // affine.for / scf.for results map to inits.
  else if (auto forOp = mlir::dyn_cast<affine::AffineForOp>(op)) {
    unsigned resultIdx = mlir::cast<OpResult>(v).getResultNumber();
    return traceShape(forOp.getInits()[resultIdx]);
  } else if (auto forOp = mlir::dyn_cast<scf::ForOp>(op)) {
    unsigned resultIdx = mlir::cast<OpResult>(v).getResultNumber();
    return traceShape(forOp.getInits()[resultIdx]);
  }
  // tensor.extract_slice
  else if (auto extractSlice = mlir::dyn_cast<tensor::ExtractSliceOp>(op)) {
    rawDims = extractSlice.getMixedSizes();
  }
  // loom.gather — DPS, output rank = 1 + input rank, must follow init.
  else if (auto gatherOp = mlir::dyn_cast<loom::GatherOp>(op)) {
    return traceShape(gatherOp.getInit());
  }
  // loom.broadcast — physical destination is the init.
  else if (auto broadcastOp = mlir::dyn_cast<loom::BroadcastOp>(op)) {
    return traceShape(broadcastOp.getInit());
  }
  // loom.sync — DPS, follows init.
  else if (auto syncOp = mlir::dyn_cast<loom::SyncOp>(op)) {
    return traceShape(syncOp.getInit());
  } else {
    if (auto tensorType = mlir::dyn_cast<RankedTensorType>(v.getType());
        tensorType && tensorType.getRank() == 0)
      return {};
    op->emitError() << "traceShape has no shape rule for op '"
                    << op->getName() << "'";
    assert(false && "traceShape missing explicit shape rule");
    llvm::report_fatal_error("traceShape missing explicit shape rule");
  }

  SmallVector<SymbolicDim, 4> canonicalDims;
  for (auto dim : rawDims) {
    auto canonical = canonicalizeOFR(dim, v.getContext());
    if (auto attr = mlir::dyn_cast<Attribute>(canonical)) {
      if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
        if (intAttr.getInt() < 0) {
          op->emitError() << "traceShape produced unresolved dynamic dimension";
          assert(false && "traceShape produced unresolved dynamic dimension");
          llvm::report_fatal_error(
              "traceShape produced unresolved dynamic dimension");
        }
      }
    }
    canonicalDims.push_back(canonical);
  }
  return canonicalDims;
}

} // namespace utils
} // namespace loom
