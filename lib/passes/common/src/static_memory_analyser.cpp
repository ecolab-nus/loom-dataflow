#include "static_memory_analyser.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/raw_ostream.h"

// Note: Reusing generated LoomOps if needed
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;
using namespace loom;

unsigned ShapeSignature::getHashValue() const {
  size_t hash = llvm::hash_value(elementType.getAsOpaquePointer());
  for (auto dim : dims) {
    if (auto val = mlir::dyn_cast<Value>(dim)) {
      hash = llvm::hash_combine(hash, val.getAsOpaquePointer());
    } else if (auto attr = mlir::dyn_cast<Attribute>(dim)) {
      if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
        hash = llvm::hash_combine(hash, llvm::hash_value(intAttr.getInt()));
      }
    }
  }
  return static_cast<unsigned>(hash);
}

void MemoryAnalysisContext::addTensor(Value v, ShapeSignature sig,
                                      Operation *defOp, int idx) {
  Bucket &bucket = buckets[sig];
  bucket.signature = sig;

  bucket.nodes.emplace_back(v, defOp, idx);
  valueToNodeMap[v] = &bucket.nodes.back();
}

int MemoryAnalysisContext::getOpIndex(Operation *op) const {
  auto it = opIndexMap.find(op);
  return it != opIndexMap.end() ? it->second : -1;
}

int MemoryAnalysisContext::getLoopEndIndex(affine::AffineForOp forOp) const {
  Operation *yield = forOp.getBody()->getTerminator();
  return getOpIndex(yield);
}

int MemoryAnalysisContext::getValueDeathIndex(Value v) const {
  int maxIdx = -1;
  for (auto &use : v.getUses()) {
    int idx = getOpIndex(use.getOwner());
    if (idx > maxIdx)
      maxIdx = idx;
  }
  // If no users, death index is definition index
  if (maxIdx == -1) {
    auto it = valueToNodeMap.find(v);
    if (it != valueToNodeMap.end()) {
      return it->second->linearIndex;
    }
  }
  return maxIdx;
}

void MemoryAnalysisContext::dump(llvm::raw_ostream &os) const {
  os << "=== Memory Analysis Context Dump ===\n";
  int bucketIdx = 0;
  OpPrintingFlags flags;
  for (auto &it : buckets) {
    const ShapeSignature &sig = it.first;
    const Bucket &bucket = it.second;
    os << "Bucket " << bucketIdx++ << ": [";
    for (size_t i = 0; i < sig.dims.size(); ++i) {
      auto dim = sig.dims[i];
      if (auto val = mlir::dyn_cast<Value>(dim)) {
        os << val;
      } else if (auto attr = mlir::dyn_cast<Attribute>(dim)) {
        if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
          os << intAttr.getInt();
        } else {
          os << "?";
        }
      }
      if (i < sig.dims.size() - 1)
        os << ", ";
    }
    os << "] (" << sig.elementType << "), Tensors: " << bucket.nodes.size()
       << "\n";
    for (const auto &node : bucket.nodes) {
      os << "  - [Liveness: " << node.linearIndex << " -> " << node.deathIndex
         << "] Value: ";
      node.value.printAsOperand(os, flags);
      os << "\n";
    }
  }
  os << "====================================\n";
}

// Helper to canonicalize OpFoldResult to ensure Attribute types are consistent
// (all index)
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

SmallVector<SymbolicDim, 4> loom::traceShape(Value v) {
  if (!v)
    return {};

  // Case 1: BlockArgument (affine.for iter_args)
  if (auto arg = mlir::dyn_cast<BlockArgument>(v)) {
    if (auto forOp = mlir::dyn_cast<affine::AffineForOp>(
            arg.getOwner()->getParentOp())) {
      unsigned argIdx = arg.getArgNumber() - 1;
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
  // Case D: affine.for (Results)
  else if (auto forOp = mlir::dyn_cast<affine::AffineForOp>(op)) {
    unsigned resultIdx = mlir::cast<OpResult>(v).getResultNumber();
    return traceShape(forOp.getInits()[resultIdx]);
  }
  // Case E: tensor.extract_slice
  else if (auto extractSlice = mlir::dyn_cast<tensor::ExtractSliceOp>(op)) {
    rawDims = extractSlice.getMixedSizes();
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

MemoryAnalysisContext loom::runMemoryAnalysis(func::FuncOp func) {
  MemoryAnalysisContext ctx;
  int linearIdx = 0;

  // 1. First Pass: Assign indices to all ops and record tensors
  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    ctx.opIndexMap[op] = linearIdx++;

    // Add op results
    for (Value res : op->getResults()) {
      if (auto tensorType = mlir::dyn_cast<RankedTensorType>(res.getType())) {
        if (mlir::isa<tensor::EmptyOp>(op))
          continue;

        auto sig = traceShape(res);
        if (sig.empty())
          continue;

        ShapeSignature signature{sig, tensorType.getElementType()};
        ctx.addTensor(res, signature, op, ctx.opIndexMap[op]);
      }
    }

    // Add BlockArguments for affine.for
    if (auto forOp = mlir::dyn_cast<affine::AffineForOp>(op)) {
      for (auto arg : forOp.getRegionIterArgs()) {
        if (auto tensorType = mlir::dyn_cast<RankedTensorType>(arg.getType())) {
          auto sig = traceShape(arg);
          if (sig.empty())
            continue;

          ShapeSignature signature{sig, tensorType.getElementType()};
          ctx.addTensor(arg, signature, op, ctx.opIndexMap[op]);
        }
      }
    }
  });

  // 2. Second Pass: Calculate death indices for all recorded tensors
  for (auto &it : ctx.buckets) {
    Bucket &bucket = it.second;
    for (auto &node : bucket.nodes) {
      node.deathIndex = ctx.getValueDeathIndex(node.value);
    }
  }

  return ctx;
}
