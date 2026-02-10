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

// ============================================================================
// VirtualBuffer
// ============================================================================

void VirtualBuffer::addMember(TensorNode *node) {
  members.push_back(node);
  node->mappedVB = this;
}

VirtualBuffer *Bucket::createVB(int id, VBType type) {
  virtualBuffers.push_back(std::make_unique<VirtualBuffer>(id, type));
  return virtualBuffers.back().get();
}

// ============================================================================
// ShapeSignature
// ============================================================================

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

// ============================================================================
// MemoryAnalysisContext — Core logic
// ============================================================================

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

// ============================================================================
// VirtualBuffer Construction — Three Axioms
// ============================================================================

/// Find the single affine.for DIRECTLY nested in affine.parallel body.
static affine::AffineForOp findInnerFor(affine::AffineParallelOp parallelOp) {
  for (auto &op : *parallelOp.getBody()) {
    if (auto forOp = mlir::dyn_cast<affine::AffineForOp>(op)) {
      return forOp;
    }
  }
  llvm::errs()
      << "Error: No directly nested affine.for found in affine.parallel body\n";
  exit(1);
}

/// Check if Init's value has any use by an op physically inside the loop body.
/// Excludes the affine.for itself.
static bool isInitPure(Value initVal, affine::AffineForOp forOp) {
  for (auto &use : initVal.getUses()) {
    Operation *user = use.getOwner();
    if (user == forOp.getOperation())
      continue;
    if (forOp->isProperAncestor(user))
      return false;
  }
  return true;
}

void MemoryAnalysisContext::buildVirtualBuffers() {
  // Find the loop structure
  affine::AffineForOp forOp = nullptr;
  for (auto &entry : opIndexMap) {
    if (auto parallelOp =
            mlir::dyn_cast<affine::AffineParallelOp>(entry.first)) {
      forOp = findInnerFor(parallelOp);
      break;
    }
  }

  if (!forOp)
    return;

  int loopStart = getOpIndex(forOp);
  int loopEnd = getLoopEndIndex(forOp);

  for (auto &bucketEntry : buckets) {
    Bucket &bucket = bucketEntry.second;

    // =======================================================================
    // Axiom 1: Phi-Fusion
    // =======================================================================
    unsigned numIterArgs = forOp.getNumIterOperands();
    affine::AffineYieldOp yieldOp =
        mlir::cast<affine::AffineYieldOp>(forOp.getBody()->getTerminator());

    for (unsigned i = 0; i < numIterArgs; ++i) {
      Value argVal = forOp.getRegionIterArgs()[i];
      TensorNode *argNode = nullptr;
      {
        auto it = valueToNodeMap.find(argVal);
        if (it != valueToNodeMap.end())
          argNode = it->second;
      }

      // Check if argNode exists and belongs to THIS bucket
      bool argInThisBucket = false;
      if (argNode) {
        for (auto &n : bucket.nodes)
          if (&n == argNode) {
            argInThisBucket = true;
            break;
          }
      }
      if (!argInThisBucket)
        continue;

      Value initVal = forOp.getInits()[i];
      Value yieldVal = yieldOp.getOperand(i);
      Value resultVal = forOp.getResults()[i];

      TensorNode *initNode = nullptr;
      TensorNode *yieldNode = nullptr;
      TensorNode *resultNode = nullptr;

      {
        auto it = valueToNodeMap.find(initVal);
        if (it != valueToNodeMap.end())
          initNode = it->second;
      }
      {
        auto it = valueToNodeMap.find(yieldVal);
        if (it != valueToNodeMap.end())
          yieldNode = it->second;
      }
      {
        auto it = valueToNodeMap.find(resultVal);
        if (it != valueToNodeMap.end())
          resultNode = it->second;
      }

      // Purity Check
      bool pure = initNode && isInitPure(initVal, forOp);
      // We also check if Init belongs to the same bucket
      bool initInThisBucket = false;
      if (initNode) {
        for (auto &n : bucket.nodes)
          if (&n == initNode) {
            initInThisBucket = true;
            break;
          }
      }

      if (pure && initInThisBucket) {
        VirtualBuffer *vb = bucket.createVB(nextVBId_++, VBType::Fused);
        vb->addMember(initNode);
        vb->addMember(argNode);
        if (yieldNode)
          vb->addMember(yieldNode);
        if (resultNode)
          vb->addMember(resultNode);

        int deathIdx = resultNode ? getValueDeathIndex(resultNode->value) : -1;
        vb->liveness = {initNode->linearIndex, std::max(loopEnd, deathIdx)};
      } else {
        VirtualBuffer *vb = bucket.createVB(nextVBId_++, VBType::LoopCarried);
        vb->addMember(argNode);
        if (yieldNode)
          vb->addMember(yieldNode);
        if (resultNode)
          vb->addMember(resultNode);

        int deathIdx = resultNode ? getValueDeathIndex(resultNode->value) : -1;
        vb->liveness = {loopStart, std::max(loopEnd, deathIdx)};
      }
    }

    // =======================================================================
    // Axiom 2: External Eternity
    // =======================================================================
    for (auto &node : bucket.nodes) {
      if (node.mappedVB)
        continue;

      // DefiningOp is outside loop
      if (forOp->isProperAncestor(node.definingOp))
        continue;

      // At least one user inside loop
      bool usedInLoop = false;
      for (auto &use : node.value.getUses()) {
        if (forOp->isProperAncestor(use.getOwner())) {
          usedInLoop = true;
          break;
        }
      }
      if (!usedInLoop)
        continue;

      VirtualBuffer *vb = bucket.createVB(nextVBId_++, VBType::Eternal);
      vb->addMember(&node);
      vb->liveness = {node.linearIndex, loopEnd};
    }

    // =======================================================================
    // Axiom 3: Standard
    // =======================================================================
    for (auto &node : bucket.nodes) {
      if (node.mappedVB)
        continue;

      VirtualBuffer *vb = bucket.createVB(nextVBId_++, VBType::Standard);
      vb->addMember(&node);
      vb->liveness = {node.linearIndex, node.deathIndex};
    }
  }
}

// ============================================================================
// Dump — Visualizer
// ============================================================================

static const char *vbTypeToStr(VBType t) {
  switch (t) {
  case VBType::Standard:
    return "Standard";
  case VBType::Fused:
    return "Fused";
  case VBType::Eternal:
    return "Eternal";
  case VBType::LoopCarried:
    return "LoopCarried";
  }
  return "???";
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

    if (!bucket.virtualBuffers.empty()) {
      os << "  VirtualBuffers: " << bucket.virtualBuffers.size() << "\n";
      for (auto &vb : bucket.virtualBuffers) {
        os << "    VB#" << vb->id << " (" << vbTypeToStr(vb->type) << ") ["
           << vb->liveness.first << " -> " << vb->liveness.second << "]: ";
        for (size_t i = 0; i < vb->members.size(); ++i) {
          vb->members[i]->value.printAsOperand(os, flags);
          if (i < vb->members.size() - 1)
            os << ", ";
        }
        os << "\n";
      }
    }
  }
  os << "====================================\n";
}

// ============================================================================
// Trace Shape — Recursive Backward Tracing
// ============================================================================

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

// ============================================================================
// Main Analysis Entry
// ============================================================================

MemoryAnalysisContext loom::runMemoryAnalysis(func::FuncOp func) {
  MemoryAnalysisContext ctx;
  int linearIdx = 0;

  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    ctx.opIndexMap[op] = linearIdx++;

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

  for (auto &it : ctx.buckets) {
    Bucket &bucket = it.second;
    for (auto &node : bucket.nodes) {
      node.deathIndex = ctx.getValueDeathIndex(node.value);
    }
  }

  // Axioms phase
  ctx.buildVirtualBuffers();

  return ctx;
}
