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
// Utilities
// ============================================================================

llvm::StringRef loom::toString(VBType type) {
  switch (type) {
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

// ============================================================================
// VirtualBuffer
// ============================================================================

void VirtualBuffer::addMember(TensorNode *node) {
  members.push_back(node);
  node->mappedVBId = id;
}

void VirtualBuffer::updateDefiningOp() {
  if (members.empty())
    return;
  TensorNode *earliest = members[0];
  for (auto *member : members) {
    if (member->linearIndex < earliest->linearIndex)
      earliest = member;
  }
  definingOp = earliest->definingOp;
}

VirtualBuffer *Bucket::createVB(int id, VBType type) {
  virtualBuffers.push_back(std::make_unique<VirtualBuffer>(id, type));
  return virtualBuffers.back().get();
}

bool Bucket::containsNode(const TensorNode *node) const {
  for (const auto &n : nodes) {
    if (&n == node)
      return true;
  }
  return false;
}

TensorNode *Bucket::findNode(Value v) {
  for (auto &node : nodes) {
    if (node.value == v)
      return &node;
  }
  return nullptr;
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

void ShapeSignature::print(llvm::raw_ostream &os) const {
  os << "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    auto dim = dims[i];
    if (auto val = mlir::dyn_cast<Value>(dim)) {
      os << val;
    } else if (auto attr = mlir::dyn_cast<Attribute>(dim)) {
      if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
        os << intAttr.getInt();
      } else {
        os << "?";
      }
    }
    if (i < dims.size() - 1)
      os << ", ";
  }
  os << "] (" << elementType << ")";
}

// ============================================================================
// MemoryAnalysisContext — Core logic
// ============================================================================

void MemoryAnalysisContext::setOpIndex(Operation *op, int idx) {
  opIndexMap_[op] = idx;
}

void MemoryAnalysisContext::addTensor(Value v, ShapeSignature sig,
                                      Operation *defOp, int idx) {
  setOpIndex(defOp, idx);
  if (!v)
    return;

  Bucket &bucket = buckets_[sig];
  bucket.signature = sig;

  bucket.nodes.emplace_back(v, defOp, idx);
  valueToNodeMap_[v] = &bucket.nodes.back();
}

int MemoryAnalysisContext::getOpIndex(Operation *op) const {
  auto it = opIndexMap_.find(op);
  return it != opIndexMap_.end() ? it->second : -1;
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
    auto it = valueToNodeMap_.find(v);
    if (it != valueToNodeMap_.end()) {
      return it->second->linearIndex;
    }
  }
  return maxIdx;
}

void MemoryAnalysisContext::computeDeathIndices() {
  for (auto &it : buckets_) {
    Bucket &bucket = it.second;
    for (auto &node : bucket.nodes) {
      node.deathIndex = getValueDeathIndex(node.value);
    }
  }
}

std::optional<LoopContext> MemoryAnalysisContext::findLoopContext() const {
  for (auto &entry : opIndexMap_) {
    if (auto parallelOp =
            mlir::dyn_cast<affine::AffineParallelOp>(entry.first)) {
      for (auto &op : *parallelOp.getBody()) {
        if (auto forOp = mlir::dyn_cast<affine::AffineForOp>(op)) {
          return LoopContext{forOp, getOpIndex(forOp), getLoopEndIndex(forOp)};
        }
      }
    }
  }
  return std::nullopt;
}

// ============================================================================
// VirtualBuffer Construction — Three Axioms
// ============================================================================

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

void MemoryAnalysisContext::applyPhiFusionAxiom(
    Bucket &bucket, const LoopContext &loopContext) {
  affine::AffineForOp forOp = loopContext.forOp;
  unsigned numIterArgs = forOp.getNumIterOperands();
  affine::AffineYieldOp yieldOp =
      mlir::cast<affine::AffineYieldOp>(forOp.getBody()->getTerminator());

  for (unsigned i = 0; i < numIterArgs; ++i) {
    Value argVal = forOp.getRegionIterArgs()[i];
    TensorNode *argNode = bucket.findNode(argVal);

    if (!argNode)
      continue;

    Value initVal = forOp.getInits()[i];
    Value yieldVal = yieldOp.getOperand(i);
    Value resultVal = forOp.getResults()[i];

    TensorNode *initNode = bucket.findNode(initVal);
    TensorNode *yieldNode = bucket.findNode(yieldVal);
    TensorNode *resultNode = bucket.findNode(resultVal);

    bool pure = initNode && isInitPure(initVal, forOp);

    if (pure) {
      VirtualBuffer *vb = bucket.createVB(nextVBId_++, VBType::Fused);
      vb->addMember(initNode);
      vb->addMember(argNode);
      if (yieldNode)
        vb->addMember(yieldNode);
      if (resultNode)
        vb->addMember(resultNode);

      int deathIdx = resultNode ? getValueDeathIndex(resultNode->value) : -1;
      vb->liveness = {initNode->linearIndex,
                      std::max(loopContext.endIndex, deathIdx)};
      vb->updateDefiningOp();
    } else {
      VirtualBuffer *vb = bucket.createVB(nextVBId_++, VBType::LoopCarried);
      vb->addMember(argNode);
      if (yieldNode)
        vb->addMember(yieldNode);
      if (resultNode)
        vb->addMember(resultNode);

      int deathIdx = resultNode ? getValueDeathIndex(resultNode->value) : -1;
      vb->liveness = {loopContext.startIndex,
                      std::max(loopContext.endIndex, deathIdx)};
      vb->updateDefiningOp();
    }
  }
}

void MemoryAnalysisContext::applyExternalEternityAxiom(
    Bucket &bucket, const LoopContext &loopContext) {
  affine::AffineForOp forOp = loopContext.forOp;
  for (auto &node : bucket.nodes) {
    if (node.mappedVBId.has_value())
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
    vb->liveness = {node.linearIndex, loopContext.endIndex};
    vb->updateDefiningOp();
  }
}

void MemoryAnalysisContext::applyStandardAxiom(Bucket &bucket) {
  for (auto &node : bucket.nodes) {
    if (node.mappedVBId.has_value())
      continue;

    VirtualBuffer *vb = bucket.createVB(nextVBId_++, VBType::Standard);
    vb->addMember(&node);
    vb->liveness = {node.linearIndex, node.deathIndex};
    vb->updateDefiningOp();
  }
}

void MemoryAnalysisContext::buildVirtualBuffers() {
  auto loopOpt = findLoopContext();
  if (!loopOpt)
    return;

  LoopContext loop = loopOpt.value();

  for (auto &bucketEntry : buckets_) {
    Bucket &bucket = bucketEntry.second;
    applyPhiFusionAxiom(bucket, loop);
    applyExternalEternityAxiom(bucket, loop);
    applyStandardAxiom(bucket);
  }
}

// ============================================================================
// Interference Graph
// ============================================================================

InterferenceGraph::InterferenceGraph(const Bucket &bucket,
                                     const MemoryAnalysisContext &ctx)
    : bucket_(bucket), ctx_(ctx) {}

int InterferenceGraph::classifyOverlap(const VirtualBuffer &vbA,
                                       const VirtualBuffer &vbB) const {
  const VirtualBuffer *a = &vbA;
  const VirtualBuffer *b = &vbB;
  if (b->liveness.birth < a->liveness.birth)
    std::swap(a, b);

  int startB = b->liveness.birth;
  int endA = a->liveness.death;

  if (startB < endA)
    return 1; // Strict Overlap
  if (endA < startB)
    return -1; // Disjoint
  return 0;    // Touching
}

mlir::OpOperand *
InterferenceGraph::findOperandRelation(const VirtualBuffer &vbA,
                                       const VirtualBuffer &vbB) const {
  Operation *opB = vbB.definingOp;
  if (!opB)
    return nullptr;

  for (TensorNode *memberA : vbA.members) {
    for (OpOperand &operand : opB->getOpOperands()) {
      if (operand.get() == memberA->value)
        return &operand;
    }
  }
  return nullptr;
}

static void collectDimPositions(AffineExpr expr,
                                llvm::DenseSet<unsigned> &positions) {
  if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr)) {
    positions.insert(dimExpr.getPosition());
    return;
  }
  if (auto binExpr = mlir::dyn_cast<AffineBinaryOpExpr>(expr)) {
    collectDimPositions(binExpr.getLHS(), positions);
    collectDimPositions(binExpr.getRHS(), positions);
  }
}

bool InterferenceGraph::checkHandoffInterference(
    const VirtualBuffer &vbA, const VirtualBuffer &vbB) const {
  OpOperand *operandA = findOperandRelation(vbA, vbB);
  if (!operandA)
    return false;

  auto linalgOp = mlir::dyn_cast<linalg::LinalgOp>(vbB.definingOp);
  if (!linalgOp)
    return true; // Conservative for non-linalg

  if (linalgOp.isDpsInit(operandA))
    return false; // Destination passing style, safe

  AffineMap mapA = linalgOp.getMatchingIndexingMap(operandA);
  OpOperand *initOperand = linalgOp.getDpsInitOperand(0);
  AffineMap mapB = linalgOp.getMatchingIndexingMap(initOperand);

  if (mapA != mapB)
    return true;

  auto iteratorTypes = linalgOp.getIteratorTypesArray();
  llvm::DenseSet<unsigned> positions;
  for (AffineExpr expr : mapA.getResults()) {
    collectDimPositions(expr, positions);
  }

  for (unsigned pos : positions) {
    if (pos < iteratorTypes.size() &&
        iteratorTypes[pos] == mlir::utils::IteratorType::reduction)
      return true;
  }

  return false;
}

void InterferenceGraph::build() {
  int n = bucket_.virtualBuffers.size();
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      VirtualBuffer &vbI = *bucket_.virtualBuffers[i];
      VirtualBuffer &vbJ = *bucket_.virtualBuffers[j];

      const VirtualBuffer *a = &vbI;
      const VirtualBuffer *b = &vbJ;
      if (b->liveness.birth < a->liveness.birth)
        std::swap(a, b);

      int overlap = classifyOverlap(*a, *b);
      bool interfere = false;
      if (overlap == 1) {
        interfere = true;
      } else if (overlap == 0) {
        interfere = checkHandoffInterference(*a, *b);
      }

      if (interfere) {
        edges_.insert({std::min(vbI.id, vbJ.id), std::max(vbI.id, vbJ.id)});
      }
    }
  }
}

void InterferenceGraph::dump(llvm::raw_ostream &os) const {
  if (edges_.empty())
    return;
  os << "    --- Interference Graph (" << bucket_.virtualBuffers.size()
     << " VBs, " << edges_.size() << " edges) ---\n";
  for (auto &edge : edges_) {
    os << "      VB#" << edge.first << " -- VB#" << edge.second << "\n";
  }
}

bool InterferenceGraph::interferes(int vbIdA, int vbIdB) const {
  return edges_.count({std::min(vbIdA, vbIdB), std::max(vbIdA, vbIdB)});
}

void MemoryAnalysisContext::buildInterferenceGraphs() {
  for (auto &it : buckets_) {
    Bucket &bucket = it.second;
    if (bucket.virtualBuffers.size() < 2)
      continue;
    bucket.interferenceGraph =
        std::make_unique<InterferenceGraph>(bucket, *this);
    bucket.interferenceGraph->build();
  }
}

// ============================================================================
// Dump — Visualizer
// ============================================================================

void MemoryAnalysisContext::dump(llvm::raw_ostream &os) const {
  os << "=== Memory Analysis Context Dump ===\n";
  int bucketIdx = 0;
  OpPrintingFlags flags;
  for (auto &it : buckets_) {
    const ShapeSignature &sig = it.first;
    const Bucket &bucket = it.second;
    os << "Bucket " << bucketIdx++ << ": ";
    sig.print(os);
    os << ", Tensors: " << bucket.nodes.size() << "\n";
    for (const auto &node : bucket.nodes) {
      os << "  - [Liveness: " << node.linearIndex << " -> " << node.deathIndex
         << "] Value: ";
      node.value.printAsOperand(os, flags);
      os << "\n";
    }

    if (!bucket.virtualBuffers.empty()) {
      os << "  VirtualBuffers: " << bucket.virtualBuffers.size() << "\n";
      for (auto &vb : bucket.virtualBuffers) {
        os << "    VB#" << vb->id << " (" << toString(vb->type) << ") ["
           << vb->liveness.birth << " -> " << vb->liveness.death << "]: ";
        for (size_t i = 0; i < vb->members.size(); ++i) {
          vb->members[i]->value.printAsOperand(os, flags);
          if (i < vb->members.size() - 1)
            os << ", ";
        }
        os << "\n";
      }
    }
    if (bucket.interferenceGraph) {
      bucket.interferenceGraph->dump(os);
    }
  }
  os << "====================================\n";
}

// ============================================================================
// Main Analysis Entry
// ============================================================================

MemoryAnalysisContext loom::runMemoryAnalysis(func::FuncOp func) {
  MemoryAnalysisContext ctx;
  int linearIdx = 0;

  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    ctx.setOpIndex(op, linearIdx++);

    for (Value res : op->getResults()) {
      if (auto tensorType = mlir::dyn_cast<RankedTensorType>(res.getType())) {
        if (mlir::isa<tensor::EmptyOp>(op))
          continue;

        auto sig = utils::traceShape(res);
        if (sig.empty())
          continue;

        ShapeSignature signature{sig, tensorType.getElementType()};
        ctx.addTensor(res, signature, op, ctx.getOpIndex(op));
      }
    }

    if (auto forOp = mlir::dyn_cast<affine::AffineForOp>(op)) {
      for (auto arg : forOp.getRegionIterArgs()) {
        if (auto tensorType = mlir::dyn_cast<RankedTensorType>(arg.getType())) {
          auto sig = utils::traceShape(arg);
          if (sig.empty())
            continue;

          ShapeSignature signature{sig, tensorType.getElementType()};
          ctx.addTensor(arg, signature, op, ctx.getOpIndex(op));
        }
      }
    }
  });

  ctx.computeDeathIndices();

  // Axioms phase
  ctx.buildVirtualBuffers();
  ctx.buildInterferenceGraphs();

  return ctx;
}
