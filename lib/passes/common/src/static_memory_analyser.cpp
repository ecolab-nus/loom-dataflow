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

  auto setVBDefiningOp = [](VirtualBuffer *vb) {
    if (vb->members.empty())
      return;
    TensorNode *earliest = vb->members[0];
    for (auto *member : vb->members) {
      if (member->linearIndex < earliest->linearIndex)
        earliest = member;
    }
    vb->definingOp = earliest->definingOp;
  };

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
        setVBDefiningOp(vb);
      } else {
        VirtualBuffer *vb = bucket.createVB(nextVBId_++, VBType::LoopCarried);
        vb->addMember(argNode);
        if (yieldNode)
          vb->addMember(yieldNode);
        if (resultNode)
          vb->addMember(resultNode);

        int deathIdx = resultNode ? getValueDeathIndex(resultNode->value) : -1;
        vb->liveness = {loopStart, std::max(loopEnd, deathIdx)};
        setVBDefiningOp(vb);
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
      setVBDefiningOp(vb);
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
      setVBDefiningOp(vb);
    }
  }
}

// ============================================================================
// Interference Graph
// ============================================================================

InterferenceGraph::InterferenceGraph(Bucket &bucket,
                                     const MemoryAnalysisContext &ctx)
    : bucket_(bucket), ctx_(ctx) {}

int InterferenceGraph::classifyOverlap(const VirtualBuffer &vbA,
                                       const VirtualBuffer &vbB) const {
  // Precondition: vbB.liveness.first >= vbA.liveness.first
  // We handle the ordering outside or here. Let's handle it here for safety.
  const VirtualBuffer *a = &vbA;
  const VirtualBuffer *b = &vbB;
  if (b->liveness.first < a->liveness.first)
    std::swap(a, b);

  int startB = b->liveness.first;
  int endA = a->liveness.second;

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
  // Dimension 0: Role lookup
  OpOperand *operandA = findOperandRelation(vbA, vbB);
  if (!operandA)
    return false;

  auto linalgOp = mlir::dyn_cast<linalg::LinalgOp>(vbB.definingOp);
  if (!linalgOp)
    return true; // Conservative for non-linalg

  // Dimension 1: Role Check (outs? )
  if (linalgOp.isDpsInit(operandA))
    return false; // Destination passing style, safe

  // Dimension 2: Indexing Map Alignment
  AffineMap mapA = linalgOp.getMatchingIndexingMap(operandA);
  // Assume single result for now as per Loom patterns
  OpOperand *initOperand = linalgOp.getDpsInitOperand(0);
  AffineMap mapB = linalgOp.getMatchingIndexingMap(initOperand);

  if (mapA != mapB)
    return true;

  // Dimension 3: Iterator Safety
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
      if (b->liveness.first < a->liveness.first)
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
  for (auto &it : buckets) {
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
    ctx.opIndexMap[op] = linearIdx++;

    for (Value res : op->getResults()) {
      if (auto tensorType = mlir::dyn_cast<RankedTensorType>(res.getType())) {
        if (mlir::isa<tensor::EmptyOp>(op))
          continue;

        auto sig = utils::traceShape(res);
        if (sig.empty())
          continue;

        ShapeSignature signature{sig, tensorType.getElementType()};
        ctx.addTensor(res, signature, op, ctx.opIndexMap[op]);
      }
    }

    if (auto forOp = mlir::dyn_cast<affine::AffineForOp>(op)) {
      for (auto arg : forOp.getRegionIterArgs()) {
        if (auto tensorType = mlir::dyn_cast<RankedTensorType>(arg.getType())) {
          auto sig = utils::traceShape(arg);
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
  ctx.buildInterferenceGraphs();

  return ctx;
}
