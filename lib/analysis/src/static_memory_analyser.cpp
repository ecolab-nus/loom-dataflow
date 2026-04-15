#include "static_memory_analyser.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
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
    if (auto val = dim.dyn_cast<Value>()) {
      hash = llvm::hash_combine(hash, val.getAsOpaquePointer());
    } else if (auto attr = dim.dyn_cast<Attribute>()) {
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
    if (auto val = dim.dyn_cast<Value>()) {
      os << val;
    } else if (auto attr = dim.dyn_cast<Attribute>()) {
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
  if (idx >= (int)indexToOpMap_.size())
    indexToOpMap_.resize(idx + 1, nullptr);
  indexToOpMap_[idx] = op;
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

Operation *MemoryAnalysisContext::getOpFromIndex(int index) const {
  if (index < 0 || index >= (int)indexToOpMap_.size())
    return nullptr;
  return indexToOpMap_[index];
}

int MemoryAnalysisContext::getLoopEndIndex(Operation *loopOp) const {
  Operation *yield = loopOp->getRegion(0).front().getTerminator();
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
  for (auto &[sig, bucket] : buckets_) {
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
          return LoopContext{forOp.getOperation(), getOpIndex(forOp),
                             getLoopEndIndex(forOp)};
        }
        if (auto forOp = mlir::dyn_cast<scf::ForOp>(op)) {
          return LoopContext{forOp.getOperation(), getOpIndex(forOp),
                             getLoopEndIndex(forOp)};
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
static bool isInitPure(Value initVal, Operation *loopOp) {
  for (auto &use : initVal.getUses()) {
    Operation *user = use.getOwner();
    if (user == loopOp)
      continue;
    if (loopOp->isProperAncestor(user))
      return false;
  }
  return true;
}

void MemoryAnalysisContext::applyPhiFusionAxiom(
    Bucket &bucket, const LoopContext &loopContext) {
  Operation *loopOp = loopContext.loopOp;

  // Extract loop properties generically for both affine.for and scf.for.
  unsigned numIterArgs;
  ValueRange regionIterArgs;
  ValueRange inits;
  ResultRange results = loopOp->getResults();
  Operation *yieldOp;

  if (auto affFor = mlir::dyn_cast<affine::AffineForOp>(loopOp)) {
    numIterArgs = affFor.getNumIterOperands();
    regionIterArgs = affFor.getRegionIterArgs();
    inits = affFor.getInits();
    yieldOp = affFor.getBody()->getTerminator();
  } else if (auto scfFor = mlir::dyn_cast<scf::ForOp>(loopOp)) {
    numIterArgs = scfFor.getInitArgs().size();
    regionIterArgs = scfFor.getRegionIterArgs();
    inits = scfFor.getInits();
    yieldOp = scfFor.getBody()->getTerminator();
  } else {
    return;
  }

  for (unsigned i = 0; i < numIterArgs; ++i) {
    Value argVal = regionIterArgs[i];
    TensorNode *argNode = bucket.findNode(argVal);
    if (!argNode)
      continue;

    Value initVal = inits[i];
    Value yieldVal = yieldOp->getOperand(i);
    Value resultVal = results[i];

    TensorNode *initNode = bucket.findNode(initVal);
    TensorNode *yieldNode = bucket.findNode(yieldVal);
    TensorNode *resultNode = bucket.findNode(resultVal);

    bool pure = initNode && isInitPure(initVal, loopOp);
    VirtualBuffer *vb = bucket.createVB(
        nextVBId_++, pure ? VBType::Fused : VBType::LoopCarried);

    // Phase 1: Core members (init and iterarg)
    vb->addMember(argNode);
    if (pure)
      vb->addMember(initNode);

    // Phase 2: Conditionally merge yield (handoff logic)
    bool mergedYield = false;
    if (yieldNode && yieldNode != argNode && yieldNode != initNode) {
      int argDeath = getValueDeathIndex(argVal);
      // Handoff check: yield birth >= arg death
      if (yieldNode->linearIndex >= argDeath) {
        vb->addMember(yieldNode);
        mergedYield = true;
      }
    }

    // Phase 3: Unconditionally add return
    if (resultNode && resultNode != argNode && resultNode != initNode &&
        resultNode != yieldNode) {
      vb->addMember(resultNode);
    }

    // Phase 4: Compute final liveness
    int birth = argNode->linearIndex;
    if (pure)
      birth = std::min(birth, initNode->linearIndex);
    else
      birth = std::min(birth, loopContext.startIndex);

    int death = loopContext.endIndex;
    if (resultNode)
      death = std::max(death, resultNode->deathIndex);

    vb->liveness = {birth, death};
    vb->updateDefiningOp();

    // Record split info if NOT merged
    if (!mergedYield && yieldVal) {
      splitYields_.push_back({i, argVal, yieldVal});
    }
  }
}

void MemoryAnalysisContext::applyExternalEternityAxiom(
    Bucket &bucket, const LoopContext &loopContext) {
  Operation *loopOp = loopContext.loopOp;
  for (auto &node : bucket.nodes) {
    if (node.mappedVBId.has_value())
      continue;

    // DefiningOp is outside loop
    if (loopOp->isProperAncestor(node.definingOp))
      continue;

    // At least one user inside loop
    bool usedInLoop = false;
    for (auto &use : node.value.getUses()) {
      if (loopOp->isProperAncestor(use.getOwner())) {
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

  for (auto &[sig, bucket] : buckets_) {
    if (loopOpt) {
      applyPhiFusionAxiom(bucket, *loopOpt);
      applyExternalEternityAxiom(bucket, *loopOpt);
    }
    applyStandardAxiom(bucket);
  }
}

void MemoryAnalysisContext::fuseRelayVirtualBuffers() {
  for (auto &[sig, bucket] : buckets_) {
    if (bucket.virtualBuffers.size() < 2)
      continue;

    InterferenceGraph graph(bucket, *this);

    bool changed = true;
    while (changed) {
      changed = false;
      int n = bucket.virtualBuffers.size();
      int mergeA = -1, mergeB = -1;

      for (int i = 0; i < n && mergeA == -1; ++i) {
        if (bucket.virtualBuffers[i]->type != VBType::Standard)
          continue;
        for (int j = i + 1; j < n; ++j) {
          if (bucket.virtualBuffers[j]->type != VBType::Standard)
            continue;

          if (graph.canRelay(*bucket.virtualBuffers[i],
                             *bucket.virtualBuffers[j])) {
            mergeA = i;
            mergeB = j;
            break;
          }
        }
      }

      if (mergeA != -1) {
        // Merge B into A
        auto &vbA = bucket.virtualBuffers[mergeA];
        auto &vbB = bucket.virtualBuffers[mergeB];

        for (TensorNode *member : vbB->members) {
          vbA->addMember(member);
        }
        vbA->liveness.birth =
            std::min(vbA->liveness.birth, vbB->liveness.birth);
        vbA->liveness.death =
            std::max(vbA->liveness.death, vbB->liveness.death);
        vbA->updateDefiningOp();

        bucket.virtualBuffers.erase(bucket.virtualBuffers.begin() + mergeB);
        changed = true;
      }
    }
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

  // Check DPS semantics: if the operand is a DPS init, it aliases the result
  // and is safe to share the buffer. Covers loom.gather, all linalg ops, and
  // any other op implementing DestinationStyleOpInterface.
  if (auto dpsOp = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(
          vbB.definingOp)) {
    if (dpsOp.isDpsInit(operandA))
      return false;
  }

  auto linalgOp = mlir::dyn_cast<linalg::LinalgOp>(vbB.definingOp);
  if (!linalgOp)
    return true; // Conservative for non-linalg/non-DPS ops

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

bool InterferenceGraph::canRelay(const VirtualBuffer &vbA,
                                 const VirtualBuffer &vbB) const {
  int overlap = classifyOverlap(vbA, vbB);
  if (overlap != 0)
    return false; // Not touching
  return !checkHandoffInterference(vbA, vbB);
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
  for (auto &[sig, bucket] : buckets_) {
    if (bucket.virtualBuffers.size() < 2)
      continue;
    bucket.interferenceGraph =
        std::make_unique<InterferenceGraph>(bucket, *this);
    bucket.interferenceGraph->build();
  }
}

void MemoryAnalysisContext::solveColoring() {
  for (auto &[sig, bucket] : buckets_) {
    if (bucket.virtualBuffers.empty())
      continue;
    ColoringSolver(bucket).solve();
  }
}

void MemoryAnalysisContext::buildAllocationPlan() {
  for (auto &[sig, bucket] : buckets_) {
    // Record color count
    allocationPlan_.colorCountPerBucket[sig] = bucket.maxColorsRequired;

    // Build physical buffer slots
    std::vector<LoomAllocationPlan::PhysicalBufferSlot> slots;
    slots.reserve(bucket.maxColorsRequired);
    for (int c = 0; c < bucket.maxColorsRequired; ++c) {
      slots.push_back({c, sig, sig.elementType});
    }
    allocationPlan_.bucketAllocations[sig] = std::move(slots);

    // Map each tensor value to its assignment
    for (const auto &vb : bucket.virtualBuffers) {
      LoomAllocationPlan::Assignment assignment{sig, vb->color};
      for (TensorNode *member : vb->members) {
        allocationPlan_.tensorToBufferMap[member->value] = assignment;
      }
    }
  }
}

// ============================================================================
// Coloring Solver — First-Fit Greedy
// ============================================================================

ColoringSolver::ColoringSolver(Bucket &bucket) : bucket_(bucket) {}

int ColoringSolver::solve() {
  // 1. Gather pointers to all VBs
  std::vector<VirtualBuffer *> sorted;
  sorted.reserve(bucket_.virtualBuffers.size());
  for (auto &vb : bucket_.virtualBuffers)
    sorted.push_back(vb.get());

  // 2. Sort: ascending by liveness.birth, then descending by duration
  // (liveness.end - liveness.start)
  std::sort(sorted.begin(), sorted.end(),
            [](const VirtualBuffer *a, const VirtualBuffer *b) {
              if (a->liveness.birth != b->liveness.birth)
                return a->liveness.birth < b->liveness.birth;
              int durA = a->liveness.death - a->liveness.birth;
              int durB = b->liveness.death - b->liveness.birth;
              return durA > durB; // longer duration first when birth is equal
            });

  // 3. Greedy coloring
  int maxColor = -1;
  for (VirtualBuffer *vb : sorted) {
    // Collect colors of interfering neighbors
    std::set<int> usedColors;
    for (VirtualBuffer *other : sorted) {
      if (other->color < 0)
        continue;
      if (bucket_.interferenceGraph &&
          bucket_.interferenceGraph->interferes(vb->id, other->id))
        usedColors.insert(other->color);
    }

    // First-fit: find the smallest non-negative integer not in usedColors
    int c = 0;
    while (usedColors.count(c))
      ++c;

    vb->color = c;
    if (c > maxColor)
      maxColor = c;
  }

  int numColors = maxColor + 1;
  bucket_.maxColorsRequired = numColors;
  return numColors;
}

// ============================================================================
// Dump — Visualizer
// ============================================================================

void MemoryAnalysisContext::dump(llvm::raw_ostream &os) const {
  os << "=== Memory Analysis Context Dump ===\n";
  int bucketIdx = 0;
  OpPrintingFlags flags;
  for (const auto &[sig, bucket] : buckets_) {
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
      for (const auto &vb : bucket.virtualBuffers) {
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
    if (bucket.maxColorsRequired > 0) {
      os << "    --- Coloring (" << bucket.maxColorsRequired
         << " colors) ---\n";
      for (const auto &vb : bucket.virtualBuffers) {
        os << "      VB#" << vb->id << " -> Color " << vb->color << "\n";
      }
    }
  }
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
        // Allow empty sig (rank-0 tensor) through — still a valid signature.
        if (sig.empty() && tensorType.getRank() != 0)
          continue;

        ShapeSignature signature{sig, tensorType.getElementType()};
        ctx.addTensor(res, signature, op, ctx.getOpIndex(op));
      }
    }

    // Collect iter_args from affine.for and scf.for loops.
    auto collectIterArgs = [&](auto forOp) {
      for (auto arg : forOp.getRegionIterArgs()) {
        if (auto tensorType = mlir::dyn_cast<RankedTensorType>(arg.getType())) {
          auto sig = utils::traceShape(arg);
          if (sig.empty() && tensorType.getRank() != 0)
            continue;

          ShapeSignature signature{sig, tensorType.getElementType()};
          ctx.addTensor(arg, signature, op, ctx.getOpIndex(op));
        }
      }
    };

    if (auto forOp = mlir::dyn_cast<affine::AffineForOp>(op))
      collectIterArgs(forOp);
    else if (auto forOp = mlir::dyn_cast<scf::ForOp>(op))
      collectIterArgs(forOp);
  });

  ctx.computeDeathIndices();

  // Axioms phase
  ctx.buildVirtualBuffers();
  ctx.fuseRelayVirtualBuffers();
  ctx.buildInterferenceGraphs();
  ctx.solveColoring();
  ctx.buildAllocationPlan();

  return ctx;
}
