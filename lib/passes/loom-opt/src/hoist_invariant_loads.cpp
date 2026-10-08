/**
 * @file hoist_invariant_loads.cpp
 * @brief Hoist DRAM loads out of the innermost loop of an independent nest.
 * @details
 * Runs on bufferized IR. A load group is, at the top level of a loop body:
 *
 *   %a = loom.alloc ... on @L1          (static shape, single take)
 *   %t = loom.semaphore_take %a
 *   loom.copy %src, %t src_mem_space @mem_DRAM ...
 *   ... read-only uses of %t ...
 *   loom.semaphore_give %t
 *
 * where %src and the copy's region operands are computed by pure ops. If such
 * a group does not depend on the loop's induction variable, the alloc, take,
 * copy and their address computation move in front of the loop and the give
 * after it, so the data is read once and reused by every iteration (the TT
 * lowering then reserves/pushes the CB once and pops it after the loop).
 *
 * The loops considered are perfect nests of `scf.for` marked
 * `loom.independent` (iterations independent, set by ExploreParallelSets).
 * Within such a nest the loop whose invariant loads save the most DRAM traffic
 * is first swapped innermost, which is legal because the iterations are
 * independent and the bounds do not depend on the nest.
 */

#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

// Loom dialect headers
#include "LoomDialect.h.inc"
#include "LoomEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "LoomAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#define DEBUG_TYPE "hoist-invariant-loads"

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kIndependentAttr = "loom.independent";

struct LoadGroup {
  loom::AllocOp alloc;
  loom::SemaphoreTakeOp take;
  loom::CopyOp copy;
  loom::SemaphoreGiveOp give;
  llvm::SetVector<Operation *> slice; // body ops computing the copy operands
  int64_t bytes = 0;
};

static bool isIndependentLoop(scf::ForOp loop) {
  return loop->hasAttr(kIndependentAttr) && loop.getInitArgs().empty() &&
         matchPattern(loop.getLowerBound(), m_Zero()) &&
         matchPattern(loop.getStep(), m_One());
}

/// The perfect nest starting at `top`: each loop body holds only the next loop.
static SmallVector<scf::ForOp> perfectNest(scf::ForOp top) {
  SmallVector<scf::ForOp> nest{top};
  while (true) {
    Block *body = nest.back().getBody();
    if (body->getOperations().size() != 2)
      break;
    auto inner = dyn_cast<scf::ForOp>(body->front());
    if (!inner || !isIndependentLoop(inner))
      break;
    nest.push_back(inner);
  }
  return nest;
}

static bool isNestTop(scf::ForOp loop) {
  auto parent = dyn_cast<scf::ForOp>(loop->getParentOp());
  return !(parent && isIndependentLoop(parent) &&
           parent.getBody()->getOperations().size() == 2);
}

static bool isDramSpace(std::optional<SymbolRefAttr> space) {
  return space && space->getRootReference().getValue() == "mem_DRAM";
}

/// Whether `user` only reads `buf`.
static bool isReadOnlyUse(Operation *user, Value buf) {
  if (auto copy = dyn_cast<loom::CopyOp>(user))
    return copy.getDestination() != buf;
  if (auto matmul = dyn_cast<loom::MatmulOp>(user))
    return matmul.getOuts() != buf;
  if (auto dps = dyn_cast<DestinationStyleOpInterface>(user))
    return llvm::none_of(dps.getDpsInits(), [&](Value v) { return v == buf; });
  return false;
}

/// Add to `slice` the pure ops of `body` that `v` is computed from. Fails if
/// `v` depends on an op of the body that is not pure or not at its top level.
static bool collectSlice(Value v, Block *body,
                         llvm::SetVector<Operation *> &slice) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return true; // block argument: induction variables are checked separately
  Operation *bodyOwner = body->getParentOp();
  if (!bodyOwner->isAncestor(def))
    return true; // defined outside the loop
  if (def->getBlock() != body || def->getNumRegions() != 0 ||
      !isMemoryEffectFree(def))
    return false;
  if (!slice.insert(def))
    return true;
  return llvm::all_of(def->getOperands(), [&](Value operand) {
    return collectSlice(operand, body, slice);
  });
}

static Value rootMemref(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto cast = dyn_cast<memref::ReinterpretCastOp>(def))
      v = cast.getSource();
    else if (auto view = dyn_cast<ViewLikeOpInterface>(def))
      v = view.getViewSource();
    else
      break;
  }
  return v;
}

/// DRAM memrefs written by a copy anywhere under `scope`.
static llvm::SmallPtrSet<Value, 8> writtenDramRoots(Operation *scope) {
  llvm::SmallPtrSet<Value, 8> roots;
  scope->walk([&](loom::CopyOp copy) {
    if (isDramSpace(copy.getDstMemSpace()))
      roots.insert(rootMemref(copy.getDestination()));
  });
  return roots;
}

/// Load groups at the top level of `loop`'s body.
static SmallVector<LoadGroup> collectLoadGroups(scf::ForOp loop,
                                                Operation *nestTop) {
  SmallVector<LoadGroup> groups;
  Block *body = loop.getBody();
  llvm::SmallPtrSet<Value, 8> written = writtenDramRoots(nestTop);
  for (Operation &op : *body) {
    auto copy = dyn_cast<loom::CopyOp>(op);
    if (!copy || !isDramSpace(copy.getSrcMemSpace()))
      continue;
    if (written.contains(rootMemref(copy.getSource())))
      continue; // the nest also writes this DRAM buffer
    auto take = copy.getDestination().getDefiningOp<loom::SemaphoreTakeOp>();
    if (!take || take->getBlock() != body)
      continue;
    auto alloc = take.getSource().getDefiningOp<loom::AllocOp>();
    if (!alloc || alloc->getBlock() != body || !alloc->hasOneUse() ||
        !alloc.getSizes().empty())
      continue;
    auto type = dyn_cast<MemRefType>(take.getResult().getType());
    if (!type || !type.hasStaticShape())
      continue;

    LoadGroup group;
    group.alloc = alloc;
    group.take = take;
    group.copy = copy;
    bool ok = true;
    for (Operation *user : take.getResult().getUsers()) {
      if (user == copy.getOperation())
        continue;
      if (auto give = dyn_cast<loom::SemaphoreGiveOp>(user)) {
        if (group.give || give->getBlock() != body) {
          ok = false;
          break;
        }
        group.give = give;
        continue;
      }
      Operation *topLevel = body->findAncestorOpInBlock(*user);
      if (!isReadOnlyUse(user, take.getResult()) || !topLevel ||
          topLevel->isBeforeInBlock(copy)) {
        ok = false;
        break;
      }
    }
    if (!ok || !group.give || !group.give->isBeforeInBlock(body->getTerminator()))
      continue;
    for (Value operand : copy->getOperands()) {
      if (operand == take.getResult())
        continue;
      if (!collectSlice(operand, body, group.slice)) {
        ok = false;
        break;
      }
    }
    if (!ok)
      continue;
    group.bytes = type.getNumElements() *
                  ((type.getElementTypeBitWidth() + 7) / 8);
    groups.push_back(std::move(group));
  }
  return groups;
}

/// Whether the group's address computation uses `iv`.
static bool dependsOn(const LoadGroup &group, Value iv) {
  auto uses = [&](Operation *op) {
    return llvm::is_contained(op->getOperands(), iv);
  };
  return uses(group.copy) || llvm::any_of(group.slice, uses);
}

/// Swap the iteration spaces of two loops of a perfect nest (`inner` nested in
/// `outer`): bounds and induction-variable uses trade places.
static void swapLoops(scf::ForOp outer, scf::ForOp inner) {
  Value outerUb = outer.getUpperBound();
  Value innerUb = inner.getUpperBound();
  outer.setUpperBound(innerUb);
  inner.setUpperBound(outerUb);
  Value outerIv = outer.getInductionVar();
  Value innerIv = inner.getInductionVar();
  SmallVector<OpOperand *> outerUses, innerUses;
  for (OpOperand &use : outerIv.getUses())
    outerUses.push_back(&use);
  for (OpOperand &use : innerIv.getUses())
    innerUses.push_back(&use);
  for (OpOperand *use : outerUses)
    use->set(innerIv);
  for (OpOperand *use : innerUses)
    use->set(outerIv);
}

static void processNest(SmallVector<scf::ForOp> nest) {
  scf::ForOp top = nest.front();
  scf::ForOp innermost = nest.back();
  // Loop bounds must not depend on the nest for the loops to be reordered.
  for (scf::ForOp loop : nest)
    if (top->isAncestor(loop.getUpperBound().getParentBlock()->getParentOp()))
      return;

  SmallVector<LoadGroup> groups = collectLoadGroups(innermost, top);
  if (groups.empty())
    return;

  // Pick the loop whose invariant loads save the most traffic.
  int best = -1;
  int64_t bestSaving = 0;
  for (auto [i, loop] : llvm::enumerate(nest)) {
    std::optional<int64_t> trip = getConstantIntValue(loop.getUpperBound());
    if (!trip || *trip <= 1)
      continue;
    int64_t saving = 0;
    for (const LoadGroup &group : groups)
      if (!dependsOn(group, loop.getInductionVar()))
        saving += group.bytes * (*trip - 1);
    // Prefer the innermost loop on ties (no reordering needed).
    if (saving > bestSaving ||
        (saving == bestSaving && saving > 0 && loop == innermost)) {
      best = i;
      bestSaving = saving;
    }
  }
  if (best < 0)
    return;
  if (nest[best] != innermost) {
    LLVM_DEBUG(llvm::dbgs() << "swap loop " << best << " innermost\n");
    swapLoops(nest[best], innermost);
  }

  // The innermost loop now iterates the chosen dimension.
  Value iv = innermost.getInductionVar();
  for (LoadGroup &group : groups) {
    if (dependsOn(group, iv))
      continue;
    llvm::SmallPtrSet<Operation *, 16> toMove(group.slice.begin(),
                                              group.slice.end());
    toMove.insert(group.alloc);
    toMove.insert(group.take);
    toMove.insert(group.copy);
    SmallVector<Operation *> ordered;
    for (Operation &op : *innermost.getBody())
      if (toMove.contains(&op))
        ordered.push_back(&op);
    for (Operation *op : ordered)
      op->moveBefore(innermost);
    group.give->moveAfter(innermost);
    LLVM_DEBUG(llvm::dbgs() << "hoisted load of " << group.bytes
                            << " bytes\n");
  }
}

class HoistInvariantLoadsPass
    : public PassWrapper<HoistInvariantLoadsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistInvariantLoadsPass)

  StringRef getArgument() const override { return "loom-hoist-invariant-loads"; }
  StringRef getDescription() const override {
    return "Hoist loop-invariant DRAM loads out of independent loop nests";
  }

  void runOnOperation() override {
    SmallVector<SmallVector<scf::ForOp>> nests;
    getOperation().walk([&](scf::ForOp loop) {
      if (isIndependentLoop(loop) && isNestTop(loop))
        nests.push_back(perfectNest(loop));
    });
    for (SmallVector<scf::ForOp> &nest : nests)
      processNest(nest);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createHoistInvariantLoadsPass() {
  return std::make_unique<HoistInvariantLoadsPass>();
}
