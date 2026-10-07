/**
 * @file parallel_set_exploration.cpp
 * @brief Explore which loops of a kernel run spatially.
 * @details
 * The Helion frontend makes the outermost hl.tile an affine.parallel (the only
 * loop the spatial mapper distributes over cores) and every inner hl.tile a
 * sequential scf.for. That fixes the work decomposition to the source's loop
 * nesting. This exploration adds variants with a different spatial set:
 *
 * - Promote: a perfectly nested chain of scf.for loops directly inside the
 *   affine.parallel (only pure index ops between them, no iter_args, lb 0,
 *   step 1, bounds independent of the loop nest) can join the parallel loop
 *   when every L1->DRAM store in its body writes a location indexed by its IV
 *   (iterations write disjoint outputs) and no stored-to buffer is read back.
 * - Demote: an affine.parallel dimension can become a sequential scf.for
 *   (always legal), placed innermost so loops whose bounds depend on it (e.g.
 *   causal loops) stay inside it.
 *
 * Each spatial set becomes a function clone; the original function is kept
 * unchanged under its name. To keep the downstream mapping enumeration
 * tractable, a set has at most max(3, P) dimensions, contains only loops with
 * guaranteed work (minimum trip count > 1), and its guaranteed parallel work
 * (product of minimum trip counts) covers every core; its mappings are then
 * enumerated at full occupancy only.
 */

#include "hardware_info.h"
#include "affine_utils.h"
#include "loop_iv_dependency.h"
#include "utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "LoomDialect.h.inc"
#include "LoomEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "LoomAttributes.h.inc"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#define DEBUG_TYPE "parallel-set-exploration"

using namespace mlir;

namespace {

/// The loop structure of one function: the root affine.parallel, the
/// promotable scf.for chain directly inside it, and the index ops between
/// chain levels (hoisted above the root when a variant is built).
struct LoopNest {
  affine::AffineParallelOp root;
  SmallVector<scf::ForOp> chain;
  SmallVector<Operation *> hoistOps;
};

/// Side-effect-free index computations that may move above the root parallel
/// loop. Divisions are not speculatable in general (division by zero), but
/// the bounds here divide by block sizes, which are positive.
static bool isHoistableIndexOp(Operation *op) {
  if (!isa<arith::ArithDialect>(op->getDialect()))
    return false;
  if (!isMemoryEffectFree(op) || op->getNumRegions() != 0)
    return false;
  return llvm::all_of(op->getResultTypes(),
                      [](Type t) { return t.isIndex(); });
}

static bool isConstantIndex(Value v, int64_t expected) {
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
    return c.value() == expected;
  return false;
}

static bool definedOutside(Value v, Operation *root,
                           const llvm::SmallPtrSetImpl<Operation *> &hoisted) {
  if (auto arg = dyn_cast<BlockArgument>(v))
    return !root->isAncestor(arg.getOwner()->getParentOp());
  Operation *def = v.getDefiningOp();
  return !root->isAncestor(def) || hoisted.contains(def);
}

/// Every DRAM-backed func-argument memref written by a loom.copy under
/// `scope`, with the subview offsets of each write.
struct StoreInfo {
  Value buffer;
  SmallVector<Value> offsets;
};

static Value rootBuffer(Value v) {
  while (auto sv = v.getDefiningOp<loom::SubviewOp>())
    v = sv.getSource();
  return v;
}

static SmallVector<StoreInfo> collectStores(Operation *scope) {
  SmallVector<StoreInfo> stores;
  scope->walk([&](loom::CopyOp copy) {
    Value dst = copy.getDestination();
    auto sv = dst.getDefiningOp<loom::SubviewOp>();
    if (!sv)
      return;
    Value buf = rootBuffer(dst);
    if (!isa<BlockArgument>(buf))
      return;
    StoreInfo info;
    info.buffer = buf;
    for (Value off : sv.getOffsets())
      info.offsets.push_back(off);
    stores.push_back(std::move(info));
  });
  return stores;
}

/// Find the root parallel loop and its promotable loop chain.
static std::optional<LoopNest> analyzeLoopNest(func::FuncOp func) {
  LoopNest nest;
  func.walk([&](affine::AffineParallelOp par) {
    if (!nest.root && !par->getParentOfType<affine::AffineParallelOp>())
      nest.root = par;
  });
  if (!nest.root)
    return std::nullopt;
  // The root must sit directly in the function body so that values hoisted
  // in front of it are valid affine symbols.
  if (nest.root->getParentOp() != func.getOperation())
    return std::nullopt;
  for (unsigned d = 0; d < nest.root.getNumDims(); ++d) {
    if (nest.root.getSteps()[d] != 1 ||
        !nest.root.getLowerBoundMap(d).isSingleConstant() ||
        nest.root.getLowerBoundMap(d).getSingleConstantResult() != 0 ||
        nest.root.getUpperBoundMap(d).getNumResults() != 1)
      return std::nullopt;
  }
  // Cross-core reductions (loom.gather) tie the spatial structure to the
  // source; leave such kernels alone.
  bool hasGather = false;
  func.walk([&](loom::GatherOp) { hasGather = true; });
  if (hasGather)
    return std::nullopt;

  // Buffers that are both stored to and read inside the nest make promotion
  // unsafe (a later iteration could read an earlier one's output).
  SmallVector<StoreInfo> allStores = collectStores(nest.root);
  llvm::SmallPtrSet<Value, 4> storedBuffers;
  for (const StoreInfo &s : allStores)
    storedBuffers.insert(s.buffer);
  bool readsStored = false;
  nest.root->walk([&](loom::CopyOp copy) {
    if (storedBuffers.contains(rootBuffer(copy.getSource())))
      readsStored = true;
  });

  llvm::SmallPtrSet<Operation *, 8> hoisted;
  Block *cur = nest.root.getBody();
  while (!readsStored) {
    scf::ForOp inner;
    SmallVector<Operation *> pending;
    bool perfect = true;
    for (Operation &op : cur->without_terminator()) {
      if (auto f = dyn_cast<scf::ForOp>(op)) {
        if (inner) {
          perfect = false;
          break;
        }
        inner = f;
        continue;
      }
      if (!inner && isHoistableIndexOp(&op) &&
          llvm::all_of(op.getOperands(), [&](Value v) {
            return definedOutside(v, nest.root, hoisted);
          })) {
        pending.push_back(&op);
        hoisted.insert(&op);
        continue;
      }
      perfect = false;
      break;
    }
    if (!perfect || !inner)
      break;
    if (inner.getNumRegionIterArgs() != 0 ||
        !isConstantIndex(inner.getLowerBound(), 0) ||
        !isConstantIndex(inner.getStep(), 1) ||
        !definedOutside(inner.getUpperBound(), nest.root, hoisted))
      break;
    // Iterations must write disjoint outputs: every store is indexed by the
    // loop's IV.
    Value iv = inner.getInductionVar();
    bool disjoint = true;
    for (const StoreInfo &s : collectStores(inner)) {
      auto deps = loom::utils::collectLoopIVDependencies(s.offsets);
      if (!llvm::any_of(deps, [&](const auto &d) { return d.iv == iv; })) {
        disjoint = false;
        break;
      }
    }
    if (!disjoint)
      break;
    nest.hoistOps.append(pending.begin(), pending.end());
    nest.chain.push_back(inner);
    cur = inner.getBody();
  }
  return nest;
}

/// Smallest trip count a loop bound can take: constants, and
/// ceildiv(extent, block) with block a loom.sym bounded by upper_bound.
static int64_t minTripCount(Value ub) {
  if (auto c = ub.getDefiningOp<arith::ConstantIndexOp>())
    return c.value();
  if (auto cdiv = ub.getDefiningOp<arith::CeilDivUIOp>()) {
    auto extent = cdiv.getLhs().getDefiningOp<arith::ConstantIndexOp>();
    auto sym = cdiv.getRhs().getDefiningOp<loom::SymOp>();
    if (extent && sym && sym.getUpperBound()) {
      int64_t ubBlock = sym.getUpperBound()->getSExtValue();
      if (ubBlock > 0)
        return (extent.value() + ubBlock - 1) / ubBlock;
    }
  }
  return 1;
}

/// The bound of root dimension `d` as a single SSA value when the map is a
/// bare symbol or constant (the frontend's form); null otherwise.
static Value rootDimBoundValue(affine::AffineParallelOp root, unsigned d) {
  AffineMap map = root.getUpperBoundMap(d);
  AffineExpr e = map.getResult(0);
  if (auto s = dyn_cast<AffineSymbolExpr>(e))
    return root.getUpperBoundsOperands()[map.getNumDims() + s.getPosition()];
  if (auto dimE = dyn_cast<AffineDimExpr>(e))
    return root.getUpperBoundsOperands()[dimE.getPosition()];
  return nullptr;
}

static std::string loopName(Value ub, unsigned fallbackIdx) {
  if (ub) {
    if (auto sym = loom_affine::traceToLoomSymRef(ub)) {
      StringRef name = sym->getLeafReference().getValue();
      name.consume_front("tile_");
      return name.str();
    }
  }
  return "d" + std::to_string(fallbackIdx);
}

/// One loop of the nest: a root dimension (parDim >= 0) or a chain loop.
struct Item {
  int parDim = -1;
  int chainIdx = -1;
  int64_t minTrip = 1;
  std::string name;
};

/// Rebuild `func`'s nest with `spatial` (indices into `items`, in nest order)
/// as the parallel dimensions; all other items become sequential loops: chain
/// loops in their original order, then demoted root dimensions.
static LogicalResult rebuildNest(func::FuncOp func, ArrayRef<Item> items,
                                 ArrayRef<unsigned> spatial,
                                 size_t expectedChain) {
  std::optional<LoopNest> nestOpt = analyzeLoopNest(func);
  if (!nestOpt || nestOpt->chain.size() < expectedChain)
    return failure();
  LoopNest &nest = *nestOpt;
  nest.chain.resize(expectedChain);
  affine::AffineParallelOp root = nest.root;
  MLIRContext *ctx = func.getContext();
  Location loc = root.getLoc();
  OpBuilder b(ctx);

  // Hoist the inter-level index ops (only those feeding kept chain levels are
  // needed, but all of them are pure, so hoisting every one is harmless).
  for (Operation *op : nest.hoistOps)
    op->moveBefore(root);

  b.setInsertionPoint(root);
  auto boundOf = [&](const Item &it) -> Value {
    if (it.chainIdx >= 0)
      return nest.chain[it.chainIdx].getUpperBound();
    Value v = rootDimBoundValue(root, it.parDim);
    if (v)
      return v;
    return affine::AffineApplyOp::create(b, loc,
                                         root.getUpperBoundMap(it.parDim),
                                         root.getUpperBoundsOperands());
  };

  SmallVector<Value> ubOperands;
  SmallVector<AffineMap> lbMaps, ubMaps;
  SmallVector<int64_t> steps;
  for (unsigned i = 0; i < spatial.size(); ++i)
    ubOperands.push_back(boundOf(items[spatial[i]]));
  for (unsigned i = 0; i < spatial.size(); ++i) {
    lbMaps.push_back(b.getConstantAffineMap(0));
    ubMaps.push_back(
        AffineMap::get(0, ubOperands.size(), b.getAffineSymbolExpr(i), ctx));
    steps.push_back(1);
  }
  auto newPar = affine::AffineParallelOp::create(
      b, loc, TypeRange{}, ArrayRef<arith::AtomicRMWKind>{}, lbMaps,
      ValueRange{}, ubMaps, ubOperands, steps);
  for (NamedAttribute attr : root->getDiscardableAttrs())
    newPar->setAttr(attr.getName(), attr.getValue());

  // IV of each item in the new nest.
  DenseMap<unsigned, Value> newIV;
  for (unsigned i = 0; i < spatial.size(); ++i)
    newIV[spatial[i]] = newPar.getIVs()[i];

  SmallVector<unsigned> sequential;
  for (unsigned i = 0; i < items.size(); ++i)
    if (items[i].chainIdx >= 0 && !llvm::is_contained(spatial, i))
      sequential.push_back(i);
  for (unsigned i = 0; i < items.size(); ++i)
    if (items[i].parDim >= 0 && !llvm::is_contained(spatial, i))
      sequential.push_back(i);

  b.setInsertionPointToStart(newPar.getBody());
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Block *innermost = newPar.getBody();
  for (unsigned idx : sequential) {
    const Item &it = items[idx];
    Value ub;
    if (it.chainIdx >= 0) {
      ub = nest.chain[it.chainIdx].getUpperBound();
    } else {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPoint(newPar);
      ub = boundOf(it);
    }
    auto forOp = scf::ForOp::create(b, loc, c0, ub, c1);
    newIV[idx] = forOp.getInductionVar();
    innermost = forOp.getBody();
    b.setInsertionPoint(innermost->getTerminator());
  }

  // Move the body of the innermost kept chain loop (or the root body when the
  // chain is empty) into the new innermost loop.
  Block *oldBody = expectedChain ? nest.chain.back().getBody() : root.getBody();
  Operation *dstTerm = innermost->getTerminator();
  for (Operation &op : llvm::make_early_inc_range(oldBody->without_terminator()))
    op.moveBefore(dstTerm);

  for (unsigned i = 0; i < items.size(); ++i) {
    Value oldIV = items[i].chainIdx >= 0
                      ? nest.chain[items[i].chainIdx].getInductionVar()
                      : root.getIVs()[items[i].parDim];
    oldIV.replaceAllUsesWith(newIV[i]);
  }
  root.erase();
  return success();
}

} // namespace

namespace loom {

void ExploreParallelSets(ModuleOp module, const HardwareInfo &hardwareInfo) {
  int64_t cores = 1;
  for (const SpatialDimInfo &d : hardwareInfo.spatialDimInfoVec)
    cores *= d.size.value_or(1);

  OpBuilder builder(module.getContext());
  for (func::FuncOp func : loom::utils::collectFunctions(module)) {
    std::optional<LoopNest> nest = analyzeLoopNest(func);
    if (!nest || nest->chain.empty())
      continue;

    affine::AffineParallelOp root = nest->root;
    const unsigned P = root.getNumDims();
    SmallVector<Item> items;
    for (unsigned d = 0; d < P; ++d) {
      Item it;
      it.parDim = d;
      Value ub = rootDimBoundValue(root, d);
      it.minTrip = ub ? minTripCount(ub) : 1;
      it.name = loopName(ub, d);
      items.push_back(it);
    }
    for (unsigned j = 0; j < nest->chain.size(); ++j) {
      Item it;
      it.chainIdx = j;
      Value ub = nest->chain[j].getUpperBound();
      it.minTrip = minTripCount(ub);
      it.name = loopName(ub, P + j);
      items.push_back(it);
    }

    const unsigned n = items.size();
    const unsigned maxDims = std::max(3u, P);
    const unsigned identity = (1u << P) - 1;
    if (n > 16)
      continue;
    Operation *insertAfter = func;
    for (unsigned mask = 1; mask < (1u << n); ++mask) {
      if (mask == identity)
        continue;
      SmallVector<unsigned> spatial;
      int64_t work = 1;
      for (unsigned i = 0; i < n; ++i) {
        if (mask & (1u << i)) {
          spatial.push_back(i);
          work *= items[i].minTrip;
        }
      }
      // A loop whose minimum trip count is 1 gives no guaranteed parallel
      // work (it depends on the block size); new sets leave such loops
      // sequential, and the original function covers keeping them spatial.
      if (llvm::any_of(spatial,
                       [&](unsigned i) { return items[i].minTrip <= 1; }))
        continue;
      if (spatial.size() > maxDims || work < cores)
        continue;

      std::string name = func.getName().str() + "__par";
      for (unsigned i : spatial)
        name += "_" + items[i].name;
      size_t chainLen = nest->chain.size();
      func::FuncOp clone = loom::utils::cloneFunc(
          builder, func, name, /*moduleAttrs=*/nullptr,
          [&](func::FuncOp f) {
            return rebuildNest(f, items, spatial, chainLen);
          },
          insertAfter);
      if (clone) {
        // Work covers every core: partial-occupancy mappings are not useful.
        clone->setAttr("loom.full_occupancy_only", builder.getUnitAttr());
        insertAfter = clone;
        LLVM_DEBUG(llvm::dbgs() << "parallel set variant: " << name << "\n");
      }
    }
  }
}

} // namespace loom
