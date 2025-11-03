//===- tile_scf_for_to_l1.cpp ----------------------------------*- C++ -*-===//
// Tile scf.for loops so that per-tile memory fits the single df.memory (L1).
// This pass expects fully bufferized IR (no tensor types anywhere).
//===----------------------------------------------------------------------===//

/**
 * @file tile_scf_for_to_l1.cpp
 * @brief Implementation of tiling `scf.for` loops to respect L1 capacity.
 * @details
 * Algorithm outline:
 * - Validate the module has exactly one `df.memory`; record its `size` (bytes)
 *   and `label`.
 * - Require the entire module to be fully bufferized: no tensor types are
 *   allowed (operands, results, or block arguments). If any tensor is found,
 *   the pass fails.
 * - For each function, visit `scf.for` loops and compute per-iteration memory
 *   by summing byte sizes of all static `memref.alloc` inside the loop body
 *   that are explicitly annotated as local to the single `df.memory` (i.e.,
 *   `tmd.alloc = {local=true, memory_name = <label>}`). Other allocs are
 *   ignored for the L1 capacity check.
 * - Pick the largest power-of-two tile factor `t` such that
 *     `t * perIterBytes <= L1SizeBytes`.
 *   If `perIterBytes == 0`, skip the loop; if no positive `t` exists, fail.
 * - Compute the loop trip count `N` when statically provable with constant
 *   lb/ub/step and require exact divisibility by the tile factor; otherwise
 *   fail the pass.
 * - Rewrite the loop into an outer `scf.for` over tiles and an inner
 *   `scf.for` over the tile span. Replace the original loop induction variable
 *   with the absolute IV: `lb + tileIdx * tileSpan + innerIV`. Thread
 *   original iter_args/results through the outer `scf.for` directly (no
 *   tensors or temporary buffers are created by this pass).
 */

#include "tile_scf_for_to_l1.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "DataflowDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "DataflowTypes.h.inc"
#define GET_OP_CLASSES
#include "DataflowOps.h.inc"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/WithColor.h"

using namespace mlir;

namespace tmd {
namespace passes {

namespace {

struct SingleMemoryInfo {
  std::string label;
  uint64_t sizeBytes = 0;
};

static FailureOr<SingleMemoryInfo> requireSingleMemory(ModuleOp module) {
  int memCount = 0;
  tmd::df::MemoryOp first;
  module.walk([&](tmd::df::MemoryOp m) {
    if (memCount == 0)
      first = m;
    ++memCount;
  });
  if (memCount != 1) {
    llvm::WithColor::error(llvm::errs())
        << "Expected exactly one df.memory in module\n";
    return failure();
  }
  SingleMemoryInfo info;
  if (auto l = first.getLabelAttr())
    info.label = l.getValue().str();
  else
    info.label = "mem0";
  info.sizeBytes = (uint64_t)first.getSize();
  return info;
}

static std::optional<unsigned> getElementBitWidth(Type elemTy) {
  if (auto i = llvm::dyn_cast<IntegerType>(elemTy))
    return i.getWidth();
  if (auto f = llvm::dyn_cast<FloatType>(elemTy))
    return f.getWidth();
  if (auto idx = llvm::dyn_cast<IndexType>(elemTy))
    return 64u; // Assume 64-bit index
  return std::nullopt;
}

static FailureOr<uint64_t> getStaticMemrefByteSize(MemRefType mt) {
  if (!mt.hasStaticShape())
    return failure();
  auto maybeBits = getElementBitWidth(mt.getElementType());
  if (!maybeBits)
    return failure();
  uint64_t elems = 1;
  for (int64_t d : mt.getShape())
    elems *= static_cast<uint64_t>(d);
  uint64_t bytesPerElem = (*maybeBits + 7) / 8;
  return elems * bytesPerElem;
}

static bool hasAllocOnMemory(memref::AllocOp alloc, StringRef label) {
  if (auto dict = alloc->getAttrOfType<DictionaryAttr>("tmd.alloc")) {
    auto local = dict.get("local");
    auto memName = dict.get("memory_name");
    auto b = llvm::dyn_cast_or_null<BoolAttr>(local);
    auto s = llvm::dyn_cast_or_null<StringAttr>(memName);
    return b && b.getValue() && s && s.getValue() == label;
  }
  return false;
}

static std::optional<int64_t> getConstIndex(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
    return c.value();
  if (auto c2 = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = llvm::dyn_cast<IntegerAttr>(c2.getValue()))
      return ia.getInt();
  }
  return std::nullopt;
}

static std::optional<int64_t> getTripCount(scf::ForOp forOp) {
  // Only support constant lb/ub/step where (ub - lb) is divisible by step.
  auto lb = getConstIndex(forOp.getLowerBound());
  auto ub = getConstIndex(forOp.getUpperBound());
  auto st = getConstIndex(forOp.getStep());
  if (!lb || !ub || !st)
    return std::nullopt;
  if (*st <= 0)
    return std::nullopt;
  int64_t span = *ub - *lb;
  if (span <= 0)
    return 0ll;
  if ((span % *st) != 0)
    return std::nullopt;
  int64_t n = span / *st;
  return n;
}

static uint64_t largestPowerOfTwoLE(uint64_t n) {
  if (n == 0)
    return 0;
  uint64_t p = 1;
  while ((p << 1) > p && (p << 1) <= n)
    p <<= 1;
  return p;
}

class TileScfForToL1Pass
    : public PassWrapper<TileScfForToL1Pass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileScfForToL1Pass)

  StringRef getArgument() const override { return "tmd-tile-scf-for-to-l1"; }
  StringRef getDescription() const override {
    return "Tile scf.for loops with an outer affine.for so each tile fits in "
           "L1";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<affine::AffineDialect, arith::ArithDialect, func::FuncDialect,
                scf::SCFDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    auto memInfoOr = requireSingleMemory(module);
    if (failed(memInfoOr)) {
      signalPassFailure();
      return;
    }
    SingleMemoryInfo memInfo = *memInfoOr;

    // Enforce fully bufferized IR: no tensor types anywhere.
    bool foundTensor = false;
    module.walk([&](Operation *op) -> WalkResult {
      if (foundTensor)
        return WalkResult::interrupt();
      auto checkType = [&](Type t) {
        if (llvm::isa<TensorType>(t)) {
          foundTensor = true;
          return true;
        }
        return false;
      };
      for (Type t : op->getResultTypes())
        if (checkType(t))
          return WalkResult::interrupt();
      for (Value v : op->getOperands())
        if (checkType(v.getType()))
          return WalkResult::interrupt();
      for (Region &r : op->getRegions())
        for (Block &b : r)
          for (BlockArgument a : b.getArguments())
            if (checkType(a.getType()))
              return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (foundTensor) {
      llvm::WithColor::error(llvm::errs())
          << "This pass requires bufferized IR (no tensor types).\n";
      signalPassFailure();
      return;
    }

    bool changedAny = false;

    module.walk([&](func::FuncOp func) {
      SmallVector<scf::ForOp, 8> loops;
      func.walk([&](scf::ForOp f) { loops.push_back(f); });

      for (scf::ForOp forOp : loops) {
        // Compute per-iteration alloc bytes of L1-local buffers only.
        uint64_t perIterBytes = 0;
        bool badAlloc = false;
        forOp.getBody()->walk([&](memref::AllocOp a) {
          // Only count allocs explicitly placed in the single L1 memory.
          if (!hasAllocOnMemory(a, memInfo.label))
            return WalkResult::advance();
          auto mt = llvm::dyn_cast<MemRefType>(a.getType());
          if (!mt)
            return WalkResult::advance();
          // Prefer the annotated size in bytes if present.
          uint64_t bytes = 0;
          if (auto dict = a->getAttrOfType<DictionaryAttr>("tmd.alloc")) {
            if (auto any = dict.get("size"))
              if (auto szAttr = llvm::dyn_cast<IntegerAttr>(any))
                bytes = static_cast<uint64_t>(szAttr.getInt());
          }
          if (bytes == 0) {
            auto sz = getStaticMemrefByteSize(mt);
            if (failed(sz)) {
              badAlloc = true;
              return WalkResult::interrupt();
            }
            bytes = *sz;
          }
          perIterBytes += bytes;
          return WalkResult::advance();
        });
        if (badAlloc) {
          llvm::WithColor::error(llvm::errs())
              << "Alloc sizing failed for L1-local alloc under scf.for in "
                 "function '"
              << func.getSymName() << "'\n";
          signalPassFailure();
          return;
        }
        if (perIterBytes == 0)
          continue; // nothing to tile

        uint64_t maxTiles = memInfo.sizeBytes / perIterBytes;
        if (maxTiles == 0) {
          llvm::WithColor::error(llvm::errs())
              << "Per-iteration memory exceeds L1 size in function '"
              << func.getSymName() << "'\n";
          signalPassFailure();
          return;
        }
        uint64_t tileFactor = largestPowerOfTwoLE(maxTiles);
        if (tileFactor == 0) {
          signalPassFailure();
          return;
        }

        // Prove trip count and divisibility.
        auto trip = getTripCount(forOp);
        if (!trip) {
          llvm::WithColor::error(llvm::errs())
              << "Trip count is not statically provable for scf.for in "
                 "function '"
              << func.getSymName() << "'\n";
          signalPassFailure();
          return;
        }
        if ((*trip % static_cast<int64_t>(tileFactor)) != 0) {
          llvm::WithColor::error(llvm::errs())
              << "Trip count not divisible by tile factor in function '"
              << func.getSymName() << "'\n";
          signalPassFailure();
          return;
        }

        // Start rewriting
        OpBuilder b(forOp);
        Location loc = forOp.getLoc();

        // Compute tile span and tile count (as constants when possible)
        Value tileCountVal = b.create<arith::ConstantIndexOp>(
            loc, *trip / static_cast<int64_t>(tileFactor));
        int64_t stepInt = *getConstIndex(forOp.getStep());
        int64_t lbInt = *getConstIndex(forOp.getLowerBound());
        int64_t tileSpanInt = stepInt * static_cast<int64_t>(tileFactor);
        Value tileSpan = b.create<arith::ConstantIndexOp>(loc, tileSpanInt);

        // Create outer scf.for over tiles, threading original iter_args.
        b.setInsertionPoint(forOp);
        Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
        Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
        SmallVector<Value, 4> outerInit(forOp.getInitArgs().begin(),
                                        forOp.getInitArgs().end());
        auto outer = b.create<scf::ForOp>(loc, c0, tileCountVal, c1, outerInit);

        // Prepare inside outer: compute inner loop bounds
        Block &outerBody = outer.getRegion().front();
        b.setInsertionPointToStart(&outerBody);
        Value tIdx = outer.getInductionVar();
        Value innerLB = b.create<arith::ConstantIndexOp>(loc, 0);
        Value innerUB = tileSpan;

        // Gather current carried values from outer body args for inner init
        SmallVector<Value, 4> innerInitArgs;
        innerInitArgs.reserve(outerBody.getNumArguments() - 1);
        for (unsigned i = 1; i < outerBody.getNumArguments(); ++i)
          innerInitArgs.push_back(outerBody.getArgument(i));

        // Create inner scf.for with same iter_args/result arity as original
        Value innerStep = b.create<arith::ConstantIndexOp>(loc, stepInt);
        auto inner = b.create<scf::ForOp>(loc, innerLB, innerUB, innerStep,
                                          innerInitArgs);

        // Move original for body into inner and remap IV and carried args
        IRMapping mapper;
        Block &oldBody = forOp.getRegion().front();
        Block &newBody = inner.getRegion().front();
        unsigned numCarried = forOp.getInitArgs().size();
        for (unsigned i = 0; i < numCarried; ++i)
          mapper.map(oldBody.getArgument(i + 1), newBody.getArgument(i + 1));
        b.setInsertionPointToStart(&newBody);
        // Map original IV to absolute IV directly as
        // absIV = lb + tIdx * tileSpan + innerLocalIV using a single
        // affine.apply
        AffineExpr d0 = getAffineDimExpr(0, b.getContext()); // tIdx
        AffineExpr d1 = getAffineDimExpr(1, b.getContext()); // inner iv
        AffineMap absMap = AffineMap::get(2, 0, d0 * tileSpanInt + d1 + lbInt);
        Value absIV =
            b.create<affine::AffineApplyOp>(
                 loc, absMap, ValueRange{tIdx, inner.getInductionVar()})
                .getResult();
        mapper.map(forOp.getInductionVar(), absIV);
        for (Operation &op : llvm::make_early_inc_range(oldBody)) {
          if (isa<scf::YieldOp>(&op))
            continue;
          b.clone(op, mapper);
        }

        // Build new yield from mapped operands of old yield and replace
        // terminator
        auto oldYield = cast<scf::YieldOp>(oldBody.getTerminator());
        SmallVector<Value, 4> newYieldVals;
        for (Value v : oldYield.getOperands()) {
          Value mapped = mapper.lookupOrNull(v);
          if (!mapped) {
            llvm::WithColor::error(llvm::errs())
                << "Internal error: missing mapping for yield operand in tiled "
                   "loop\n";
            signalPassFailure();
            return;
          }
          newYieldVals.push_back(mapped);
        }
        // Ensure the body has exactly one terminator: if the last operation is
        // a terminator, remove it before inserting a new scf.yield with the
        // remapped operands. Avoid calling getTerminator() here because the
        // builder may not have inserted a terminator yet when iter_args exist.
        if (!newBody.empty() &&
            newBody.back().hasTrait<OpTrait::IsTerminator>())
          newBody.back().erase();
        b.setInsertionPointToEnd(&newBody);
        b.create<scf::YieldOp>(loc, newYieldVals);

        // After cloning, compose upstream affine.apply chains into users so
        // that the temporary absolute IV affine.apply is fused into existing
        // affine maps, eliminating the extra affine.apply where possible.
        {
          SmallVector<affine::AffineApplyOp, 16> applyOps;
          newBody.walk(
              [&](affine::AffineApplyOp aa) { applyOps.push_back(aa); });
          for (affine::AffineApplyOp aa : applyOps) {
            AffineMap map = aa.getAffineMap();
            SmallVector<Value, 8> operands(aa.getMapOperands().begin(),
                                           aa.getMapOperands().end());
            affine::fullyComposeAffineMapAndOperands(&map, &operands);
            affine::canonicalizeMapAndOperands(&map, &operands);
            OpBuilder bb(aa);
            auto newApply =
                bb.create<affine::AffineApplyOp>(aa.getLoc(), map, operands);
            aa.replaceAllUsesWith(newApply.getResult());
            aa.erase();
          }
          // Erase any affine.apply ops that became dead after composition.
          SmallVector<affine::AffineApplyOp, 16> toErase;
          newBody.walk([&](affine::AffineApplyOp aa) {
            if (aa->use_empty())
              toErase.push_back(aa);
          });
          for (affine::AffineApplyOp aa : toErase)
            aa.erase();
        }

        // Outer loop yields the results produced by the inner loop.
        // Remove any existing terminator first, then insert a new one.
        if (!outerBody.empty() &&
            outerBody.back().hasTrait<OpTrait::IsTerminator>())
          outerBody.back().erase();
        b.setInsertionPointToEnd(&outerBody);
        b.create<scf::YieldOp>(loc, inner.getResults());

        // Replace original loop with the outer tiled loop results.
        if (forOp.getNumResults() == outer.getNumResults())
          forOp.replaceAllUsesWith(outer.getResults());
        else if (forOp.getNumResults() == 0)
          forOp.replaceAllUsesWith(ValueRange{});
        else {
          llvm::WithColor::error(llvm::errs())
              << "Unexpected arity mismatch when replacing loop results\n";
          signalPassFailure();
          return;
        }
        forOp.erase();

        changedAny = true;
      }
    });

    (void)changedAny;
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createTileScfForToL1Pass() {
  return std::make_unique<TileScfForToL1Pass>();
}

void registerTileScfForToL1Pass() { PassRegistration<TileScfForToL1Pass>(); }

} // namespace passes
} // namespace tmd
