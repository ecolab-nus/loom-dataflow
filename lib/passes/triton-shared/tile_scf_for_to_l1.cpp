//===- tile_scf_for_to_l1.cpp ----------------------------------*- C++ -*-===//
// Tile scf.for loops so that per-tile memory fits the single df.memory (L1).
//===----------------------------------------------------------------------===//

/**
 * @file tile_scf_for_to_l1.cpp
 * @brief Implementation of tiling `scf.for` loops to respect L1 capacity.
 * @details
 * Algorithm outline:
 * - Validate the module has exactly one `df.memory`; record its `size` (bytes)
 *   and `label`.
 * - For each function, visit `scf.for` loops and compute per-iteration memory
 *   by summing byte sizes of all static `memref.alloc` inside the loop body.
 *   Require all such allocs to carry `tmd.alloc.memory_name == <label>` and
 *   `local=true`. If any alloc violates, fail the pass.
 * - Pick the largest power-of-two tile factor `t` such that
 *     `t * perIterBytes <= L1SizeBytes`.
 *   If `perIterBytes == 0`, skip the loop; if no positive `t` exists, fail.
 * - Compute the loop trip count `N` when statically provable with constant
 *   lb/ub/step and require exact divisibility by the tile factor; otherwise
 *   fail the pass.
 * - Rewrite the loop into an outer `affine.for` over tiles and an inner
 *   `scf.for` over the tile span. Replace the original loop induction variable
 *   with the inner loop IV. Keep inner `scf.for` unchanged otherwise.
 */

#include "tile_scf_for_to_l1.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    bufferization::BufferizationDialect, func::FuncDialect,
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

    bool changedAny = false;

    module.walk([&](func::FuncOp func) {
      SmallVector<scf::ForOp, 8> loops;
      func.walk([&](scf::ForOp f) { loops.push_back(f); });

      for (scf::ForOp forOp : loops) {
        // Compute per-iteration alloc bytes and validate alloc targets.
        uint64_t perIterBytes = 0;
        bool badAlloc = false;
        forOp.getBody()->walk([&](memref::AllocOp a) {
          if (!hasAllocOnMemory(a, memInfo.label)) {
            badAlloc = true;
            return WalkResult::interrupt();
          }
          auto mt = llvm::dyn_cast<MemRefType>(a.getType());
          if (!mt) {
            badAlloc = true;
            return WalkResult::interrupt();
          }
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
              << "Alloc validation or sizing failed under scf.for in function '"
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
        auto idxTy = b.getIndexType();

        // Compute tile span and tile count (as constants when possible)
        Value tileCountVal = b.create<arith::ConstantIndexOp>(
            loc, *trip / static_cast<int64_t>(tileFactor));
        int64_t stepInt = *getConstIndex(forOp.getStep());
        int64_t lbInt = *getConstIndex(forOp.getLowerBound());
        int64_t tileSpanInt = stepInt * static_cast<int64_t>(tileFactor);
        Value tileSpan = b.create<arith::ConstantIndexOp>(loc, tileSpanInt);

        // Create carry buffers BEFORE the outer loop and seed with init args
        b.setInsertionPoint(forOp);
        SmallVector<Value, 4> carryAllocs;
        for (Value initVal : forOp.getInitArgs()) {
          auto tTy = llvm::dyn_cast<RankedTensorType>(initVal.getType());
          if (!tTy || !tTy.hasStaticShape()) {
            llvm::WithColor::error(llvm::errs())
                << "Only ranked static tensor iter_args supported for tiling\n";
            signalPassFailure();
            return;
          }
          MemRefType mty =
              MemRefType::get(tTy.getShape(), tTy.getElementType());
          auto alloc = b.create<memref::AllocOp>(loc, mty);
          {
            NamedAttrList nl;
            nl.append("local", BoolAttr::get(module.getContext(), true));
            nl.append("memory_name",
                      StringAttr::get(module.getContext(), memInfo.label));
            alloc->setAttr("tmd.alloc",
                           DictionaryAttr::get(module.getContext(), nl));
          }
          auto midInit = b.create<bufferization::MaterializeInDestinationOp>(
              loc, initVal, alloc);
          midInit->setAttr("writable", UnitAttr::get(module.getContext()));
          carryAllocs.push_back(alloc);
        }

        // Create outer affine.for t in [0, tileCount)
        auto lbMap = AffineMap::getConstantMap(0, b.getContext());
        auto ubMap = AffineMap::get(/*dimCount=*/1, /*symCount=*/0,
                                    getAffineDimExpr(0, b.getContext()));
        auto outer = b.create<affine::AffineForOp>(
            loc, /*lowerBoundOperands=*/ValueRange{}, lbMap,
            /*upperBoundOperands=*/ValueRange{tileCountVal}, ubMap,
            /*step=*/1);

        // Prepare inside outer: compute tile base = lb + t * tileSpan
        b.setInsertionPointToStart(outer.getBody());
        Value tIdx = outer.getInductionVar();
        Value innerLB = b.create<arith::ConstantIndexOp>(loc, 0);
        Value innerUB = tileSpan;

        // Build init args for inner from current carry buffers
        SmallVector<Value, 4> iterArgs;
        iterArgs.reserve(carryAllocs.size());
        for (Value alloc : carryAllocs) {
          auto mt = llvm::cast<MemRefType>(alloc.getType());
          auto tt = RankedTensorType::get(mt.getShape(), mt.getElementType());
          auto toT = b.create<bufferization::ToTensorOp>(loc, tt, alloc);
          iterArgs.push_back(toT.getResult());
        }

        // Create inner scf.for with same iter_args/result arity
        auto inner = b.create<scf::ForOp>(loc, innerLB, innerUB,
                                          forOp.getStep(), iterArgs);

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

        // After inner loop, update carry buffers with the inner results
        b.setInsertionPointAfter(inner);
        for (auto it : llvm::enumerate(inner.getResults())) {
          Value res = it.value();
          Value alloc = carryAllocs[it.index()];
          auto midUpdate = b.create<bufferization::MaterializeInDestinationOp>(
              loc, res, alloc);
          midUpdate->setAttr("writable", UnitAttr::get(module.getContext()));
        }

        // After the outer loop, read back final carried tensors and replace
        // uses
        b.setInsertionPointAfter(outer);
        SmallVector<Value, 4> finals;
        finals.reserve(carryAllocs.size());
        for (Value alloc : carryAllocs) {
          auto mt = llvm::cast<MemRefType>(alloc.getType());
          auto tt = RankedTensorType::get(mt.getShape(), mt.getElementType());
          auto toT = b.create<bufferization::ToTensorOp>(loc, tt, alloc);
          finals.push_back(toT.getResult());
        }
        if (forOp.getNumResults() == finals.size())
          forOp.replaceAllUsesWith(finals);
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
