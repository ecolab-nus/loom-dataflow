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

#include <cassert>

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

#include "DataflowDialect.h.inc"
#include "llvm/ADT/Twine.h"
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

static bool IsFullyBufferized(ModuleOp& module) {
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
  return !foundTensor;
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

static FailureOr<uint64_t> ComputePerIterMemoryBytes(scf::ForOp forOp, StringRef memoryLabel) {
  uint64_t totalBytes = 0;

  auto walkResult = forOp.getBody()->walk([&](memref::AllocOp allocOp) {
  // Only count allocs explicitly placed in the single L1 memory.
  if (!hasAllocOnMemory(allocOp, memoryLabel)) return WalkResult::advance();

  auto memrefType = llvm::dyn_cast<MemRefType>(allocOp.getType());
  if (!memrefType) return WalkResult::advance();

  // Prefer the annotated size in bytes if present.
  uint64_t bytes = 0;
  if (auto dict = allocOp->getAttrOfType<DictionaryAttr>("tmd.alloc")) {
    if (auto sizeAttr = dict.get("size"))
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(sizeAttr))
        bytes = static_cast<uint64_t>(intAttr.getInt());
  }

  // Fallback: compute static memref byte size
  if (bytes == 0) {
    auto sizeOrFailure = getStaticMemrefByteSize(memrefType);
    if (failed(sizeOrFailure)) // Cannot compute size for this allocation
      return WalkResult::interrupt();
    bytes = *sizeOrFailure;
  }

  totalBytes += bytes;
  return WalkResult::advance();
  });

  // Check if walk was interrupted due to unsized allocation
  if (walkResult.wasInterrupted())
    return failure();

  return totalBytes;
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


/**
 * @brief Implement FromAffineFor factory method
 */
FailureOr<LoopDescriptor> LoopDescriptor::FromAffineFor(affine::AffineForOp op, uint64_t tileFactor) {
  LoopDescriptor desc;
  desc.type = AFFINE_FOR;
  desc.op = op.getOperation();
  desc.inductionVar = op.getInductionVar();
  desc.tileFactor = tileFactor;
  
  // Extract bounds and step
  desc.lowerBoundMap = op.getLowerBoundMap();
  desc.upperBoundMap = op.getUpperBoundMap();
  desc.lbOperands.append(op.getLowerBoundOperands().begin(), op.getLowerBoundOperands().end());
  desc.ubOperands.append(op.getUpperBoundOperands().begin(), op.getUpperBoundOperands().end());
  desc.step = op.getStepAsInt();
  desc.tripCount = -1;
  
  return desc;
}

/**
 * @brief Implement FromScfFor factory method
 */
FailureOr<LoopDescriptor> LoopDescriptor::FromScfFor(scf::ForOp op, uint64_t tileFactor) {
  LoopDescriptor desc;
  desc.type = SCF_FOR;
  desc.op = op.getOperation();
  desc.inductionVar = op.getInductionVar();
  desc.tileFactor = tileFactor;
  
  // Extract bounds and step
  desc.lowerBound = op.getLowerBound();
  desc.upperBound = op.getUpperBound();
  desc.stepValue = op.getStep();
  
  // Compute trip count (reuse existing function)
  auto tripOpt = getTripCount(op);
  if (!tripOpt) {
    llvm::WithColor::error(llvm::errs()) 
        << "Scf.for trip count is not statically provable\n";
    return failure();
  }
  desc.tripCount = *tripOpt;
  
  // Extract step value
  auto stepOpt = getConstIndex(desc.stepValue);
  if (!stepOpt) {
    llvm::WithColor::error(llvm::errs()) 
        << "Scf.for step is not constant\n";
    return failure();
  }
  desc.step = *stepOpt;
  
  // Check divisibility
  if (desc.tripCount % static_cast<int64_t>(tileFactor) != 0) {
    llvm::WithColor::error(llvm::errs()) 
        << "Scf.for trip count (" << desc.tripCount 
        << ") not divisible by tile factor (" << tileFactor << ")\n";
    return failure();
  }
  
  return desc;
}


// SingleLoopTillingManager: original single loop tiling implementation (kept for backward compatibility)
class SingleLoopTillingManager {
  private:
    struct TilingConfig {
      int64_t tripCount;
      int64_t tileFactor;
      int64_t lowerBound;
      int64_t step;
      int64_t tileCount;
      int64_t tileSpan;
    };
  
  public:
    /**
     * @brief Constructor for SingleLoopTillingManager
     * @param forOp The original scf.for loop to be transformed
     * @param trip The static iteration count of the loop
     * @param tileFactor The tile size (must divide trip)
     */
    SingleLoopTillingManager(mlir::scf::ForOp& forOp, int64_t trip, uint64_t tileFactor) 
      : forOp_(forOp), op_builder_(forOp), loc_(forOp.getLoc()) {
      tiling_config_ = ComputeTilingConfig(trip, tileFactor);
    }
  
    /**
     * @brief Compute the tiling configuration
     * @param trip The static iteration count of the loop
     * @param tileFactor The tile size (must divide trip)
     * @return The tiling configuration
     */
    TilingConfig ComputeTilingConfig(int64_t trip, uint64_t tileFactor) {
      TilingConfig config;
      config.tripCount = trip;
      config.tileFactor = static_cast<int64_t>(tileFactor);
      config.lowerBound = *getConstIndex(forOp_.getLowerBound());
      config.step = *getConstIndex(forOp_.getStep());
      config.tileCount = trip / static_cast<int64_t>(tileFactor);
      config.tileSpan = config.step * static_cast<int64_t>(tileFactor);
      
      return config;
    }
  
    /**
     * @brief Create the outer tile loop
     * @return The outer loop and its body block on success, failure otherwise
     */
    FailureOr<std::pair<scf::ForOp, Block*>> CreateOuterTileLoop() {
      Location loc = forOp_.getLoc();
      
      Value c0 = op_builder_.create<arith::ConstantIndexOp>(loc, 0);
      Value c1 = op_builder_.create<arith::ConstantIndexOp>(loc, 1);
      Value tileCount = op_builder_.create<arith::ConstantIndexOp>(loc, tiling_config_.tileCount);
      
      SmallVector<Value, 4> initArgs(forOp_.getInitArgs());
      auto outer = op_builder_.create<scf::ForOp>(loc, c0, tileCount, c1, initArgs);
      
      if (!outer) {
        llvm::WithColor::error(llvm::errs()) 
            << "Failed to create outer tile loop\n";
        return failure();
      }
      
      return std::make_pair(outer, &outer.getRegion().front());
    }
  
    /**
     * @brief Create the inner local iteration loop
     * @param outerBody The body block of the outer loop
     * @return The inner loop on success, failure otherwise
     */
    FailureOr<scf::ForOp> CreateInnerLocalLoop(Block *outerBody) {
      if (!outerBody) {
        llvm::WithColor::error(llvm::errs()) 
            << "Invalid outer body block\n";
        return failure();
      }
      
      op_builder_.setInsertionPointToStart(outerBody);
  
      Value innerLB = op_builder_.create<arith::ConstantIndexOp>(loc_, 0);
      Value innerUB = op_builder_.create<arith::ConstantIndexOp>(loc_, tiling_config_.tileSpan);
      Value innerStep = op_builder_.create<arith::ConstantIndexOp>(loc_, tiling_config_.step);
  
      // Collect the outer block arguments as the inner initial values
      SmallVector<Value, 4> innerInit;
      for (unsigned i = 1; i < outerBody->getNumArguments(); ++i)
        innerInit.push_back(outerBody->getArgument(i));
  
      auto inner = op_builder_.create<scf::ForOp>(loc_, innerLB, innerUB, innerStep, innerInit);
      
      if (!inner) {
        llvm::WithColor::error(llvm::errs()) 
            << "Failed to create inner local loop\n";
        return failure();
      }
      
      return inner;
    }
  
    /**
     * @brief Clone the loop body to the inner loop and remap all references
     * @param innerFor The inner loop
     * @param tileIndexIV The tile index induction variable
     * @return The remapped yield operands on success, failure otherwise
     */
    FailureOr<SmallVector<Value, 4>> CloneAndRemapLoopBody(scf::ForOp innerFor, Value tileIndexIV) {
      Block &oldBody = forOp_.getRegion().front();
      Block &newBody = innerFor.getRegion().front();
      IRMapping mapper;
      
      // 1. Map the iteration parameters
      unsigned numCarried = forOp_.getInitArgs().size();
      for (unsigned i = 0; i < numCarried; ++i)
        mapper.map(oldBody.getArgument(i + 1), newBody.getArgument(i + 1));
      
      // 2. Map the induction variables to the absolute IV
      op_builder_.setInsertionPointToStart(&newBody);
      AffineExpr d0 = getAffineDimExpr(0, op_builder_.getContext());
      AffineExpr d1 = getAffineDimExpr(1, op_builder_.getContext());
      AffineMap absMap = AffineMap::get(
          2, 0, d0 * tiling_config_.tileSpan + d1 + tiling_config_.lowerBound);
      
      Value absIV = op_builder_.create<affine::AffineApplyOp>(
          forOp_.getLoc(), absMap, 
          ValueRange{tileIndexIV, innerFor.getInductionVar()}).getResult();
      mapper.map(forOp_.getInductionVar(), absIV);
      
      // 3. Clone all operations (except yield)
      for (Operation &op : llvm::make_early_inc_range(oldBody)) {
        if (isa<scf::YieldOp>(&op))
          continue;
        op_builder_.clone(op, mapper);
      }
      
      // 4. Remap the yield operands (with error checking)
      auto oldYield = cast<scf::YieldOp>(oldBody.getTerminator());
      SmallVector<Value, 4> newYieldVals;
      for (Value v : oldYield.getOperands()) {
        Value mapped = mapper.lookupOrNull(v);
        if (!mapped) {
          llvm::WithColor::error(llvm::errs())
              << "Internal error: missing mapping for yield operand in tiled loop\n";
          return failure();
        }
        newYieldVals.push_back(mapped);
      }
      
      return newYieldVals;
    }
  
    /**
     * @brief Optimize the affine expressions in the loop body
     * @param loopBody The loop body to optimize
     */
    void OptimizeAffineExpressions(Block &loopBody) {
      // Collect all affine.apply operations
      SmallVector<affine::AffineApplyOp, 16> applyOps;
      loopBody.walk([&](affine::AffineApplyOp aa) { applyOps.push_back(aa); });
      
      // Compose and canonicalize
      for (affine::AffineApplyOp aa : applyOps) {
        AffineMap map = aa.getAffineMap();
        SmallVector<Value, 8> operands(aa.getMapOperands());
        
        affine::fullyComposeAffineMapAndOperands(&map, &operands);
        affine::canonicalizeMapAndOperands(&map, &operands);
        
        OpBuilder builder(aa);
        auto newApply = builder.create<affine::AffineApplyOp>(
            aa.getLoc(), map, operands);
        aa.replaceAllUsesWith(newApply.getResult());
        aa.erase();
      }
      
      // Clean up dead operations
      SmallVector<affine::AffineApplyOp, 16> deadOps;
      loopBody.walk([&](affine::AffineApplyOp aa) {
        if (aa->use_empty())
          deadOps.push_back(aa);
      });
      for (auto aa : deadOps)
        aa.erase();
    }
  
    /**
     * @brief Finalize the outer loop and replace the original loop
     * @param outerFor The outer tile loop
     * @param innerFor The inner local iteration loop
     * @return success on success, failure otherwise
     */
    LogicalResult FinalizeAndReplaceLoop(scf::ForOp outerFor, scf::ForOp innerFor) {
      Block &outerBody = outerFor.getRegion().front();
  
      // 1. Build the outer yield
      if (!outerBody.empty() && outerBody.back().hasTrait<OpTrait::IsTerminator>())
        outerBody.back().erase();
      
      op_builder_.setInsertionPointToEnd(&outerBody);
      op_builder_.create<scf::YieldOp>(loc_, innerFor.getResults());
  
      // 2. Verify the result count matches
      if (forOp_.getNumResults() != outerFor.getNumResults() && forOp_.getNumResults() != 0)
        return failure();
  
      // 3. Replace the original loop
      if (forOp_.getNumResults() == outerFor.getNumResults())
        forOp_.replaceAllUsesWith(outerFor.getResults());
      else
        forOp_.replaceAllUsesWith(ValueRange{});
  
      forOp_.erase();
      return success();
    }
  
    /**
     * @brief Execute the complete tiling transformation
     * @return success on success, failure otherwise
     */
    LogicalResult Transform() {
      // 1. Create the outer tile loop
      auto outerResult = CreateOuterTileLoop();
      if (failed(outerResult))
        return failure();
      auto [outer, outerBody] = *outerResult;
      
      // 2. Create the inner local iteration loop
      auto innerResult = CreateInnerLocalLoop(outerBody);
      if (failed(innerResult))
        return failure();
      auto inner = *innerResult;
  
      // 3. Clone and remap the loop body
      auto yieldValsResult = CloneAndRemapLoopBody(inner, outer.getInductionVar());
      if (failed(yieldValsResult))
        return failure();
      auto yieldVals = *yieldValsResult;
      
      // 4. Build the inner loop yield
      Block& innerBody = inner.getRegion().front();
      if (!innerBody.empty() && innerBody.back().hasTrait<OpTrait::IsTerminator>())
        innerBody.back().erase();
      op_builder_.setInsertionPointToEnd(&innerBody);
      op_builder_.create<scf::YieldOp>(loc_, yieldVals);
  
      // 5. Optimize the affine expressions
      OptimizeAffineExpressions(innerBody);
  
      // 6. Finalize the transformation and replace the original loop
      return FinalizeAndReplaceLoop(outer, inner);
    }
  
  
  private:
    mlir::scf::ForOp forOp_;          /// The original loop to be transformed
    OpBuilder op_builder_;             /// IR builder
    Location loc_;                     /// Source code location information
    TilingConfig tiling_config_;       /// Tiling configuration parameters
  };


/**
 * @brief Implement DiscoverNestedLoops
 */
FailureOr<SmallVector<Operation*>> TillingManager::DiscoverNestedLoops() {
  SmallVector<Operation*> loops;
  
  // 1. Find the outermost affine.for (no parent affine.for)
  affine::AffineForOp outermost = nullptr;
  func_.walk([&](affine::AffineForOp affineFor) {
    if (!outermost && !affineFor->getParentOfType<affine::AffineForOp>()) {
      outermost = affineFor;
    }
  });
  
  if (!outermost) {
    llvm::WithColor::error(llvm::errs()) 
        << "No outermost affine.for found in function\n";
    return failure();
  }
  loops.push_back(outermost.getOperation());
  
  // 2. Find the second affine.for in the body of the first affine.for
  affine::AffineForOp second = nullptr;
  outermost.getBody()->walk([&](affine::AffineForOp affineFor) {
    if (!second && affineFor != outermost) {
      second = affineFor;
    }
  });
  
  if (!second) {
    llvm::WithColor::error(llvm::errs()) 
        << "No second affine.for found in outermost affine.for body\n";
    return failure();
  }
  loops.push_back(second.getOperation());
  
  // 3. Skip intermediate affine.parallel, find the innermost scf.for
  scf::ForOp scfFor = nullptr;
  second.getBody()->walk([&](scf::ForOp f) {
    if (!scfFor) {
      scfFor = f;
    }
  });
  
  if (!scfFor) {
    llvm::WithColor::error(llvm::errs()) 
        << "No scf.for found in nested structure\n";
    return failure();
  }
  loops.push_back(scfFor.getOperation());
  
  return loops;
}

/**
 * @brief Implement ValidateLoopStructure
 */
LogicalResult TillingManager::ValidateLoopStructure(ArrayRef<Operation*> loops, ArrayRef<uint64_t> tileFactors) {
  if (loops.size() != tileFactors.size()) {
    llvm::WithColor::error(llvm::errs()) 
        << "Loop count (" << loops.size() << ") does not match tile factor count (" 
        << tileFactors.size() << ")\n";
    return failure();
  }
  
  // Check that the first two must be affine.for
  if (!isa<affine::AffineForOp>(loops[0]) || !isa<affine::AffineForOp>(loops[1])) {
    llvm::WithColor::error(llvm::errs()) 
        << "Expected first two loops to be affine.for\n";
    return failure();
  }
  
  // Check that the third must be scf.for
  if (!isa<scf::ForOp>(loops[2])) {
    llvm::WithColor::error(llvm::errs()) 
        << "Expected third loop to be scf.for\n";
    return failure();
  }
  
  // Build loopDescriptors_
  loopDescriptors_.clear();
  for (size_t i = 0; i < loops.size(); ++i) {
    if (auto affineFor = dyn_cast<affine::AffineForOp>(loops[i])) {
      auto desc = LoopDescriptor::FromAffineFor(affineFor, tileFactors[i]);
      if (failed(desc)) {
        llvm::WithColor::error(llvm::errs()) 
            << "Failed to create LoopDescriptor for affine.for at level " << i << "\n";
        return failure();
      }
      loopDescriptors_.push_back(*desc);
    } else if (auto scfFor = dyn_cast<scf::ForOp>(loops[i])) {
      auto desc = LoopDescriptor::FromScfFor(scfFor, tileFactors[i]);
      if (failed(desc)) {
        llvm::WithColor::error(llvm::errs()) 
            << "Failed to create LoopDescriptor for scf.for at level " << i << "\n";
        return failure();
      }
      loopDescriptors_.push_back(*desc);
    } else {
      llvm::WithColor::error(llvm::errs()) 
          << "Unknown loop type at level " << i << "\n";
      return failure();
    }
  }
  
  return success();
}

/**
 * @brief Implement SetupTiling
 */
LogicalResult TillingManager::SetupTiling(ArrayRef<uint64_t> tileFactors) {
  tileFactors_.clear();
  tileFactors_.append(tileFactors.begin(), tileFactors.end());
  
  // Automatically discover loop structure
  auto loopsResult = DiscoverNestedLoops();
  if (failed(loopsResult)) {
    return failure();
  }
  auto loops = *loopsResult;
  
  // Validate and build loopDescriptors_
  if (failed(ValidateLoopStructure(loops, tileFactors))) {
    return failure();
  }
  
  return success();
}

/**
 * @brief Implement TileTwoAffineFors - use MLIR built-in API
 */
FailureOr<SmallVector<affine::AffineForOp, 4>>
TillingManager::TileTwoAffineFors(const LoopDescriptor& outerDesc, 
                                   const LoopDescriptor& innerDesc,
                                   unsigned outerTileSize, 
                                   unsigned innerTileSize) {
  auto outerLoop = cast<affine::AffineForOp>(outerDesc.op);
  auto innerLoop = cast<affine::AffineForOp>(innerDesc.op);
  
  // Prepare loop array and tile size array
  SmallVector<affine::AffineForOp> loops = {outerLoop, innerLoop};
  SmallVector<unsigned> tileSizes = {outerTileSize, innerTileSize};
  
  // Call MLIR built-in API (note the third parameter is a pointer)
  SmallVector<affine::AffineForOp, 4> tiledNest;
  if (failed(affine::tilePerfectlyNested(loops, tileSizes, &tiledNest))) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to tile two affine.for loops using MLIR built-in API\n";
    return failure();
  }
  
  // tiledNest contains tiled loops, the order should be:
  // [outerTile, outerLocal, innerTile, innerLocal]
  // But need to validate the actual returned structure
  if (tiledNest.size() < 4) {
    llvm::WithColor::error(llvm::errs())
        << "Unexpected tiled nest structure: expected 4 loops, got " 
        << tiledNest.size() << "\n";
    return failure();
  }
  
  return tiledNest;
}

/**
 * @brief Compute multi-level loop tile factors
 * @param func 目标函数
 * @param memInfo L1 memory information
 * @param m outer affine.for tile factor (default 4)
 * @param n inner affine.for tile factor (default 4)
 * @param k innermost scf.for tile factor (default 4)
 * @return {m, n, k} three tile factors, or failure
 * 
 * TODO: Implement automatic calculation logic based on L1 size
 * Currently using the default values (all 4)
 */
static FailureOr<SmallVector<uint64_t, 3>> 
ComputeMultiLevelTileFactors(func::FuncOp func, const SingleMemoryInfo& memInfo,
                              uint64_t m = 4, uint64_t n = 4, uint64_t k = 4) {
  (void)func;      // Not used yet
  (void)memInfo;   // Not used yet
  
  SmallVector<uint64_t, 3> factors = {m, n, k};
  
  llvm::outs() << "[ComputeMultiLevelTileFactors] Using tile factors: "
               << "m=" << factors[0] << ", n=" << factors[1] << ", k=" << factors[2] << "\n";
  
  return factors;
}

/**
 * @brief Implement Transform main流程 - 两步 tiling
 * First step: use MLIR built-in API to tile the outermost two affine.for loops
 * Second step: use SingleLoopTillingManager to tile the innermost scf.for loop
 */
LogicalResult TillingManager::Transform() {
  if (loopDescriptors_.empty()) {
    llvm::WithColor::error(llvm::errs()) 
        << "No loop descriptors available. Call SetupTiling() first.\n";
    return failure();
  }
  
  // Validate loop structure: should be two affine.for and one scf.for
  if (loopDescriptors_.size() != 3) {
    llvm::WithColor::error(llvm::errs()) 
        << "Expected 3 loop descriptors (2 affine.for + 1 scf.for), got " 
        << loopDescriptors_.size() << "\n";
    return failure();
  }
  
  if (loopDescriptors_[0].type != LoopDescriptor::AFFINE_FOR ||
      loopDescriptors_[1].type != LoopDescriptor::AFFINE_FOR ||
      loopDescriptors_[2].type != LoopDescriptor::SCF_FOR) {
    llvm::WithColor::error(llvm::errs()) 
        << "Expected loop structure: affine.for, affine.for, scf.for\n";
    return failure();
  }

  // First step: use MLIR built-in API to tile the outermost two affine.for loops
  auto& outerDesc = loopDescriptors_[0];
  auto& innerAffineDesc = loopDescriptors_[1];
  
  auto tiledAffineResult = TileTwoAffineFors(
      outerDesc, innerAffineDesc, 
      static_cast<unsigned>(tileFactors_[0]), 
      static_cast<unsigned>(tileFactors_[1]));
  
  if (failed(tiledAffineResult)) {
    llvm::WithColor::error(llvm::errs()) 
        << "Failed to tile two affine.for loops using MLIR built-in API\n";
    return failure();
  }
  
  auto tiledAffineNest = *tiledAffineResult;
  // tiledAffineNest should contain 4 loops: [outerTile, outerLocal, innerTile, innerLocal]
  if (tiledAffineNest.size() != 4) {
    llvm::WithColor::error(llvm::errs()) 
        << "Unexpected tiled affine nest size: expected 4, got " 
        << tiledAffineNest.size() << "\n";
    return failure();
  }
  
  // MLIR built-in API has automatically handled IV remapping, the IVs in the original loops have been replaced
  // tiledAffineNest 包含：[outerTile, outerLocal, innerTile, innerLocal]
  auto innerLocalLoop = tiledAffineNest[3];
  
  // Get the body of the innermost affine local loop, this is the insertion point for the second step tiling
  Block* innermostAffineBody = innerLocalLoop.getBody();
  
  // Second step: find the scf.for in the innermost body after tiling, use SingleLoopTillingManager to tile
  scf::ForOp scfForToTile = nullptr;
  innermostAffineBody->walk([&](scf::ForOp scfFor) {
    if (!scfForToTile) {
      scfForToTile = scfFor;
    }
  });
  
  if (!scfForToTile) {
    llvm::WithColor::error(llvm::errs()) 
        << "No scf.for found in tiled affine nest for second step tiling\n";
    return failure();
  }
  
  // Validate the found scf.for is the one we expect
  auto& scfDesc = loopDescriptors_[2];
  if (scfForToTile.getOperation() != scfDesc.op) {
    llvm::WithColor::warning(llvm::errs()) 
        << "Found scf.for does not match expected one, proceeding anyway\n";
  }
  
  // Compute the tile factor for the scf.for
  uint64_t scfTileFactor = tileFactors_[2];
  
  // Compute trip count
  auto tripOpt = getTripCount(scfForToTile);
  if (!tripOpt) {
    llvm::WithColor::error(llvm::errs()) 
        << "Scf.for trip count is not statically provable\n";
    return failure();
  }
  
  // Check divisibility
  if (*tripOpt % static_cast<int64_t>(scfTileFactor) != 0) {
    llvm::WithColor::error(llvm::errs()) 
        << "Scf.for trip count (" << *tripOpt 
        << ") not divisible by tile factor (" << scfTileFactor << ")\n";
    return failure();
  }
  
  // Use SingleLoopTillingManager to tile
  SingleLoopTillingManager scfTilingManager(scfForToTile, *tripOpt, scfTileFactor);
  if (failed(scfTilingManager.Transform())) {
    llvm::WithColor::error(llvm::errs()) 
        << "Failed to tile scf.for using SingleLoopTillingManager\n";
    return failure();
  }
  
  // Delete the original outermost affine.for (MLIR built-in API should have already handled it, but for safety check)
  // Actually MLIR built-in API should have already deleted the original loop, so this might not need to be manually deleted
  // But for clarity, we check if the original loop still exists
  if (outerDesc.op->getParentOp()) {
    // If the original loop still exists, it means MLIR API did not delete it, we need to manually delete it
    outerDesc.op->erase();
  }
  
  return success();
}


class TileScfForToL1Pass
    : public PassWrapper<TileScfForToL1Pass, OperationPass<ModuleOp>> {
private:
  template <typename T>
  void TMD_RETURN_ON_FAILURE(const T& result, StringRef reason) {
    if (!result) {
      llvm::WithColor::error(llvm::errs()) << "[Error] " << reason << "\n";
      signalPassFailure();
      return;
    }
  }

  template <typename T>
  void TMD_RETURN_ON_FAILURE(const FailureOr<T>& result, StringRef reason) {
    if (failed(result)) {
      llvm::WithColor::error(llvm::errs()) << "[Error] " << reason << "\n";
      signalPassFailure();
      return;
    }
  }
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
    TMD_RETURN_ON_FAILURE(memInfoOr, "Failed to require single memory");
    SingleMemoryInfo memInfo = *memInfoOr;

    // Enforce fully bufferized IR: no tensor types anywhere.
    TMD_RETURN_ON_FAILURE(IsFullyBufferized(module), "IR is not fully bufferized");

    bool changedAny = false;

    module.walk([&](func::FuncOp func) {
      // Try using func-level multi-level tiling
      llvm::outs() << "[TileScfForToL1Pass] Processing function: " << func.getSymName() << "\n";
      
      // Compute multi-level tile factors
      auto tileFactorsResult = ComputeMultiLevelTileFactors(func, memInfo);
      
      if (succeeded(tileFactorsResult)) {
        auto tileFactors = *tileFactorsResult;
        
        // Create func-level TillingManager
        TillingManager manager(func);
        
        // Configure tiling scheme
        if (succeeded(manager.SetupTiling(tileFactors))) {
          // Execute transformation
          if (succeeded(manager.Transform())) {
            llvm::outs() << "[TileScfForToL1Pass] Successfully applied multi-level tiling to function: " 
                         << func.getSymName() << "\n";
            changedAny = true;
            return;  // Successfully completed multi-level tiling, skip fallback
          } else {
            llvm::outs() << "[TileScfForToL1Pass] Multi-level tiling Transform() failed, falling back to single-loop tiling\n";
          }
        } else {
          llvm::outs() << "[TileScfForToL1Pass] Multi-level tiling SetupTiling() failed, falling back to single-loop tiling\n";
        }
      } else {
        llvm::outs() << "[TileScfForToL1Pass] ComputeMultiLevelTileFactors() failed, falling back to single-loop tiling\n";
      }
      
      // Fallback: use the original single-loop tiling logic
      llvm::outs() << "[TileScfForToL1Pass] Applying single-loop tiling to function: " << func.getSymName() << "\n";
      
      SmallVector<scf::ForOp, 8> loops;
      func.walk([&](scf::ForOp f) { loops.push_back(f); });

      for (auto& forOp : loops) {
        auto perIterBytes = ComputePerIterMemoryBytes(forOp, memInfo.label);
        TMD_RETURN_ON_FAILURE(perIterBytes, "Failed to compute per-iteration memory bytes");
        uint64_t perIterBytesVal = *perIterBytes;

        if (perIterBytesVal == 0)
          continue; // nothing to tile

        uint64_t maxTiles = memInfo.sizeBytes / perIterBytesVal;
        TMD_RETURN_ON_FAILURE(maxTiles, (Twine("Per-iteration memory exceeds L1 size in function '") + 
        func.getSymName() + "'").str());
        
        uint64_t tileFactor = largestPowerOfTwoLE(maxTiles);
        TMD_RETURN_ON_FAILURE(tileFactor, (Twine("Failed to compute tile factor in function '") + 
        func.getSymName() + "'").str());

        // Prove trip count and divisibility.
        auto trip = getTripCount(forOp);
        TMD_RETURN_ON_FAILURE(trip, (Twine("Trip count is not statically provable for scf.for in function '") + func.getSymName() + "'").str());
        TMD_RETURN_ON_FAILURE(!(*trip % static_cast<int64_t>(tileFactor)), (Twine("Trip count not divisible by tile factor in function '") + func.getSymName() + "'").str());

        // Start rewriting
        SingleLoopTillingManager manager(forOp, *trip, tileFactor);
        if (failed(manager.Transform())) {
          signalPassFailure();  // Handled by the caller
          return;
        }
        changedAny = true;
      }
    });

    (void)changedAny;
  }
};


std::unique_ptr<mlir::Pass> createTileScfForToL1Pass() {
  return std::make_unique<TileScfForToL1Pass>();
}

void registerTileScfForToL1Pass() { PassRegistration<TileScfForToL1Pass>(); }

} // namespace passes
} // namespace tmd
