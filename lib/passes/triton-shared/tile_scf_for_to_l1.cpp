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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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
 * @brief Extract M, N, K dimensions from a linalg.matmul operation.
 * @param matmul The matmul operation
 * @return A tuple of (M, N, K) dimensions, or failure if dimensions cannot be extracted
 */
static FailureOr<std::tuple<int64_t, int64_t, int64_t>>
extractMatmulDimensions(linalg::MatmulOp matmul) {
  // Get operands: A (M x K), B (K x N), C (M x N)
  if (matmul.getNumDpsInputs() != 2 || matmul.getNumDpsInits() != 1)
    return failure();

  Value aOperand = matmul.getDpsInputOperand(0)->get();
  Value bOperand = matmul.getDpsInputOperand(1)->get();
  Value cOperand = matmul.getDpsInitOperand(0)->get();

  auto aType = llvm::dyn_cast<MemRefType>(aOperand.getType());
  auto bType = llvm::dyn_cast<MemRefType>(bOperand.getType());
  auto cType = llvm::dyn_cast<MemRefType>(cOperand.getType());

  if (!aType || !bType || !cType)
    return failure();

  if (!aType.hasStaticShape() || !bType.hasStaticShape() ||
      !cType.hasStaticShape())
    return failure();

  // A is M x K, B is K x N, C is M x N
  auto aShape = aType.getShape();
  auto bShape = bType.getShape();
  auto cShape = cType.getShape();

  if (aShape.size() != 2 || bShape.size() != 2 || cShape.size() != 2)
    return failure();

  int64_t M = aShape[0]; // A's first dimension
  int64_t K = aShape[1]; // A's second dimension (should match B's first)
  int64_t N = bShape[1]; // B's second dimension

  // Verify consistency
  if (aShape[1] != bShape[0] || aShape[0] != cShape[0] ||
      bShape[1] != cShape[1]) {
    return failure();
  }

  return std::make_tuple(M, N, K);
}

/**
 * @brief Prefetch A and B matrix blocks to L1 memory.
 * @param forOp The original inner scf.for loop
 * @param insertionPoint The operation before which to insert prefetch code
 * @param aOperand The A matrix operand (contains memory address info: base, offset, shape, strides)
 * @param bOperand The B matrix operand (contains memory address info: base, offset, shape, strides)
 * @param M The M dimension of the block
 * @param N The N dimension of the block
 * @param K The K dimension of the block
 * @param memoryLabel The memory label (default "L1")
 * @return A pair of (aL1Block, bL1Block) - the allocated L1 memory blocks
 */
static std::pair<Value, Value> prefetchABBlocksToL1(
    scf::ForOp forOp, Operation *insertionPoint, Value aOperand, Value bOperand,
    int64_t M, int64_t N, int64_t K, StringRef memoryLabel = "L1") {
  Location loc = forOp.getLoc();
  OpBuilder builder(forOp);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(insertionPoint);

  // Get element type from aOperand
  auto aType = llvm::cast<MemRefType>(aOperand.getType());
  Type elemType = aType.getElementType();

  // Calculate byte size for L1 allocation
  // Get element bit width
  auto getElementBitWidth = [](Type elemTy) -> uint64_t {
    if (auto i = llvm::dyn_cast<IntegerType>(elemTy))
      return i.getWidth();
    if (auto f = llvm::dyn_cast<FloatType>(elemTy))
      return f.getWidth();
    if (auto idx = llvm::dyn_cast<IndexType>(elemTy))
      return 64u; // Assume 64-bit index
    return 32u; // Default to 32 bits
  };

  auto getByteSize = [&](int64_t rows, int64_t cols) -> uint64_t {
    uint64_t elems = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);
    uint64_t bitsPerElem = getElementBitWidth(elemType);
    uint64_t bytesPerElem = (bitsPerElem + 7) / 8;
    return elems * bytesPerElem;
  };

  auto ctx = builder.getContext();

  // Create L1 memory allocation for A block (M x K)
  uint64_t aBlockBytes = getByteSize(M, K);
  MemRefType aBlockType = MemRefType::get({M, K}, elemType);
  NamedAttribute allocAttr = NamedAttribute(
      StringAttr::get(ctx, "tmd.alloc"),
      DictionaryAttr::get(ctx, {
          NamedAttribute(StringAttr::get(ctx, "local"), BoolAttr::get(ctx, true)),
          NamedAttribute(StringAttr::get(ctx, "memory_name"),
                       StringAttr::get(ctx, memoryLabel)),
          NamedAttribute(StringAttr::get(ctx, "size"),
                       IntegerAttr::get(IntegerType::get(ctx, 64), aBlockBytes))}));
  Value aL1Block = builder.create<memref::AllocOp>(
      loc, aBlockType, ValueRange{}, ArrayRef<NamedAttribute>{allocAttr});

  // Create L1 memory allocation for B block (K x N)
  uint64_t bBlockBytes = getByteSize(K, N);
  MemRefType bBlockType = MemRefType::get({K, N}, elemType);
  NamedAttribute bAllocAttr = NamedAttribute(
      StringAttr::get(ctx, "tmd.alloc"),
      DictionaryAttr::get(ctx, {
          NamedAttribute(StringAttr::get(ctx, "local"), BoolAttr::get(ctx, true)),
          NamedAttribute(StringAttr::get(ctx, "memory_name"),
                       StringAttr::get(ctx, memoryLabel)),
          NamedAttribute(StringAttr::get(ctx, "size"),
                       IntegerAttr::get(IntegerType::get(ctx, 64), bBlockBytes))}));
  Value bL1Block = builder.create<memref::AllocOp>(
      loc, bBlockType, ValueRange{}, ArrayRef<NamedAttribute>{bAllocAttr});

  // Copy A and B blocks to L1
  // Find the original memref sources (they might be from reinterpret_cast)
  Value aSource = aOperand;
  Value bSource = bOperand;

  // If operands come from reinterpret_cast, get the source
  if (auto aCast = aOperand.getDefiningOp<memref::ReinterpretCastOp>()) {
    aSource = aCast.getSource();
  }
  if (auto bCast = bOperand.getDefiningOp<memref::ReinterpretCastOp>()) {
    bSource = bCast.getSource();
  }

  // Check if aOperand is already M x K
  auto aOperandType = llvm::cast<MemRefType>(aOperand.getType());
  if (aOperandType.getShape()[0] == M && aOperandType.getShape()[1] == K) {
    // Already the right size
    builder.create<memref::CopyOp>(loc, aOperand, aL1Block);
  } else {
    // Need to extract from source - this is complex and depends on the original layout
    // For now, we'll assume aOperand can be used directly
    // In a full implementation, we'd need to create a new reinterpret_cast
    builder.create<memref::CopyOp>(loc, aOperand, aL1Block);
  }

  // Same for B block (K x N)
  auto bOperandType = llvm::cast<MemRefType>(bOperand.getType());
  if (bOperandType.getShape()[0] == K && bOperandType.getShape()[1] == N) {
    builder.create<memref::CopyOp>(loc, bOperand, bL1Block);
  } else {
    builder.create<memref::CopyOp>(loc, bOperand, bL1Block);
  }

  return {aL1Block, bL1Block};
}

class TillingManager {
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
     * @brief Constructor for TillingManager
     * @param forOp The original scf.for loop to be transformed
     * @param trip The static iteration count of the loop
     * @param tileFactor The tile size (must divide trip)
     */
    TillingManager(mlir::scf::ForOp& forOp, int64_t trip, uint64_t tileFactor) 
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

    /**
     * @brief Tile a matmul operation inside the scf.for loop with fixed tile size.
     * @param matmulOp The matmul operation to tile
     * @param tileSize The tile size for each dimension (default 32)
     * @return success on success, failure otherwise
     */
    LogicalResult TileMatmulInLoop(linalg::MatmulOp matmulOp, int64_t tileSize) {
      // Extract dimensions
      auto dimsOr = extractMatmulDimensions(matmulOp);
      if (failed(dimsOr))
        return failure();
      auto [M, N, K] = *dimsOr;

      // Verify divisibility
      if (M % tileSize != 0 || N % tileSize != 0 || K % tileSize != 0) {
        llvm::WithColor::error(llvm::errs())
            << "Matmul dimensions (" << M << ", " << N << ", " << K
            << ") are not divisible by tile size " << tileSize << "\n";
        return failure();
      }

      int64_t numMTiles = M / tileSize;
      int64_t numNTiles = N / tileSize;
      int64_t numKTiles = K / tileSize;

      // Get matmul operands
      Value aOperand = matmulOp.getDpsInputOperand(0)->get();
      Value bOperand = matmulOp.getDpsInputOperand(1)->get();
      Value cOperand = matmulOp.getDpsInitOperand(0)->get();

      OpBuilder::InsertionGuard guard(op_builder_);

      // Find matmul in scf.for body and get its position
      Block *scfForBody = &forOp_.getRegion().front();
      Operation *matmulOpPtr = matmulOp.getOperation();
      
      // Get scf.for's iter_arg (C buffer)
      Value cIterArg = nullptr;
      if (forOp_.getNumRegionIterArgs() > 0) {
        cIterArg = scfForBody->getArgument(1); // First iter_arg is at index 1
      } else {
        llvm::WithColor::error(llvm::errs())
            << "scf.for must have at least one iter_arg for C buffer\n";
        return failure();
      }

      // Get the original A and B operands from the matmul
      // These are likely from reinterpret_cast operations in the scf.for body
      
      // Step 1: Load A and B blocks to L1 (before matmul, not at start of scf.for body)
      // Insert before matmul to ensure aOperand and bOperand are already defined
      // NOTE: Prefetch is already done in bufferization phase, so we reuse aOperand and bOperand
      // which are already L1 memrefs (%alloc_2 and %alloc_4 in after_bufferization.mlir)
      // auto [aL1Block, bL1Block] = prefetchABBlocksToL1(
      //     forOp_, matmulOpPtr, aOperand, bOperand, M, N, K, "L1");
      
      // Reuse the existing L1 blocks from aOperand and bOperand
      // These are already prefetched to L1 in the bufferization phase
      Value aL1Block = aOperand;
      Value bL1Block = bOperand;
      
      // Create constants (these will be used in nested loops, so create them early)
      op_builder_.setInsertionPoint(matmulOpPtr);
      Value tileSizeVal = op_builder_.create<arith::ConstantIndexOp>(loc_, tileSize);
      Value strideOne = op_builder_.create<arith::ConstantIndexOp>(loc_, 1);
      
      // Get element type and context for later use
      auto aType = llvm::cast<MemRefType>(aOperand.getType());
      Type elemType = aType.getElementType();
      auto ctx = op_builder_.getContext();
      
      // Step 2: Create 3 nested affine.for loops (m, n, k) to replace matmul
      // Insert after L1 allocations (aL1Block and bL1Block are now defined)
      // The matmul will be replaced, so we insert the loops at matmul's position
      // Note: aL1Block and bL1Block are defined before matmul, so they're accessible
      // in all nested loops (m, n, k) that we create here
      op_builder_.setInsertionPoint(matmulOpPtr);
      
      // Create m loop (outermost)
      AffineMap mLbMap = AffineMap::getConstantMap(0, ctx);
      AffineMap mUbMap = AffineMap::getConstantMap(numMTiles, ctx);
      auto mLoop = affine::AffineForOp::create(
          op_builder_, loc_, ValueRange{}, mLbMap, ValueRange{}, mUbMap, 1);
      Block *mLoopBody = mLoop.getBody();
      if (mLoopBody->empty() || !isa<affine::AffineYieldOp>(mLoopBody->back())) {
        OpBuilder termBuilder = OpBuilder::atBlockEnd(mLoopBody);
        termBuilder.create<affine::AffineYieldOp>(loc_);
      }
      Value mIV = mLoopBody->getArgument(0);
      
      // Create n loop (inside m loop)
      op_builder_.setInsertionPoint(mLoopBody, std::prev(mLoopBody->end()));
      AffineMap nLbMap = AffineMap::getConstantMap(0, ctx);
      AffineMap nUbMap = AffineMap::getConstantMap(numNTiles, ctx);
      auto nLoop = affine::AffineForOp::create(
          op_builder_, loc_, ValueRange{}, nLbMap, ValueRange{}, nUbMap, 1);
      Block *nLoopBody = nLoop.getBody();
      if (nLoopBody->empty() || !isa<affine::AffineYieldOp>(nLoopBody->back())) {
        OpBuilder termBuilder = OpBuilder::atBlockEnd(nLoopBody);
        termBuilder.create<affine::AffineYieldOp>(loc_);
      }
      Value nIV = nLoopBody->getArgument(0);
      
      // Initialize C tile (32x32) from iter_arg
      op_builder_.setInsertionPointToStart(nLoopBody);
      
      // Calculate offsets for C tile extraction
      AffineExpr d0 = getAffineDimExpr(0, ctx);
      AffineExpr c32 = getAffineConstantExpr(tileSize, ctx);
      AffineMap mOffsetMap = AffineMap::get(1, 0, d0 * c32, ctx);
      AffineMap nOffsetMap = AffineMap::get(1, 0, d0 * c32, ctx);
      
      Value mOffset = op_builder_.create<affine::AffineApplyOp>(
          loc_, mOffsetMap, ValueRange{mIV}).getResult();
      Value nOffset = op_builder_.create<affine::AffineApplyOp>(
          loc_, nOffsetMap, ValueRange{nIV}).getResult();
      
      // Create C tile by subview from iter_arg
      MemRefType cTileType = MemRefType::get({tileSize, tileSize}, elemType);
      Value cTile = op_builder_.create<memref::AllocOp>(
          loc_, cTileType, ValueRange{}, ArrayRef<NamedAttribute>{});
      
      // Initialize C tile from iter_arg (extract 32x32 region)
      // Use SubViewOp with dynamic offsets
      // tileSizeVal and strideOne are defined before n loop, so accessible here
      SmallVector<Value> offsets = {mOffset, nOffset};
      SmallVector<Value> sizes = {tileSizeVal, tileSizeVal};
      SmallVector<Value> strides = {strideOne, strideOne};
      
      // Create subview with dynamic offsets/sizes
      auto cTileView = op_builder_.create<memref::SubViewOp>(
          loc_, cIterArg, offsets, sizes, strides);
      op_builder_.create<memref::CopyOp>(loc_, cTileView, cTile);
      
      // Create k loop (inside n loop)
      op_builder_.setInsertionPoint(nLoopBody, std::prev(nLoopBody->end()));
      AffineMap kLbMap = AffineMap::getConstantMap(0, ctx);
      AffineMap kUbMap = AffineMap::getConstantMap(numKTiles, ctx);
      auto kLoop = affine::AffineForOp::create(
          op_builder_, loc_, ValueRange{}, kLbMap, ValueRange{}, kUbMap, 1);
      Block *kLoopBody = kLoop.getBody();
      if (kLoopBody->empty() || !isa<affine::AffineYieldOp>(kLoopBody->back())) {
        OpBuilder termBuilder = OpBuilder::atBlockEnd(kLoopBody);
        termBuilder.create<affine::AffineYieldOp>(loc_);
      }
      Value kIV = kLoopBody->getArgument(0);
      
      // In k loop: extract 32x32 tiles from L1 blocks and compute
      // Note: mOffset, nOffset, tileSizeVal, strideOne are defined in n loop, 
      // so they're accessible in k loop (which is inside n loop)
      op_builder_.setInsertionPointToStart(kLoopBody);
      
      // Calculate k offset (kIV is from k loop, so it's accessible)
      AffineMap kOffsetMap = AffineMap::get(1, 0, d0 * c32, ctx);
      Value kOffset = op_builder_.create<affine::AffineApplyOp>(
          loc_, kOffsetMap, ValueRange{kIV}).getResult();
      
      // Extract A tile (32x32) from L1 A block
      // All values (mOffset, tileSizeVal, strideOne) are from n loop, accessible here
      SmallVector<Value> aOffsets = {mOffset, kOffset};
      SmallVector<Value> aSizes = {tileSizeVal, tileSizeVal};
      SmallVector<Value> aStrides = {strideOne, strideOne};
      auto aTileView = op_builder_.create<memref::SubViewOp>(
          loc_, aL1Block, aOffsets, aSizes, aStrides);
      
      // Extract B tile (32x32) from L1 B block
      // All values (nOffset, tileSizeVal, strideOne) are from n loop, accessible here
      SmallVector<Value> bOffsets = {kOffset, nOffset};
      SmallVector<Value> bSizes = {tileSizeVal, tileSizeVal};
      SmallVector<Value> bStrides = {strideOne, strideOne};
      auto bTileView = op_builder_.create<memref::SubViewOp>(
          loc_, bL1Block, bOffsets, bSizes, bStrides);
      
      // Execute 32x32 matrix multiplication (accumulates into cTile)
      // Insert before k loop terminator
      Operation *kTerminator = kLoopBody->getTerminator();
      op_builder_.setInsertionPoint(kTerminator);
      op_builder_.create<linalg::MatmulOp>(
          loc_, ValueRange{aTileView, bTileView}, ValueRange{cTile});
      
      // Step 3: Write back C tile to iter_arg (after n loop, before n loop terminator)
      op_builder_.setInsertionPoint(nLoopBody, std::prev(nLoopBody->end()));
      op_builder_.create<memref::CopyOp>(loc_, cTile, cTileView);
      
      // Replace the original matmul with the new structure
      matmulOpPtr->erase();

      return success();
    }

  private:
    mlir::scf::ForOp forOp_;          ///< The original loop to be transformed
    OpBuilder op_builder_;             ///< IR builder
    Location loc_;                     ///< Source code location information
    TilingConfig tiling_config_;       ///< Tiling configuration parameters
  };
  

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
                linalg::LinalgDialect, scf::SCFDialect, memref::MemRefDialect>();
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
      SmallVector<scf::ForOp, 8> loops;
      func.walk([&](scf::ForOp f) { loops.push_back(f); });

      for (auto& forOp : loops) {
        // Check if this loop contains a matmul operation
        linalg::MatmulOp matmulOp = nullptr;
        for (Operation &op : forOp.getBody()->getOperations()) {
          if (auto mm = llvm::dyn_cast<linalg::MatmulOp>(&op)) {
            matmulOp = mm;
            break;
          }
        }

        // If matmul found, apply matmul tiling with fixed tile size
        if (matmulOp) {
          TillingManager manager(forOp, 0, 1); // Dummy values, not used for matmul tiling
          if (failed(manager.TileMatmulInLoop(matmulOp, 32))) {
            signalPassFailure();
            return;
          }
          changedAny = true;
          continue;
        }

        // Otherwise, apply regular L1 tiling
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
        TillingManager manager(forOp, *trip, tileFactor);
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
