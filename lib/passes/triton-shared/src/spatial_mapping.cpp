/**
 * @file spatial_mapping.cpp
 * @brief Implementation for spatial mapping discovery and enumeration.
 * @details
 * Provided functionality
 * - `collectSpatialDims`: parse `df.spatial_dim` declarations from the DF
 *   module into simple name/size pairs.
 * - `mapSpatialDimsToAffine`: greedily tile outermost `affine.parallel` loops
 *   with factors derived from spatial dimension sizes and mark inner loops with
 *   `loom.mapped_to`.
 * - `enumerateSpatialMappings`: produce clones for all unique bucketings and
 *   per-iterator permutations of spatial dims over the first outermost
 *   `affine.parallel` in each function.
 * - `enumerateTritonSharedSpatialMappings`: Triton-specific enumeration that
 *   associates used program grid indices {x,y,z} with hardware spatial dims and
 *   records the mapping as function attributes; also rewrites the legacy grid
 *   ABI to a compact form based on spatial IDs.
 *
 * Limitations
 * - Currently enumerates only the first outermost `affine.parallel` per
 *   function. Extending to multiple regions requires nested enumeration.
 * - Dynamic spatial sizes are treated as factor=1 during tiling.
 */

#include "spatial_mapping.h"
#include "affine_tile.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/IRMapping.h"
#include "affine_parallel_to_for.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <algorithm>
#include <numeric>

#include "DataflowDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "DataflowTypes.h.inc"
#define GET_OP_CLASSES
#include "DataflowOps.h.inc"

using namespace mlir;

namespace loom_affine {

static LogicalResult GetSpatialDimInfo(loom::df::SpatialDimOp sdOp, llvm::SmallVector<loom_affine::SpatialDimInfo>& dimVec) {
  loom_affine::SpatialDimInfo info;
  // Read declared name and size from the op properties.
  // Use getSymNameAttr() since we changed the attribute to sym_name for Symbol trait
  if (auto nameAttr = sdOp.getSymNameAttr()) {
    info.name = nameAttr.getValue().str();
    // Use the symbol name for SymbolRefAttr
    info.symbolName = nameAttr.getValue().str();
  } else {
    info.name = "dim";
    info.symbolName = "dim";
  }
  uint64_t sz = sdOp.getSize();
  if (sz > 0)
    info.size = static_cast<int64_t>(sz);
  else
    info.size = std::nullopt;
  dimVec.push_back(std::move(info));
  return success();
}


static std::pair<bool, bool> AnalyzeInterconnectDirection(AffineMap map) {
  if (map.getNumResults() < 2) return {false, false};

  bool d0Connected = false;
  bool d1Connected = false;
  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    AffineExpr expr = map.getResult(i);

    if (i == 0 && expr != getAffineDimExpr(0, map.getContext())) d0Connected = true;
    if (i == 1 && expr != getAffineDimExpr(1, map.getContext())) d1Connected = true;
  }

  return {d0Connected, d1Connected};
}


LogicalResult GetHardwareInfoForExploration(mlir::ModuleOp dfModule, 
  HardwareInfo &hardwareInfo) {
  bool res = false;
  bool d0Connected = false;
  bool d1Connected = false;
  dfModule.walk([&](Operation *op) {
    if (auto sd = dyn_cast<loom::df::SpatialDimOp>(op)) {
      res = res || failed(GetSpatialDimInfo(sd, hardwareInfo.spatialDimInfoVec));
    }
    else if (auto ic = dyn_cast<loom::df::InterconnectsOp>(op)) {
      AffineMap map = ic.getMapAttr().getValue();
      auto [x, y] = AnalyzeInterconnectDirection(map);
      if (x && !y) {
        d0Connected = true;
      }
      if (y && !x) {
        d1Connected = true;
      }
      res = res || x || y;
    }
  });
  hardwareInfo.hasBidirInterconnect = d0Connected && d1Connected;
  return success(res);
}


// ------------------------------------------------------------
// Helper functions

/**
 * \brief Compose and canonicalize all affine.apply operations in a function.
 *
 * This fuses chains of affine.apply so that no affine.apply consumes the
 * result of another affine.apply, eliminating temporaries like `%4` used only
 * by a subsequent affine.apply. It also simplifies maps and operands.
 */
static void composeAndCanonicalizeAffineApplies(func::FuncOp func) {
  SmallVector<affine::AffineApplyOp> applies;
  func.walk([&](affine::AffineApplyOp op) { applies.push_back(op); });
  // Process in program order to maximize folding as we go.
  for (affine::AffineApplyOp op : applies) {
    OpBuilder b(op);
    AffineMap map = op.getAffineMap();
    SmallVector<Value> operands(op.getOperands().begin(),
                                op.getOperands().end());
    // Compose nested affine.apply producers and simplify.
    affine::fullyComposeAffineMapAndOperands(&map, &operands);
    affine::canonicalizeMapAndOperands(&map, &operands);
    // If nothing changed, continue.
    bool sameMap = (map == op.getAffineMap());
    bool sameOperands =
        operands.size() == op.getNumOperands() &&
        std::equal(operands.begin(), operands.end(), op.getOperands().begin());
    if (sameMap && sameOperands)
      continue;
    auto newOp = b.create<affine::AffineApplyOp>(op.getLoc(), map, operands);
    op.replaceAllUsesWith(newOp.getResult());
    op.erase();
  }

  // Clean up any trivially dead ops introduced by the rewrites (e.g.,
  // affine.applys or arithmetic ops that became unused).
  SmallVector<Operation *> toErase;
  func.walk([&](Operation *op) {
    if (mlir::isOpTriviallyDead(op))
      toErase.push_back(op);
  });
  for (Operation *op : toErase)
    op->erase();
}

/**
 * @brief Recursively enumerate all possible bucketing of dimensions into parallel iterators.
 * 
 * @param dimIdx The current dimension index.
 * @param numDims The total number of dimensions.
 * @param currentBuckets The current bucketing.
 * @param out The result.
 */
static void EnumerateBucketingRec(unsigned dimIdx, unsigned numDims,
  loom_affine::DimBuckets& currentBuckets, llvm::SmallVector<loom_affine::DimBuckets>& out) {
  // If we have assigned all dimensions, add the current bucketing to the result.
  if (dimIdx == numDims) {
    out.push_back(currentBuckets);
    return;
  }
  // For each iterator, add the current dimension to the bucket and recurse.
  for (unsigned it = 0; it < currentBuckets.size(); ++it) {
    currentBuckets[it].push_back(dimIdx);
    EnumerateBucketingRec(dimIdx + 1, numDims, currentBuckets, out);
    currentBuckets[it].pop_back();
  }
}


/**
 * @brief Generate all possible bucketing of dimensions into parallel iterators by calling EnumerateBucketingRec.
 * 
 * @param numParelleIter The number of parallel iterators.
 * @param numDims The total number of dimensions.
 * @return The result.
 */
static llvm::SmallVector<loom_affine::DimBuckets> GenerateAllPossibleParallelBuckets(unsigned numParelleIter, unsigned numDims) {
  llvm::SmallVector<loom_affine::DimBuckets> bucketing_results;
  loom_affine::DimBuckets currentBuckets(numParelleIter);
  EnumerateBucketingRec(0, numDims, currentBuckets, bucketing_results);
  return bucketing_results;
}


/**
 * @brief Generate all possible mappings by performing a Cartesian product of each iterator's dims in the permuted bucketing.
 * 
 * @param iterIdx The current iterator index.
 * @param current The current bucketing.
 * @param bucketsPerIter The permuted bucketing for each iterator.
 * @param out The result.
 */
static void CartesianProductOfBuckets(unsigned iterIdx,
  loom_affine::DimBuckets &current,
  const llvm::SmallVector<loom_affine::DimBuckets> &bucketsPerIter,
  llvm::SmallVector<loom_affine::DimBuckets> &out) {
  // If we have processed all iterators, add the current bucketing to the result.
  if (iterIdx == current.size()) {
      out.push_back(current);
      return;
  }

  const auto &buckets = bucketsPerIter[iterIdx];
  auto saved = current[iterIdx];

  if (buckets.empty()) {
    current[iterIdx].clear();
    CartesianProductOfBuckets(iterIdx + 1, current, bucketsPerIter, out);
    current[iterIdx] = saved;
    return;
  }

  for (const auto &dims : buckets) {
    current[iterIdx] = dims;
    CartesianProductOfBuckets(iterIdx + 1, current, bucketsPerIter, out);
  }

  current[iterIdx] = saved;
}


/**
 * @brief Generate all possible mappings by calling CartesianProductOfBuckets.
 * 
 * @param permutedBucketsPerIter The permuted bucketing for each iterator.
 * @return The result.
 */
static llvm::SmallVector<loom_affine::DimBuckets> GenerateAllPossibleMappings(const llvm::SmallVector<loom_affine::DimBuckets>& permutedBucketsPerIter) {
  llvm::SmallVector<loom_affine::DimBuckets> result;
  loom_affine::DimBuckets currentBuckets(permutedBucketsPerIter.size());
  CartesianProductOfBuckets(0, currentBuckets, permutedBucketsPerIter, result);
  return result;
}


/**
 * @brief Permute the dimensions in each iterator's bucket.
 * 
 * @param baseBuckets The base bucketing.
 * @param skipPermutation Whether to skip permutation if the number of dimensions is 2 and there is a bidirectional interconnect.
 * @return The complete mappings.
 */
static llvm::SmallVector<loom_affine::DimBuckets> PermuteBucket(const loom_affine::DimBuckets& baseBuckets, const HardwareInfo& hardwareInfo) {
  const unsigned numIters = static_cast<unsigned>(baseBuckets.size());
  llvm::SmallVector<loom_affine::DimBuckets> permutedBucketsPerIter(numIters);

  for (unsigned it = 0; it < numIters; ++it) {
    SmallVector<unsigned> dims = baseBuckets[it];
    if (dims.size() <= 1 
        || (dims.size() == hardwareInfo.spatialDimInfoVec.size() && hardwareInfo.skipPermutation())) {
      permutedBucketsPerIter[it].push_back(dims);
    } else {
      std::sort(dims.begin(), dims.end());
      do {
        permutedBucketsPerIter[it].push_back(dims);
      } while (std::next_permutation(dims.begin(), dims.end()));
    }
  }

  return GenerateAllPossibleMappings(permutedBucketsPerIter);
}


/**
 * @brief Get the outermost parallel iterator in a function.
 * 
 * @param func The function to get the outermost parallel iterator from.
 * @return The outermost parallel iterator.
 */
static affine::AffineParallelOp getOutermostParallel(func::FuncOp func) {
  affine::AffineParallelOp result = nullptr;
  func.walk([&](affine::AffineParallelOp par) {
    if (!par->getParentOfType<affine::AffineParallelOp>() && !result)
      result = par;
  });
  return result;
}


/**
 * @brief Apply a tiling mapping to a kernel function.
 * 
 * @param func The function to apply the tiling mapping to.
 * @param mapping The tiling mapping to apply.
 * @param dims The dimensions to map.
 * @param suffix The suffix to add to the function name.
 * @return success if the mapping is applied successfully, failure otherwise.
 */
static LogicalResult applyMappingToFunction(func::FuncOp func,
                                            const loom_affine::DimBuckets &mapping,
                                            const llvm::SmallVector<loom_affine::SpatialDimInfo>& dims,
                                            affine::AffineParallelOp &tar_forOp, std::string &suffix) {
  suffix.clear();

  MLIRContext *ctx = func.getContext();
  const unsigned numIter = static_cast<unsigned>(mapping.size());
  for (unsigned iterIdx = 0; iterIdx < numIter; ++iterIdx) {
    for (unsigned dimIdx : mapping[iterIdx]) {
      const auto &sd = dims[dimIdx];
      int64_t factor = sd.size.value_or(1);
      loom_affine::TiledParallels tiled_parallels{};
      if (failed(tileAffineParallel(tar_forOp, factor, iterIdx, tiled_parallels)))
        return failure();
      // Use SymbolRefAttr to reference the df.spatial_dim operation
      // Use the symbolName from SpatialDimInfo, defaulting to "dim" if empty
      StringRef symbolName = sd.symbolName.empty() ? StringRef("dim") : StringRef(sd.symbolName);
      tiled_parallels.tiled_new_->setAttr("loom.mapped_to",
                        SymbolRefAttr::get(ctx, symbolName));
      if (!suffix.empty())
        suffix += "_";
      suffix += "d" + std::to_string(dimIdx) + "i" + std::to_string(iterIdx);
      tar_forOp = tiled_parallels.tiled_org_;
    }
  }
  return success();
}
// End of Helper Functions
// ------------------------------------------------------------



/**
 * @copydoc loom_affine::EnumerateSpatialMappings
 */
OwningOpRef<ModuleOp>
EnumerateSpatialMappings(ModuleOp affineModule,
                                      const HardwareInfo& hardwareInfo) {
  MLIRContext *ctx = affineModule.getContext();
  OpBuilder builder(ctx);
  auto out = ModuleOp::create(affineModule.getLoc());
  // 复制原始 module 的 attributes
  if (!affineModule->getAttrs().empty()) {
    out->setAttrs(affineModule->getAttrs());
  }

  for (func::FuncOp func : affineModule.getOps<func::FuncOp>()) {
    SmallVector<affine::AffineParallelOp> roots;
    func.walk([&](affine::AffineParallelOp par) {
      if (!par->getParentOfType<affine::AffineParallelOp>())
        roots.push_back(par);
    });
    if (roots.empty()) {
      IRMapping map;
      builder.setInsertionPointToEnd(out.getBody());
      (void)builder.clone(*func, map);
      continue;
    }

    affine::AffineParallelOp root = roots.front();
    const unsigned P = root.getNumDims();

    // Reuse the same enumeration over bucketings/permutations within iterators
    // as in enumerateSpatialMappings, but additionally enumerate all
    // permutations of the remaining outer parallel iterators as well.

    const unsigned D = static_cast<unsigned>(hardwareInfo.spatialDimInfoVec.size());
    if (D == 0) {
      // No spatial dims to map: convert the outer parallel to a canonical
      // nested-for in identity order [0..P-1] and emit a single clone.
      SmallVector<unsigned> order(P);
      std::iota(order.begin(), order.end(), 0);
      IRMapping map;
      builder.setInsertionPointToEnd(out.getBody());
      auto clonedFunc = cast<func::FuncOp>(builder.clone(*func, map));
      affine::AffineParallelOp currentOuter = nullptr;
      clonedFunc.walk([&](affine::AffineParallelOp par) {
        if (!par->getParentOfType<affine::AffineParallelOp>() && !currentOuter)
          currentOuter = par;
      });
      if (!currentOuter)
        continue;
      (void)ConvertParallelToNested(currentOuter, order);
      // Canonical case: append "__for" to indicate outer-for materialization.
      clonedFunc.setName((func.getName() + "__for").str());
      continue;
    }

    // Generate partitions of D dims into P buckets, then permutations within
    // each bucket, as before.
    llvm::SmallVector<loom_affine::DimBuckets> allBuckets =
        GenerateAllPossibleParallelBuckets(P, D);

    for (auto &bucketing : allBuckets) {
      auto mappings = PermuteBucket(bucketing, hardwareInfo);
      for (const auto &mapping : mappings) {
        SmallVector<unsigned> order(P);
        std::iota(order.begin(), order.end(), 0);

        SmallVector<unsigned> orderCopy = order;
        do {
          IRMapping map;
          builder.setInsertionPointToEnd(out.getBody());
          auto clonedFunc = cast<func::FuncOp>(builder.clone(*func, map));

          std::string mappingSuffix;
          affine::AffineParallelOp tar_forOp = getOutermostParallel(clonedFunc);
          if (!tar_forOp) {
            clonedFunc.erase();
            continue;
          }

          if (failed(applyMappingToFunction(clonedFunc, mapping, hardwareInfo.spatialDimInfoVec,
                                            tar_forOp, mappingSuffix))) {
            clonedFunc.erase();
            continue;
          }

          if (!tar_forOp ||
              failed(ConvertParallelToNested(tar_forOp, orderCopy))) {
            clonedFunc.erase();
            continue;
          }

          composeAndCanonicalizeAffineApplies(clonedFunc);

          std::string newName = func.getName().str();
          newName += "__";
          if (!mappingSuffix.empty())
            newName += mappingSuffix;
          newName += "__f";
          for (unsigned idx : orderCopy)
            newName += std::to_string(idx);
          clonedFunc.setName(newName);

        } while (std::next_permutation(orderCopy.begin(), orderCopy.end()));
      }
    }
  }

  return out;
}

} // namespace loom_affine
