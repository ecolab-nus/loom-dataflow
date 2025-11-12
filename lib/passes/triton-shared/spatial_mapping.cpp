/**
 * @file spatial_mapping.cpp
 * @brief Implementation for spatial mapping discovery and enumeration.
 * @details
 * Provided functionality
 * - `collectSpatialDims`: parse `df.spatial_dim` declarations from the DF
 *   module into simple name/size pairs.
 * - `mapSpatialDimsToAffine`: greedily tile outermost `affine.parallel` loops
 *   with factors derived from spatial dimension sizes and mark inner loops with
 *   `tmd.mapped_to`.
 * - `enumerateSpatialMappings`: produce clones for all unique bucketings and
 *   per-iterator permutations of spatial dims over the first outermost
 *   `affine.parallel` in each function.
 * - `enumerateSpatialMappingsWithOuterFors`: additionally convert the
 *   remaining parallel iterators to a canonical `affine.for` nest (identity
 *   iterator order; no enumeration over permutations).
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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/IRMapping.h"
#include "affine_parallel_to_for.h"
#include "llvm/ADT/BitVector.h"
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

namespace tmd_affine {

LogicalResult collectSpatialDims(ModuleOp dfModule,
                                 llvm::SmallVectorImpl<SpatialDimInfo> &out) {
  bool foundAny = false;
  dfModule.walk([&](Operation *op) {
    if (auto sd = dyn_cast<tmd::df::SpatialDimOp>(op)) {
      SpatialDimInfo info;
      // Read declared name and size from the op properties.
      if (auto nameAttr = sd.getNameAttr())
        info.name = nameAttr.getValue().str();
      else
        info.name = "dim";
      uint64_t sz = sd.getSize();
      if (sz > 0)
        info.size = static_cast<int64_t>(sz);
      else
        info.size = std::nullopt;
      out.push_back(std::move(info));
      foundAny = true;
    }
  });
  return success(foundAny);
}

static LogicalResult markInnerMapped(affine::AffineParallelOp inner,
                                     StringRef dimName) {
  /**
   * @brief Tag the inner loop with its mapped spatial dimension.
   *
   * @param inner   The inner `affine.parallel` produced by tiling.
   * @param dimName Hardware spatial dimension name to record.
   * @return success always.
   */
  inner->setAttr("tmd.mapped_to", StringAttr::get(inner.getContext(), dimName));
  return success();
}

static bool hasSufficientExtent(affine::AffineParallelOp par, unsigned dim,
                                std::optional<int64_t> needed) {
  /**
   * @brief Check if a parallel iterator has enough static extent for tiling.
   *
   * @param par    The target `affine.parallel` operation.
   * @param dim    Iterator index within `par` to check.
   * @param needed Required static factor; `std::nullopt` means dynamic and is
   *               conservatively considered sufficient.
   * @return True if static range is >= needed or if unknown (conservative).
   */
  if (!needed.has_value())
    return true; // dynamic spatial size considered infinite
  if (auto maybeRanges = par.getConstantRanges()) {
    auto ranges = *maybeRanges;
    if (dim < ranges.size())
      return ranges[dim] >= *needed;
  }
  // If we cannot query statically, be conservative-allow; the tiler will fail
  // later if it's incompatible.
  return true;
}

/**
 * @copydoc tmd_affine::mapSpatialDimsToAffine
 */
LogicalResult mapSpatialDimsToAffine(ModuleOp affineModule,
                                     llvm::ArrayRef<SpatialDimInfo> dims,
                                     unsigned tileDimIndex) {
  if (dims.empty())
    return success();

  unsigned consumed = 0;
  for (func::FuncOp func : affineModule.getOps<func::FuncOp>()) {
    // Greedy walk: consider outermost affine.parallel first.
    SmallVector<affine::AffineParallelOp> candidates;
    func.walk([&](affine::AffineParallelOp op) {
      if (op->getParentOfType<affine::AffineParallelOp>())
        return; // only outermost
      candidates.push_back(op);
    });

    for (affine::AffineParallelOp par : candidates) {
      // Map as many spatial dims as fit this parallel op, one at a time.
      while (consumed < dims.size()) {
        const SpatialDimInfo &sd = dims[consumed];
        if (!hasSufficientExtent(par, tileDimIndex, sd.size))
          break;

        int64_t factor = 1;
        if (sd.size.has_value())
          factor = std::max<int64_t>(1, *sd.size);
        else
          factor = 1; // dynamic: use factor 1; still mark mapping

        TiledParallels tiled{};
        if (failed(tileAffineParallel(par, factor, tileDimIndex, tiled))) {
          // If tiling fails, stop trying to map this par; move to next par.
          break;
        }
        // Mark inner as mapped to this spatial dim.
        (void)markInnerMapped(tiled.inner, sd.name);

        // The next iteration can continue mapping the new inner body. We set
        // par to the inner for potential further mappings of the same loop.
        par = tiled.inner;
        ++consumed;
        if (consumed >= dims.size())
          break;
      }
      if (consumed >= dims.size())
        break;
    }
    if (consumed >= dims.size())
      break;
  }

  // Success if we consumed all spatial dims, or at least mapped some.
  return success(consumed > 0);
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
  tmd_affine::DimBuckets& currentBuckets, llvm::SmallVector<tmd_affine::DimBuckets>& out) {
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
static llvm::SmallVector<tmd_affine::DimBuckets> GenerateAllPossibleParallelBuckets(unsigned numParelleIter, unsigned numDims) {
  llvm::SmallVector<tmd_affine::DimBuckets> bucketing_results;
  tmd_affine::DimBuckets currentBuckets(numParelleIter);
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
  tmd_affine::DimBuckets &current,
  const llvm::SmallVector<tmd_affine::DimBuckets> &bucketsPerIter,
  llvm::SmallVector<tmd_affine::DimBuckets> &out) {
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
static llvm::SmallVector<tmd_affine::DimBuckets> GenerateAllPossibleMappings(const llvm::SmallVector<tmd_affine::DimBuckets>& permutedBucketsPerIter) {
  llvm::SmallVector<tmd_affine::DimBuckets> result;
  tmd_affine::DimBuckets currentBuckets(permutedBucketsPerIter.size());
  CartesianProductOfBuckets(0, currentBuckets, permutedBucketsPerIter, result);
  return result;
}


/**
 * @brief Permute the dimensions in each iterator's bucket.
 * 
 * @param baseBuckets The base bucketing.
 * @return The complete mappings.
 */
static llvm::SmallVector<tmd_affine::DimBuckets> PermuteBucket(const tmd_affine::DimBuckets& baseBuckets) {
  const unsigned numIters = static_cast<unsigned>(baseBuckets.size());
  llvm::SmallVector<tmd_affine::DimBuckets> permutedBucketsPerIter(numIters);

  for (unsigned it = 0; it < numIters; ++it) {
    SmallVector<unsigned> dims = baseBuckets[it];
    if (dims.size() <= 1) {
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
                                            const tmd_affine::DimBuckets &mapping,
                                            ArrayRef<tmd_affine::SpatialDimInfo> dims,
                                            std::string &suffix) {
  suffix.clear();
  affine::AffineParallelOp currentOuter = getOutermostParallel(func);
  if (!currentOuter)
    return failure();

  MLIRContext *ctx = func.getContext();
  const unsigned numIter = static_cast<unsigned>(mapping.size());
  for (unsigned iterIdx = 0; iterIdx < numIter; ++iterIdx) {
    for (unsigned dimIdx : mapping[iterIdx]) {
      const auto &sd = dims[dimIdx];
      int64_t factor = sd.size.value_or(1);
      tmd_affine::TiledParallels tp{};
      if (failed(tileAffineParallel(currentOuter, factor, iterIdx, tp)))
        return failure();
      tp.inner->setAttr("tmd.mapped_to",
                        StringAttr::get(ctx, sd.name.empty() ? "dim" : sd.name));
      if (!suffix.empty())
        suffix += "_";
      suffix += "d" + std::to_string(dimIdx) + "i" + std::to_string(iterIdx);
      currentOuter = tp.outer;
    }
  }
  return success();
}
// End of Helper Functions
// ------------------------------------------------------------

/**
 * @copydoc tmd_affine::enumerateSpatialMappings
 */
OwningOpRef<ModuleOp> enumerateSpatialMappings(ModuleOp affineModule,
                                               ArrayRef<SpatialDimInfo> dims) {
  MLIRContext *ctx = affineModule.getContext();
  OpBuilder builder(ctx);

  // Create an output module that will hold all clones.
  auto out = ModuleOp::create(affineModule.getLoc());

  for (func::FuncOp func : affineModule.getOps<func::FuncOp>()) {
    // Collect outermost affine.parallel candidates in function order.
    SmallVector<affine::AffineParallelOp> roots;
    func.walk([&](affine::AffineParallelOp par) {
      if (!par->getParentOfType<affine::AffineParallelOp>())
        roots.push_back(par);
    });

    if (roots.empty()) {
      // Just clone function unchanged.
      IRMapping map;
      builder.setInsertionPointToEnd(out.getBody());
      (void)builder.clone(*func, map);
      continue;
    }

    // For simplicity, only enumerate for the first outermost par in the func.
    // Extensions can lift this to multiple regions by nested enumeration.
    affine::AffineParallelOp root = roots.front();
    const unsigned P = root.getNumDims();
    const unsigned D = static_cast<unsigned>(dims.size());
    if (D == 0) {
      IRMapping map;
      builder.setInsertionPointToEnd(out.getBody());
      (void)builder.clone(*func, map);
      continue;
    }

    llvm::SmallVector<tmd_affine::DimBuckets> allBuckets =
        GenerateAllPossibleParallelBuckets(P, D);

    for (auto &bucketing : allBuckets) {
      auto mappings = PermuteBucket(bucketing);
      for (const auto &mapping : mappings) {
        IRMapping map;
        builder.setInsertionPointToEnd(out.getBody());
        auto clonedFunc = cast<func::FuncOp>(builder.clone(*func, map));

        std::string mappingSuffix;
        if (failed(applyMappingToFunction(clonedFunc, mapping, dims,
                                          mappingSuffix))) {
          clonedFunc.erase();
          continue;
        }

        composeAndCanonicalizeAffineApplies(clonedFunc);

        std::string newName = func.getName().str();
        newName += "__";
        if (!mappingSuffix.empty())
          newName += mappingSuffix;
        clonedFunc.setName(newName);
      }
    }
  }

  return out;
}

/**
 * @copydoc tmd_affine::enumerateSpatialMappingsWithOuterFors
 */
OwningOpRef<ModuleOp>
enumerateSpatialMappingsWithOuterFors(ModuleOp affineModule,
                                      ArrayRef<SpatialDimInfo> dims) {
  MLIRContext *ctx = affineModule.getContext();
  OpBuilder builder(ctx);
  auto out = ModuleOp::create(affineModule.getLoc());

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

    const unsigned D = static_cast<unsigned>(dims.size());
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
      (void)convertOutermostParallelToNestedFors(currentOuter, order);
      // Canonical case: append "__for" to indicate outer-for materialization.
      clonedFunc.setName((func.getName() + "__for").str());
      continue;
    }

    // Generate partitions of D dims into P buckets, then permutations within
    // each bucket, as before.
    llvm::SmallVector<tmd_affine::DimBuckets> allBuckets =
        GenerateAllPossibleParallelBuckets(P, D);

    for (auto &bucketing : allBuckets) {
      auto mappings = PermuteBucket(bucketing);
      for (const auto &mapping : mappings) {
        SmallVector<unsigned> order(P);
        std::iota(order.begin(), order.end(), 0);

        SmallVector<unsigned> orderCopy = order;
        do {
          IRMapping map;
          builder.setInsertionPointToEnd(out.getBody());
          auto clonedFunc = cast<func::FuncOp>(builder.clone(*func, map));

          std::string mappingSuffix;
          if (failed(applyMappingToFunction(clonedFunc, mapping, dims,
                                            mappingSuffix))) {
            clonedFunc.erase();
            continue;
          }

          affine::AffineParallelOp outer = getOutermostParallel(clonedFunc);
          if (!outer ||
              failed(convertOutermostParallelToNestedFors(outer, orderCopy))) {
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


/**
 * @copydoc tmd_affine::enumerateTritonSharedSpatialMappings
 */
OwningOpRef<ModuleOp> enumerateTritonSharedSpatialMappings(
    ModuleOp module, ArrayRef<SpatialDimInfo> dims, unsigned numGridDims) {
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);
  auto out = ModuleOp::create(module.getLoc());

  // Precompute spatial dim metadata attrs (names, sizes) once.
  SmallVector<Attribute> spatialNameAttrs;
  SmallVector<Attribute> spatialSizeAttrs;
  spatialNameAttrs.reserve(dims.size());
  spatialSizeAttrs.reserve(dims.size());
  for (const SpatialDimInfo &sd : dims) {
    spatialNameAttrs.push_back(StringAttr::get(ctx, sd.name));
    int64_t sz = sd.size.has_value() ? *sd.size : static_cast<int64_t>(-1);
    spatialSizeAttrs.push_back(IntegerAttr::get(IntegerType::get(ctx, 64), sz));
  }

  const unsigned S = static_cast<unsigned>(dims.size());
  (void)numGridDims; // Grid dims are fixed to x,y,z; analyze usage below.

  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    // Analyze which program_id.{x,y,z} are used (ABI args 12,13,14).
    bool gridUsed[3] = {false, false, false};
    SmallVector<Attribute> gridUsedAttrs;
    gridUsedAttrs.reserve(3);
    unsigned totalArgs = func.getNumArguments();
    if (totalArgs >= 15) {
      for (unsigned i = 0; i < 3; ++i) {
        BlockArgument pid = func.getArgument(12 + i);
        gridUsed[i] = !pid.use_empty();
      }
    }
    for (unsigned i = 0; i < 3; ++i)
      gridUsedAttrs.push_back(
          IntegerAttr::get(IntegerType::get(ctx, 1), gridUsed[i] ? 1 : 0));

    // Build list of used grid indices in x(0), y(1), z(2) order.
    SmallVector<unsigned> usedGridIdx;
    for (unsigned i = 0; i < 3; ++i)
      if (gridUsed[i])
        usedGridIdx.push_back(i);

    // If no spatial dims or no used grid dims, just clone and mark unused.
    if (S == 0 || usedGridIdx.empty()) {
      IRMapping map;
      builder.setInsertionPointToEnd(out.getBody());
      auto clonedFunc = cast<func::FuncOp>(builder.clone(*func, map));
      clonedFunc->setAttr("tmd.spatial_dim_names",
                          ArrayAttr::get(ctx, spatialNameAttrs));
      clonedFunc->setAttr("tmd.spatial_dim_sizes",
                          ArrayAttr::get(ctx, spatialSizeAttrs));
      SmallVector<Attribute> threeEmpty{ArrayAttr::get(ctx, {}),
                                        ArrayAttr::get(ctx, {}),
                                        ArrayAttr::get(ctx, {})};
      clonedFunc->setAttr("tmd.grid_to_spatial",
                          ArrayAttr::get(ctx, threeEmpty));
      clonedFunc->setAttr("tmd.grid_used", ArrayAttr::get(ctx, gridUsedAttrs));
      // Build suffix tokens.
      std::string suffix;
      for (unsigned g = 0; g < 3; ++g) {
        if (!suffix.empty())
          suffix += "_";
        suffix += std::string("g") + std::to_string(g) +
                  (gridUsed[g] ? "none" : "unused");
      }
      clonedFunc.setName((func.getName() + "__" + suffix).str());
      continue;
    }

    // Partition S spatial dims among K buckets, where K = number of used grids.
    const unsigned K = static_cast<unsigned>(usedGridIdx.size());
    SmallVector<SmallVector<SmallVector<unsigned>>> bucketings;
    SmallVector<SmallVector<unsigned>> buckets(K);
    std::function<void(unsigned)> placeSpatial = [&](unsigned d) {
      if (d == S) {
        bucketings.push_back(buckets);
        return;
      }
      for (unsigned k = 0; k < K; ++k) {
        buckets[k].push_back(d);
        placeSpatial(d + 1);
        buckets[k].pop_back();
      }
    };
    placeSpatial(0);

    for (auto &bucketing : bucketings) {
      // Permute within each bucket and enumerate choices.
      SmallVector<SmallVector<SmallVector<unsigned>>> permsPerGrid(K);
      for (unsigned k = 0; k < K; ++k) {
        SmallVector<unsigned> b = bucketing[k];
        if (b.size() <= 1) {
          permsPerGrid[k].push_back(b);
        } else {
          std::sort(b.begin(), b.end());
          do {
            permsPerGrid[k].push_back(b);
          } while (std::next_permutation(b.begin(), b.end()));
        }
      }

      SmallVector<unsigned> choiceIdx(K, 0);
      std::function<void(unsigned)> choose = [&](unsigned k) {
        if (k == K) {
          // Build per-grid ordered lists for x,y,z (3 buckets total).
          SmallVector<Attribute> gridToSpatial(3);
          for (unsigned i = 0; i < 3; ++i)
            gridToSpatial[i] = ArrayAttr::get(ctx, {});

          for (unsigned pos = 0; pos < K; ++pos) {
            unsigned gdim = usedGridIdx[pos];
            SmallVector<Attribute> ints;
            for (unsigned sidx : permsPerGrid[pos][choiceIdx[pos]]) {
              ints.push_back(IntegerAttr::get(IntegerType::get(ctx, 64),
                                              static_cast<int64_t>(sidx)));
            }
            gridToSpatial[gdim] = ArrayAttr::get(ctx, ints);
          }

          // Build suffix per grid dim.
          std::string suffix;
          for (unsigned g = 0; g < 3; ++g) {
            if (!suffix.empty())
              suffix += "_";
            auto arr = cast<ArrayAttr>(gridToSpatial[g]);
            if (!gridUsed[g]) {
              suffix += std::string("g") + std::to_string(g) + "unused";
            } else if (arr.empty()) {
              suffix += std::string("g") + std::to_string(g) + "none";
            } else {
              suffix += std::string("g") + std::to_string(g) + "s";
              for (Attribute a : arr) {
                auto ia = cast<IntegerAttr>(a);
                suffix += "d" + std::to_string(static_cast<int>(ia.getInt()));
              }
            }
          }

          // Clone function and attach attributes.
          IRMapping map;
          builder.setInsertionPointToEnd(out.getBody());
          auto clonedFunc = cast<func::FuncOp>(builder.clone(*func, map));
          clonedFunc->setAttr("tmd.spatial_dim_names",
                              ArrayAttr::get(ctx, spatialNameAttrs));
          clonedFunc->setAttr("tmd.spatial_dim_sizes",
                              ArrayAttr::get(ctx, spatialSizeAttrs));
          clonedFunc->setAttr("tmd.grid_to_spatial",
                              ArrayAttr::get(ctx, gridToSpatial));
          clonedFunc->setAttr("tmd.grid_used",
                              ArrayAttr::get(ctx, gridUsedAttrs));

          // Replace ABI grid args with computed values from spatial ids.
          //
          // Rewrite summary:
          // - Original ABI args:
          //     %arg9..%arg11  = grid_size.{x,y,z} (i32)
          //     %arg12..%arg14 = program_id.{x,y,z} (i32)
          // - We append S spatial-id args (i32), one per spatial dim.
          // - Spatial sizes are compile-time constants (from DF), so:
          //     grid_size[g] = Π_j size(spatial_dim_j)  → arith.constant
          //     program_id[g] = Σ_j id_j * Π_{k>j} size(spatial_dim_k)
          //   is implemented as affine.apply with constant coefficients.
          if (clonedFunc.getNumArguments() >= 15) {
            OpBuilder entryBuilder(clonedFunc.getBody());
            entryBuilder.setInsertionPointToStart(
                &clonedFunc.getBody().front());
            Location loc = clonedFunc.getLoc();
            Type i32Ty = entryBuilder.getI32Type();

            // 1) Append only spatial id args (i32), one per spatial dimension,
            //    at the end of the signature.
            const unsigned oldNumArgs = clonedFunc.getNumArguments();
            SmallVector<BlockArgument> spatialIdArgs;
            spatialIdArgs.reserve(S);
            for (unsigned sIdx = 0; sIdx < S; ++sIdx) {
              unsigned idx = clonedFunc.getNumArguments();
              (void)clonedFunc.insertArgument(idx, i32Ty, /*argAttrs=*/{}, loc);
              // Tag the argument with the spatial dimension name for clarity.
              clonedFunc.setArgAttr(idx, "tmd.spatial_dim_name",
                                    StringAttr::get(ctx, dims[sIdx].name));
              spatialIdArgs.push_back(clonedFunc.getArgument(idx));
            }

            // 2) Use index-typed views of IDs to feed affine.apply.
            SmallVector<Value> spatialIdIdx(S);
            for (unsigned sIdx = 0; sIdx < S; ++sIdx)
              spatialIdIdx[sIdx] = entryBuilder.create<arith::IndexCastOp>(
                  loc, entryBuilder.getIndexType(), spatialIdArgs[sIdx]);

            Value c0 = entryBuilder.create<arith::ConstantIntOp>(loc, 0, 32);

            // 3) Build grid_size[g] as a single arith.constant product.
            auto makeGridSize = [&](unsigned g) -> Value {
              auto arr = cast<ArrayAttr>(gridToSpatial[g]);
              int64_t prodStatic = 1;
              for (Attribute a : arr) {
                unsigned sIdx =
                    static_cast<unsigned>(cast<IntegerAttr>(a).getInt());
                (void)sIdx;
                prodStatic *= *dims[sIdx].size;
              }
              return entryBuilder.create<arith::ConstantIntOp>(loc, prodStatic,
                                                               32);
            };

            // 4) Build program_id[g] by linearizing assigned spatial IDs with
            //    constant strides using affine.apply.
            auto makeGridId = [&](unsigned g) -> Value {
              auto arr = cast<ArrayAttr>(gridToSpatial[g]);
              if (arr.empty())
                return c0;
              // Compute strides Π_{k>j} size(spatial_dim_k) in assigned order.
              SmallVector<int64_t> strides(arr.size(), 1);
              int64_t running = 1;
              for (int i = static_cast<int>(arr.size()) - 1; i >= 0; --i) {
                strides[i] = running;
                unsigned sIdx =
                    static_cast<unsigned>(cast<IntegerAttr>(arr[i]).getInt());
                running *= *dims[sIdx].size;
              }
              // Construct affine map sum_j (d_j * stride_j) over id operands
              // d_j.
              SmallVector<AffineExpr> d;
              d.reserve(arr.size());
              for (unsigned i = 0; i < arr.size(); ++i)
                d.push_back(getAffineDimExpr(i, ctx));
              AffineExpr sum = getAffineConstantExpr(0, ctx);
              for (unsigned i = 0; i < arr.size(); ++i) {
                AffineExpr term = d[i];
                if (strides[i] != 1)
                  term = term * getAffineConstantExpr(strides[i], ctx);
                sum = sum + term;
              }
              AffineMap map = AffineMap::get(arr.size(), 0, sum, ctx);
              SmallVector<Value> args;
              args.reserve(arr.size());
              for (unsigned i = 0; i < arr.size(); ++i) {
                unsigned sIdx =
                    static_cast<unsigned>(cast<IntegerAttr>(arr[i]).getInt());
                args.push_back(spatialIdIdx[sIdx]);
              }
              Value idxVal =
                  entryBuilder.create<affine::AffineApplyOp>(loc, map, args);
              return entryBuilder.create<arith::IndexCastOp>(loc, i32Ty,
                                                             idxVal);
            };

            // 5) Materialize replacement values for all 3 grid dimensions.
            SmallVector<Value, 3> newGridSizes(3);
            SmallVector<Value, 3> newGridIds(3);
            for (unsigned g = 0; g < 3; ++g) {
              newGridSizes[g] = makeGridSize(g);
              newGridIds[g] = makeGridId(g);
            }

            // 6) Replace old ABI uses with new values.
            for (unsigned g = 0; g < 3; ++g) {
              unsigned sizeIdx = 9 + g;
              unsigned idIdx = 12 + g;
              if (sizeIdx < oldNumArgs)
                clonedFunc.getArgument(sizeIdx).replaceAllUsesWith(
                    newGridSizes[g]);
              if (idIdx < oldNumArgs)
                clonedFunc.getArgument(idIdx).replaceAllUsesWith(newGridIds[g]);
            }

            // 7) Erase the 6 legacy ABI arguments.
            llvm::BitVector bv(clonedFunc.getNumArguments());
            bv.set(9);
            bv.set(10);
            bv.set(11);
            bv.set(12);
            bv.set(13);
            bv.set(14);
            (void)clonedFunc.eraseArguments(bv);
          }

          if (!suffix.empty())
            clonedFunc.setName((func.getName() + "__" + suffix).str());
          return;
        }
        for (unsigned p = 0; p < permsPerGrid[k].size(); ++p) {
          choiceIdx[k] = p;
          choose(k + 1);
        }
      };
      choose(0);
    }
  }

  return out;
}

} // namespace tmd_affine
