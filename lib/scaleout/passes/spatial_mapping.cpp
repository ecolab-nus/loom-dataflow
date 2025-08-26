#include "spatial_mapping.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

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
      info.name =
          "dim"; // placeholder; SSA result name is not directly accessible here
      // The op carries an i64 size attribute; negative meaning not provided
      // is treated as dynamic/unbounded.
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
  inner->setAttr("tmd.mapped_to", StringAttr::get(inner.getContext(), dimName));
  return success();
}

static bool hasSufficientExtent(affine::AffineParallelOp par, unsigned dim,
                                std::optional<int64_t> needed) {
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

} // namespace tmd_affine

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include <algorithm>
#include <numeric>

namespace tmd_affine {

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

    // Step 1: generate all partitions of D dims into P ordered buckets.
    SmallVector<SmallVector<SmallVector<unsigned>>> bucketings;
    SmallVector<SmallVector<unsigned>> buckets(P);
    std::function<void(unsigned)> placeDim = [&](unsigned d) {
      if (d == D) {
        bucketings.push_back(buckets);
        return;
      }
      for (unsigned it = 0; it < P; ++it) {
        buckets[it].push_back(d);
        placeDim(d + 1);
        buckets[it].pop_back();
      }
    };
    placeDim(0);

    // Step 2: for each partition, permute within each bucket and apply.
    for (auto &bucketing : bucketings) {
      SmallVector<SmallVector<SmallVector<unsigned>>> permsPerIter(P);
      for (unsigned it = 0; it < P; ++it) {
        SmallVector<unsigned> b = bucketing[it];
        if (b.size() <= 1) {
          permsPerIter[it].push_back(b);
        } else {
          std::sort(b.begin(), b.end());
          do {
            permsPerIter[it].push_back(b);
          } while (std::next_permutation(b.begin(), b.end()));
        }
      }

      SmallVector<unsigned> choiceIdx(P, 0);
      std::function<void(unsigned)> choose = [&](unsigned it) {
        if (it == P) {
          // Build chosen order per iterator.
          SmallVector<SmallVector<unsigned>> ordered(P);
          for (unsigned j = 0; j < P; ++j)
            ordered[j] = permsPerIter[j][choiceIdx[j]];

          // Clone func and apply tiling: iterate iterators 0..P-1, and within
          // each, tile per dim in that iterator's ordered list.
          IRMapping map;
          builder.setInsertionPointToEnd(out.getBody());
          auto clonedFunc = cast<func::FuncOp>(builder.clone(*func, map));
          affine::AffineParallelOp currentOuter = nullptr;
          clonedFunc.walk([&](affine::AffineParallelOp par) {
            if (!par->getParentOfType<affine::AffineParallelOp>() &&
                !currentOuter)
              currentOuter = par;
          });
          if (!currentOuter)
            return;

          std::string suffix;
          for (unsigned iter = 0; iter < P; ++iter) {
            for (unsigned dIdx : ordered[iter]) {
              const SpatialDimInfo &sd = dims[dIdx];
              int64_t factor = 1;
              if (sd.size.has_value())
                factor = std::max<int64_t>(1, *sd.size);
              TiledParallels tp{};
              if (failed(tileAffineParallel(currentOuter, factor, iter, tp)))
                return;
              tp.inner->setAttr(
                  "tmd.mapped_to",
                  StringAttr::get(ctx, sd.name.empty() ? "dim" : sd.name));
              if (!suffix.empty())
                suffix += "_";
              suffix += "d" + std::to_string(dIdx) + "i" + std::to_string(iter);
              currentOuter = tp.outer;
            }
          }
          clonedFunc.setName((func.getName() + "__" + suffix).str());
          return;
        }
        for (unsigned k = 0; k < permsPerIter[it].size(); ++k) {
          choiceIdx[it] = k;
          choose(it + 1);
        }
      };
      choose(0);
    }
  }

  return out;
}

} // namespace tmd_affine
