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

    // Enumerate over all permutations of spatial dimensions (order matters).
    SmallVector<unsigned> perm(D);
    std::iota(perm.begin(), perm.end(), 0);

    auto enumerateAssignmentsForPerm = [&](ArrayRef<unsigned> permOrder) {
      // All assignments of dims to iterators, allowing reuse (P^D choices).
      SmallVector<unsigned> assignment;
      std::function<void()> rec = [&]() {
        if (assignment.size() == D) {
          // Clone and sequentially apply tiling per permuted dim.
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

          SmallVector<affine::AffineParallelOp> createdInners;
          bool failedAny = false;
          for (unsigned step = 0; step < D; ++step) {
            const SpatialDimInfo &sd = dims[permOrder[step]];
            int64_t factor = 1;
            if (sd.size.has_value())
              factor = std::max<int64_t>(1, *sd.size);
            unsigned iterIdx = assignment[step];
            TiledParallels tp{};
            if (failed(tileAffineParallel(currentOuter, factor, iterIdx, tp))) {
              failedAny = true;
              break;
            }
            createdInners.push_back(tp.inner);
            currentOuter = tp.outer;
          }
          if (failedAny)
            return;

          // Annotate per step with mapped dim name.
          for (unsigned step = 0; step < createdInners.size(); ++step) {
            StringRef name = dims[permOrder[step]].name;
            createdInners[step]->setAttr(
                "tmd.mapped_to",
                StringAttr::get(ctx, name.empty() ? "dim" : name));
          }

          // Name suffix encodes (dimIndex->iterIdx) per step.
          std::string suffix;
          for (unsigned step = 0; step < D; ++step) {
            if (step)
              suffix += "_";
            suffix += "d" + std::to_string(permOrder[step]) + "i" +
                      std::to_string(assignment[step]);
          }
          clonedFunc.setName((func.getName() + "__" + suffix).str());
          return;
        }
        for (unsigned it = 0; it < P; ++it) {
          assignment.push_back(it);
          rec();
          assignment.pop_back();
        }
      };
      rec();
    };

    do {
      enumerateAssignmentsForPerm(perm);
    } while (std::next_permutation(perm.begin(), perm.end()));
  }

  return out;
}

} // namespace tmd_affine
