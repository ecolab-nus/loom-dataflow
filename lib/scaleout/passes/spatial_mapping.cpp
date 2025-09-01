#include "spatial_mapping.h"
#include "affine_tile.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/BitVector.h"

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

namespace tmd_affine {

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
