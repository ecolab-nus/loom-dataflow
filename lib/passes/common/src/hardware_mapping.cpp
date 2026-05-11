#include "affine_utils.h"
#include "hardware_info.h"
#include "hw_dim_splitter.h"
#include "index_mapping.h"
#include "mapping_prioritizer.h"
#include "utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include <algorithm>
#include <numeric>

#define SKIP_TEMPORAL_EXPLORATION // Toggle this to skip temporal loop exploration


#include "LoomDialect.h.inc"
#include "LoomEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "LoomAttributes.h.inc"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace {

struct HWDimMapping {
  unsigned hwDimIdx;
  int64_t hwDimSize;
  Value iv;
};

struct ReductionAxisInfo {
  unsigned parallelIterIdx; // Index in the affine.parallel (e.g., 2 for K)
  Value parallelIV;         // The block argument (%arg5)
  scf::IfOp ifOp;           // The scf.if operation
  Value conditionSSA;       // The arith.cmpi result
};

static std::optional<ReductionAxisInfo>
detectReductionAxis(affine::AffineParallelOp root) {
  std::optional<ReductionAxisInfo> result;
  root.walk([&](loom::GatherOp gatherOp) {
    if (result)
      return;
    Value across = gatherOp.getAcross();
    if (!across)
      return;
    auto ivs = root.getIVs();
    for (unsigned i = 0; i < ivs.size(); ++i) {
      if (ivs[i] == across) {
        // Find the enclosing scf.if so its condition can be rewritten later.
        scf::IfOp ifOp = gatherOp->getParentOfType<scf::IfOp>();
        Value cond = ifOp ? ifOp.getCondition() : Value{};
        result = {i, ivs[i], ifOp, cond};
        return;
      }
    }
  });
  return result;
}

static void markLoopsSequential(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();
  func.walk([ctx](Operation *op) {
    if (isa<affine::AffineForOp, scf::ForOp>(op)) {
      if (!op->hasAttr("loom.iter_type")) {
        op->setAttr("loom.iter_type",
                    loom::IterTypeAttr::get(ctx, loom::IterType::Sequential));
      }
    }
  });
}

static void markLoopsTemporal(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();
  func.walk([ctx](Operation *op) {
    if (isa<affine::AffineForOp, scf::ForOp>(op)) {
      if (!op->hasAttr("loom.iter_type")) {
        op->setAttr("loom.iter_type",
                    loom::IterTypeAttr::get(ctx, loom::IterType::Temporal));
      }
    }
  });
}

// Build split description string in "x4x2_y8" format from a HWDimSplit.
// Groups logical dims by sourceIdx, emits short name + factors inner-to-outer.
static std::string buildSplitDesc(const loom::HWDimSplit &split) {
  // Collect factors per source dim, ordered by level.
  llvm::DenseMap<unsigned, llvm::SmallVector<int64_t>> factorsBySource;
  llvm::SmallVector<unsigned> sourceOrder;
  for (const auto &ld : split.logicalDims) {
    if (factorsBySource.find(ld.sourceIdx) == factorsBySource.end())
      sourceOrder.push_back(ld.sourceIdx);
    factorsBySource[ld.sourceIdx].push_back(ld.size);
  }
  // sourceOrder already reflects insertion order (level ASC, sourceIdx ASC),
  // but we want dims sorted by sourceIdx for the name.
  std::sort(sourceOrder.begin(), sourceOrder.end());

  std::string desc;
  for (unsigned i = 0; i < sourceOrder.size(); ++i) {
    if (i > 0)
      desc += "_";
    unsigned srcIdx = sourceOrder[i];
    // Extract short name: "dim_x" -> "x", "dim_y" -> "y"
    std::string shortName;
    for (const auto &ld : split.logicalDims) {
      if (ld.sourceIdx == srcIdx) {
        llvm::StringRef ref(ld.sourceName);
        if (ref.starts_with("dim_"))
          shortName = ref.drop_front(4).str();
        else
          shortName = ref.str();
        break;
      }
    }
    // Factors for this source are already in level order (inner first)
    // because logicalDims is sorted by (level, sourceIdx).
    const auto &factors = factorsBySource[srcIdx];
    for (int64_t f : factors)
      desc += shortName + std::to_string(f);
  }
  return desc;
}

static affine::AffineParallelOp getOutermostParallel(func::FuncOp func) {
  affine::AffineParallelOp result = nullptr;
  func.walk([&](affine::AffineParallelOp par) {
    if (!par->getParentOfType<affine::AffineParallelOp>() && !result)
      result = par;
  });
  return result;
}

static LogicalResult applyMappingToFunction(
    func::FuncOp func, const loom::DimBuckets &mapping,
    const llvm::SmallVector<loom::SpatialDimInfo> &dims,
    affine::AffineParallelOp &tar_forOp, std::string &suffix,
    llvm::SmallVector<loom::ParallelToHWMapping> &mappingInfo,
    std::optional<ReductionAxisInfo> reductionInfo) {
  suffix.clear();
  mappingInfo.clear();

  MLIRContext *ctx = func.getContext();
  const unsigned numIter = static_cast<unsigned>(mapping.size());

  llvm::DenseMap<unsigned, llvm::SmallVector<HWDimMapping>> iterIdxToHWMappings;

  for (unsigned iterIdx = 0; iterIdx < numIter; ++iterIdx) {
    for (unsigned dimIdx : mapping[iterIdx]) {
      const auto &sd = dims[dimIdx];
      int64_t factor = sd.size.value_or(1);

      // Find the loom.sym referenced in the UB of dimension iterIdx BEFORE
      // tiling (tileAffineParallel replaces the parallel op with constant UBs
      // on the outer loop, so we must capture symbol info here).
      // The per-dim UB AffineExpr uses symbol positions that index into the
      // shared allUbOps operand pool starting at ubMap.getNumDims().
      std::optional<SymbolRefAttr> blockSym;
      {
        AffineMap ubMapForDim = tar_forOp.getUpperBoundMap(iterIdx);
        AffineExpr ubExpr = ubMapForDim.getResult(0);
        auto allUbOps = tar_forOp.getUpperBoundsOperands();
        unsigned nd = ubMapForDim.getNumDims();
        ubExpr.walk([&](AffineExpr e) {
          if (!blockSym) {
            if (auto symE = llvm::dyn_cast<AffineSymbolExpr>(e)) {
              unsigned opIdx = nd + symE.getPosition();
              if (opIdx < allUbOps.size())
                blockSym = loom_affine::traceToLoomSymRef(allUbOps[opIdx]);
            }
          }
        });
      }

      loom_affine::TiledParallels tiled_parallels{};
      if (failed(loom_affine::tileAffineParallel(tar_forOp, factor, iterIdx,
                                                 tiled_parallels)))
        return failure();

      StringRef symbolName =
          sd.symbolName.empty() ? StringRef("dim") : StringRef(sd.symbolName);
      tiled_parallels.tiled_new_->setAttr("loom.physical_dim",
                                          SymbolRefAttr::get(ctx, symbolName));
      tiled_parallels.tiled_new_->setAttr(
          "loom.logical_level",
          IntegerAttr::get(IntegerType::get(ctx, 64),
                           static_cast<int64_t>(sd.level)));
      tiled_parallels.tiled_new_->setAttr(
          "loom.iter_type",
          loom::IterTypeAttr::get(ctx, loom::IterType::Spatial));
      if (blockSym)
        tiled_parallels.tiled_new_->setAttr("loom.block_sym", *blockSym);

      loom::ParallelToHWMapping mapInfo;
      mapInfo.parallelIterIdx = iterIdx;
      mapInfo.hwDimIdx = dimIdx;
      mapInfo.hwDimSize = factor;
      mappingInfo.push_back(mapInfo);

      HWDimMapping hwMap;
      hwMap.hwDimIdx = dimIdx;
      hwMap.hwDimSize = factor;
      hwMap.iv = tiled_parallels.tiled_new_.getBody()->getArgument(0);
      iterIdxToHWMappings[iterIdx].push_back(hwMap);

      tar_forOp = tiled_parallels.tiled_org_;
    }
  }

  // Construct suffix ordered by dimIdx (d) instead of iterIdx (i).
  SmallVector<std::pair<unsigned, unsigned>> diPairs;
  for (const auto &info : mappingInfo)
    diPairs.push_back({info.hwDimIdx, info.parallelIterIdx});
  llvm::sort(diPairs);
  for (const auto &pair : diPairs) {
    if (!suffix.empty())
      suffix += "_";
    suffix += "d" + std::to_string(pair.first) + "i" + std::to_string(pair.second);
  }

  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(tar_forOp.getBody());
  Location loc = tar_forOp.getLoc();

  llvm::DenseMap<unsigned, Value> reconstructedIVByIterIdx;

  for (unsigned i = 0; i < tar_forOp.getNumDims(); ++i) {
    Value waveIV = tar_forOp.getBody()->getArgument(i);
    Value reconstructedIV = nullptr;

    auto it = iterIdxToHWMappings.find(i);
    if (it != iterIdxToHWMappings.end()) {
      auto &hwmVec = it->second;
      SmallVector<const HWDimMapping *> nonUnitHWMappings;
      int64_t totalCores = 1;
      for (const auto &hwm : hwmVec) {
        totalCores *= hwm.hwDimSize;
        if (hwm.hwDimSize != 1)
          nonUnitHWMappings.push_back(&hwm);
      }

      if (nonUnitHWMappings.size() >= 2) {
        reconstructedIV = loom::emitGlobalIndex2d(
            builder, loc, nonUnitHWMappings[0]->iv, nonUnitHWMappings[1]->iv,
            nonUnitHWMappings[0]->hwDimSize, waveIV, 1,
            static_cast<unsigned>(totalCores));
      } else if (nonUnitHWMappings.size() == 1) {
        int64_t factor = nonUnitHWMappings[0]->hwDimSize;
        AffineExpr d0 = builder.getAffineDimExpr(0);
        AffineExpr d1 = builder.getAffineDimExpr(1);
        AffineMap map = AffineMap::get(2, 0, d0 + d1 * factor, ctx);
        reconstructedIV = affine::AffineApplyOp::create(
            builder, loc, map, ValueRange{nonUnitHWMappings[0]->iv, waveIV});
      }
    }

    reconstructedIVByIterIdx[i] = reconstructedIV ? reconstructedIV : waveIV;

    if (reconstructedIV) {
      for (auto &use : llvm::make_early_inc_range(waveIV.getUses())) {
        if (use.getOwner() == reconstructedIV.getDefiningOp())
          continue;
        use.set(reconstructedIV);
      }
    }
  }

  // Enhancement 2 & 3: Handle reduction axis
  if (reductionInfo) {
    loom::MeshCoordinateSystem meshCoords;
    for (auto &entry : iterIdxToHWMappings) {
      for (const auto &hwm : entry.second) {
        unsigned srcIdx = dims[hwm.hwDimIdx].sourceIdx;
        loom::AxisLinearIndex &axis =
            (srcIdx == 0) ? meshCoords.xAxis : meshCoords.yAxis;
        axis.sourceIdx = srcIdx;
        axis.ivs.push_back(hwm.iv);
        axis.tileSizes.push_back(hwm.hwDimSize);
        axis.logicalDimIndices.push_back(hwm.hwDimIdx);
      }
    }

    // Sort each axis by logical level (innermost first) to match
    // fromEnclosingLoops ordering and emitLinearIndex assumptions.
    auto sortAxis = [&](loom::AxisLinearIndex &axis) {
      if (axis.ivs.size() <= 1)
        return;
      SmallVector<unsigned> indices(axis.ivs.size());
      std::iota(indices.begin(), indices.end(), 0);
      std::sort(indices.begin(), indices.end(), [&](unsigned a, unsigned b) {
        return dims[axis.logicalDimIndices[a]].level <
               dims[axis.logicalDimIndices[b]].level;
      });
      SmallVector<Value> sortedIVs;
      SmallVector<int64_t> sortedTileSizes;
      SmallVector<unsigned> sortedLogicalDimIndices;
      for (unsigned idx : indices) {
        sortedIVs.push_back(axis.ivs[idx]);
        sortedTileSizes.push_back(axis.tileSizes[idx]);
        sortedLogicalDimIndices.push_back(axis.logicalDimIndices[idx]);
      }
      axis.ivs = std::move(sortedIVs);
      axis.tileSizes = std::move(sortedTileSizes);
      axis.logicalDimIndices = std::move(sortedLogicalDimIndices);
    };
    sortAxis(meshCoords.xAxis);
    sortAxis(meshCoords.yAxis);

    unsigned reductionIterIdx = reductionInfo->parallelIterIdx;
    auto it = iterIdxToHWMappings.find(reductionIterIdx);
    if (it != iterIdxToHWMappings.end() && !it->second.empty()) {
      SmallVector<int64_t, 2> gatherArea(2, 1);
      llvm::DenseMap<unsigned, int64_t> ulXOverrides;
      llvm::DenseMap<unsigned, int64_t> lrXOverrides;
      llvm::DenseMap<unsigned, int64_t> ulYOverrides;
      llvm::DenseMap<unsigned, int64_t> lrYOverrides;

      for (const auto &hwm : it->second) {
        unsigned srcIdx = dims[hwm.hwDimIdx].sourceIdx;
        loom::AxisLinearIndex &axis =
            (srcIdx == 0) ? meshCoords.xAxis : meshCoords.yAxis;

        std::optional<unsigned> levelIdx;
        for (unsigned i = 0; i < axis.logicalDimIndices.size(); ++i) {
          if (axis.logicalDimIndices[i] == hwm.hwDimIdx) {
            levelIdx = i;
            break;
          }
        }
        if (!levelIdx)
          continue;

        if (srcIdx < gatherArea.size())
          gatherArea[srcIdx] *= hwm.hwDimSize;

        if (srcIdx == 0) {
          ulXOverrides[*levelIdx] = 0;
          lrXOverrides[*levelIdx] = hwm.hwDimSize - 1;
        } else if (srcIdx == 1) {
          ulYOverrides[*levelIdx] = 0;
          lrYOverrides[*levelIdx] = hwm.hwDimSize - 1;
        }
      }

      Value reconstructedAcross =
          reconstructedIVByIterIdx.lookup(reductionIterIdx);
      if (reductionInfo->ifOp && reconstructedAcross) {
        OpBuilder condBuilder(reductionInfo->ifOp);
        Location ifLoc = reductionInfo->ifOp.getLoc();
        Value c0 = arith::ConstantIndexOp::create(condBuilder, ifLoc, 0);
        Value newCond = arith::CmpIOp::create(
            condBuilder, ifLoc, arith::CmpIPredicate::eq, reconstructedAcross,
            c0);
        reductionInfo->ifOp.getConditionMutable().assign(newCond);
      }

      func.walk([&](loom::GatherOp gatherOp) {
        OpBuilder gBuilder(gatherOp);
        Location gatherLoc = gatherOp.getLoc();
        Value ul_x = meshCoords.emitLinearIndexWithMultiOverride(
            gBuilder, gatherLoc, meshCoords.xAxis, ulXOverrides);
        Value ul_y = meshCoords.emitLinearIndexWithMultiOverride(
            gBuilder, gatherLoc, meshCoords.yAxis, ulYOverrides);
        Value lr_x = meshCoords.emitLinearIndexWithMultiOverride(
            gBuilder, gatherLoc, meshCoords.xAxis, lrXOverrides);
        Value lr_y = meshCoords.emitLinearIndexWithMultiOverride(
            gBuilder, gatherLoc, meshCoords.yAxis, lrYOverrides);
        auto staticAreaAttr = gBuilder.getDenseI64ArrayAttr(gatherArea);

        loom::GatherOp::create(gBuilder, gatherLoc, gatherOp.getSource(),
                               gatherOp.getDestination(), gatherOp.getAcross(),
                               ValueRange{}, staticAreaAttr, ul_x, ul_y,
                               lr_x, lr_y);
        gatherOp.erase();
      });
    }
  }

  return success();
}

} // namespace

namespace loom {

OwningOpRef<ModuleOp>
EnumerateSpatialMappings(ModuleOp affineModule,
                         const HardwareInfo &hardwareInfo) {
  MLIRContext *ctx = affineModule.getContext();
  OpBuilder builder(ctx);
  auto out = ModuleOp::create(affineModule.getLoc());
  if (!affineModule->getAttrs().empty()) {
    out->setAttrs(affineModule->getAttrs());
  }

  llvm::SmallVector<func::FuncOp> allFuncs =
      loom::utils::collectFunctions(affineModule);

  for (func::FuncOp func : allFuncs) {
    ModuleOp parentModule = loom::utils::getParentModule(func);
    DictionaryAttr moduleAttrs = nullptr;
    if (parentModule) {
      moduleAttrs = parentModule->getAttrDictionary();
    }
    SmallVector<affine::AffineParallelOp> roots;
    func.walk([&](affine::AffineParallelOp par) {
      if (!par->getParentOfType<affine::AffineParallelOp>())
        roots.push_back(par);
    });

    if (roots.empty()) {
      builder.setInsertionPointToEnd(out.getBody());
      (void)loom::utils::cloneFunc(
          builder, func, func.getName(), moduleAttrs,
          [](func::FuncOp) { return success(); }, nullptr);
      continue;
    }

    affine::AffineParallelOp root = roots.front();
    const unsigned P = root.getNumDims();
    const unsigned D =
        static_cast<unsigned>(hardwareInfo.spatialDimInfoVec.size());

    if (D < 2 || P < 1) {
      SmallVector<unsigned> order(P);
      std::iota(order.begin(), order.end(), 0);

      builder.setInsertionPointToEnd(out.getBody());
      std::string newName = (func.getName() + "__for").str();

      auto clonedFunc = loom::utils::cloneFunc(
          builder, func, newName, moduleAttrs,
          [&](func::FuncOp cloned) -> LogicalResult {
            markLoopsSequential(cloned);
            affine::AffineParallelOp currentOuter = nullptr;
            cloned.walk([&](affine::AffineParallelOp par) {
              if (!par->getParentOfType<affine::AffineParallelOp>() &&
                  !currentOuter)
                currentOuter = par;
            });
            if (!currentOuter)
              return failure();
            if (P > 0) {
              if (failed(loom_affine::ConvertParallelToNested(currentOuter,
                                                              order)))
                return failure();
            }

            markLoopsTemporal(cloned);

            // Assign memory attributes to copy ops
            cloned.walk([&](Operation *op) {
              if (auto copyTo = dyn_cast<loom::CopyToTensorOp>(op)) {
                copyTo.setMemoryAttr(SymbolRefAttr::get(ctx, "L1"));
              } else if (auto copyFrom = dyn_cast<loom::CopyFromTensorOp>(op)) {
                copyFrom.setMemoryAttr(SymbolRefAttr::get(ctx, "DRAM"));
              }
            });

            return success();
          },
          nullptr);

      if (!clonedFunc)
        continue;
      continue;
    }

    loom::MappingPrioritizer prioritizer;
    loom::AxisScores scores = prioritizer.computeAxisScores(func, root);
    auto reductionInfoMain = detectReductionAxis(root);
    std::optional<unsigned> reductionIdx;
    if (reductionInfoMain)
      reductionIdx = reductionInfoMain->parallelIterIdx;

    auto buildLogicalDimInfos = [](const loom::HWDimSplit &split) {
      SmallVector<loom::SpatialDimInfo> logicalDimInfos;
      for (const auto &ld : split.logicalDims) {
        loom::SpatialDimInfo sdi;
        sdi.name = ld.sourceName;
        sdi.size = ld.size;
        sdi.symbolName = ld.sourceName;
        sdi.level = ld.level;
        sdi.sourceIdx = ld.sourceIdx;
        logicalDimInfos.push_back(sdi);
      }
      return logicalDimInfos;
    };

    auto applyOneCandidate =
        [&](const loom::DimBuckets &mapping,
            const SmallVector<loom::SpatialDimInfo> &logicalDimInfos,
            const std::string &splitDesc) {
          SmallVector<unsigned> order(P);
          std::iota(order.begin(), order.end(), 0);

          SmallVector<unsigned> orderCopy = order;
#ifndef SKIP_TEMPORAL_EXPLORATION
          do {
#endif
            builder.setInsertionPointToEnd(out.getBody());
            std::string mappingSuffix;
            std::string newNamePrefix = func.getName().str();
            newNamePrefix += "__";

            auto clonedFunc = loom::utils::cloneFunc(
                builder, func, "", moduleAttrs,
                [&](func::FuncOp cloned) -> LogicalResult {
                  markLoopsSequential(cloned);
                  affine::AffineParallelOp tar_forOp =
                      getOutermostParallel(cloned);
                  if (!tar_forOp) {
                    return failure();
                  }

                  auto reductionInfoCloned = detectReductionAxis(tar_forOp);

                  llvm::SmallVector<loom::ParallelToHWMapping> hwMappingInfo;
                  if (failed(applyMappingToFunction(
                          cloned, mapping, logicalDimInfos, tar_forOp,
                          mappingSuffix, hwMappingInfo, reductionInfoCloned))) {
                    return failure();
                  }

                  if (!tar_forOp ||
                      failed(loom_affine::ConvertParallelToNested(
                          tar_forOp, orderCopy))) {
                    return failure();
                  }

                  markLoopsTemporal(cloned);

                  // Assign memory attributes to copy ops
                  cloned.walk([&](Operation *op) {
                    if (auto copyTo = dyn_cast<loom::CopyToTensorOp>(op)) {
                      copyTo.setMemoryAttr(SymbolRefAttr::get(ctx, "L1"));
                    } else if (auto copyFrom =
                                   dyn_cast<loom::CopyFromTensorOp>(op)) {
                      copyFrom.setMemoryAttr(SymbolRefAttr::get(ctx, "DRAM"));
                    }
                  });

                  loom::utils::composeAndCanonicalizeAffineApplies(cloned);

                  return success();
                },
                nullptr);

            if (clonedFunc) {
              std::string finalName = newNamePrefix + splitDesc;
              if (!mappingSuffix.empty())
                finalName += "__" + mappingSuffix;
              finalName += "__f";
              for (unsigned idx : orderCopy)
                finalName += std::to_string(idx);
              clonedFunc.setName(finalName);
            }

#ifndef SKIP_TEMPORAL_EXPLORATION
          } while (std::next_permutation(orderCopy.begin(), orderCopy.end()));
#endif
        };

    loom::HWDimSplitter splitter;
    auto pPlus1Splits = splitter.generateAllSplits(
        P + 1, hardwareInfo.spatialDimInfoVec, /*allowSizeOne=*/true);

    for (const auto &split : pPlus1Splits) {
      SmallVector<unsigned> level0Indices, nonLevel0Indices;
      for (unsigned i = 0; i < split.logicalDims.size(); ++i) {
        if (split.logicalDims[i].level == 0)
          level0Indices.push_back(i);
        else
          nonLevel0Indices.push_back(i);
      }

      assert(level0Indices.size() == 2 &&
             "framework currently assumes D == 2");

      SmallVector<loom::SpatialDimInfo> logicalDimInfos =
          buildLogicalDimInfos(split);
      std::string splitDesc = buildSplitDesc(split);

      auto mappings = prioritizer.generateLevel0PairClaimMappings(
          scores, level0Indices, nonLevel0Indices, reductionIdx);

      for (const auto &mapping : mappings)
        applyOneCandidate(mapping, logicalDimInfos, splitDesc);
    }
  }

  return out;
}

} // namespace loom
