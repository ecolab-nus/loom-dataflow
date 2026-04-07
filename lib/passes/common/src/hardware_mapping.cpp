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

#include "LoomDialect.h.inc"
#include "LoomEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "LoomAttributes.h.inc"

#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace {

struct HWDimMapping {
  unsigned hwDimIdx;
  int64_t hwDimSize;
  Value iv;
};

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
    llvm::SmallVector<loom::ParallelToHWMapping> &mappingInfo) {
  suffix.clear();
  mappingInfo.clear();

  MLIRContext *ctx = func.getContext();
  const unsigned numIter = static_cast<unsigned>(mapping.size());

  llvm::DenseMap<unsigned, llvm::SmallVector<HWDimMapping>> iterIdxToHWMappings;

  for (unsigned iterIdx = 0; iterIdx < numIter; ++iterIdx) {
    for (unsigned dimIdx : mapping[iterIdx]) {
      const auto &sd = dims[dimIdx];
      int64_t factor = sd.size.value_or(1);
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
      if (!suffix.empty())
        suffix += "_";
      suffix += "d" + std::to_string(dimIdx) + "i" + std::to_string(iterIdx);

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

  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(tar_forOp.getBody());
  Location loc = tar_forOp.getLoc();

  for (unsigned i = 0; i < tar_forOp.getNumDims(); ++i) {
    Value waveIV = tar_forOp.getBody()->getArgument(i);
    Value reconstructedIV = nullptr;

    auto it = iterIdxToHWMappings.find(i);
    if (it != iterIdxToHWMappings.end()) {
      auto &hwmVec = it->second;
      int64_t totalCores = 1;
      for (const auto &hwm : hwmVec)
        totalCores *= hwm.hwDimSize;

      if (hwmVec.size() >= 2) {
        reconstructedIV = loom::emitGlobalIndex2d(
            builder, loc, hwmVec[0].iv, hwmVec[1].iv, hwmVec[0].hwDimSize,
            waveIV, 1, static_cast<unsigned>(totalCores));
      } else if (hwmVec.size() == 1) {
        int64_t factor = hwmVec[0].hwDimSize;
        AffineExpr d0 = builder.getAffineDimExpr(0);
        AffineExpr d1 = builder.getAffineDimExpr(1);
        AffineMap map = AffineMap::get(2, 0, d0 + d1 * factor, ctx);
        reconstructedIV = affine::AffineApplyOp::create(
            builder, loc, map, ValueRange{hwmVec[0].iv, waveIV});
      }
    }

    if (reconstructedIV) {
      for (auto &use : llvm::make_early_inc_range(waveIV.getUses())) {
        if (use.getOwner() == reconstructedIV.getDefiningOp())
          continue;
        use.set(reconstructedIV);
      }
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
      (void)loom::utils::cloneFuncWithConstraints(
          builder, func, func.getName(), moduleAttrs, "EnumerateHWMapping",
          [](func::FuncOp) { return success(); }, nullptr);
      continue;
    }

    affine::AffineParallelOp root = roots.front();
    const unsigned P = root.getNumDims();
    const unsigned D =
        static_cast<unsigned>(hardwareInfo.spatialDimInfoVec.size());

    if (D == 0) {
      SmallVector<unsigned> order(P);
      std::iota(order.begin(), order.end(), 0);

      builder.setInsertionPointToEnd(out.getBody());
      std::string newName = (func.getName() + "__for").str();

      auto clonedFunc = loom::utils::cloneFuncWithConstraints(
          builder, func, newName, moduleAttrs, "EnumerateHWMapping",
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
            if (failed(
                    loom_affine::ConvertParallelToNested(currentOuter, order)))
              return failure();

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

    // Generate all HW dim splits to produce P logical dims from D physical dims.
    loom::HWDimSplitter splitter;
    auto allSplits =
        splitter.generateAllSplits(P, hardwareInfo.spatialDimInfoVec);

    loom::MappingPrioritizer prioritizer;
    auto weights = prioritizer.computeIterWeights(func, root);
    // Bijective mappings depend only on weights and P, not on the split.
    auto bijectiveMappings =
        prioritizer.generateBijectiveMappings(weights, P);

    for (const auto &split : allSplits) {
      // Convert LogicalDims to SpatialDimInfo for applyMappingToFunction.
      SmallVector<loom::SpatialDimInfo> logicalDimInfos;
      for (const auto &ld : split.logicalDims) {
        loom::SpatialDimInfo sdi;
        sdi.name = ld.sourceName;
        sdi.size = ld.size;
        sdi.symbolName = ld.sourceName;
        sdi.level = ld.level;
        logicalDimInfos.push_back(sdi);
      }

      std::string splitDesc = buildSplitDesc(split);

      for (const auto &mapping : bijectiveMappings) {
        SmallVector<unsigned> order(P);
        std::iota(order.begin(), order.end(), 0);

        SmallVector<unsigned> orderCopy = order;
        do {
            builder.setInsertionPointToEnd(out.getBody());
            std::string mappingSuffix;
            std::string newNamePrefix = func.getName().str();
            newNamePrefix += "__";

            auto clonedFunc = loom::utils::cloneFuncWithConstraints(
                builder, func, "", moduleAttrs, "EnumerateHWMapping",
                [&](func::FuncOp cloned) -> LogicalResult {
                  markLoopsSequential(cloned);
                  affine::AffineParallelOp tar_forOp =
                      getOutermostParallel(cloned);
                  if (!tar_forOp) {
                    return failure();
                  }

                  llvm::SmallVector<loom::ParallelToHWMapping> hwMappingInfo;
                  if (failed(applyMappingToFunction(
                          cloned, mapping, logicalDimInfos,
                          tar_forOp, mappingSuffix, hwMappingInfo))) {
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
                  loom_affine::flattenCeilDivInForBounds(cloned);
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

        } while (std::next_permutation(orderCopy.begin(), orderCopy.end()));
      }
    }
  }

  return out;
}

} // namespace loom
