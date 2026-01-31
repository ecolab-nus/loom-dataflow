#include "affine_utils.h"
#include "constraint_space_utils.h"
#include "hardware_info.h"
#include "index_mapping.h"
#include "mapping_enumeration.h"
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
      tiled_parallels.tiled_new_->setAttr("loom.mapped_to",
                                          SymbolRefAttr::get(ctx, symbolName));
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
        reconstructedIV = builder.create<affine::AffineApplyOp>(
            loc, map, ValueRange{hwmVec[0].iv, waveIV});
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

static LogicalResult addL1CacheConstraints(func::FuncOp func,
                                           loom::ConstraintSpaceOp csOp,
                                           int64_t l1Size) {
  if (l1Size <= 0 || !csOp)
    return success();

  auto allocInfos = loom::utils::collectL1AllocInfos(func);
  if (allocInfos.empty())
    return success();

  llvm::SmallVector<StringRef> symVarNames;
  llvm::StringMap<unsigned> symVarToIndex;

  for (const auto &info : allocInfos) {
    for (StringRef dimName : info.dims) {
      if (symVarToIndex.find(dimName) == symVarToIndex.end()) {
        symVarToIndex[dimName] = symVarNames.size();
        symVarNames.push_back(dimName);
      }
    }
  }

  llvm::SmallVector<loom::lcs::Monomial> monomials;
  for (const auto &info : allocInfos) {
    loom::lcs::Monomial m;
    for (StringRef dimName : info.dims) {
      m.varIndices.push_back(symVarToIndex[dimName]);
    }
    m.coeff = info.elemSize;
    monomials.push_back(m);
  }

  loom::lcs::addPolynomialConstraint(csOp, symVarNames, monomials, l1Size);
  return success();
}

static LogicalResult
addIntraCorePipelineConstraints(loom::ConstraintSpaceOp csOp,
                                const loom::HardwareInfo &hardwareInfo) {
  if (!csOp)
    return success();

  if (!hardwareInfo.matUnits.empty()) {
    int64_t align = hardwareInfo.matUnits[0].shape[0];
    loom::lcs::updateRangeLowerBounds(csOp, align);
    loom::lcs::addAlignConstraintsForAllVars(csOp, align);
  }

  if (hardwareInfo.matUnitCount > 0 && !hardwareInfo.matUnits.empty()) {
    int64_t align = hardwareInfo.matUnits[0].shape[0];
    llvm::SmallVector<StringRef> symVarNames;
    for (Operation &op : csOp.getBodyBlock()->getOperations()) {
      if (auto symVar = dyn_cast<loom::SymbolicVarOp>(&op)) {
        symVarNames.push_back(symVar.getName());
      }
    }
    if (symVarNames.size() >= 3) {
      loom::lcs::addPipelineParallelismConstraint(
          csOp, symVarNames, hardwareInfo.matUnitCount, align);
    }
  }

  return success();
}

static LogicalResult addInterCoreBalanceConstraints(
    loom::ConstraintSpaceOp csOp,
    const llvm::SmallVector<loom::ParallelToHWMapping> &mappingInfo) {
  if (!csOp || mappingInfo.empty())
    return success();

  llvm::StringMap<int64_t> varUpperBounds;
  llvm::SmallVector<StringRef> symVarNamesOrdered;
  for (Operation &op : csOp.getBodyBlock()->getOperations()) {
    if (auto symVar = dyn_cast<loom::SymbolicVarOp>(&op)) {
      symVarNamesOrdered.push_back(symVar.getName());
    }
    if (auto rangeOp = dyn_cast<loom::RangeOp>(&op)) {
      if (auto symVar =
              rangeOp.getVariable().getDefiningOp<loom::SymbolicVarOp>()) {
        varUpperBounds[symVar.getName()] = rangeOp.getUpperBound();
      }
    }
  }

  auto getVarNameForIter = [&](unsigned iterIdx) -> StringRef {
    if (iterIdx < symVarNamesOrdered.size())
      return symVarNamesOrdered[iterIdx];
    return "";
  };

  llvm::DenseMap<unsigned, int64_t> iterToTotalCores;
  for (const auto &mapping : mappingInfo) {
    auto it = iterToTotalCores.find(mapping.parallelIterIdx);
    if (it == iterToTotalCores.end()) {
      iterToTotalCores[mapping.parallelIterIdx] = mapping.hwDimSize;
    } else {
      it->second *= mapping.hwDimSize;
    }
  }

  for (const auto &entry : iterToTotalCores) {
    unsigned iterIdx = entry.first;
    int64_t totalCores = entry.second;

    StringRef varName = getVarNameForIter(iterIdx);
    if (varName.empty())
      continue;

    auto it = varUpperBounds.find(varName);
    if (it == varUpperBounds.end())
      continue;

    int64_t dimUB = it->second;

    llvm::SmallVector<loom::lcs::Monomial> monomials;
    loom::lcs::Monomial m;
    m.varIndices.push_back(0);
    m.coeff = totalCores;
    monomials.push_back(m);

    loom::lcs::addPolynomialConstraint(csOp, {varName}, monomials, dimUB);
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

  MappingEnumerator enumerator(hardwareInfo);

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
          [](func::FuncOp, loom::ConstraintSpaceOp) { return success(); },
          nullptr);
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
          [&](func::FuncOp cloned,
              loom::ConstraintSpaceOp csOp) -> LogicalResult {
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

            if (failed(
                    addL1CacheConstraints(cloned, csOp, hardwareInfo.l1Size)))
              return failure();

            if (failed(addIntraCorePipelineConstraints(csOp, hardwareInfo)))
              return failure();

            return success();
          },
          nullptr);

      if (!clonedFunc)
        continue;
      continue;
    }

    llvm::SmallVector<loom::DimBuckets> allBucketingResults =
        enumerator.generateAllPossibleBuckets(P, D);

    for (auto &bucketing : allBucketingResults) {
      auto mappings = enumerator.permuteBuckets(bucketing);
      for (const auto &mapping : mappings) {
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
              [&](func::FuncOp cloned,
                  loom::ConstraintSpaceOp csOp) -> LogicalResult {
                markLoopsSequential(cloned);
                affine::AffineParallelOp tar_forOp =
                    getOutermostParallel(cloned);
                if (!tar_forOp) {
                  return failure();
                }

                llvm::SmallVector<loom::ParallelToHWMapping> hwMappingInfo;
                if (failed(applyMappingToFunction(
                        cloned, mapping, hardwareInfo.spatialDimInfoVec,
                        tar_forOp, mappingSuffix, hwMappingInfo))) {
                  return failure();
                }

                if (!tar_forOp || failed(loom_affine::ConvertParallelToNested(
                                      tar_forOp, orderCopy))) {
                  return failure();
                }

                markLoopsTemporal(cloned);

                loom::utils::composeAndCanonicalizeAffineApplies(cloned);

                if (failed(addIntraCorePipelineConstraints(csOp, hardwareInfo)))
                  return failure();

                if (failed(addL1CacheConstraints(cloned, csOp,
                                                 hardwareInfo.l1Size)))
                  return failure();

                if (failed(addInterCoreBalanceConstraints(csOp, hwMappingInfo)))
                  return failure();

                return success();
              },
              nullptr);

          if (clonedFunc) {
            std::string finalName = newNamePrefix;
            if (!mappingSuffix.empty())
              finalName += mappingSuffix;
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
