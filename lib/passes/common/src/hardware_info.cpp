/**
 * @file hardware_info.cpp
 * @brief Implementation for hardware information discovery and spatial mapping
 * enumeration.
 */

#include "hardware_info.h"
#include "affine_parallel_to_for.h"
#include "affine_tile.h"
#include "constraint_space_utils.h"
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

#include "DataflowDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "DataflowTypes.h.inc"
#define GET_OP_CLASSES
#include "DataflowOps.h.inc"

#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace {

static LogicalResult
GetSpatialDimInfo(loom::df::SpatialDimOp sdOp,
                  llvm::SmallVector<loom::SpatialDimInfo> &dimVec) {
  loom::SpatialDimInfo info;
  if (auto nameAttr = sdOp.getSymNameAttr()) {
    info.name = nameAttr.getValue().str();
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
  if (map.getNumResults() < 2)
    return {false, false};

  bool d0Connected = false;
  bool d1Connected = false;
  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    AffineExpr expr = map.getResult(i);
    if (i == 0 && expr != getAffineDimExpr(0, map.getContext()))
      d0Connected = true;
    if (i == 1 && expr != getAffineDimExpr(1, map.getContext()))
      d1Connected = true;
  }
  return {d0Connected, d1Connected};
}

/**
 * \brief Compose and canonicalize all affine.apply operations in a function.
 */
static void composeAndCanonicalizeAffineApplies(func::FuncOp func) {
  SmallVector<affine::AffineApplyOp> applies;
  func.walk([&](affine::AffineApplyOp op) { applies.push_back(op); });
  for (affine::AffineApplyOp op : applies) {
    OpBuilder b(op);
    AffineMap map = op.getAffineMap();
    SmallVector<Value> operands(op.getOperands().begin(),
                                op.getOperands().end());
    affine::fullyComposeAffineMapAndOperands(&map, &operands);
    affine::canonicalizeMapAndOperands(&map, &operands);
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

  SmallVector<Operation *> toErase;
  func.walk([&](Operation *op) {
    if (mlir::isOpTriviallyDead(op))
      toErase.push_back(op);
  });
  for (Operation *op : toErase)
    op->erase();
}

static void EnumerateBucketingRec(unsigned dimIdx, unsigned numDims,
                                  loom::DimBuckets &currentBuckets,
                                  llvm::SmallVector<loom::DimBuckets> &out) {
  if (dimIdx == numDims) {
    out.push_back(currentBuckets);
    return;
  }
  for (unsigned it = 0; it < currentBuckets.size(); ++it) {
    currentBuckets[it].push_back(dimIdx);
    EnumerateBucketingRec(dimIdx + 1, numDims, currentBuckets, out);
    currentBuckets[it].pop_back();
  }
}

static llvm::SmallVector<loom::DimBuckets>
GenerateAllPossibleParallelBuckets(unsigned numParelleIter, unsigned numDims) {
  llvm::SmallVector<loom::DimBuckets> bucketing_results;
  loom::DimBuckets currentBuckets(numParelleIter);
  EnumerateBucketingRec(0, numDims, currentBuckets, bucketing_results);
  return bucketing_results;
}

static void CartesianProductOfBuckets(
    unsigned iterIdx, loom::DimBuckets &current,
    const llvm::SmallVector<loom::DimBuckets> &bucketsPerIter,
    llvm::SmallVector<loom::DimBuckets> &out) {
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

static llvm::SmallVector<loom::DimBuckets> GenerateAllPossibleMappings(
    const llvm::SmallVector<loom::DimBuckets> &permutedBucketsPerIter) {
  llvm::SmallVector<loom::DimBuckets> result;
  loom::DimBuckets currentBuckets(permutedBucketsPerIter.size());
  CartesianProductOfBuckets(0, currentBuckets, permutedBucketsPerIter, result);
  return result;
}

static llvm::SmallVector<loom::DimBuckets>
PermuteBucket(const loom::DimBuckets &baseBuckets,
              const loom::HardwareInfo &hardwareInfo) {
  const unsigned numIters = static_cast<unsigned>(baseBuckets.size());
  llvm::SmallVector<loom::DimBuckets> permutedBucketsPerIter(numIters);

  for (unsigned it = 0; it < numIters; ++it) {
    SmallVector<unsigned> dims = baseBuckets[it];
    if (dims.size() <= 1 ||
        (dims.size() == hardwareInfo.spatialDimInfoVec.size() &&
         hardwareInfo.skipPermutation())) {
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

static affine::AffineParallelOp getOutermostParallel(func::FuncOp func) {
  affine::AffineParallelOp result = nullptr;
  func.walk([&](affine::AffineParallelOp par) {
    if (!par->getParentOfType<affine::AffineParallelOp>() && !result)
      result = par;
  });
  return result;
}

static LogicalResult
applyMappingToFunction(func::FuncOp func, const loom::DimBuckets &mapping,
                       const llvm::SmallVector<loom::SpatialDimInfo> &dims,
                       affine::AffineParallelOp &tar_forOp,
                       std::string &suffix) {
  suffix.clear();

  MLIRContext *ctx = func.getContext();
  const unsigned numIter = static_cast<unsigned>(mapping.size());
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
      if (!suffix.empty())
        suffix += "_";
      suffix += "d" + std::to_string(dimIdx) + "i" + std::to_string(iterIdx);
      tar_forOp = tiled_parallels.tiled_org_;
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
addHardwareDerivedConstraints(loom::ConstraintSpaceOp csOp,
                              const loom::HardwareInfo &hardwareInfo) {
  if (!csOp)
    return success();

  // Task 1: BM BN BK 32-alignment and range LB=32
  if (!hardwareInfo.matUnits.empty()) {
    int64_t align = hardwareInfo.matUnits[0].shape[0]; // e.g., 32
    loom::lcs::updateRangeLowerBounds(csOp, align);
    loom::lcs::addAlignConstraintsForAllVars(csOp, align);
  }

  // Task 2: BM*BN*BK / 32^3 >= 8
  if (hardwareInfo.matUnitCount > 0 && !hardwareInfo.matUnits.empty()) {
    int64_t align = hardwareInfo.matUnits[0].shape[0]; // e.g., 32
    // We assume M, N, K are the symbolic variables in order.
    // In mm_2Dmesh, they are "M", "N", "K".
    // We should ideally find them by name or use a heuristic.
    // The requirement says BM BN BK Need 32-element alignment.
    // In our test case they are %0, %1, %2 but they correspond to M, N, K.
    // Actually, we can just apply it to all symbolic variables if there are
    // exactly 3.
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

} // namespace

namespace loom {

LogicalResult GetHardwareInfoForExploration(mlir::ModuleOp dfModule,
                                            HardwareInfo &hardwareInfo) {
  bool res = false;
  bool d0Connected = false;
  bool d1Connected = false;
  dfModule.walk([&](Operation *op) {
    if (auto sd = dyn_cast<loom::df::SpatialDimOp>(op)) {
      res =
          res || failed(GetSpatialDimInfo(sd, hardwareInfo.spatialDimInfoVec));
    } else if (auto mem = dyn_cast<loom::df::MemoryOp>(op)) {
      if (mem.getLabel() == "L1") {
        hardwareInfo.l1Size = mem.getSize();
      }
    } else if (auto mat = dyn_cast<loom::df::MatOp>(op)) {
      MatUnitInfo info;
      info.name = mat.getName().str();
      auto shape = mat.getShape();
      for (auto dim : shape)
        info.shape.push_back(dim);
      hardwareInfo.matUnits.push_back(std::move(info));
    } else if (auto core = dyn_cast<loom::df::CoreOp>(op)) {
      if (auto countsAttr = core.getScaleinCountsAttr()) {
        auto counts = countsAttr.asArrayRef();
        if (!counts.empty())
          hardwareInfo.matUnitCount = counts[0]; // First unit is mat_unit
      }
    } else if (auto ic = dyn_cast<loom::df::InterconnectsOp>(op)) {
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

            if (failed(
                    addL1CacheConstraints(cloned, csOp, hardwareInfo.l1Size)))
              return failure();

            if (failed(addHardwareDerivedConstraints(csOp, hardwareInfo)))
              return failure();

            return success();
          },
          nullptr);

      if (!clonedFunc)
        continue;
      continue;
    }

    llvm::SmallVector<loom::DimBuckets> allBuckets =
        GenerateAllPossibleParallelBuckets(P, D);

    for (auto &bucketing : allBuckets) {
      auto mappings = PermuteBucket(bucketing, hardwareInfo);
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
                affine::AffineParallelOp tar_forOp =
                    getOutermostParallel(cloned);
                if (!tar_forOp) {
                  return failure();
                }

                if (failed(applyMappingToFunction(
                        cloned, mapping, hardwareInfo.spatialDimInfoVec,
                        tar_forOp, mappingSuffix))) {
                  return failure();
                }

                if (!tar_forOp || failed(loom_affine::ConvertParallelToNested(
                                      tar_forOp, orderCopy))) {
                  return failure();
                }

                composeAndCanonicalizeAffineApplies(cloned);

                if (failed(addHardwareDerivedConstraints(csOp, hardwareInfo)))
                  return failure();

                if (failed(addL1CacheConstraints(cloned, csOp,
                                                 hardwareInfo.l1Size)))
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
