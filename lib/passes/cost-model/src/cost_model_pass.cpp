#include "Passes.h"

#include "cost_model_types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Error.h"

namespace loom {
namespace cost_model {

#define GEN_PASS_DEF_LOOMCOSTMODEL
#include "Passes.h.inc"

namespace {

constexpr int kDefaultTopK = 1;
constexpr double kDefaultDramBandwidth = 15.0;
constexpr double kDefaultDramLatency = 454.0;
constexpr double kDefaultNocBandwidth = 27.88;
constexpr double kDefaultAllLinksBandwidth = 18.235;
constexpr double kDefaultNocLatency = 344.0;
constexpr double kDefaultAllLinksLatency = 586.0;
constexpr double kDefaultMatrixGflops = 89.6;
constexpr double kDefaultVectorGflops = 100.0;
constexpr double kDefaultElementBytes = 2.0;
constexpr double kDefaultMatrixUnit = 8.0;
constexpr double kDefaultVectorUnit = 1.0;

double computeTotalCost(const InnermostLoopSummary &summary) {
  return summary.totalWorkloadDurationNs > 0.0 ? summary.totalWorkloadDurationNs
                                               : summary.estimatedDurationNs;
}

void setDoubleAttr(mlir::func::FuncOp func, llvm::StringRef name, double value) {
  auto type = mlir::Float64Type::get(func.getContext());
  func->setAttr(name, mlir::FloatAttr::get(type, value));
}

void setRankAttr(mlir::func::FuncOp func, std::int64_t rank) {
  auto type = mlir::IntegerType::get(func.getContext(), 64);
  func->setAttr("loom.cost.rank", mlir::IntegerAttr::get(type, rank));
}

void annotateCostAttrs(mlir::func::FuncOp func,
                       const InnermostLoopSummary &summary) {
  setDoubleAttr(func, "loom.cost.memory_ns", summary.memoryDuration);
  setDoubleAttr(func, "loom.cost.noc_ns", summary.nocDuration);
  setDoubleAttr(func, "loom.cost.compute_ns", summary.computeDuration);
  setDoubleAttr(func, "loom.cost.estimated_ns", summary.estimatedDurationNs);
  setDoubleAttr(func, "loom.cost.total_ns", computeTotalCost(summary));
}

class LoomCostModelPass
    : public impl::LoomCostModelBase<LoomCostModelPass> {
public:
  LoomCostModelPass() = default;

  explicit LoomCostModelPass(const CostModelOptions &options) {
    topK = options.topK;
    dramBandwidthGbps = options.dramBandwidthGbps;
    dramLatencyNs = options.dramLatencyNs;
    nocBandwidthGbps = options.nocBandwidthGbps;
    allLinksBandwidthGbps = options.allLinksBandwidthGbps;
    nocLatencyNs = options.nocLatencyNs;
    allLinksLatencyNs = options.allLinksLatencyNs;
    matrixGflops = options.matrixGflops;
    vectorGflops = options.vectorGflops;
    elementBytes = options.elementBytes;
    matrixUnit = options.matrixUnit;
    vectorUnit = options.vectorUnit;
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    HardwareSpec spec;
    mergeHardwareSpecFromDF(module, spec);
    applyOptionOverrides(spec);

    auto workloadsOrErr = buildCoreSequentialWorkloads(module);
    if (!workloadsOrErr) {
      module.emitError() << "cost-model workload extraction failed: "
                         << llvm::toString(workloadsOrErr.takeError());
      signalPassFailure();
      return;
    }

    struct CandidateResult {
      mlir::func::FuncOp func;
      std::string funcName;
      InnermostLoopSummary summary;
      double totalCost = 0.0;
    };

    std::vector<CandidateResult> candidates;

    for (const FunctionWorkload &functionWorkload : *workloadsOrErr) {
      if (!functionWorkload.funcOp || !functionWorkload.hasComputeWorkload) {
        continue;
      }

      InnermostLoopSummary summary =
          estimateLongestInnermostLoop(functionWorkload.workloads, spec);
      annotateCostAttrs(functionWorkload.funcOp, summary);

      CandidateResult result;
      result.func = functionWorkload.funcOp;
      result.funcName = functionWorkload.functionName;
      result.summary = summary;
      result.totalCost = computeTotalCost(summary);
      candidates.push_back(result);
    }

    if (candidates.empty()) {
      module.emitRemark("loom-cost-model found no candidate functions");
      return;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const CandidateResult &lhs, const CandidateResult &rhs) {
                if (lhs.totalCost != rhs.totalCost) {
                  return lhs.totalCost < rhs.totalCost;
                }
                return lhs.funcName < rhs.funcName;
              });

    for (std::size_t i = 0; i < candidates.size(); ++i) {
      setRankAttr(candidates[i].func, static_cast<std::int64_t>(i + 1));
    }

    const int keepCount = std::max(0, static_cast<int>(topK));
    std::vector<mlir::func::FuncOp> toErase;
    for (std::size_t i = static_cast<std::size_t>(keepCount);
         i < candidates.size(); ++i) {
      toErase.push_back(candidates[i].func);
    }

    for (mlir::func::FuncOp func : toErase) {
      func.erase();
    }
  }

private:
  void applyOptionOverrides(HardwareSpec &spec) const {
    auto overrideIfSet = [](double optionValue, double defaultValue,
                            double &target) {
      if (std::abs(optionValue - defaultValue) > 1e-12) {
        target = optionValue;
      }
    };

    overrideIfSet(dramBandwidthGbps, kDefaultDramBandwidth,
                  spec.dramBandwidthGBps);
    overrideIfSet(dramLatencyNs, kDefaultDramLatency, spec.dramLatencyNs);
    overrideIfSet(nocBandwidthGbps, kDefaultNocBandwidth, spec.nocBandwidthGBps);
    overrideIfSet(allLinksBandwidthGbps, kDefaultAllLinksBandwidth,
                  spec.allLinksBandwidthGBps);
    overrideIfSet(nocLatencyNs, kDefaultNocLatency, spec.nocLatencyNs);
    overrideIfSet(allLinksLatencyNs, kDefaultAllLinksLatency,
                  spec.allLinksLatencyNs);
    overrideIfSet(matrixGflops, kDefaultMatrixGflops, spec.matrixGflops);
    overrideIfSet(vectorGflops, kDefaultVectorGflops, spec.vectorGflops);
    overrideIfSet(elementBytes, kDefaultElementBytes, spec.elementBytes);
    overrideIfSet(matrixUnit, kDefaultMatrixUnit, spec.matrixUnit);
    overrideIfSet(vectorUnit, kDefaultVectorUnit, spec.vectorUnit);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLoomCostModelPass() {
  return std::make_unique<LoomCostModelPass>();
}

std::unique_ptr<mlir::Pass>
createLoomCostModelPass(const CostModelOptions &options) {
  return std::make_unique<LoomCostModelPass>(options);
}

} // namespace cost_model
} // namespace loom
