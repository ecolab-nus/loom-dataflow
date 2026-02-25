#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace loom {
namespace cost_model {

struct CoreIndex {
  std::int64_t x = 0;
  std::int64_t y = 0;

  bool operator==(const CoreIndex &other) const noexcept {
    return x == other.x && y == other.y;
  }
  bool operator!=(const CoreIndex &other) const noexcept {
    return !(*this == other);
  }
};

struct MemoryLoad {
  CoreIndex core{};
  CoreIndex startCore{};
  CoreIndex requestedCore{};
  std::vector<std::int64_t> sourceLayout;
  std::vector<std::int64_t> sourceStrides;
  std::string guardReason;
};

struct MemoryStore {
  CoreIndex core{};
  CoreIndex endCore{};
  std::vector<std::int64_t> targetLayout;
  std::vector<std::int64_t> targetStrides;
  std::string guardReason;
};

struct NoCWorkload {
  CoreIndex core{};
  CoreIndex start{};
  CoreIndex end{};
  std::int64_t bytes = 0;
  std::string reason;
  std::string network;
};

struct MulticastWorkload {
  CoreIndex core{};
  CoreIndex start{};
  CoreIndex end{};
  std::int64_t bytes = 0;
  std::string reason;
  std::string network;
};

struct ComputeWorkload {
  CoreIndex core{};
  std::string opName;
  std::vector<std::int64_t> outputLayout;
  std::vector<std::int64_t> outputStrides;
  std::int64_t totalCalculations = 0;
};

struct LoopRange {
  std::string inductionVar;
  std::int64_t lowerBound = 0;
  std::int64_t upperBound = 0;
  std::int64_t step = 1;
  bool hasLowerBound = false;
  bool hasUpperBound = false;
  bool hasStep = false;
};

struct LoopBegin {
  std::string loopType;
  std::vector<LoopRange> ranges;
};

struct LoopEnd {
  std::string loopType;
};

using SequentialEvent =
    std::variant<LoopBegin, LoopEnd, MemoryLoad, ComputeWorkload, MemoryStore,
                 NoCWorkload, MulticastWorkload>;

struct CoreSequentialWorkload {
  CoreIndex core{};
  std::vector<SequentialEvent> events;
};

struct FunctionWorkload {
  std::string functionName;
  mlir::func::FuncOp funcOp;
  std::vector<CoreSequentialWorkload> workloads;
  bool hasComputeWorkload = false;
};

struct HardwareSpec {
  double dramBandwidthGBps = 15.0;
  double dramLatencyNs = 454.0;
  double nocBandwidthGBps = 27.88;
  double allLinksBandwidthGBps = 18.235;
  double nocLatencyNs = 344.0;
  double allLinksLatencyNs = 586.0;
  double matrixGflops = 128.0 * 0.7;
  double vectorGflops = 100.0;
  double elementBytes = 2.0;
  double matrixUnit = 8.0;
  double vectorUnit = 1.0;
};

struct InnermostLoopSummary {
  CoreIndex core{};
  double memoryDuration = 0.0;
  double nocDuration = 0.0;
  double computeDuration = 0.0;
  double estimatedDurationNs = 0.0;
  double totalWorkloadDurationNs = 0.0;
};

llvm::Expected<std::vector<FunctionWorkload>>
buildCoreSequentialWorkloads(mlir::ModuleOp module);

void mergeHardwareSpecFromDF(mlir::ModuleOp module, HardwareSpec &spec);

InnermostLoopSummary
estimateLongestInnermostLoop(const std::vector<CoreSequentialWorkload> &workloads,
                             const HardwareSpec &spec);

} // namespace cost_model
} // namespace loom
