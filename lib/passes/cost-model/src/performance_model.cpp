#include "cost_model_types.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace loom {
namespace cost_model {
namespace {

double productOfDims(const std::vector<std::int64_t> &dims) {
  double product = 1.0;
  bool hasPositiveDim = false;
  for (std::int64_t dim : dims) {
    if (dim <= 0) {
      continue;
    }
    product *= static_cast<double>(dim);
    hasPositiveDim = true;
    if (!std::isfinite(product)) {
      return std::numeric_limits<double>::max();
    }
  }
  return hasPositiveDim ? product : 1.0;
}

double estimateMemoryDurationNs(const std::vector<std::int64_t> &layout,
                                double elementBytes,
                                double bandwidthGBps,
                                double latencyNs) {
  const double elements = productOfDims(layout);
  const double bytes = elements * std::max(elementBytes, 0.0);
  if (bandwidthGBps <= 0.0) {
    return latencyNs;
  }
  return latencyNs + bytes / bandwidthGBps;
}

double estimateMemoryDurationNs(const MemoryLoad &load,
                                const HardwareSpec &spec) {
  return estimateMemoryDurationNs(load.sourceLayout, spec.elementBytes,
                                  spec.dramBandwidthGBps, spec.dramLatencyNs);
}

double estimateMemoryDurationNs(const MemoryStore &store,
                                const HardwareSpec &spec) {
  return estimateMemoryDurationNs(store.targetLayout, spec.elementBytes,
                                  spec.dramBandwidthGBps, spec.dramLatencyNs);
}

double estimateNoCDurationNs(std::int64_t bytes, const HardwareSpec &spec,
                             bool allLinks = false) {
  const double positiveBytes =
      static_cast<double>(std::max<std::int64_t>(bytes, 0));
  const double bw = allLinks ? spec.allLinksBandwidthGBps : spec.nocBandwidthGBps;
  const double latency = allLinks ? spec.allLinksLatencyNs : spec.nocLatencyNs;
  if (bw <= 0.0) {
    return latency;
  }
  return latency + positiveBytes / bw;
}

double estimateNoCDurationNs(const NoCWorkload &workload,
                             const HardwareSpec &spec) {
  std::string lowered = workload.network;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  const bool isAllLinks = lowered.find("all_links") != std::string::npos;
  return estimateNoCDurationNs(workload.bytes, spec, isAllLinks);
}

double estimateNoCDurationNs(const MulticastWorkload &workload,
                             const HardwareSpec &spec) {
  std::string lowered = workload.network;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  const bool isAllLinks = lowered.find("all_links") != std::string::npos;
  return estimateNoCDurationNs(workload.bytes, spec, isAllLinks);
}

double estimateComputeDurationNs(const ComputeWorkload &compute,
                                 const HardwareSpec &spec) {
  const auto &layout = compute.outputLayout;
  const double ops = static_cast<double>(compute.totalCalculations);

  if (compute.opName.find("matmul") != std::string::npos) {
    if (layout.size() < 2) {
      return 0.0;
    }
    const std::int64_t blockM = layout[0] / 32;
    const std::int64_t blockN = layout[1] / 32;
    const double parallelUnit =
        std::min<double>(spec.matrixUnit,
                         static_cast<double>(std::max<std::int64_t>(
                             1, blockM * blockN)));
    double throughput = spec.matrixGflops * parallelUnit;
    if (blockM != blockN) {
      throughput *= 0.7;
    }
    if (throughput <= 0.0) {
      return 0.0;
    }
    return ops / throughput;
  }

  if (compute.opName.find("vec") != std::string::npos ||
      compute.opName.find("dot") != std::string::npos) {
    const double throughput = spec.vectorGflops * spec.vectorUnit;
    if (throughput <= 0.0) {
      return 0.0;
    }
    return ops / throughput;
  }

  return 0.0;
}

std::optional<std::int64_t> computeTripCount(const LoopRange &range) {
  if (!range.hasLowerBound || !range.hasUpperBound || !range.hasStep ||
      range.step == 0) {
    return std::nullopt;
  }
  if (range.step <= 0) {
    return std::nullopt;
  }
  if (range.upperBound <= range.lowerBound) {
    return static_cast<std::int64_t>(0);
  }
  const std::int64_t distance = range.upperBound - range.lowerBound;
  if (distance > std::numeric_limits<std::int64_t>::max() - (range.step - 1)) {
    return std::nullopt;
  }
  return (distance + (range.step - 1)) / range.step;
}

double computeLoopIterationCount(const LoopBegin &loop) {
  if (loop.ranges.empty()) {
    return 1.0;
  }

  bool hasValidRange = false;
  long double totalIterations = 1.0L;
  for (const LoopRange &range : loop.ranges) {
    auto tripCount = computeTripCount(range);
    if (!tripCount) {
      return 1.0;
    }
    hasValidRange = true;
    totalIterations *= static_cast<long double>(*tripCount);
    if (!std::isfinite(totalIterations)) {
      return std::numeric_limits<double>::max();
    }
  }

  if (!hasValidRange) {
    return 1.0;
  }

  const long double clamped =
      std::min<long double>(totalIterations,
                            static_cast<long double>(
                                std::numeric_limits<double>::max()));
  return static_cast<double>(clamped);
}

class PerformanceModel {
public:
  explicit PerformanceModel(const HardwareSpec &spec) : spec_(spec) {}

  double memory(const MemoryLoad &load) const {
    return estimateMemoryDurationNs(load, spec_);
  }

  double memory(const MemoryStore &store) const {
    return estimateMemoryDurationNs(store, spec_);
  }

  double noc(const NoCWorkload &workload) const {
    return estimateNoCDurationNs(workload, spec_);
  }

  double noc(const MulticastWorkload &workload) const {
    return estimateNoCDurationNs(workload, spec_);
  }

  double compute(const ComputeWorkload &workload) const {
    return estimateComputeDurationNs(workload, spec_);
  }

private:
  const HardwareSpec &spec_;
};

enum class NoCLinkType {
  Horizontal,
  Vertical,
};

NoCLinkType classifyNoCLink(const CoreIndex &start, const CoreIndex &end,
                            const std::string &network) {
  std::string lowered = network;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  auto contains = [&](const char *token) {
    return lowered.find(token) != std::string::npos;
  };

  if (contains("all_links")) {
    const std::int64_t dx = std::abs(end.x - start.x);
    const std::int64_t dy = std::abs(end.y - start.y);
    return dx >= dy ? NoCLinkType::Horizontal : NoCLinkType::Vertical;
  }
  if (contains("ver") || contains("vec")) {
    return NoCLinkType::Vertical;
  }
  if (contains("hor")) {
    return NoCLinkType::Horizontal;
  }

  const std::int64_t dx = std::abs(end.x - start.x);
  const std::int64_t dy = std::abs(end.y - start.y);
  return dy > dx ? NoCLinkType::Vertical : NoCLinkType::Horizontal;
}

class LoopAnalyzer {
public:
  explicit LoopAnalyzer(const HardwareSpec &spec) : model_(spec) {}

  InnermostLoopSummary
  analyze(const std::vector<CoreSequentialWorkload> &workloads) {
    for (const auto &coreWork : workloads) {
      processCore(coreWork);
    }
    bestSummary_.totalWorkloadDurationNs = worstCoreDurationNs_;
    return bestSummary_;
  }

private:
  struct Frame {
    double memoryCount = 0.0;
    double nocCount = 0.0;
    double computeCount = 0.0;
    double memoryDurationNs = 0.0;
    double nocHorizontalDurationNs = 0.0;
    double nocVerticalDurationNs = 0.0;
    double computeDurationNs = 0.0;
    double nestedDurationNs = 0.0;
    double iterationCount = 1.0;
    bool hasNestedLoop = false;

    bool hasWork() const {
      return memoryCount > 0 || nocCount > 0 || computeCount > 0;
    }

    double nocDurationNs() const {
      return std::max(nocHorizontalDurationNs, nocVerticalDurationNs);
    }

    double directDuration() const {
      return std::max({memoryDurationNs, nocDurationNs(), computeDurationNs});
    }
  };

  void processCore(const CoreSequentialWorkload &coreWork) {
    std::vector<Frame> loopStack;
    loopStack.reserve(8);
    double coreTotalDurationNs = 0.0;

    for (const auto &event : coreWork.events) {
      std::visit(
          [&](const auto &payload) {
            handleEvent(payload, loopStack, coreWork.core, coreTotalDurationNs);
          },
          event);
    }

    worstCoreDurationNs_ = std::max(worstCoreDurationNs_, coreTotalDurationNs);
  }

  template <typename Payload>
  void handleEvent(const Payload &payload, std::vector<Frame> &loopStack,
                   const CoreIndex &core, double &coreTotalDurationNs) {
    using T = std::decay_t<Payload>;
    if constexpr (std::is_same_v<T, LoopBegin>) {
      handleLoopBegin(payload, loopStack);
    } else if constexpr (std::is_same_v<T, LoopEnd>) {
      handleLoopEnd(loopStack, core, coreTotalDurationNs);
    } else if constexpr (std::is_same_v<T, MemoryLoad>) {
      accumulateMemory(payload, loopStack);
    } else if constexpr (std::is_same_v<T, MemoryStore>) {
      accumulateMemory(payload, loopStack);
    } else if constexpr (std::is_same_v<T, NoCWorkload>) {
      accumulateNoC(payload, loopStack);
    } else if constexpr (std::is_same_v<T, MulticastWorkload>) {
      accumulateNoC(payload, loopStack);
    } else if constexpr (std::is_same_v<T, ComputeWorkload>) {
      accumulateCompute(payload, loopStack);
    }
  }

  void handleLoopBegin(const LoopBegin &loop, std::vector<Frame> &loopStack) {
    if (!loopStack.empty()) {
      loopStack.back().hasNestedLoop = true;
    }
    Frame frame;
    frame.iterationCount = computeLoopIterationCount(loop);
    loopStack.push_back(frame);
  }

  void handleLoopEnd(std::vector<Frame> &loopStack, const CoreIndex &core,
                     double &coreTotalDurationNs) {
    if (loopStack.empty()) {
      return;
    }

    Frame completed = loopStack.back();
    loopStack.pop_back();
    if (!completed.hasNestedLoop) {
      recordCandidate(completed, core);
    }

    const double directDuration = completed.directDuration();
    const double nocDuration = completed.nocDurationNs();
    const double iterations = std::max(1.0, completed.iterationCount);
    const double totalDuration =
        completed.computeDurationNs + completed.memoryDurationNs + nocDuration +
        completed.nestedDurationNs +
        (iterations - 1) * std::max(directDuration, completed.nestedDurationNs);

    if (!loopStack.empty()) {
      loopStack.back().nestedDurationNs += totalDuration;
    } else {
      coreTotalDurationNs += totalDuration;
    }
  }

  template <typename MemoryEvent>
  void accumulateMemory(const MemoryEvent &event, std::vector<Frame> &loopStack) {
    if (loopStack.empty()) {
      return;
    }
    auto &current = loopStack.back();
    current.memoryCount += 1;
    current.memoryDurationNs += model_.memory(event);
  }

  template <typename NoCEvent>
  void accumulateNoC(const NoCEvent &event, std::vector<Frame> &loopStack) {
    if (loopStack.empty()) {
      return;
    }

    auto &current = loopStack.back();
    current.nocCount += 1;
    const double duration = model_.noc(event);
    const NoCLinkType link =
        classifyNoCLink(event.start, event.end, event.network);
    if (link == NoCLinkType::Horizontal) {
      current.nocHorizontalDurationNs += duration;
    } else {
      current.nocVerticalDurationNs += duration;
    }
  }

  void accumulateCompute(const ComputeWorkload &compute,
                         std::vector<Frame> &loopStack) {
    if (loopStack.empty()) {
      return;
    }

    auto &current = loopStack.back();
    current.computeCount += 1;
    current.computeDurationNs += model_.compute(compute);
  }

  void recordCandidate(const Frame &frame, const CoreIndex &core) {
    if (!frame.hasWork()) {
      return;
    }

    InnermostLoopSummary summary;
    summary.core = core;
    summary.memoryDuration = frame.memoryDurationNs;
    summary.nocDuration = frame.nocDurationNs();
    summary.computeDuration = frame.computeDurationNs;
    summary.estimatedDurationNs = frame.directDuration();

    if (!hasCandidate_ ||
        summary.estimatedDurationNs > bestSummary_.estimatedDurationNs) {
      bestSummary_ = summary;
      hasCandidate_ = true;
    }
  }

  PerformanceModel model_;
  InnermostLoopSummary bestSummary_{};
  bool hasCandidate_ = false;
  double worstCoreDurationNs_ = 0.0;
};

} // namespace

InnermostLoopSummary
estimateLongestInnermostLoop(const std::vector<CoreSequentialWorkload> &workloads,
                             const HardwareSpec &spec) {
  LoopAnalyzer analyzer(spec);
  return analyzer.analyze(workloads);
}

} // namespace cost_model
} // namespace loom
