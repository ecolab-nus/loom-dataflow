#ifndef LOOM_COST_MODEL_PASSES
#define LOOM_COST_MODEL_PASSES

include "mlir/Pass/PassBase.td"

def LoomCostModel : Pass<"loom-cost-model", "mlir::ModuleOp"> {
  let summary = "Estimate per-function runtime and keep top-k candidates";
  let description = [{
    Build per-core sequential workloads from post-OSB style IR (loom.copy /
    memref.copy patterns), estimate runtime with a TMDSimulator-derived model,
    annotate candidate functions with loom.cost.* attributes, and prune to top-k.
  }];
  let constructor = "loom::cost_model::createLoomCostModelPass()";
  let options = [
    Option<"topK", "top-k", "int", "1",
           "Keep only the best-k candidate kernels">,
    Option<"dramBandwidthGbps", "dram-bandwidth-gbps", "double", "15.0",
           "DRAM bandwidth override">,
    Option<"dramLatencyNs", "dram-latency-ns", "double", "454.0",
           "DRAM latency override">,
    Option<"nocBandwidthGbps", "noc-bandwidth-gbps", "double", "27.88",
           "NoC bandwidth override">,
    Option<"allLinksBandwidthGbps", "all-links-bandwidth-gbps", "double", "18.235",
           "All-links interconnect bandwidth override">,
    Option<"nocLatencyNs", "noc-latency-ns", "double", "344.0",
           "NoC latency override">,
    Option<"allLinksLatencyNs", "all-links-latency-ns", "double", "586.0",
           "All-links interconnect latency override">,
    Option<"matrixGflops", "matrix-gflops", "double", "89.6",
           "Matrix compute throughput override">,
    Option<"vectorGflops", "vector-gflops", "double", "100.0",
           "Vector compute throughput override">,
    Option<"elementBytes", "element-bytes", "double", "2.0",
           "Element size in bytes used by memory model">,
    Option<"matrixUnit", "matrix-unit", "double", "8.0",
           "Max matrix parallel unit">,
    Option<"vectorUnit", "vector-unit", "double", "1.0",
           "Vector parallel unit">,
  ];
}

#endif // LOOM_COST_MODEL_PASSES
