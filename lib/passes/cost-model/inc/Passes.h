#ifndef LOOM_PASSES_COST_MODEL_H
#define LOOM_PASSES_COST_MODEL_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace loom {
namespace cost_model {

struct CostModelOptions {
  int topK = 1;
  double dramBandwidthGbps = 15.0;
  double dramLatencyNs = 454.0;
  double nocBandwidthGbps = 27.88;
  double allLinksBandwidthGbps = 18.235;
  double nocLatencyNs = 344.0;
  double allLinksLatencyNs = 586.0;
  double matrixGflops = 89.6;
  double vectorGflops = 100.0;
  double elementBytes = 2.0;
  double matrixUnit = 8.0;
  double vectorUnit = 1.0;
};

#define GEN_PASS_DECL
#include "Passes.h.inc"

std::unique_ptr<mlir::Pass> createLoomCostModelPass();
std::unique_ptr<mlir::Pass>
createLoomCostModelPass(const CostModelOptions &options);

#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

} // namespace cost_model
} // namespace loom

#endif // LOOM_PASSES_COST_MODEL_H
