#ifndef LOOM_PASSES_TT_OPT_H
#define LOOM_PASSES_TT_OPT_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace loom {
namespace passes {

#define GEN_PASS_DECL
#include "Passes.h.inc"

std::unique_ptr<mlir::Pass> createFuseZeroFillMatmulPass();

#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

} // namespace passes
} // namespace loom

#endif // LOOM_PASSES_TT_OPT_H
