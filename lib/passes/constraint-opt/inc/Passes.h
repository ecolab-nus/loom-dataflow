#ifndef LOOM_PASSES_CONSTRAINT_OPT_H
#define LOOM_PASSES_CONSTRAINT_OPT_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace loom {
namespace constraint_opt {

#define GEN_PASS_DECL
#include "Passes.h.inc"

std::unique_ptr<mlir::Pass> createLoomConstraintCanonicalizePass();
std::unique_ptr<mlir::Pass> createLoomConstraintFactorizePass();
std::unique_ptr<mlir::Pass> createLoomConstraintDecomposePass();
std::unique_ptr<mlir::Pass> createLoomConstraintLinearizePass();
std::unique_ptr<mlir::Pass> createLoomCompressIntermediateVarPass();
std::unique_ptr<mlir::Pass> createLoomConstraintSimplifyPass();

#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

} // namespace constraint_opt
} // namespace loom

#endif // LOOM_PASSES_CONSTRAINT_OPT_H
