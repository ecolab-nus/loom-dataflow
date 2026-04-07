#ifndef LOOM_PASSES_TRITON_SHARED_H
#define LOOM_PASSES_TRITON_SHARED_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include <memory>

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace loom {
namespace passes {

// Map: func_name → {symbol_name → concrete_value}
// Used to pass external solver results directly to the materialize pass.
using BlockSizeMap = llvm::StringMap<llvm::StringMap<int64_t>>;

#define GEN_PASS_DECL
#include "Passes.h.inc"

// Pass factory functions
std::unique_ptr<mlir::Pass> createTritonSharedExploreSpatialMappingsPass();
std::unique_ptr<mlir::Pass> createHoistBlockLoadingPass();
std::unique_ptr<mlir::Pass> createAnnotateSubviewReusePass();
std::unique_ptr<mlir::Pass> createEnumerateCopyBroadcastPass();
std::unique_ptr<mlir::Pass> createMaterializePass();
std::unique_ptr<mlir::Pass> createMaterializePass(const BlockSizeMap &blockSizes);
std::unique_ptr<mlir::Pass> createBridgeToOSBPass();
std::unique_ptr<mlir::Pass> createMemoryBindingPass();
std::unique_ptr<mlir::Pass> createLinalgDestinationSpecializationPass();
std::unique_ptr<mlir::Pass> createFoldRedundantExtractSlicePass();
std::unique_ptr<mlir::Pass> createSinkFillOpsPass();
std::unique_ptr<mlir::Pass> createLowerAffineWithAttrPass();

// Pass registration functions
void registerTritonSharedExploreSpatialMappingsPass();

#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

} // namespace passes
} // namespace loom

#endif // LOOM_PASSES_TRITON_SHARED_H
