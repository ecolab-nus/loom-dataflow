#ifndef LOOM_PASSES_TRITON_SHARED_H
#define LOOM_PASSES_TRITON_SHARED_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace loom {
namespace passes {

#define GEN_PASS_DECL
#include "Passes.h.inc"

// Pass factory functions
std::unique_ptr<mlir::Pass> createTritonSharedAffinizePass();
std::unique_ptr<mlir::Pass> createTritonSharedGridToParallelPass();
std::unique_ptr<mlir::Pass>
createTritonSharedExploreSpatialMappingsPass(bool withOuterFors = true);
std::unique_ptr<mlir::Pass> createHoistBlockLoadingPass();
std::unique_ptr<mlir::Pass> createAnnotateSubviewReusePass();
std::unique_ptr<mlir::Pass>
createEnumerateCopyBroadcastPass(bool analysisOnly = false);
std::unique_ptr<mlir::Pass> createTileScfForToL1Pass();
std::unique_ptr<mlir::Pass> createMaterializePass();
std::unique_ptr<mlir::Pass> createBridgeToOSBPass();
std::unique_ptr<mlir::Pass> createStaticizeTypesPass();
std::unique_ptr<mlir::Pass> createConstDedupCleanupPass();
std::unique_ptr<mlir::Pass> createMemoryBindingPass();
std::unique_ptr<mlir::Pass> createLinalgDestinationSpecializationPass();
std::unique_ptr<mlir::Pass> createLowerAffineWithAttrPass();

// Pass registration functions
void registerTritonSharedAffinizePass();
void registerTritonSharedGridToParallelPass();
void registerTritonSharedExploreSpatialMappingsPass();
void registerHoistBlockLoadingPass();
void registerAnnotateSubviewReusePass();
void registerEnumerateCopyBroadcastPass();
void registerTileScfForToL1Pass();
void registerMaterializePass();
void registerStaticizeTypesPass();
void registerConstDedupCleanupPass();
void registerMemoryBindingPass();
void registerLinalgDestinationSpecializationPass();
void registerLowerAffineWithAttrPass();

#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

} // namespace passes
} // namespace loom

#endif // LOOM_PASSES_TRITON_SHARED_H
