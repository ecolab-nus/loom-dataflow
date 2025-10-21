/**
 * @file triton_shared_spatial_mapping_pass.cpp
 * @brief Implementation of spatial mapping enumeration as an MLIR pass.
 */

#include "triton_shared_spatial_mapping_pass.h"

#include "spatial_mapping.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include "DataflowDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "DataflowTypes.h.inc"
#define GET_OP_CLASSES
#include "DataflowOps.h.inc"

using namespace mlir;

namespace {

struct TritonSharedExploreSpatialMappingsPass
    : public PassWrapper<TritonSharedExploreSpatialMappingsPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TritonSharedExploreSpatialMappingsPass)

  TritonSharedExploreSpatialMappingsPass() = default;
  explicit TritonSharedExploreSpatialMappingsPass(bool withOuterFors)
      : withOuterFors(withOuterFors) {}

  StringRef getArgument() const override {
    return "tmd-triton-shared-explore-spatial-mappings";
  }
  StringRef getDescription() const override {
    return "Enumerate spatial mappings for Triton-shared kernels using DF dims";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    (void)module.getContext();

    // Collect spatial dimensions from DF ops in this module.
    llvm::SmallVector<tmd_affine::SpatialDimInfo, 8> spatialDims;
    if (failed(tmd_affine::collectSpatialDims(module, spatialDims)))
      return; // No DF spatial dims; silently no-op

    // Create an enumerated output module using existing utilities.
    OwningOpRef<ModuleOp> enumerated =
        withOuterFors
            ? tmd_affine::enumerateSpatialMappingsWithOuterFors(module,
                                                                spatialDims)
            : tmd_affine::enumerateSpatialMappings(module, spatialDims);

    // If enumeration produced no functions, keep the original functions.
    bool producedAnyFunc = false;
    for (Operation &op : *enumerated->getBody())
      if (isa<func::FuncOp>(&op)) {
        producedAnyFunc = true;
        break;
      }
    if (!producedAnyFunc)
      return; // leave module unchanged

    // Erase all non-DF top-level ops from the original module; keep DF.
    SmallVector<Operation *, 16> toErase;
    for (Operation &op : *module.getBody()) {
      Dialect *dialect = op.getDialect();
      if (dialect && dialect->getNamespace() == StringRef("df"))
        continue;
      toErase.push_back(&op);
    }
    for (auto it = toErase.rbegin(); it != toErase.rend(); ++it)
      (*it)->erase();

    // Insert enumerated clones after the last DF op to keep DF at the top.
    OpBuilder builder(module.getBodyRegion());
    Operation *after = nullptr;
    for (Operation &op : *module.getBody()) {
      Dialect *dialect = op.getDialect();
      if (dialect && dialect->getNamespace() == StringRef("df"))
        after = &op;
    }
    if (after)
      builder.setInsertionPointAfter(after);
    else
      builder.setInsertionPointToStart(module.getBody());

    IRMapping mapping;
    for (Operation &op : *enumerated->getBody()) {
      Dialect *dialect = op.getDialect();
      if (dialect && dialect->getNamespace() == StringRef("df"))
        continue; // skip DF decls from the enumerated module
      builder.clone(op, mapping);
    }
  }

  bool withOuterFors = true;
};

} // namespace

std::unique_ptr<mlir::Pass>
tmd::passes::createTritonSharedExploreSpatialMappingsPass(bool withOuterFors) {
  return std::make_unique<TritonSharedExploreSpatialMappingsPass>(
      withOuterFors);
}

void tmd::passes::registerTritonSharedExploreSpatialMappingsPass() {
  PassRegistration<TritonSharedExploreSpatialMappingsPass>();
}
