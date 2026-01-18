/**
 * @file triton_shared_spatial_mapping_pass.cpp
 * @brief Implementation of spatial mapping enumeration as an MLIR pass.
 */

#include "Passes.h"

#include "enumerate_hw_mapping.h"

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
    return "loom-triton-shared-explore-spatial-mappings";
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
    loom_affine::HardwareInfo hardwareInfo;
    if (failed(loom_affine::GetHardwareInfoForExploration(module, hardwareInfo)))
      return; // Failed to collect hardware information from DF module; silently no-op

    // Collect all functions first to avoid iterator invalidation
    SmallVector<func::FuncOp> funcs(module.getOps<func::FuncOp>().begin(),
                                     module.getOps<func::FuncOp>().end());
    
    if (funcs.empty())
      return;

    // Use EnumerateSpatialMappings to generate the enumerated functions,
    // then insert them directly into the original module.
    OwningOpRef<ModuleOp> enumerated = loom_affine::EnumerateSpatialMappings(module, hardwareInfo);

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
    // Note: Module attributes are automatically preserved since we operate
    // directly on the existing module without creating a new one.
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
loom::passes::createTritonSharedExploreSpatialMappingsPass(bool withOuterFors) {
  return std::make_unique<TritonSharedExploreSpatialMappingsPass>(
      withOuterFors);
}


