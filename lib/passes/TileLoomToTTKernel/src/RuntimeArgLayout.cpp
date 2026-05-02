/**
 * @file RuntimeArgLayout.cpp
 * @brief Runtime argument ABI layout for TileLoomToTTKernel kernels.
 */

#include "FuncOpToTTKernel.h"
#include "TTKernelAttrs.h"
#include "TTKernelUtils.h"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#define GET_OP_CLASSES
#include "LoomEnums.h.inc"
#include "LoomAttributes.h.inc"
#include "LoomOps.h.inc"

using namespace mlir;
using namespace mlir::loom;
using namespace tt::ttkernel;

DataMovementKernelSpec
mlir::loom::getDataMovementKernelSpec(DataMovementKernelRole role) {
  switch (role) {
  case DataMovementKernelRole::Reader:
    return {"riscv1", 0, "DataMovementProcessor::RISCV_1",
            "NOC::RISCV_0_default"};
  case DataMovementKernelRole::Writer:
    return {"riscv0", 1, "DataMovementProcessor::RISCV_0",
            "NOC::RISCV_1_default"};
  }
  llvm_unreachable("unknown data movement kernel role");
}

std::optional<KernelRuntimeRole>
mlir::loom::getKernelRuntimeRole(func::FuncOp func) {
  if (!func || func.getName().contains("__host"))
    return std::nullopt;

  if (auto threadAttr =
          func->getAttrOfType<ThreadTypeAttr>(ThreadTypeAttr::name)) {
    if (threadAttr.getValue() == ThreadType::Compute)
      return KernelRuntimeRole::Compute;
  }
  if (func.getName().ends_with("__compute"))
    return KernelRuntimeRole::Compute;

  if (auto dmRole = getDataMovementKernelRoleForFuncName(func.getName())) {
    return *dmRole == DataMovementKernelRole::Reader
               ? KernelRuntimeRole::Reader
               : KernelRuntimeRole::Writer;
  }
  if (auto dmSpec = resolveDataMovementKernelSpec(func)) {
    if (dmSpec->processorAttrValue == kDMProcessorRISCV1)
      return KernelRuntimeRole::Reader;
    if (dmSpec->processorAttrValue == kDMProcessorRISCV0)
      return KernelRuntimeRole::Writer;
  }

  return std::nullopt;
}

FailureOr<RuntimeArgLayout>
mlir::loom::buildRuntimeArgLayout(func::FuncOp func) {
  RuntimeArgLayout layout;
  auto role = getKernelRuntimeRole(func);
  if (!role)
    return func.emitError()
           << "cannot build TTKernel runtime layout for function without "
              "kernel role";

  SmallVector<::loom::CopyOp, 8> copyBindings;
  if (failed(collectCopyBindingOps(func, copyBindings)))
    return failure();

  auto addCopyBindingBase = [&](int64_t slot) {
    layout.add(RuntimeArgKey::copyBinding(slot, CopyBindingRuntimeField::Cb));
    layout.add(
        RuntimeArgKey::copyBinding(slot, CopyBindingRuntimeField::BaseAddr));
  };

  auto addCopyBindingMcast = [&](int64_t slot) {
    layout.add(RuntimeArgKey::copyBinding(
        slot, CopyBindingRuntimeField::McastDestStartX));
    layout.add(RuntimeArgKey::copyBinding(
        slot, CopyBindingRuntimeField::McastDestStartY));
    layout.add(RuntimeArgKey::copyBinding(
        slot, CopyBindingRuntimeField::McastDestEndX));
    layout.add(RuntimeArgKey::copyBinding(
        slot, CopyBindingRuntimeField::McastDestEndY));
    layout.add(RuntimeArgKey::copyBinding(slot,
                                          CopyBindingRuntimeField::McastDestNum));
    layout.add(RuntimeArgKey::copyBinding(
        slot, CopyBindingRuntimeField::McastSenderNocX));
    layout.add(RuntimeArgKey::copyBinding(
        slot, CopyBindingRuntimeField::McastSenderNocY));
    layout.add(RuntimeArgKey::copyBinding(
        slot, CopyBindingRuntimeField::McastSenderSemaphore));
    layout.add(RuntimeArgKey::copyBinding(
        slot, CopyBindingRuntimeField::McastReceiverSemaphore));
  };

  for (::loom::CopyOp copyOp : copyBindings) {
    auto slot = getCopyBindingSlot(copyOp.getOperation());
    if (!slot)
      return copyOp.emitOpError("missing required copy binding slot");

    if (*role == KernelRuntimeRole::Compute) {
      layout.add(
          RuntimeArgKey::copyBinding(*slot, CopyBindingRuntimeField::Cb));
      continue;
    }

    if (isDramToL1Copy(copyOp)) {
      addCopyBindingBase(*slot);
      if (hasRuntimeBroadcast(copyOp))
        addCopyBindingMcast(*slot);
      continue;
    }

    if (isL1ToDramCopy(copyOp)) {
      addCopyBindingBase(*slot);
      continue;
    }

    return copyOp.emitOpError()
           << "copy binding slot " << *slot
           << " is neither DRAM->L1 nor L1->DRAM";
  }

  if (*role == KernelRuntimeRole::Writer && func->hasAttr(kHasReduceAttrName)) {
    layout.add(RuntimeArgKey::reduce(ReduceRuntimeField::ReadySemaphore));
    layout.add(RuntimeArgKey::reduce(ReduceRuntimeField::TokenSemaphore));
    layout.add(RuntimeArgKey::reduce(ReduceRuntimeField::TokenMcastDestStartX));
    layout.add(RuntimeArgKey::reduce(ReduceRuntimeField::TokenMcastDestStartY));
    layout.add(RuntimeArgKey::reduce(ReduceRuntimeField::TokenMcastDestEndX));
    layout.add(RuntimeArgKey::reduce(ReduceRuntimeField::TokenMcastDestEndY));
  }

  if (*role == KernelRuntimeRole::Compute) {
    if (auto scalarSiteCountAttr =
            func->getAttrOfType<IntegerAttr>(kScalarSiteCountAttrName)) {
      for (int64_t siteId = 0; siteId < scalarSiteCountAttr.getInt(); ++siteId)
        layout.add(RuntimeArgKey::scalarSite(siteId));
    }
  }

  if (hasMappedParallel(func)) {
    layout.add(RuntimeArgKey::coreCoord(CoreCoordRuntimeField::X));
    layout.add(RuntimeArgKey::coreCoord(CoreCoordRuntimeField::Y));
  }

  if (*role == KernelRuntimeRole::Compute ||
      *role == KernelRuntimeRole::Writer) {
    for (int64_t slot : collectInternalSemaphoreSlots(func))
      layout.add(RuntimeArgKey::internalCb(slot));
  }

  return layout;
}
