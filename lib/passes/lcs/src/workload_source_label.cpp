#include "workload_source_label.h"
#include "hw_op_registry.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

namespace loom {
namespace lcs {

std::string makeWorkloadLabel(mlir::Operation *label_op,
                              llvm::ArrayRef<mlir::Value> operands,
                              mlir::AsmState &asm_state) {
  std::string label;
  llvm::raw_string_ostream os(label);
  os << label_op->getName().getStringRef() << "(";
  for (auto [idx, operand] : llvm::enumerate(operands)) {
    if (idx != 0)
      os << ", ";
    operand.printAsOperand(os, asm_state);
    mlir::FailureOr<int64_t> localMemKind =
        getLocalMemKind(operand.getType(), label_op, idx);
    if (mlir::succeeded(localMemKind) && *localMemKind != 0)
      os << ": " << *localMemKind;
  }
  os << ")";
  return os.str();
}

mlir::SmallVector<mlir::Value>
getLinalgCompactOperands(mlir::linalg::LinalgOp op) {
  mlir::SmallVector<mlir::Value> operands;
  for (mlir::Value input : op.getDpsInputs())
    operands.push_back(input);
  for (mlir::Value init : op.getDpsInits())
    operands.push_back(init);
  return operands;
}

std::string makeNamedLinalgWorkloadLabel(mlir::linalg::LinalgOp op,
                                         mlir::AsmState &asm_state) {
  return makeWorkloadLabel(op.getOperation(), getLinalgCompactOperands(op),
                           asm_state);
}

std::string makeGenericPayloadWorkloadLabel(mlir::Operation *payload_op,
                                            mlir::linalg::LinalgOp generic_op,
                                            mlir::AsmState &asm_state) {
  return makeWorkloadLabel(payload_op, getLinalgCompactOperands(generic_op),
                           asm_state);
}

std::string makeDataMoverWorkloadLabel(mlir::Operation *data_mover_op,
                                       mlir::AsmState &asm_state) {
  mlir::SmallVector<mlir::Value> operands;
  if (auto copyOp = llvm::dyn_cast<loom::CopyOp>(data_mover_op)) {
    operands = {copyOp.getSource(), copyOp.getDestination()};
  } else if (auto gatherOp = llvm::dyn_cast<loom::GatherOp>(data_mover_op)) {
    operands = {gatherOp.getSource(), gatherOp.getDestination()};
  } else {
    llvm_unreachable("expected loom.copy or loom.gather");
  }
  return makeWorkloadLabel(data_mover_op, operands, asm_state);
}

} // namespace lcs
} // namespace loom
