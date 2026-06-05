#include "workload_source_label.h"

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

std::string makeCopyWorkloadLabel(mlir::Operation *copy_op,
                                  mlir::AsmState &asm_state) {
  auto op = llvm::cast<loom::CopyOp>(copy_op);
  mlir::SmallVector<mlir::Value> operands{op.getSource(),
                                          op.getDestination()};
  return makeWorkloadLabel(copy_op, operands, asm_state);
}

} // namespace lcs
} // namespace loom
