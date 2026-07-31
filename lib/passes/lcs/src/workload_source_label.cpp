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
                              mlir::AsmState &asm_state,
                              llvm::ArrayRef<std::optional<int64_t>>
                                  operandMemKinds) {
  std::string label;
  llvm::raw_string_ostream os(label);
  os << label_op->getName().getStringRef() << "(";
  for (auto [idx, operand] : llvm::enumerate(operands)) {
    if (idx != 0)
      os << ", ";
    operand.printAsOperand(os, asm_state);
    std::optional<int64_t> localMemKind;
    if (idx < operandMemKinds.size() && operandMemKinds[idx]) {
      localMemKind = operandMemKinds[idx];
    } else {
      mlir::FailureOr<int64_t> typeMemKind =
          getLocalMemKind(operand.getType(), label_op, idx);
      if (mlir::succeeded(typeMemKind))
        localMemKind = *typeMemKind;
    }
    if (localMemKind && *localMemKind != 0)
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
  mlir::SmallVector<std::optional<int64_t>> operandMemKinds;
  if (auto copyOp = llvm::dyn_cast<loom::CopyOp>(data_mover_op)) {
    operands = {copyOp.getSource(), copyOp.getDestination()};
    operandMemKinds = {
        copyOp.getSrcMemKindAttr()
            ? std::optional<int64_t>(copyOp.getSrcMemKindAttr().getInt())
            : std::nullopt,
        copyOp.getDstMemKindAttr()
            ? std::optional<int64_t>(copyOp.getDstMemKindAttr().getInt())
            : std::nullopt,
    };
  } else if (auto gatherOp = llvm::dyn_cast<loom::GatherOp>(data_mover_op)) {
    operands = {gatherOp.getSource(), gatherOp.getDestination()};
    operandMemKinds.resize(operands.size());
  } else {
    llvm_unreachable("expected loom.copy or loom.gather");
  }
  return makeWorkloadLabel(data_mover_op, operands, asm_state,
                           operandMemKinds);
}

namespace {

mlir::FailureOr<std::string> formatOperandAccessList(
    mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
    mlir::AsmState &asmState,
    llvm::ArrayRef<std::optional<int64_t>> operandMemKinds = {}) {
  std::string result;
  llvm::raw_string_ostream os(result);
  for (auto [index, operand] : llvm::enumerate(operands)) {
    if (index)
      os << ";";
    operand.printAsOperand(os, asmState);
    mlir::FailureOr<int64_t> typeMemKind =
        getLocalMemKind(operand.getType(), op, index);
    if (mlir::failed(typeMemKind))
      return mlir::failure();
    int64_t memKind =
        index < operandMemKinds.size() && operandMemKinds[index]
            ? *operandMemKinds[index]
            : *typeMemKind;
    os << ": " << memKind;
  }
  return result;
}

} // namespace

mlir::FailureOr<OperandAccessMetadata>
makeLinalgOperandAccessMetadata(mlir::linalg::LinalgOp op,
                                mlir::AsmState &asmState) {
  mlir::SmallVector<mlir::Value> inputs = op.getDpsInputs();
  mlir::OperandRange dpsInits = op.getDpsInits();
  mlir::SmallVector<mlir::Value> inits(dpsInits.begin(), dpsInits.end());
  auto read = formatOperandAccessList(op.getOperation(), inputs, asmState);
  auto write = formatOperandAccessList(op.getOperation(), inits, asmState);
  if (mlir::failed(read) || mlir::failed(write))
    return mlir::failure();
  return OperandAccessMetadata{*read, *write};
}

mlir::FailureOr<OperandAccessMetadata>
makeDataMoverOperandAccessMetadata(mlir::Operation *op,
                                   mlir::AsmState &asmState) {
  mlir::Value source;
  mlir::Value destination;
  std::optional<int64_t> sourceKind;
  std::optional<int64_t> destinationKind;
  if (auto copy = llvm::dyn_cast<loom::CopyOp>(op)) {
    source = copy.getSource();
    destination = copy.getDestination();
    if (copy.getSrcMemKindAttr())
      sourceKind = copy.getSrcMemKindAttr().getInt();
    if (copy.getDstMemKindAttr())
      destinationKind = copy.getDstMemKindAttr().getInt();
  } else if (auto gather = llvm::dyn_cast<loom::GatherOp>(op)) {
    source = gather.getSource();
    destination = gather.getDestination();
  } else {
    op->emitError() << "staged-etg: executable data mover has no operand "
                       "access adapter";
    return mlir::failure();
  }

  auto read = formatOperandAccessList(op, {source}, asmState, {sourceKind});
  auto write =
      formatOperandAccessList(op, {destination}, asmState, {destinationKind});
  if (mlir::failed(read) || mlir::failed(write))
    return mlir::failure();
  return OperandAccessMetadata{*read, *write};
}

} // namespace lcs
} // namespace loom
