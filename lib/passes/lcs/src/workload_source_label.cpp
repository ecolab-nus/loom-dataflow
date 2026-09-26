#include "workload_source_label.h"
#include "hw_op_registry.h"
#include "compute_binding.h"
#include "ssa_utils.h"
#include "utils.h"

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

std::string makeDataMoverWorkloadLabel(mlir::Operation *data_mover_op,
                                       mlir::AsmState &asm_state) {
  mlir::SmallVector<mlir::Value> operands;
  if (auto copyOp = llvm::dyn_cast<loom::CopyOp>(data_mover_op))
    operands = {copyOp.getSource(), copyOp.getDestination()};
  else if (auto gatherOp = llvm::dyn_cast<loom::GatherOp>(data_mover_op))
    operands = {gatherOp.getSource(), gatherOp.getDestination()};
  else
    llvm_unreachable("expected loom.copy or loom.gather");
  return makeWorkloadLabel(data_mover_op, operands, asm_state);
}

namespace {

/// `%ssa: <memory>;...` where the memory is the explicit endpoint if given,
/// else the memory of the allocation the operand is backed by.
mlir::FailureOr<std::string> formatOperandAccessList(
    mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
    mlir::AsmState &asmState,
    llvm::ArrayRef<mlir::SymbolRefAttr> endpoints = {}) {
  std::string result;
  llvm::raw_string_ostream os(result);
  for (auto [index, operand] : llvm::enumerate(operands)) {
    if (index)
      os << ";";
    operand.printAsOperand(os, asmState);
    llvm::StringRef memory;
    if (index < endpoints.size() && endpoints[index])
      memory = loom::utils::memoryName(endpoints[index].getLeafReference());
    else if (auto alloc = loom::utils::traceToRootAllocOp(operand))
      memory = alloc.getMemory().getLeafReference();
    if (memory.empty()) {
      op->emitError() << "staged-etg: cannot determine the memory of operand "
                      << index;
      return mlir::failure();
    }
    os << ": " << memory;
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
makeGenericSiteAccessMetadata(const ComputeBindingSite &site,
                              mlir::AsmState &asmState) {
  auto linalgOp = mlir::cast<mlir::linalg::LinalgOp>(site.enclosing_op);
  mlir::SmallVector<mlir::Value> operands = getLinalgCompactOperands(linalgOp);
  mlir::SmallVector<mlir::Value> reads;
  mlir::SmallVector<mlir::Value> writes;
  for (const BindingSiteValue &input : site.inputs)
    if (input.kind == BindingSiteValue::ExternalOperand)
      reads.push_back(operands[input.dps_index]);
  for (const BindingSiteValue &output : site.outputs)
    for (unsigned dpsIndex : output.external_dps_indices)
      writes.push_back(operands[dpsIndex]);
  auto read = formatOperandAccessList(site.payload_op, reads, asmState);
  auto write = formatOperandAccessList(site.payload_op, writes, asmState);
  if (mlir::failed(read) || mlir::failed(write))
    return mlir::failure();
  return OperandAccessMetadata{*read, *write};
}

mlir::FailureOr<OperandAccessMetadata>
makeDataMoverOperandAccessMetadata(mlir::Operation *op,
                                   mlir::AsmState &asmState) {
  mlir::Value source;
  mlir::Value destination;
  mlir::SymbolRefAttr sourceMemory;
  mlir::SymbolRefAttr destinationMemory;
  if (auto copy = llvm::dyn_cast<loom::CopyOp>(op)) {
    source = copy.getSource();
    destination = copy.getDestination();
    sourceMemory = copy.getSrcMemSpaceAttr();
    destinationMemory = copy.getDstMemSpaceAttr();
  } else if (auto gather = llvm::dyn_cast<loom::GatherOp>(op)) {
    source = gather.getSource();
    destination = gather.getDestination();
    sourceMemory = gather.getSrcMemSpaceAttr();
    destinationMemory = gather.getDstMemSpaceAttr();
  } else {
    op->emitError() << "staged-etg: executable data mover has no operand "
                       "access adapter";
    return mlir::failure();
  }

  auto read = formatOperandAccessList(op, {source}, asmState, {sourceMemory});
  auto write = formatOperandAccessList(op, {destination}, asmState,
                                       {destinationMemory});
  if (mlir::failed(read) || mlir::failed(write))
    return mlir::failure();
  return OperandAccessMetadata{*read, *write};
}

} // namespace lcs
} // namespace loom
