#ifndef LOOM_LCS_WORKLOAD_SOURCE_LABEL_H
#define LOOM_LCS_WORKLOAD_SOURCE_LABEL_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <string>

namespace loom {
namespace lcs {

struct OperandAccessMetadata {
  std::string read;
  std::string write;
};

std::string makeWorkloadLabel(mlir::Operation *label_op,
                              llvm::ArrayRef<mlir::Value> operands,
                              mlir::AsmState &asm_state,
                              llvm::ArrayRef<std::optional<int64_t>>
                                  operand_mem_kinds = {});

mlir::SmallVector<mlir::Value>
getLinalgCompactOperands(mlir::linalg::LinalgOp op);

std::string makeNamedLinalgWorkloadLabel(mlir::linalg::LinalgOp op,
                                         mlir::AsmState &asm_state);

std::string makeGenericPayloadWorkloadLabel(mlir::Operation *payload_op,
                                            mlir::linalg::LinalgOp generic_op,
                                            mlir::AsmState &asm_state);

std::string makeDataMoverWorkloadLabel(mlir::Operation *data_mover_op,
                                       mlir::AsmState &asm_state);

mlir::FailureOr<OperandAccessMetadata>
makeLinalgOperandAccessMetadata(mlir::linalg::LinalgOp op,
                                mlir::AsmState &asm_state);

mlir::FailureOr<OperandAccessMetadata>
makeDataMoverOperandAccessMetadata(mlir::Operation *data_mover_op,
                                   mlir::AsmState &asm_state);

} // namespace lcs
} // namespace loom

#endif // LOOM_LCS_WORKLOAD_SOURCE_LABEL_H
