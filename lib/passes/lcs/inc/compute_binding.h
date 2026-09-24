#ifndef LOOM_LCS_COMPUTE_BINDING_H
#define LOOM_LCS_COMPUTE_BINDING_H

#include "hw_op_registry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/LogicalResult.h"
#include <vector>

namespace loom::lcs {

struct BindingSiteValue {
  enum Kind { ExternalOperand, InternalValue, Constant } kind;
  mlir::Value value;
  unsigned dps_index = 0;
  std::vector<unsigned> external_dps_indices;
};

struct ComputeBindingSite {
  mlir::Operation *enclosing_op = nullptr;
  mlir::Operation *payload_op = nullptr;
  size_t id = 0;
  bool is_named = false;
  HWOpKey semantic_key;
  std::vector<BindingSiteValue> inputs;
  std::vector<BindingSiteValue> outputs;
};

bool isComputeLinalg(mlir::Operation *op);
mlir::FailureOr<std::vector<ComputeBindingSite>>
analyzeComputeBindingSites(mlir::func::FuncOp function);

bool candidateMatchesSite(const HWComputeFunc &candidate,
                          const ComputeBindingSite &site);

} // namespace loom::lcs

#endif
