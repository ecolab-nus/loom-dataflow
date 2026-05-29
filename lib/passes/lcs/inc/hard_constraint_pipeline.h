#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace loom {
namespace lcs {

class HWOpRegistry;
struct ConstraintScope;

class HardConstraintPipeline {
public:
  static void pushAll(mlir::func::FuncOp funcOp, const HWOpRegistry *registry,
                      ConstraintScope &scope);
};

} // namespace lcs
} // namespace loom
