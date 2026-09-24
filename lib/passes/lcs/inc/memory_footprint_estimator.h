#pragma once

#include "expr.h"
#include "target.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include <cstdint>
#include <string>
#include <vector>

namespace loom {
namespace lcs {

class HWOpRegistry;

struct MemoryFootprint {
  std::string memory;
  std::vector<Expr> load_bytes;
  std::vector<Expr> compute_bytes;
  std::vector<Expr> store_bytes;
  int64_t capacity_bytes = 0;

  bool empty() const {
    return load_bytes.empty() && compute_bytes.empty() && store_bytes.empty();
  }
};

struct MemoryFootprintResult {
  std::vector<MemoryFootprint> memory_footprints;
};

class MemoryFootprintEstimator {
public:
  static mlir::FailureOr<MemoryFootprintResult>
  estimateFromFunc(mlir::func::FuncOp funcOp, const HWOpRegistry *registry,
                   Target target);
};

} // namespace lcs
} // namespace loom
