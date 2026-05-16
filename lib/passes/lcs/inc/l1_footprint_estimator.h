#pragma once

#include "expr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cstdint>
#include <string>
#include <vector>

namespace loom {
namespace lcs {

class HWOpRegistry;

struct L1FootprintByScope {
  std::vector<Expr> load;
  std::vector<Expr> compute;
  std::vector<Expr> store;
  int64_t capacity = 0;

  bool empty() const {
    return load.empty() && compute.empty() && store.empty();
  }
};

struct L1FootprintResult {
  L1FootprintByScope l1_footprint;
  std::string datatype;
};

class L1FootprintEstimator {
public:
  static L1FootprintResult estimateFromFunc(mlir::func::FuncOp funcOp,
                                            const HWOpRegistry *registry);
};

} // namespace lcs
} // namespace loom
