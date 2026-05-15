#pragma once

#include "expr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <string>
#include <vector>

namespace loom {
namespace lcs {

struct L1FootprintByScope {
  std::vector<Expr> load;
  std::vector<Expr> compute;
  std::vector<Expr> store;

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
  static L1FootprintResult estimateFromFunc(mlir::func::FuncOp funcOp);
};

} // namespace lcs
} // namespace loom
