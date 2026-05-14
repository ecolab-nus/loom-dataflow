#pragma once

#include "expr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <string>
#include <vector>

namespace loom {
namespace lcs {

struct L1FootprintResult {
  std::vector<Expr> l1_footprint;
  std::string datatype;
};

class L1FootprintEstimator {
public:
  static L1FootprintResult estimateFromFunc(mlir::func::FuncOp funcOp);
};

} // namespace lcs
} // namespace loom

