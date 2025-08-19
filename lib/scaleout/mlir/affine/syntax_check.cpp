#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace tmd_affine_analysis {

// Verify the simplified IR contract from README.
LogicalResult runSyntaxCheck(func::FuncOp funcOp) {
  bool sawParallel = false;
  for (Operation &op : funcOp.getBody().front()) {
    if (isa<affine::AffineParallelOp>(&op)) {
      sawParallel = true;
      LogicalResult bodyOK = success();
      op.walk([&](Operation *nested) {
        if (isa<ModuleOp>(nested) || isa<func::FuncOp>(nested))
          return;
        if (isa<affine::AffineApplyOp>(nested)) {
          bodyOK = failure();
          return;
        }
        if (nested->getDialect() &&
            llvm::isa<affine::AffineDialect>(nested->getDialect()))
          return;
        if (isa<func::ReturnOp>(nested))
          return;
        bodyOK = failure();
      });
      if (failed(bodyOK))
        return funcOp.emitOpError(
            "violates Analysis IR contract: contains disallowed ops");
    }
  }
  if (!sawParallel)
    return funcOp.emitOpError(
        "expected outermost affine.parallel over core grid");
  return success();
}

} // namespace tmd_affine_analysis
