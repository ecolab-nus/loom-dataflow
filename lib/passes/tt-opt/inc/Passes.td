#ifndef LOOM_TT_OPT_PASSES
#define LOOM_TT_OPT_PASSES

include "mlir/Pass/PassBase.td"

def FoldZeroFillLinalg : Pass<"tt-fold-zero-fill-linalg", "mlir::ModuleOp"> {
  let summary = "Fold linalg.fill(0) feeding linalg outs operands";
  let description = [{
    Detects linalg destination operands initialized by a `linalg.fill` with
    constant zero, removes the redundant fill when there is no intervening use,
    and preserves the existing linalg.matmul/batch_matmul lowering to Loom ops.
  }];
  let constructor = "loom::passes::createFoldZeroFillLinalgPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "loom::LoomDialect"
  ];
}

#endif // LOOM_TT_OPT_PASSES
