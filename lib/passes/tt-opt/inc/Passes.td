#ifndef LOOM_TT_OPT_PASSES
#define LOOM_TT_OPT_PASSES

include "mlir/Pass/PassBase.td"

def FuseZeroFillMatmul : Pass<"tt-fuse-zero-fill-matmul", "mlir::ModuleOp"> {
  let summary = "Fuse linalg.fill(0) and linalg.matmul into loom.matmul";
  let description = [{
    Detects the pattern where a `linalg.matmul` op's `outs` operand is defined 
    by a `linalg.fill` with constant-zero, and replaces both with a single 
    `loom.matmul` op.
  }];
  let constructor = "loom::passes::createFuseZeroFillMatmulPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "loom::LoomDialect"
  ];
}

#endif // LOOM_TT_OPT_PASSES
