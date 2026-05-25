#ifndef LOOM_TT_OPT_PASSES
#define LOOM_TT_OPT_PASSES

include "mlir/Pass/PassBase.td"

def ConvertZeroFillLinalgMatmulToLoom
    : Pass<"tt-convert-zero-fill-linalg-matmul-to-loom", "mlir::ModuleOp"> {
  let summary = "Convert zero-initialized linalg matmul ops to Loom ops";
  let description = [{
    Detects same-block zero linalg.fill initialization for linalg.matmul and
    linalg.batch_matmul and rewrites the matmul op to loom.matmul or
    loom.batch_matmul without directly erasing the fill.
  }];
  let constructor =
      "loom::passes::createConvertZeroFillLinalgMatmulToLoomPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "loom::LoomDialect"
  ];
}

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

def SplitBinaryScalarChain
    : Pass<"tt-split-binary-scalar-chain", "mlir::ModuleOp"> {
  let summary = "Split safe fused binary scalar chains in linalg.generic ops";
  let description = [{
    Detects all-parallel single-output linalg.generic ops containing a safe
    binary scalar chain and splits them into adjacent generics that use the
    original output buffer as the intermediate storage.
  }];
  let constructor = "loom::passes::createSplitBinaryScalarChainPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif // LOOM_TT_OPT_PASSES
