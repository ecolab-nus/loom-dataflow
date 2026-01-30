#ifndef LOOM_CONSTRAINT_OPT_PASSES
#define LOOM_CONSTRAINT_OPT_PASSES

include "mlir/Pass/PassBase.td"

def LoomConstraintCanonicalize : Pass<"loom-constraint-canonicalize", "mlir::ModuleOp"> {
  let summary = "Canonicalize polynomial constraints";
  let description = [{
    Normalizes polynomial constraints by sorting monomials and merging like terms.
  }];
  let constructor = "loom::constraint_opt::createLoomConstraintCanonicalizePass()";
}

def LoomConstraintFactorize : Pass<"loom-constraint-factorize", "mlir::ModuleOp"> {
  let summary = "Factorize polynomial constraints";
  let description = [{
    Applies greedy factorization to polynomial constraints to reduce dimensionality.
  }];
  let constructor = "loom::constraint_opt::createLoomConstraintFactorizePass()";
}

def LoomConstraintDecompose : Pass<"loom-constraint-decompose", "mlir::ModuleOp"> {
  let summary = "Decompose non-linear polynomial constraints";
  let description = [{
    Decomposes higher-degree polynomial constraints into a sequence of binary operations
    using intermediate variables.
  }];
  let constructor = "loom::constraint_opt::createLoomConstraintDecomposePass()";
}

def LoomConstraintLinearize : Pass<"loom-constraint-linearize", "mlir::ModuleOp"> {
  let summary = "Linearize polynomial constraints";
  let description = [{
    Relax non-linear polynomial constraints into linear ones using McCormick envelopes.
  }];
  let constructor = "loom::constraint_opt::createLoomConstraintLinearizePass()";
}

def LoomCompressIntermediateVar : Pass<"loom-compress-intermediate-var", "mlir::ModuleOp"> {
  let summary = "Compress intermediate variables";
  let description = [{
    Eliminates redundant intermediate variables by folding linear relations.
  }];
  let constructor = "loom::constraint_opt::createLoomCompressIntermediateVarPass()";
}

def LoomConstraintSimplify : Pass<"loom-constraint-simplify", "mlir::ModuleOp"> {
  let summary = "Simplify constraint space";
  let description = [{
    Removes redundant constraints and simplifies the polyhedral representation of the space.
  }];
  let constructor = "loom::constraint_opt::createLoomConstraintSimplifyPass()";
}

#endif // LOOM_CONSTRAINT_OPT_PASSES
