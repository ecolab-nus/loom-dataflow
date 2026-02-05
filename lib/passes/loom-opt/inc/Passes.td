#ifndef LOOM_TRITON_SHARED_PASSES
#define LOOM_TRITON_SHARED_PASSES

include "mlir/Pass/PassBase.td"



def TritonSharedExploreSpatialMappings : Pass<"loom-triton-shared-explore-spatial-mappings", "mlir::ModuleOp"> {
  let summary = "Enumerate spatial mappings";
  let description = [{
    Enumerate mappings from hardware df.spatial_dim declarations to the 
    outermost affine.parallel iterators; clone per mapping, annotate inner 
    loops with loom.mapped_to, and insert outer affine.for waves when needed.
  }];
  let constructor = "loom::passes::createTritonSharedExploreSpatialMappingsPass()";
  let options = [
    Option<"withOuterFors", "with-outer-fors", "bool", "true",
           "Convert remaining affine.parallel to affine.for with all permutations">
  ];
}

def HoistBlockLoading : Pass<"loom-hoist-block-loading", "mlir::ModuleOp"> {
  let summary = "Hoist block loading operations";
  let description = [{
    Hoist block loading operations from innermost scf.for loops to outer loop 
    levels. For each function, identifies loading blocks and clones the function 
    per loading block, hoisting each block to reduce redundant memory accesses.
  }];
  let constructor = "loom::passes::createHoistBlockLoadingPass()";
}

def AnnotateSubviewReuse : Pass<"loom-annotate-subview-reuse", "mlir::ModuleOp"> {
  let summary = "Analyze reuse pattern on loom.subview";
  let description = [{
    Attach a loom.reuse dictionary to each loom.subview describing 
    how its offset varies with surrounding iterators, grouped by spatial 
    (affine.parallel), temporal (affine.for), and sequential (scf.for).
  }];
  let constructor = "loom::passes::createAnnotateSubviewReusePass()";
}

def EnumerateCopyBroadcast : Pass<"loom-enumerate-copy-broadcast", "mlir::ModuleOp"> {
  let summary = "Enumerate copy interconnect broadcast choices";
  let description = [{
    Annotate memref.alloc with loom.alloc and enumerate per-memref.copy 
    mapping choices: local memory copies and broadcasts along dimensions 
    with spatial total-reuse.
  }];
  let constructor = "loom::passes::createEnumerateCopyBroadcastPass()";
  let options = [
    Option<"analysisOnly", "analysis-only", "bool", "false",
           "Only attach loom.copy.candidates; do not clone functions">
  ];
}


def Materialize : Pass<"loom-materialize", "mlir::ModuleOp"> {
  let summary = "Materialize module attributes";
  let description = [{
    Replace loom.get_module_attribute operations with arith.constant values 
    by looking up the attribute in the module.
  }];
  let constructor = "loom::passes::createMaterializePass()";
}

def SubviewToReinterpretCast : Pass<"loom-subview-to-reinterpret-cast", "mlir::ModuleOp"> {
  let summary = "Convert loom.subview to loom.reinterpret_cast";
  let description = [{
    Linearize multi-dimensional offsets via affine.apply and adjust strides.
  }];
  let constructor = "loom::passes::createSubviewToReinterpretCastPass()";
}




def LinalgDestinationSpecializationPass : Pass<"loom-linalg-destination-specialization", "mlir::ModuleOp"> {
  let summary = "Specialize LinalgOp destinations";
  let description = [{
    Identify and eliminate redundant elementwise accumulation operations by
    folding them into the outs semantics of producer LinalgOps (e.g., matmul).
    This converts patterns like `matmul(A, B, fill(0)) + generic.add(iter_args)`
    into `matmul(A, B, iter_args)`.

    Currently supports arith.addf accumulation. Designed for future extension
    to other operators (addi, maximumf, mulf, etc.).
  }];
  let constructor = "loom::passes::createLinalgDestinationSpecializationPass()";
}

def MemoryBindingPass : Pass<"loom-memory-binding", "mlir::ModuleOp"> {
  let summary = "Bind memory allocations to tensor operations";
  let description = [{
    Transform bufferization patterns to loom dialect operations that explicitly
    bind physical memory allocations with tensor semantics for dataflow analysis.
  }];
  let constructor = "loom::passes::createMemoryBindingPass()";
}

#endif // LOOM_TRITON_SHARED_PASSES
