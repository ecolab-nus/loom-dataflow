#ifndef LOOM_TRITON_SHARED_PASSES
#define LOOM_TRITON_SHARED_PASSES

include "mlir/Pass/PassBase.td"

def TritonSharedAffinize : Pass<"loom-triton-shared-affinize", "mlir::ModuleOp"> {
  let summary = "Affinize Triton-shared indices";
  let description = [{
    Rewrite arithmetic index expressions into affine.apply, convert eligible 
    loads/stores to affine form, and express memref.reinterpret_cast offsets 
    via affine maps. Treats trailing grid/thread arguments as dims/symbols to 
    expose GPU-style indexing to affine.
  }];
  let constructor = "loom::passes::createTritonSharedAffinizePass()";
}

def TritonSharedGridToParallel : Pass<"loom-triton-shared-grid-to-parallel", "mlir::ModuleOp"> {
  let summary = "Replace grid indices with affine.parallel";
  let description = [{
    Replace the last three grid index arguments with a 3-D affine.parallel 
    with dynamic uppers (sizeX,sizeY,sizeZ); erase index args from the 
    signature and replace their uses by the parallel IVs.
  }];
  let constructor = "loom::passes::createTritonSharedGridToParallelPass()";
}

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

def AnnotateReinterpretCastReuse : Pass<"loom-annotate-reinterpret-cast-reuse", "mlir::ModuleOp"> {
  let summary = "Analyze reuse pattern on loom.reinterpret_cast";
  let description = [{
    Attach a loom.reuse dictionary to each memref.reinterpret_cast describing 
    how its offset varies with surrounding iterators, grouped by spatial 
    (affine.parallel), temporal (affine.for), and sequential (scf.for).
  }];
  let constructor = "loom::passes::createAnnotateReinterpretCastReusePass()";
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

def TileScfForToL1 : Pass<"loom-tile-scf-for-to-l1", "mlir::ModuleOp"> {
  let summary = "Tile scf.for loops to fit L1";
  let description = [{
    Tile scf.for loops so that per-tile memory fits within the single df.memory 
    (L1) capacity. Computes per-iteration memory from memref.alloc operations 
    annotated with loom.alloc.
  }];
  let constructor = "loom::passes::createTileScfForToL1Pass()";
}

def Materialize : Pass<"loom-materialize", "mlir::ModuleOp"> {
  let summary = "Materialize module attributes";
  let description = [{
    Replace loom.get_module_attribute operations with arith.constant values 
    by looking up the attribute in the module.
  }];
  let constructor = "loom::passes::createMaterializePass()";
}

def StaticizeTypes : Pass<"loom-staticize-types", "mlir::ModuleOp"> {
  let summary = "Convert dynamic types to static";
  let description = [{
    Convert dynamic memref and tensor types to static types by materializing 
    dimension sizes from symbolic values.
  }];
  let constructor = "loom::passes::createStaticizeTypesPass()";
}

def LoomToMemRefLowering : Pass<"loom-to-memref-lowering", "mlir::ModuleOp"> {
  let summary = "Lower loom ops to memref";
  let description = [{
    Lower loom dialect operations to memref dialect operations.
  }];
  let constructor = "loom::passes::createLoomToMemRefLoweringPass()";
}

def ConstDedupCleanup : Pass<"loom-const-cleanup", "mlir::ModuleOp"> {
  let summary = "Deduplicate and cleanup constants";
  let description = [{
    Deduplicate arith.constant and arith.constant_index operations by value 
    and type, remove unused constants, and fold constant operands into 
    affine.apply operations.
  }];
  let constructor = "loom::passes::createConstDedupCleanupPass()";
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
