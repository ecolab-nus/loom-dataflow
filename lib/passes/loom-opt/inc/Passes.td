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
}


def Materialize : Pass<"loom-materialize", "mlir::ModuleOp"> {
  let summary = "Materialize module attributes";
  let description = [{
    Replace loom.get_module_attribute operations with arith.constant values 
    by looking up the attribute in the module.
  }];
  let constructor = "loom::passes::createMaterializePass()";
}

def BridgeToOSB : Pass<"loom-bridge-to-osb", "mlir::ModuleOp"> {
  let summary = "Convert loom.subview to memref.reinterpret_cast";
  let description = [{
    Linearize multi-dimensional offsets via affine.apply and adjust strides.
  }];
  let constructor = "loom::passes::createBridgeToOSBPass()";
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

def FoldRedundantExtractSlice : Pass<"loom-fold-redundant-extract-slice", "mlir::ModuleOp"> {
  let summary = "Fold redundant tensor.extract_slice operations";
  let description = [{
    Eliminate tensor.extract_slice operations that are essentially no-ops:
    offsets are all 0, strides are all 1, and the sizes match the source
    tensor's shape.
  }];
  let constructor = "loom::passes::createFoldRedundantExtractSlicePass()";
}

def LinalgGuardedElementwiseOpFusion : Pass<"loom-linalg-guarded-elementwise-fusion", "mlir::ModuleOp"> {
  let summary = "Elementwise op fusion that skips fusions producing a leading-reduction generic";
  let description = [{
    Mirrors the upstream linalg-elementwise-op-fusion pass but installs a
    control function that blocks fusion whenever the producer OR consumer
    linalg.generic has a leading (outermost) reduction iterator.  Such fused
    forms are not processable by the target hardware.
  }];
  let constructor = "loom::passes::createLinalgGuardedElementwiseOpFusionPass()";
}

def MemoryBindingPass : Pass<"loom-memory-binding", "mlir::ModuleOp"> {
  let summary = "Bind memory allocations to tensor operations";
  let description = [{
    Transform bufferization patterns to loom dialect operations that explicitly
    bind physical memory allocations with tensor semantics for dataflow analysis.
  }];
  let constructor = "loom::passes::createMemoryBindingPass()";
}

def CanonicalBufferizationToLoomPass : Pass<"loom-canonical-bufferization-to-loom", "mlir::ModuleOp"> {
  let summary = "Rewrite canonical bufferization handoff ops to loom dialect";
  let description = [{
    Convert bufferization.to_tensor and bufferization.to_buffer handoff anchors
    to loom.bufferize_to_tensor and loom.bufferize_to_memref after tensor
    canonicalization.
  }];
  let constructor = "loom::passes::createCanonicalBufferizationToLoomPass()";
}

def LowerAffineWithAttr : Pass<"loom-lower-affine-with-attr", "mlir::ModuleOp"> {
  let summary = "Lower affine to SCF while preserving loom attributes";
  let description = [{
    Lower affine.parallel to scf.parallel while preserving loom.mapped_to
    and loom.iter_type attributes as array attributes on the resulting
    scf.parallel operation.
  }];
  let constructor = "loom::passes::createLowerAffineWithAttrPass()";
}

def HandoffSyncInsertionPass : Pass<"loom-handoff-sync-insertion", "mlir::ModuleOp"> {
  let summary = "Insert tensor.empty + loom.sync around execution-core handoffs";
  let description = [{
    Discover execution-core handoff boundaries:
    - bufferization.to_tensor
    - tensor-mode loom.gather (single-op chain)
    - loom.bufferize_to_memref + loom.copy with L1->DRAM attrs (two-op chain)
    and insert tensor.empty + loom.sync around those boundaries, then rewrite
    uses of the sync input tensor to the sync result when dominated.
  }];
  let constructor = "loom::passes::createHandoffSyncInsertionPass()";
}

def LoopHandoffProxyCopyInsertionPass : Pass<"loom-loop-handoff-proxy-copy-insertion", "mlir::ModuleOp"> {
  let summary = "Insert post-loop proxy copies for communication handoffs";
  let description = [{
    For loop results consumed directly by communication boundary operations,
    insert tensor.empty + linalg.copy after the loop and rewrite only those
    communication operands to use the proxy tensor. This preserves loop Phi
    reuse while giving memory analysis a singleton handoff tensor.
  }];
  let constructor = "loom::passes::createLoopHandoffProxyCopyInsertionPass()";
}

#endif // LOOM_TRITON_SHARED_PASSES
