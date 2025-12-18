//===- LoomOps.td ------------------------------------------*- tablegen -*-===//
// Operations for the LOOM dialect (loom)
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "LoomDialect.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class Loom_Op<string mnemonic, list<Trait> traits = []>
  : Op<Loom_Dialect, mnemonic, traits> {}

/// An enhanced memref.reinterpret_cast operation.
/// 
/// The `loom.reinterpret_cast` operation creates a new memref descriptor (view) 
/// from a source memref, potentially changing the shape, offset, and strides.
/// 
/// Syntax:
///   `%result = loom.reinterpret_cast %source to offset: [offsets...], sizes: [sizes...], strides: [strides...] : type($source) to type($result)`
/// 
/// Example:
///   `%view = loom.reinterpret_cast %arg0 to offset: [%12], sizes: [%blkM_idx, %blkK_idx], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>`
def ReinterpretCastOp : Loom_Op<"reinterpret_cast", [
  Pure, 
  AttrSizedOperandSegments
]> {
  let hasVerifier = 0;

  let summary = "An enhanced memref.reinterpret_cast op that include memory reuse info";
  let description = [{
        The `df.reinterpret_cast` operation creates a new memref descriptor (view) 
        from a source memref, potentially changing the shape, offset, and strides.

        This operation extends the standard `memref.reinterpret_cast` by attaching 
        hardware-centric memory reuse attributes. These hints guide downstream 
        passes for buffer management, cache allocation, and DMA scheduling:
        
        - `sequential_reuse`: Indicates if data is reused across sequential execution steps.
        - `spatial_reuse`: Indicates data sharing across SIMD lanes or spatial hardware units.
        - `temporal_reuse`: Indicates data reuse across different time steps or loop iterations.

        Example:
        ```mlir
        %view = df.reinterpret_cast %src to memref<?x8xf32>
                reuse [seq = true, spat = false, temp = true]
                offsets: [0, %off] sizes: [%sz, 8] strides: [8, 1]
        ```
    }];
  
  let arguments = (ins 
        AnyRankedOrUnrankedMemRef:$source,
        Variadic<Index>:$offsets,
        Variadic<Index>:$sizes,
        Variadic<Index>:$strides,
        // memory reuse attributes
        DefaultValuedAttr<BoolAttr, "false">:$sequential_reuse,
        DefaultValuedAttr<BoolAttr, "false">:$spatial_reuse, 
        DefaultValuedAttr<BoolAttr, "false">:$temporal_reuse
    );

  let results = (outs AnyRankedOrUnrankedMemRef:$result);

let assemblyFormat = [{
  $source `to` `offset` `:` `[` $offsets `]` `,` 
  `sizes` `:` `[` $sizes `]` `,` 
  `strides` `:` `[` $strides `]` `,` 
  `reuse` `:` `[` `seq` `=` $sequential_reuse `,` 
                  `spat` `=` $spatial_reuse `,` 
                  `temp` `=` $temporal_reuse `]`
  `:` type($source) `to` type($result) attr-dict
}];
}

def GetBlockSizeOp : Loom_Op<"get_block_size", [Pure]> {
  let summary = "Get block size from module attributes";
  let description = [{
    This operation extracts block size values (m, n, k) from the module's
    loom.block_m, loom.block_n, loom.block_k attributes respectively.
    
    This allows symbolic representation during transformation passes,
    and will be lowered to arith.constant during constant folding.
  }];
  
  let arguments = (ins
    StrAttr:$dimension  // "m", "n", or "k"
  );
  
  let results = (outs Index:$result);
  
  let assemblyFormat = [{
    $dimension `:` type($result) attr-dict
  }];
}
