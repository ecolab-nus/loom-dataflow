//===- LoomOps.td ------------------------------------------*- tablegen -*-===//
// Operations for the LOOM dialect (loom)
//===----------------------------------------------------------------------===//

include "LoomDialect.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Loom Enum Attributes
//===----------------------------------------------------------------------===//

def IterTypeEnum : I32EnumAttr<"IterType", "Loom loop iteration type", [
  I32EnumAttrCase<"Spatial", 0, "spatial">,
  I32EnumAttrCase<"Temporal", 1, "temporal">,
  I32EnumAttrCase<"Sequential", 2, "sequential">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::loom";
}

def IterTypeAttr : EnumAttr<Loom_Dialect, IterTypeEnum, "iter_type"> {
  let assemblyFormat = "`<` $value `>`";
}

class Loom_Op<string mnemonic, list<Trait> traits = []>
  : Op<Loom_Dialect, mnemonic, traits> {}


def SubviewOp : Loom_Op<"subview", [
    Pure,
    AttrSizedOperandSegments,
    OffsetSizeAndStrideOpInterface
  ]> {
  let summary = "Create a logical subview from a memref with coordinate metadata";
  let description = [{
    The `loom.subview` operation creates a logical subview (coordinate package) from
    a source memref. The subview encapsulates:
    - A reference to the source memref (physical storage in Global Memory)
    - Offsets: logical starting coordinates
    - Sizes: logical shape of the slice
    - Strides: access pattern defining memory continuity

    This operation is purely metadata and does not generate any runtime 
    instructions. The returned `memref` type represents a "logical 
    perspective with coordinate information" in the SSA chain.

    Example:
    ```mlir
    %view_a = loom.subview %arg0[%0, %2][%0, %2][1, 1] 
              : memref<128x128xf32> to memref<?x?xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedOrUnrankedMemRef:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides,
    // memory reuse attributes
    DefaultValuedAttr<BoolAttr, "false">:$sequential_reuse,
    DefaultValuedAttr<BoolAttr, "false">:$spatial_reuse,
    DefaultValuedAttr<BoolAttr, "false">:$temporal_reuse
  );

  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    `,` `reuse` `:` `[` `seq` `=` $sequential_reuse `,`
                         `spat` `=` $spatial_reuse `,`
                         `temp` `=` $temporal_reuse `]`
    attr-dict `:` type($source) `to` type($result)
  }];

  let extraClassDeclaration = [{
    /// Get the source memref type.
    ::mlir::MemRefType getSourceType() {
      return ::llvm::dyn_cast<::mlir::MemRefType>(getSource().getType());
    }

    /// Infer the result type from source type and static parameters.
    static ::mlir::MemRefType inferResultType(
        ::mlir::MemRefType sourceType,
        ::llvm::ArrayRef<int64_t> staticOffsets,
        ::llvm::ArrayRef<int64_t> staticSizes,
        ::llvm::ArrayRef<int64_t> staticStrides);

    /// Return the number of leading operands before offsets, sizes, strides.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

    /// Return the expected rank of each of the static attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getStaticOffsets().size();
      return {rank, rank, rank};
    }
  }];

  let hasCanonicalizer = 1;
}



def AllocOp : Loom_Op<"alloc"> {
  let summary = "Allocate L1 buffer and return a memref";
  let description = [{
    The `loom.alloc` operation allocates a buffer on a specific hardware 
    memory (e.g., L1, DRAM) and returns a memref referencing the allocated memory.

    Example:
    ```mlir
    %buffer = loom.alloc(32, 32) on @L1 {buffer_count = 2} : memref<32x32xf32>
    ```
  }];

  let arguments = (ins
    Variadic<Index>:$sizes,
    DenseI64ArrayAttr:$static_sizes,
    OptionalAttr<I64Attr>:$alignment,
    DefaultValuedAttr<I64Attr, "1">:$buffer_count,
    SymbolRefAttr:$memory
  );

  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    custom<DynamicIndexList>($sizes, $static_sizes) 
    `on` $memory attr-dict `:` type($result)
  }];

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    ::mlir::SmallVector<::mlir::OpFoldResult, 4> getMixedSizes() {
      ::mlir::Builder builder(getContext());
      return ::mlir::getMixedValues(getStaticSizes(), getSizes(), builder);
    }

    /// Infer the result memref type from static sizes and element type.
    static ::mlir::MemRefType inferResultType(
        ::llvm::ArrayRef<int64_t> staticSizes,
        ::mlir::Type elementType);
  }];
}
def SemaphoreTakeOp : Loom_Op<"semaphore_take", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Take a unique synchronization semaphore over a physical buffer";
  let description = [{
    The `loom.semaphore_take` operation accepts a memref (typically from `loom.alloc`)
    and returns a memref of the same type. The returned SSA value represents a
    unique hardware synchronization semaphore. The backend can trace the source
    operand to discover the physical buffer, while treating each semaphore result
    as an independent sync barrier.

    This op is intentionally NOT Pure: it declares a MemAlloc effect so that CSE
    cannot merge two semaphore ops that share the same source alloc.

    Example:
    ```mlir
    %buf = loom.alloc [64, 128] on @L1 : memref<64x128xf32>
    %sem_q = loom.semaphore_take %buf : memref<64x128xf32>
    %sem_o = loom.semaphore_take %buf : memref<64x128xf32>
    // %sem_q and %sem_o share physical memory but have independent semaphores
    ```
  }];

  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$result);

  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($result)
  }];
}

def SemaphoreGiveOp : Loom_Op<"semaphore_give", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Give/Release a synchronization semaphore over a physical buffer";
  let description = [{
    The `loom.semaphore_give` operation accepts a semaphore memref (from `loom.semaphore_take`).
    It declares the end of liveness for the corresponding circular buffer.

    This operation has memory effects (Free) to ensure it is not optimized away or
    moved incorrectly across the lifespan of the buffer.

    Example:
    ```mlir
    loom.semaphore_give %sem_q : memref<64x128xf32>
    ```
  }];

  let arguments = (ins AnyMemRef:$source);
  let results = (outs);

  let assemblyFormat = [{
    $source attr-dict `:` type($source)
  }];
}

def CopyToTensorOp : Loom_Op<"copy_to_tensor", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Copy data from global memory to L1 and produce a tensor";
  let description = [{
    The `loom.copy_to_tensor` operation performs a data transfer from global
    memory (described by the source view) to L1 memory (identified by the
    buffer), and returns the result as a tensor.

    This operation converts physical semantics (memref + view) to logical
    semantics (tensor). It has side effects as it represents a DMA transfer.

    Input:
    - source_view: A `memref` object representing the memref reference
      and physical coordinates (Offsets, Sizes)
    - buffer: A `memref` object representing L1 placeholder

    Output: AnyRankedTensor

    Example:
    ```mlir
    %t = loom.copy_to_tensor %view, %buffer 
         : memref<?x?xf32>, memref<64x64xf32> -> tensor<64x64xf32>
    ```
  }];

  let arguments = (ins
    AnyMemRef:$source_view,
    AnyMemRef:$buffer,
    OptionalAttr<SymbolRefAttr>:$memory,
    Optional<Index>:$provenance,
    DefaultValuedAttr<ArrayAttr, "{}">:$interconnect,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1}">:$broadcast
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source_view `,` $buffer 
    ( `provenance` `:` $provenance^ )? 
    ( `on` $memory^ )?
    `,` `interconnect` `:` $interconnect 
    `,` `broadcast` `:` $broadcast 
    attr-dict `:` 
    type($source_view) `,` type($buffer) `->` type($result)
  }];

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    static ::mlir::RankedTensorType inferResultType(::mlir::MemRefType sourceType);
  }];
}

def CopyFromTensorOp : Loom_Op<"copy_from_tensor", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Copy tensor data back to global memory via a view";
  let description = [{
    The `loom.copy_from_tensor` operation transfers the contents of a tensor
    (which resides in L1 buffer) back to global memory at the location
    specified by the target view.

    This operation has side effects as it represents a DMA write-back.

    Input:
    - source_tensor: The tensor to write back (typically the final result
      from computation), using AnyRankedTensor
    - target_view: A `memref` object encapsulating the target memref 
      reference and physical coordinates (Offsets, Sizes)

    Output: void (none)

    Example:
    ```mlir
    loom.copy_from_tensor %tensor, %view : tensor<64x64xf32>, memref<?x?xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$source_tensor,
    AnyMemRef:$target_view,
    OptionalAttr<SymbolRefAttr>:$memory
  );

  let results = (outs);

  let assemblyFormat = [{
    $source_tensor `,` $target_view ( `on` $memory^ )? attr-dict `:` 
    type($source_tensor) `,` type($target_view)
  }];

  let hasCanonicalizer = 1;
}

def InitTensorOp : Loom_Op<"init_tensor", [Pure]> {
  let summary = "Create a tensor bound to a buffer";
  let description = [{
    The `loom.init_tensor` operation creates a tensor that is bound to a 
    pre-allocated buffer. This replaces `tensor.empty` to explicitly bind 
    memory allocation with tensor semantics.

    Input:
    - buffer: A `memref` from loom.alloc

    Output: AnyRankedTensor with specified shape

    Example:
    ```mlir
    %buffer = loom.alloc(32, 32) on @L1 : memref<32x32xf32>
    %t = loom.init_tensor %buffer(32, 32) : memref<32x32xf32> -> tensor<32x32xf32>
    ```
  }];

  let arguments = (ins
    AnyMemRef:$buffer,
    Variadic<Index>:$sizes,
    DenseI64ArrayAttr:$static_sizes
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $buffer ``
    custom<DynamicIndexList>($sizes, $static_sizes) attr-dict `:` 
    type($buffer) `->` type($result)
  }];

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    ::mlir::SmallVector<::mlir::OpFoldResult, 4> getMixedSizes() {
      ::mlir::Builder builder(getContext());
      return ::mlir::getMixedValues(getStaticSizes(), getSizes(), builder);
    }

    /// Infer the result tensor type from static sizes and element type.
    static ::mlir::RankedTensorType inferResultType(
        ::llvm::ArrayRef<int64_t> staticSizes,
        ::mlir::Type elementType);
  }];
}

def Loom_PackToTensorOp : Loom_Op<"pack_to_tensor", [
    Pure
  ]> {
  let summary = "Packs a 2D view into a 3D tiled tensor in a specific memory space.";
  let description = [{
    This op takes a 2D memref and a buffer memref, and produces a 3D tensor.
    It represents the physical move and layout transformation from Global Memory to L1.

    It performs tiling on the second dimension of the input view and permutes the
    outer dimensions.

    Example:
    ```mlir
    %packed_A = loom.pack_to_tensor %view, %buffer 
                inner_tiles = [64] 
                outer_dims_perm = [1, 0]
                : memref<?x?xf32>, memref<8x128x64xf32> -> tensor<8x128x64xf32>
    ```
  }];

  let arguments = (ins 
    AnyMemRef:$source,
    AnyMemRef:$buffer,
    Variadic<Index>:$inner_tiles,
    DenseI64ArrayAttr:$outer_dims_perm
  );

  let results = (outs 
    AnyRankedTensor:$result
  );

  let assemblyFormat = [{
    $source `,` $buffer 
    `inner_tiles` `=` `[` $inner_tiles `]`
    `outer_dims_perm` `=` $outer_dims_perm
    attr-dict `:` 
    type($source) `,` type($buffer) `->` type($result)
  }];
}

def CopyOp : Loom_Op<"copy", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Copy data between memory spaces";
  let description = [{
    The `loom.copy` operation performs a physical data transfer between
    two memrefs, possibly in different memory spaces (e.g., DRAM to L1).
    
    Example:
    ```mlir
    loom.copy %src, %dst src_mem_space @DRAM, dst_mem_space @L1,
              interconnect : [], broadcast : [1, 1]
              : memref<64x64xf32, strided<[512, 1], offset: ?>>,
                memref<64x64xf32>
    ```
  }];

  let arguments = (ins
    AnyMemRef:$source,
    AnyMemRef:$destination,
    OptionalAttr<SymbolRefAttr>:$src_mem_space,
    OptionalAttr<SymbolRefAttr>:$dst_mem_space,
    DefaultValuedAttr<ArrayAttr, "{}">:$interconnect,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1}">:$broadcast
  );

  let results = (outs);

  let assemblyFormat = [{
    $source `,` $destination
    (`src_mem_space` $src_mem_space^)?
    (`dst_mem_space` $dst_mem_space^)?
    `,` `interconnect` `:` $interconnect 
    `,` `broadcast` `:` $broadcast 
    attr-dict `:` type($source) `,` type($destination)
  }];
}

//===----------------------------------------------------------------------===//
// Constraint Language Operations
//===----------------------------------------------------------------------===//

/// Reference to a symbolic variable in a constraint space.
///
/// The `loom.get_symbolic_block_size` operation retrieves the value of a
/// symbolic variable defined in a constraint space. It uses a nested symbol
/// reference to reference the constraint space and the variable name.
///
/// Example:
/// ```mlir
/// %m_val = loom.get_symbolic_block_size @constraints::@M : index
/// ```
def GetSymbolicBlockSizeOp : Loom_Op<"get_symbolic_block_size", [
  Pure
]> {
  let summary = "Get the value of a symbolic variable from a constraint space";
  let description = [{
    This operation retrieves the value of a symbolic variable defined within
    a constraint space. It uses a SymbolRefAttr to reference the constraint
    space and variable name in the format @space::@var.
    
    The operation is pure and returns an index value representing the
    symbolic variable, which can be used in subsequent computations.
    
    Example:
      %m_val = loom.get_symbolic_block_size @constraints::@M : index
      
    This retrieves the value of variable "M" from the constraint space
    named "@constraints".
  }];
  
  let arguments = (ins
    SymbolRefAttr:$symbol_ref
  );
  
  let results = (outs Index:$result);
  
  let assemblyFormat = [{
    $symbol_ref `:` type($result) attr-dict
  }];
  
  let hasVerifier = 1;
}

def MatmulOp : Loom_Op<"matmul", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Loom implementation of matmul";
  let description = [{
    The `loom.matmul` operation performs matrix multiplication on memrefs.
    It is equivalent to `linalg.matmul` but specifically used for the TT platform.
    It currently only supports static shapes.

    TODO: This op should be moved to a specific TT dialect in the future.

    Example:
    ```mlir
    loom.matmul ins(%a, %b : memref<64x64xf16>, memref<64x64xf16>)
                outs(%c : memref<64x64xf16>)
    ```
  }];

  let arguments = (ins
    AnyMemRef:$lhs,
    AnyMemRef:$rhs,
    AnyMemRef:$outs
  );

  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $outs `:` type($outs) `)` attr-dict
  }];
}
