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
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"

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

    /// Rank-reducing overload: infer a result type whose shape is targetShape
    /// (a subset of staticSizes with unit dims dropped).
    static ::mlir::MemRefType inferResultType(
        ::mlir::MemRefType sourceType,
        ::llvm::ArrayRef<int64_t> staticOffsets,
        ::llvm::ArrayRef<int64_t> staticSizes,
        ::llvm::ArrayRef<int64_t> staticStrides,
        ::llvm::ArrayRef<int64_t> targetShape);

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

defvar MeshBoundsArgs = (ins 
    Optional<Index>:$ul_x, 
    Optional<Index>:$ul_y, 
    Optional<Index>:$lr_x, 
    Optional<Index>:$lr_y
);

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

def BufferizeToTensorOp : Loom_Op<"bufferize_to_tensor", [Pure]> {
  let summary = "Reinterpret a memref as a tensor (pure view, no copy)";
  let description = [{
    The `loom.bufferize_to_tensor` operation reinterprets a memref as a
    ranked tensor with the same element type and shape. It carries no
    memory effects: it is a pure metadata operation that lets downstream
    tensor-level IR consume an already-allocated L1 buffer.

    This op is emitted by the memory binding pass after `loom.copy` has
    physically moved data into the buffer. OSB can alias through it (the
    tensor result is Equivalent to the source buffer) and erase it during
    bufferization.

    Example:
    ```mlir
    loom.copy %view, %buf src_mem_space @DRAM dst_mem_space @L1,
              interconnect : [], broadcast : [1, 1]
              : memref<?x?xf16, strided<...>>, memref<?x?xf16>
    %t = loom.bufferize_to_tensor %buf[%m, %n]
         : memref<?x?xf16> -> tensor<?x?xf16>
    ```
  }];

  let arguments = (ins
    AnyMemRef:$source,
    Variadic<Index>:$sizes,
    DenseI64ArrayAttr:$static_sizes
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($sizes, $static_sizes) attr-dict `:`
    type($source) `->` type($result)
  }];

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    ::mlir::SmallVector<::mlir::OpFoldResult, 4> getMixedSizes() {
      ::mlir::Builder builder(getContext());
      return ::mlir::getMixedValues(getStaticSizes(), getSizes(), builder);
    }

    static ::mlir::RankedTensorType inferResultType(
        ::llvm::ArrayRef<int64_t> staticSizes,
        ::mlir::Type elementType);
  }];
}

def BufferizeToMemrefOp : Loom_Op<"bufferize_to_memref", [Pure]> {
  let summary = "Reinterpret a tensor as a memref (pure view, no copy)";
  let description = [{
    The `loom.bufferize_to_memref` operation reinterprets a ranked tensor
    as a memref with the same element type and shape. It carries no memory
    effects: it is a pure metadata operation.

    This op is emitted by the memory binding pass before `loom.copy` writes
    data from an L1 buffer back to DRAM. OSB erases it during bufferization
    by replacing the memref result with the underlying buffer.

    Example:
    ```mlir
    %buf = loom.bufferize_to_memref %tensor
           : tensor<?x?xf16> -> memref<?x?xf16>
    loom.copy %buf, %view src_mem_space @L1 dst_mem_space @DRAM,
              interconnect : [], broadcast : [1, 1]
              : memref<?x?xf16>, memref<?x?xf16, strided<...>>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$source);

  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($result)
  }];

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Infer a plain (identity-layout) MemRefType from static sizes and
    /// element type.  Used by the static-shape canonicalizer.
    static ::mlir::MemRefType inferResultType(
        ::llvm::ArrayRef<int64_t> staticSizes,
        ::mlir::Type elementType);
  }];
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

def ReduceSumOp : Loom_Op<"reduce_sum", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DestinationStyleOpInterface,
  AllTypesMatch<["input", "init"]>,
  AttrSizedOperandSegments
]> {
  let summary = "Reduce-sum a tensor over a 2D mesh with UB/LB bounds (DPS)";
  let description = [{
    The `loom.reduce_sum` operation performs a reduction sum on the input tensor
    across a 2D mesh grid, writing the result into the destination (`init`)
    operand following destination-passing style (DPS).

    UB and LB are each a pair of integers [x, y] that specify the upper and
    lower mesh coordinates over which to reduce.

    Example:
    ```mlir
    %result = loom.reduce_sum ins(%input) outs(%acc)
                              (UB : [2, 4], LB : [0, 0])
                              : tensor<?x?xf32> -> tensor<?x?xf32>
    ```
  }];

  let arguments = !con
  (
    (ins
      AnyTypeOf<[AnyRankedTensor, AnyMemRef]>:$input,
      AnyTypeOf<[AnyRankedTensor, AnyMemRef]>:$init
    ),
    MeshBoundsArgs
  );

  let results = (outs Variadic<AnyRankedTensor>:$result);

  let assemblyFormat = [{
    `ins` `(` $input `)`
    `outs` `(` $init `)`
    ( `region` `:` `(` `UL` `:` `[` $ul_x^ `,` $ul_y `]` `,`
        `LR` `:` `[` $lr_x `,` $lr_y `]` `)` )? attr-dict `:`
    type($input) (`->` type($result)^ )?
  }];

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }
  }];
}

def CopyOp : Loom_Op<"copy", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  AttrSizedOperandSegments
]> {
  let summary = "Copy data between memory spaces";
  let description = [{
    The `loom.copy` operation performs a physical data transfer between
    two memrefs, possibly in different memory spaces (e.g., DRAM to L1).

    The broadcast region is specified by UL (upper-left) and LR (lower-right)
    mesh coordinates. When UL == LR, no broadcast occurs (unicast). When
    they differ, the data is broadcast to all cores in the rectangular region.
    The `broadcast` attribute is a cached summary of the broadcast factor
    per dimension, kept for backward compatibility with LCS consumers.

    Example:
    ```mlir
    loom.copy %src, %dst src_mem_space @DRAM dst_mem_space @L1,
              broadcast : [1, 1]
              (UL : [%ul_x, %ul_y], LR : [%lr_x, %lr_y])
              : memref<64x64xf32, strided<[512, 1], offset: ?>>,
                memref<64x64xf32>
    ```
  }];

  let arguments = !con
  (
    (
      ins
      AnyMemRef:$source,
      AnyMemRef:$destination,
      OptionalAttr<SymbolRefAttr>:$src_mem_space,
      OptionalAttr<SymbolRefAttr>:$dst_mem_space,
      DefaultValuedAttr<I64ArrayAttr, "{1, 1}">:$broadcast
    ),
    MeshBoundsArgs
  );

  let results = (outs);

  let assemblyFormat = [{
    $source `,` $destination
    (`src_mem_space` $src_mem_space^)?
    (`dst_mem_space` $dst_mem_space^)?
    `,` `broadcast` `:` $broadcast
    ( `region` `:` `(` `UL` `:` `[` $ul_x^ `,` $ul_y `]` `,`
         `LR` `:` `[` $lr_x `,` $lr_y `]` `)` )?
    attr-dict `:` type($source) `to` type($destination)
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Constraint Language Operations
//===----------------------------------------------------------------------===//

/// Reference to a symbolic variable.
///
/// The `loom.sym` operation retrieves the value of a named symbolic variable.
/// It uses a flat SymbolRefAttr to identify the variable.
///
/// Example:
/// ```mlir
/// %m_val = loom.sym @M : index
/// ```
def SymOp : Loom_Op<"sym", [
  Pure
]> {
  let summary = "Reference a named symbolic variable";
  let description = [{
    This operation retrieves the value of a named symbolic variable.
    It returns an index value that can be used in subsequent computations.

    Example:
      %m_val = loom.sym @M : index
  }];

  let arguments = (ins
    SymbolRefAttr:$symbol_ref,
    OptionalAttr<IndexAttr>:$upper_bound,
    DefaultValuedAttr<BoolAttr, "false">:$is_reduction
  );

  let results = (outs Index:$result);

  let assemblyFormat = [{
    $symbol_ref attr-dict `:` type($result)
  }];
}

def BindOp : Loom_Op<"bind", []> {
  let summary = "Bind symbolic shape dimensions to a memref";
  let description = [{
    The `loom.bind` operation annotates a memref with its symbolic shape
    by accepting the memref and a variadic list of index SSA values
    (typically from `loom.sym`).

    This is a metadata-only operation: it does not return any SSA value
    and produces no runtime code.

    Example:
    ```mlir
    %m = loom.sym @M : index
    %n = loom.sym @N : index
    loom.bind %memref, [%m, %n] : memref<?x?xf16>
    ```
  }];

  let arguments = (ins
    AnyMemRef:$memref,
    Variadic<Index>:$symbols
  );

  let results = (outs);

  let assemblyFormat = [{
    $memref `,` `[` $symbols `]` attr-dict `:` type($memref)
  }];
}

def BindShapeOp : Loom_Op<"bind_shape", []> {
  let summary = "Bind symbolic shape dimensions to a memref (alias for loom.bind)";
  let description = [{
    The `loom.bind_shape` operation annotates a memref with its symbolic shape.
    This is functionally identical to `loom.bind` and exists as an alias.

    Example:
    ```mlir
    %m = loom.sym @M : index
    %n = loom.sym @N : index
    loom.bind_shape %memref, [%m, %n] : memref<?x?xf16>
    ```
  }];

  let arguments = (ins
    AnyMemRef:$memref,
    Variadic<Index>:$symbols
  );

  let results = (outs);

  let assemblyFormat = [{
    $memref `,` `[` $symbols `]` attr-dict `:` type($memref)
  }];
}

def BindMemOp : Loom_Op<"bind_mem", []> {
  let summary = "Label a memref with a memory region symbol";
  let description = [{
    `loom.bind_mem` annotates a memref with the memory region it belongs to,
    identified by a flat symbol reference (e.g., `@L1`, `@DRAM`).

    This is a metadata-only operation: it produces no runtime code.

    Example:
    ```mlir
    loom.bind_mem %A, @L1 : memref<?x?xf16>
    loom.bind_mem %dram_src, @DRAM : memref<?x?xf16>
    ```
  }];

  let arguments = (ins
    AnyMemRef:$memref,
    FlatSymbolRefAttr:$memory
  );

  let results = (outs);

  let assemblyFormat = [{
    $memref `,` $memory attr-dict `:` type($memref)
  }];
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

def BatchMatmulOp : Loom_Op<"batch_matmul", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Loom implementation of batch_matmul";
  let description = [{
    The `loom.batch_matmul` operation performs batch matrix multiplication on memrefs.
    It is equivalent to `linalg.batch_matmul` but specifically used for the TT platform.
    It currently only supports static shapes.

    TODO: This op should be moved to a specific TT dialect in the future.

    Example:
    ```mlir
    loom.batch_matmul ins(%a, %b : memref<2x64x64xf16>, memref<2x64x64xf16>)
                      outs(%c : memref<2x64x64xf16>)
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
