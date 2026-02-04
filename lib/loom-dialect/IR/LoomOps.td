//===- LoomOps.td ------------------------------------------*- tablegen -*-===//
// Operations for the LOOM dialect (loom)
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "LoomTypes.td"
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

// Type constraints for Loom custom types
def Loom_ViewType : Type<CPred<"::llvm::isa<::loom::ViewType>($_self)">,
                         "loom view type">,
                    BuildableType<"::loom::ViewType::get($_builder.getContext())">;

def Loom_BufferTokenType : Type<CPred<"::llvm::isa<::loom::BufferTokenType>($_self)">,
                                "loom buffer token type">,
                           BuildableType<"::loom::BufferTokenType::get($_builder.getContext())">;

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
        DenseI64ArrayAttr:$static_offsets,
        DenseI64ArrayAttr:$static_sizes,
        DenseI64ArrayAttr:$static_strides
    );

  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $source `to`
    `offset` `:` custom<DynamicIndexList>($offsets, $static_offsets)
    `,` `sizes` `:` custom<DynamicIndexList>($sizes, $static_sizes)
    `,` `strides` `:` custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($result)
  }];

  let extraClassDeclaration = [{
    /// Get the source memref type.
    ::mlir::MemRefType getSourceType() {
      return ::llvm::dyn_cast<::mlir::MemRefType>(getSource().getType());
    }
  }];
}


def ViewOp : Loom_Op<"view", [
    Pure,
    AttrSizedOperandSegments,
    OffsetSizeAndStrideOpInterface
  ]> {
  let summary = "Create a logical view from a memref with coordinate metadata";
  let description = [{
    The `loom.view` operation creates a logical view (coordinate package) from
    a source memref. The view encapsulates:
    - A reference to the source memref (physical storage in Global Memory)
    - Offsets: logical starting coordinates
    - Sizes: logical shape of the slice
    - Strides: access pattern defining memory continuity

    This operation is purely metadata and does not generate any runtime 
    instructions. The returned `memref` type represents a "logical 
    perspective with coordinate information" in the SSA chain.

    Example:
    ```mlir
    %view_a = loom.view %arg0[%0, %9][%0, %2][1, 1] 
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
  let summary = "Allocate L1 buffer and return a buffer token";
  let description = [{
    The `loom.alloc` operation allocates a buffer on a specific hardware 
    memory (e.g., L1, DRAM) and returns a buffer token representing this
    allocation record.

    The `buffer_count` attribute specifies the number of buffers for 
    double/multi-buffering scenarios.

    Example:
    ```mlir
    %token = loom.alloc(32, 32) on @L1 {buffer_count = 2} : !loom.buffer_token
    ```
  }];

  let arguments = (ins
    Variadic<Index>:$sizes,
    DenseI64ArrayAttr:$static_sizes,
    OptionalAttr<I64Attr>:$alignment,
    DefaultValuedAttr<I64Attr, "1">:$buffer_count,
    SymbolRefAttr:$memory
  );

  let results = (outs Loom_BufferTokenType:$result);

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
  }];
}

def CopyToTensorOp : Loom_Op<"copy_to_tensor", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Copy data from global memory to L1 and produce a tensor";
  let description = [{
    The `loom.copy_to_tensor` operation performs a data transfer from global
    memory (described by the source view) to L1 memory (identified by the
    buffer token), and returns the result as a tensor.

    This operation converts physical semantics (memref + view) to logical
    semantics (tensor). It has side effects as it represents a DMA transfer.

    Input:
    - source_view: A `memref` object representing the memref reference
      and physical coordinates (Offsets, Sizes)
    - buffer_token: A `!loom.buffer_token` object representing L1 placeholder

    Output: AnyRankedTensor

    Example:
    ```mlir
    %t = loom.copy_to_tensor %view, %token 
         : memref<?x?xf32>, !loom.buffer_token -> tensor<64x64xf32>
    ```
  }];

  let arguments = (ins
    AnyMemRef:$source_view,
    Loom_BufferTokenType:$buffer_token,
    OptionalAttr<SymbolRefAttr>:$memory,
    Optional<Index>:$provenance,
    DefaultValuedAttr<ArrayAttr, "{}">:$interconnect,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1}">:$broadcast
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source_view `,` $buffer_token 
    ( `provenance` `:` $provenance^ )? 
    ( `on` $memory^ )?
    `,` `interconnect` `:` $interconnect 
    `,` `broadcast` `:` $broadcast 
    attr-dict `:` 
    type($source_view) `,` type($buffer_token) `->` type($result)
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
}

def InitTensorOp : Loom_Op<"init_tensor", [Pure]> {
  let summary = "Create a tensor bound to a buffer token";
  let description = [{
    The `loom.init_tensor` operation creates a tensor that is bound to a 
    pre-allocated buffer identified by the buffer token. This replaces 
    `tensor.empty` to explicitly bind memory allocation with tensor semantics.

    Input:
    - buffer_token: A `!loom.buffer_token` from loom.alloc

    Output: AnyRankedTensor with specified shape

    Example:
    ```mlir
    %token = loom.alloc(32, 32) on @L1 : !loom.buffer_token
    %t = loom.init_tensor %token(32, 32) : !loom.buffer_token -> tensor<32x32xf32>
    ```
  }];

  let arguments = (ins
    Loom_BufferTokenType:$buffer_token,
    Variadic<Index>:$sizes,
    DenseI64ArrayAttr:$static_sizes
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $buffer_token ``
    custom<DynamicIndexList>($sizes, $static_sizes) attr-dict `:` 
    type($buffer_token) `->` type($result)
  }];

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    ::mlir::SmallVector<::mlir::OpFoldResult, 4> getMixedSizes() {
      ::mlir::Builder builder(getContext());
      return ::mlir::getMixedValues(getStaticSizes(), getSizes(), builder);
    }
  }];
}

def Loom_PackToTensorOp : Loom_Op<"pack_to_tensor", [
    Pure
  ]> {
  let summary = "Packs a 2D view into a 3D tiled tensor in a specific memory space.";
  let description = [{
    This op takes a 2D memref and a !loom.buffer_token, and produces a 3D tensor.
    It represents the physical move and layout transformation from Global Memory to L1.

    It performs tiling on the second dimension of the input view and permutes the
    outer dimensions.

    Example:
    ```mlir
    %packed_A = loom.pack_to_tensor %view, %token 
                inner_tiles = [64] 
                outer_dims_perm = [1, 0]
                : memref<?x?xf32>, !loom.buffer_token -> tensor<8x128x64xf32>
    ```
  }];

  let arguments = (ins 
    AnyMemRef:$source,
    Loom_BufferTokenType:$buffer_token,
    Variadic<Index>:$inner_tiles,
    DenseI64ArrayAttr:$outer_dims_perm
  );

  let results = (outs 
    AnyRankedTensor:$result
  );

  let assemblyFormat = [{
    $source `,` $buffer_token 
    `inner_tiles` `=` `[` $inner_tiles `]`
    `outer_dims_perm` `=` $outer_dims_perm
    attr-dict `:` 
    type($source) `,` type($buffer_token) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Constraint Language Operations
//===----------------------------------------------------------------------===//

/// Constraint space container that holds symbolic variables and constraints.
///
/// The `loom.constraint_space` operation is a top-level container for defining
/// symbolic variables and their constraints. It is used during lowering to
/// record known facts about operator dimensions (primarily block sizes).
///
/// The operation:
/// - Is isolated from above (cannot reference external SSA values)
/// - Maintains a symbol table for contained symbolic variables
/// - Contains a single region with a single block
/// - Does not require a terminator
///
/// Example:
/// ```mlir
/// loom.constraint_space @constraints {
///   %m = loom.symbolic_var "M" : index
///   %n = loom.symbolic_var "N" : index
///   loom.range %m [16, 512]
///   loom.align %m by 32
/// }
/// ```
def ConstraintSpaceOp : Loom_Op<"constraint_space", [
  IsolatedFromAbove,
  SymbolTable,
  NoTerminator,
  SingleBlock
]> {
  let summary = "Container for symbolic variables and constraints";
  let description = [{
    This operation defines a constraint space that holds symbolic variables
    and their associated constraints. It serves as a symbol table container
    during lowering passes to track dimension constraints.
    
    All symbolic variables defined within this space can be referenced by
    constraint operations (range, align, linear_constraint) within the same
    space, or by get_symbolic_block_size operations elsewhere via symbol
    references.
  }];
  
  let arguments = (ins
    SymbolNameAttr:$sym_name
  );
  
  let regions = (region SizedRegion<1>:$body);
  
  let assemblyFormat = [{
    $sym_name $body attr-dict
  }];
  
  let hasVerifier = 1;
  
  let extraClassDeclaration = [{
    /// Returns the body block of the constraint space.
    ::mlir::Block *getBodyBlock() { return &getBody().front(); }
  }];
}

/// Symbolic variable definition within a constraint space.
///
/// The `loom.symbolic_var` operation defines a symbolic variable within a
/// constraint space. The variable has a name and produces an index-typed
/// SSA value that can be used by constraint operations.
///
/// Example:
/// ```mlir
/// %m = loom.symbolic_var "M" : index
/// ```
def SymbolicVarOp : Loom_Op<"symbolic_var", [
  Pure,
  HasParent<"ConstraintSpaceOp">
]> {
  let summary = "Define a symbolic variable";
  let description = [{
    This operation defines a symbolic variable with a given name within a
    constraint space. The variable produces an index-typed SSA value that
    can be referenced by constraint operations (range, align, linear_constraint)
    within the same constraint space.
    
    The variable name is used for documentation and debugging purposes.
  }];
  
  let arguments = (ins
    StrAttr:$name
  );
  
  let results = (outs Index:$result);
  
  let assemblyFormat = [{
    $name `:` type($result) attr-dict
  }];
  
  let extraClassDeclaration = [{
    /// Returns the type of the symbolic variable (always index).
    ::mlir::Type getVarType() { return getResult().getType(); }
  }];
}

/// Range constraint for a symbolic variable.
///
/// The `loom.range` operation specifies lower and upper bounds for a
/// symbolic variable, expressing the constraint:
///   lower_bound <= variable <= upper_bound
///
/// Example:
/// ```mlir
/// loom.range %m [16, 512]
/// ```
def RangeOp : Loom_Op<"range", [
  HasParent<"ConstraintSpaceOp">
]> {
  let summary = "Specify lower and upper bounds for a variable";
  let description = [{
    This operation constrains a symbolic variable to be within a specified
    range [lower_bound, upper_bound]. Both bounds are inclusive.
    
    Semantics: lower_bound <= variable <= upper_bound
    
    Example:
      loom.range %m [16, 512]
    
    This constrains %m to be between 16 and 512 (inclusive).
  }];
  
  let arguments = (ins
    Index:$variable,
    I64Attr:$lower_bound,
    I64Attr:$upper_bound
  );
  
  let assemblyFormat = [{
    $variable `[` $lower_bound `,` $upper_bound `]` attr-dict
  }];
}

/// Alignment constraint for a symbolic variable.
///
/// The `loom.align` operation specifies that a symbolic variable must be
/// aligned to a given value, expressing the constraint:
///   variable ≡ 0 (mod alignment)
///
/// Example:
/// ```mlir
/// loom.align %m by 32
/// ```
def AlignOp : Loom_Op<"align", [
  HasParent<"ConstraintSpaceOp">
]> {
  let summary = "Specify alignment constraint for a variable";
  let description = [{
    This operation constrains a symbolic variable to be aligned to a
    specified value. The constraint is expressed as a modulo operation:
    variable must be divisible by alignment.
    
    Semantics: variable ≡ 0 (mod alignment)
    
    Example:
      loom.align %m by 32
    
    This constrains %m to be a multiple of 32.
  }];
  
  let arguments = (ins
    Index:$variable,
    I64Attr:$alignment
  );
  
  let assemblyFormat = [{
    $variable `by` $alignment attr-dict
  }];
}

/// Multi-variable linear constraint.
///
/// The `loom.linear_constraint` operation specifies a linear inequality
/// constraint over multiple symbolic variables using an AffineMap.
///
/// Each result of the AffineMap represents a constraint of the form:
///   result >= 0
///
/// Example:
/// ```mlir
/// // M + N <= 256  =>  -M - N + 256 >= 0
/// loom.linear_constraint (%m, %n) {
///   map = affine_map<(d0, d1) -> (-d0 - d1 + 256)>
/// }
/// ```
def LinearConstraintOp : Loom_Op<"linear_constraint", [
  HasParent<"ConstraintSpaceOp">
]> {
  let summary = "Specify a multi-variable linear inequality constraint";
  let description = [{
    This operation specifies a linear inequality constraint over one or more
    symbolic variables. The constraint is represented using an AffineMap,
    where each result of the map represents an inequality of the form:
    
      result >= 0
    
    This allows expressing complex linear relationships between variables.
    
    Convention:
      - To express "x + y <= c", use affine_map<(d0, d1) -> (-d0 - d1 + c)>
      - To express "x >= c", use affine_map<(d0) -> (d0 - c)>
    
    Example:
      // Constraint: M + N <= 256
      loom.linear_constraint (%m, %n) {
        map = affine_map<(d0, d1) -> (-d0 - d1 + 256)>
      }
  }];
  
  let arguments = (ins
    Variadic<Index>:$operands,
    AffineMapAttr:$map,
    DefaultValuedAttr<BoolAttr, "false">:$is_equality
  );
  
  let assemblyFormat = [{
    `(` $operands `)` attr-dict
  }];
}

def PolynomialConstraintOp : Loom_Op<"polynomial_constraint", [
  HasParent<"ConstraintSpaceOp">
]> {
  let summary = "Specify a multi-variable polynomial inequality constraint";
  let description = [{
    This operation specifies a polynomial inequality constraint over one or more
    symbolic variables. The constraint is represented as:
    
      sum(monomial.coeff * product(symbolic_vars[monomial.vars])) <= upper_bound
    
    Example:
      loom.polynomial_constraint (%m, %n, %k) {
        monomials = [
          { vars = [0, 2], coeff = 1 }, // Monomial 0: %m * %k
          { vars = [1, 2], coeff = 1 }  // Monomial 1: %n * %k
        ],
        upper_bound = 8192 : i64
      }
  }];
  
  let arguments = (ins
    Variadic<Index>:$operands,
    ArrayAttr:$monomials,
    I64Attr:$upper_bound
  );
  
  let assemblyFormat = [{
    `(` $operands `)` attr-dict
  }];
}

/// Expression within a constraint space.
///
/// The `loom.expression` operation defines a linear combination of symbolic
/// variables. It is used by the factorization pass to represent factored-out
/// sub-expressions that can be referenced as single SSA values in constraint
/// monomials.
///
/// Example:
/// ```mlir
/// %v = loom.expression(%m, %n, %c) {coeffs = [1 : i64, 1 : i64, 1 : i64]} : index
/// // Represents: M + N + C
/// ```
def ExpressionOp : Loom_Op<"expression", [
  Pure,
  HasParent<"ConstraintSpaceOp">
]> {
  let summary = "Define a symbolic expression combining variables";
  let description = [{
    This operation represents either a linear combination of symbolic variables
    (addition) or a product of two symbolic variables (multiplication).
    
    The 'logic' attribute determines whether the operation is an 'add' or 'mul'.
    - If logic is 'add' (default):
      It represents: coeff[0]*operand[0] + coeff[1]*operand[1] + ...
      There can be any number of operands.
    - If logic is 'mul':
      It represents: operand[0] * operand[1].
      There must be exactly two operands. The 'coeffs' attribute should be [1, 1].
    
    The result is an index value representing the expression, which can be
    used as an operand in polynomial_constraint or other expression operations.
  }];
  
  let arguments = (ins
    Variadic<Index>:$operands,
    I64ArrayAttr:$coeffs,
    StrAttr:$logic
  );
  
  let results = (outs Index:$result);
  
  let assemblyFormat = [{
    `(` $operands `)` attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
}

/// A lightweight handle for a local variable in the constraint system.
///
/// This operation represents a local identifier (Local ID) in the Presburger
/// arithmetic system. It does not carry any symbolic meaning outside the
/// constraint space and is used to hold intermediate results during
/// linearization or decomposition.
def IntermediateVarOp : Loom_Op<"intermediate_var", [
  Pure,
  HasParent<"ConstraintSpaceOp">
]> {
  let summary = "Define an intermediate local variable";
  let description = [{
    This operation represents a local variable within a constraint space.
    Local variables are temporary variables used to lower non-linear
    expressions or to store intermediate results during constraint simplification.
    
    The result is an index value that can be used as an operand in
    linear_constraint or polynomial_constraint operations.
  }];
  
  let arguments = (ins);
  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

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
