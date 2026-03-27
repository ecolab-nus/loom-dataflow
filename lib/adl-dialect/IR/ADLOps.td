//===- ADLOps.td -----------------------------------------------*- tablegen -*-===//
// Operations for the ADL dialect
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/SymbolInterfaces.td"
include "ADLTypes.td"

class ADL_Op<string mnemonic, list<Trait> traits = []>
  : Op<ADL_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// adl.spatial_dim
//   %x = adl.spatial_dim "x", 8
//===----------------------------------------------------------------------===//

def SpatialDimOp : ADL_Op<"spatial_dim", [Symbol]> {
  let summary = "Declare a spatial dimension with a name and a fixed size";
  let description = [{
    `adl.spatial_dim` declares a hardware spatial dimension with a user-facing
    name and a constant integer size. The result is an index-typed SSA value.
    The operation is a Symbol and can be referenced by name.

    Example:
      `%x = adl.spatial_dim "x", 8`
  }];

  let arguments = (ins StrAttr:$sym_name, I64Attr:$size);
  let results = (outs Index:$dim);

  let assemblyFormat = "$sym_name `,` $size attr-dict";
}

//===----------------------------------------------------------------------===//
// adl.memory.bank
//   %0 = adl.memory.bank "bank" {bsize = 128 : i64, nblk = 1024 : i64}
//===----------------------------------------------------------------------===//

def MemoryBankOp : ADL_Op<"memory.bank"> {
  let summary = "Declare a memory bank with block size and count";
  let description = [{
    `adl.memory.bank` declares a memory bank resource with a name, block
    size (`bsize`) and number of blocks (`nblk`). The result is an opaque
    `!adl.mem` handle.

    Example:
      `%0 = adl.memory.bank "bank" {bsize = 128 : i64, nblk = 1024 : i64}`
  }];

  let arguments = (ins StrAttr:$name, I64Attr:$bsize, I64Attr:$nblk);
  let results = (outs MemHandleType:$handle);

  // $name is printed explicitly; bsize and nblk fall into attr-dict.
  let assemblyFormat = "$name `,` attr-dict";
}

//===----------------------------------------------------------------------===//
// adl.memory.array
//   %2 = adl.memory.array "L1", (%1) of %0
//===----------------------------------------------------------------------===//

def MemoryArrayOp : ADL_Op<"memory.array", [Symbol]> {
  let summary = "Declare a memory array from spatial dimensions and a bank";
  let description = [{
    `adl.memory.array` declares a memory array by replicating a memory bank
    across one or more spatial dimensions. The operation is a Symbol.

    Example:
      `%2 = adl.memory.array "L1", (%1) of %0`
  }];

  let arguments = (ins StrAttr:$sym_name,
                       Variadic<Index>:$spatial_dims,
                       MemHandleType:$bank);
  let results = (outs MemHandleType:$handle);

  // `[` ... `]` and `of` are literal tokens; Index is a builtin buildable
  // type; MemHandleType has builderCall set so no explicit type annotation needed.
  let assemblyFormat = "$sym_name `,` `[` $spatial_dims `]` `of` $bank attr-dict";
}

//===----------------------------------------------------------------------===//
// adl.processor.compute
//   %6 = adl.processor.compute "matrix_lane", [(%5, %5)]
//===----------------------------------------------------------------------===//

def ProcessorComputeOp : ADL_Op<"processor.compute",
    [AttrSizedOperandSegments]> {
  let summary = "Declare a compute processor with memory region pairs";
  let description = [{
    `adl.processor.compute` declares a named compute processor. Memory region
    pairs specify which memory regions the processor can access. Round brackets
    delimit each (src, dst) pair; square brackets delimit the list of pairs.

    Example:
      `%6 = adl.processor.compute "matrix_lane", [(%5, %5)]`
  }];

  let arguments = (ins FlatSymbolRefAttr:$sym_name,
                       Variadic<MemHandleType>:$src_mems,
                       Variadic<MemHandleType>:$dst_mems);
  let results = (outs ArchHandleType:$handle);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// adl.processor.dmover
//   %12 = adl.processor.dmover "dram_l1_mover", [(%2, %5), (%5, %2)]
//===----------------------------------------------------------------------===//

def ProcessorDMoverOp : ADL_Op<"processor.dmover",
    [Symbol, AttrSizedOperandSegments]> {
  let summary = "Declare a data mover processor with memory region pairs";
  let description = [{
    `adl.processor.dmover` declares a named data-mover processor. Memory region
    pairs specify valid data transfer routes between memory regions.

    Example:
      `%12 = adl.processor.dmover "dram_l1_mover", [(%2, %5), (%5, %2)]`
  }];

  let arguments = (ins StrAttr:$sym_name,
                       Variadic<MemHandleType>:$src_mems,
                       Variadic<MemHandleType>:$dst_mems);
  let results = (outs ArchHandleType:$handle);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// adl.arch.compose
//   %5 = adl.arch.compose "core" arch(%3, %4) mem(%2)
//===----------------------------------------------------------------------===//

def ArchComposeOp : ADL_Op<"arch.compose", [AttrSizedOperandSegments]> {
  let summary = "Compose architectural and memory components";
  let description = [{
    `adl.arch.compose` combines architecture handles (`arch(...)`) and memory
    handles (`mem(...)`) into a new composed architecture. Either clause may
    be empty.

    Example:
      `%5 = adl.arch.compose "core", arch[%3, %4], mem[%2]`
  }];

  let arguments = (ins FlatSymbolRefAttr:$sym_name,
                       Variadic<ArchHandleType>:$arch_components,
                       Variadic<MemHandleType>:$mem_components);
  let results = (outs ArchHandleType:$handle);

  // AttrSizedOperandSegments auto-manages operandSegmentSizes and excludes
  // it from attr-dict. Both ArchHandleType and MemHandleType have builderCall
  // set, so no explicit type annotation is needed in the IR.
  let assemblyFormat = "$sym_name `,` `arch` `[` $arch_components `]` `,` `mem` `[` $mem_components `]` attr-dict";
}

//===----------------------------------------------------------------------===//
// adl.arch.scale
//   %8 = adl.arch.scale "mesh", (%6, %7) of %5
//===----------------------------------------------------------------------===//

def ArchScaleOp : ADL_Op<"arch.scale"> {
  let summary = "Scale an architecture across spatial dimensions";
  let description = [{
    `adl.arch.scale` replicates an architecture across one or more spatial
    dimensions. The result is a new `!adl.arch` handle.

    Example:
      `%8 = adl.arch.scale "mesh", (%6, %7) of %5`
  }];

  let arguments = (ins StrAttr:$name,
                       Variadic<Index>:$spatial_dims,
                       ArchHandleType:$target);
  let results = (outs ArchHandleType:$handle);

  let assemblyFormat = "$name `,` `[` $spatial_dims `]` `of` $target attr-dict";
}
