# TileLoomToTTKernel Knowledge Base

This directory implements the TileLoom-to-TTKernel lowering pipeline used by
LOOM. The pass takes tiled Loom IR plus selected `linalg`, `scf`, and `memref`
operations and rewrites them into TTKernel dialect kernels, host helpers, and
the runtime argument layout needed by downstream TT-Metal code generation.

This document is intended to be a future-maintenance guide for humans and LLMs.
It focuses on:

- where the lowering starts
- which file owns which behavior
- which functions are the real extension points
- what invariants must stay aligned across files
- how to test and debug changes without re-discovering the whole pipeline

## Quick Start

Main driver:

```bash
build/tool/tileloom-to-ttkernel/tileloom_to_ttkernel_opt \
  --loom-tileloom-to-ttkernel input.mlir -o kernel_ttkernel.mlir
```

Optional pass option:

```bash
build/tool/tileloom-to-ttkernel/tileloom_to_ttkernel_opt \
  --loom-tileloom-to-ttkernel='matmul-merge-b-reader-into-writer=1' \
  input.mlir -o kernel_ttkernel.mlir
```

Post-EmitC host signature rewrite:

```bash
build/tool/tileloom-to-ttkernel/tileloom_to_ttkernel_opt \
  --loom-post-emitc-host-signature kernel_emitc.mlir \
  -o kernel_emitc_hostsig.mlir
```

Repository shortcut:

```bash
./lower.sh
```

Important workflow note:

- `lower.sh` runs `replace.py` after TTKernel lowering. That script patches a
  few generated TTKernel spellings before the `ttkernel -> emitc` step.
- If generated MLIR starts failing in the `ttmlir-opt --convert-ttkernel-to-emitc`
  stage, inspect both the pass output and `replace.py`.
- `lower.sh` then runs `split_kernel.py`, which now emits:
  - `host_cpp.cpp`
  - `host_pybind.cpp`
  - `host_ttnn.py` (a `ttnn.generic_op` wrapper generated from `host_pybind`)

## What The Pipeline Produces

Each original TileLoom `func.func` is specialized into five variants:

- `<name>__compute`: compute kernel; original load/store structure is retained
  long enough to become CB synchronization rather than NOC traffic
- `<name>__reader`: DRAM-to-L1 / NOC reader kernel
- `<name>__writer`: L1-to-DRAM / NOC writer kernel
- `<name>__host_cpp`: vector-backed host helper
- `<name>__host_pybind`: tensor-backed host helper

The original unspecialized function is erased after cloning.

## End-To-End Lowering Flow

The main pass lives in `src/TileLoomToTTKernel.cpp`. The real execution order is:

1. Run buffer-hoisting preprocess passes.
2. Rewrite `loom.matmul` and `loom.batch_matmul` into `linalg` ops.
3. Optionally annotate matmul functions for merged B-reader/writer partitioning.
4. Specialize each original function into compute, reader, writer, and host variants.
5. Run canonicalize/CSE/SymbolDCE cleanup on the specialized functions.
6. Rewrite batch-1 `linalg.batch_matmul` into plain `linalg.matmul`.
7. Replace function arguments with `ttkernel.get_arg_val`-based values through
   `CompileArgTracker`.
8. Insert one `ttkernel.mm_init` into compute kernels that contain matmul.
9. Apply partial conversion for:
   - `scf.parallel`
   - `loom.semaphore_take`
   - `loom.semaphore_give`
   - `loom.copy`
   - `loom.gather` (writer-side cross-core transport)
   - `linalg.matmul`
   - selected `linalg.generic` (including reduction-style sum)
   - `linalg.fill`
   - selected `linalg.copy`
   - `memref.reinterpret_cast`
   - shape-preserving `memref.collapse_shape`
10. Erase leftover host lowering artifacts.
11. Run a second cleanup conversion that removes dead `loom.alloc`.
12. Remove all remaining function arguments.
13. Run final canonicalize/CSE/SymbolDCE cleanup.
14. Strip descriptor dialect ops from `df` and `adl`.

The separate `loom-post-emitc-host-signature` pass runs later, after
`ttkernel -> emitc`, to rewrite host helper signatures into C++/pybind-friendly
types.

## File Map

| File | Owns | Touch this when |
| --- | --- | --- |
| `inc/TileLoomToTTKernel.h` | public pass factories and registration | adding a new top-level pass factory or exposing a new public entrypoint |
| `src/TileLoomToTTKernel.cpp` | pass orchestration, legality, options, cleanup, post-EmitC host signature rewrite | changing pipeline order, adding a new lowering stage, changing pass options, changing legality |
| `inc/FuncOpToTTKernel.h` | `CompileArgTracker` API and specialization entrypoints | adding new compile/runtime args or changing specialization contracts |
| `src/FuncOpToTTKernel.cpp` | function cloning, compile-arg creation, host helper emission, matmul merge annotation | changing kernel split strategy, host codegen, or argument layout |
| `inc/SCFOpToTTKernel.h` | SCF conversion API | extending SCF lowering surface |
| `src/SCFOpToTTKernel.cpp` | `scf.parallel` lowering into compile-time core coords | changing how spatial loops map to core args |
| `inc/MemoryOpToTTKernel.h` | memory conversion API and alloc cleanup API | adding memory patterns or cleanup phases |
| `src/MemoryOpToTTKernel.cpp` | `loom.semaphore*`, `loom.copy`, NOC read/write, multicast, gather transport, reinterpret-cast cleanup | changing DRAM/L1 semantics, broadcast logic, semaphore behavior, gather transport protocol |
| `inc/ComputeOpToTTKernel.h` | `ReduceProtocol`, `ReduceCombineOp` enums, compute conversion API and legality predicates | exposing new compute rewrite helpers, adding combine kinds, or changing reduce protocol |
| `src/ComputeOpToTTKernel.cpp` | matmul/fill/generic/copy lowering into TTKernel ops | adding a new compute op, extending supported generic expressions, or adding a combine kind |
| `tool/tileloom-to-ttkernel/tileloom_to_ttkernel_opt.cpp` | standalone `mlir-opt`-style driver | changing CLI entrypoint or dialect registration |
| `test/Passes/tileloom_to_ttkernel_noc.mlir` | regression coverage for NOC IDs and the merge option | validating reader/writer split changes |
| `lower.sh` | local end-to-end lowering example | checking the current manual workflow |
| `split_kernel.py` | post-translation section split and host wrapper generation (`host_ttnn.py`) | changing generated file naming, host post-processing, or Python wrapper template |
| `replace.py` | TTKernel MLIR post-processing before EmitC conversion | fixing downstream textual quirks in generated MLIR |

## Public Entry Points You Should Know First

### `src/TileLoomToTTKernel.cpp`

- `createTileLoomToTTKernelPass()`
  Creates the main lowering pass.
- `createInsertMMInitPass()`
  Creates the helper pass that injects `ttkernel.mm_init`.
- `registerTileLoomToTTKernelPass()`
  Registers the main pass, the mm-init pass, and the post-EmitC host signature pass.
- `TileLoomToTTKernelPass::runOnOperation()`
  The real pipeline driver. Start here when behavior seems globally wrong.
- `PostEmitCHostSignaturePass::runOnOperation()`
  Rewrites host helper signatures after EmitC lowering.

### `src/FuncOpToTTKernel.cpp`

- `prepareMatmulBReaderMerge(ModuleOp)`
  Pre-annotates eligible original functions when
  `matmul-merge-b-reader-into-writer=1` is enabled.
- `specializeFunctionsForTTKernel(ModuleOp)`
  Creates the five specialized function variants and erases the original.
- `replaceFuncArgsWithCompileArgs(...)`
  Delegates to `CompileArgTracker` to replace original arguments with
  `ttkernel.get_arg_val` materializations.
- `removeAllFunctionArguments(func::FuncOp)`
  Final argument cleanup after all rewrites have consumed them.

### `src/SCFOpToTTKernel.cpp`

- `populateSCFOpConversionPatterns(...)`
  Registers `scf.parallel` lowering.

### `src/MemoryOpToTTKernel.cpp`

- `populateMemoryOpConversionPatterns(...)`
  Registers `loom.semaphore`, `loom.copy`, and `memref.reinterpret_cast` rewrites.
- `populateLoomAllocCleanupPatterns(...)`
  Registers the follow-up erasure of dead `loom.alloc`.

### `src/ComputeOpToTTKernel.cpp`

- `populateComputeOpConversionPatterns(...)`
  Registers compute rewrites.
- `isSupportedFlashAttentionGeneric(linalg::GenericOp)`
  Legality predicate used by the main pass to decide whether a generic op must
  be converted here.
- `shouldConvertComputeLinalgCopy(linalg::CopyOp)`
  Same idea for compute-side `linalg.copy`.

## Function-Level Ownership Guide

This section lists the most important functions to inspect before editing code.

### `src/TileLoomToTTKernel.cpp`

- `rewriteLoomLinearAlgebraToLinalg(ModuleOp)`
  Converts `loom.matmul` and `loom.batch_matmul` into `linalg` equivalents.
- `rewriteBatch1MatmulToMatmul(ModuleOp)`
  Only supports static rank-3 memrefs with batch size 1.
- `eraseHostLoweringArtifacts(ModuleOp)`
  Removes leftover host-only loop shells and dead `get_arg_val` artifacts after
  specialization and conversion.
- `InsertMMInitPass::runOnOperation()`
  Inserts exactly one `ttkernel.mm_init` in compute kernels that contain matmul.
  It currently assumes the first two memref-backed CB args are inputs and the
  last memref-backed CB arg is the output. If compile-arg ordering changes,
  revisit this first.

### `src/FuncOpToTTKernel.cpp`

- `CompileArgTracker::processInputArgs(...)`
  The most important argument-layout function in the whole directory.
  For each memref argument it creates:
  - CB handle
  - base DRAM address
  - multicast destination start/end coordinates
  - multicast destination count
  - multicast sender coordinates
  - sender/receiver semaphore addresses
  For non-compute kernels it also creates `TensorAccessor` metadata.
- `CompileArgTracker::createIndexCompileArg(...)`
  Used for `scf.parallel` induction variables that become compile-time args.
- `CompileArgTracker::setCoreCoordForDim(...)`
  Stores the per-function `x` and `y` core coordinate values.
- `inferArgToCBMemrefType(...)`
  Infers the L1 CB memref type for each function argument from `loom.copy`
  edges. This decouples DRAM tensor shapes from on-core CB shapes.
- `ensureReductionScaleInputs(func::FuncOp)`
  Injects a one-tile scale semaphore for reduction-style `linalg.generic`.
- `annotateMatmulBReaderMerge(func::FuncOp)`
  Marks the A load for the reader kernel and the B load plus store for the
  writer kernel.
- `makeComputeFunc`, `makeReaderFunc`, `makeWriterFunc`, `makeHostFunc`
  Define the specialization policy for each clone.
- `isGatherTransportOp(Operation *)` / `shouldKeepGatherOpInKernel(StringRef)`
  Explicit gather-op classification during specialization. `makeWriterFunc`
  uses this to keep gather ops in the writer clone for transport lowering.
- `TTMetalHostProgramEmitter::run()`
  Central host helper generator. If generated host code is wrong, start here.
- `TTMetalHostProgramEmitter::emitReaderRuntimeArgsForCore()`
  Builds the runtime-arg vector used for reader, writer, and compute kernels.
  If kernel argument layout changes, this must stay aligned with
  `CompileArgTracker::processInputArgs`.
- `TTMetalHostProgramEmitter::emitKernelRoles()`
  Emits the `reader.cpp`, `writer.cpp`, and `compute.cpp` kernel creation calls.

### `src/SCFOpToTTKernel.cpp`

- `ConvertSCFParallelOp::matchAndRewrite(...)`
  Rewrites `scf.parallel` into straight-line code by replacing IVs with
  compile-time args and recording mapped `x` and `y` core coordinates.
  This is where `loom.mapped_to_dims` becomes runtime-visible core metadata.

### `src/MemoryOpToTTKernel.cpp`

- `dram_read(...)`
  Emits the tiled DRAM-to-L1 NOC read loop based on a source
  `memref.reinterpret_cast`.
- `multicast_send(...)` / `multicast_receive(...)`
  Broadcast synchronization helpers for reader-side multicast.
- `ConvertLoomSemaphoreTakeOp::matchAndRewrite(...)`
  Converts semaphores into CB handles in compute kernels, but mostly passes the
  source through in reader/writer/host kernels.
- `ConvertLoomSemaphoreGiveOp::matchAndRewrite(...)`
  Lowers compute-side gives to `ttkernel.cb_pop_front`; erases them elsewhere.
- `ConvertLoomMemoryLoadOp::matchAndRewrite(...)`
  Lowers reader-kernel DRAM-to-L1 copies, including broadcast handling.
- `ConvertLoomMemoryStoreOp::matchAndRewrite(...)`
  Lowers writer-kernel L1-to-DRAM copies.
- `ConvertLoomComputeLoadOp::matchAndRewrite(...)`
  Erases compute-side load copies because compute op lowering handles CB
  synchronization.
- `ConvertLoomComputeStoreOp::matchAndRewrite(...)`
  Erases compute-side store copies; matmul lowering itself materializes packed
  output into the destination CB.
- `ConvertLoomGatherTransportOp::matchAndRewrite(...)`
  Writer-only gather transport lowering. Emits cross-core NOC payload sends
  (worker side) and semaphore-based reducer synchronization using the selected
  protocol.
- `emitWorkerReduceTransport(...)`
  Worker-side payload send with protocol-specific token/slot handling.
- `emitReducerReduceTransportSync(...)`
  Reducer-side sync: multi-slot bulk wait or single-slot per-worker handshake.
- `ConvertReinterpretCastOp::matchAndRewrite(...)`
  Final cleanup for `memref.reinterpret_cast`.
- `ConvertLoomAllocOp::matchAndRewrite(...)`
  Final cleanup for dead `loom.alloc`.

### `src/ComputeOpToTTKernel.cpp`

- `ConvertLinalgMatmulOp::matchAndRewrite(...)`
  Main matmul lowering. Emits:
  - `ttkernel.matmul_block_init_short`
  - `ttkernel.cb_wait_front`
  - `ttkernel.tile_regs_acquire`
  - `ttkernel.experimental.matmul_block`
  - packing of tile-register results into the output CB
- `findMatmulOutputMaterializationScope(...)`
  Decides where output packing should occur if the result is consumed outside
  the immediate block.
- `ConvertLinalgFillOp::matchAndRewrite(...)`
  Lowers `linalg.fill` into SFPU tile generation plus pack loop.
- `ConvertLinalgCopyOp::matchAndRewrite(...)`
  Handles compute-side CB-to-CB copies when input/output tile counts match.
- `ConvertFlashAttentionGenericOp::matchAndRewrite(...)`
  Dispatches supported `linalg.generic` ops into reduction or elementwise
  lowering.
- `rewriteReduceGeneric(...)`
  Lowers reduction-style `linalg.generic` using TTKernel reduce ops.
- `rewriteElementwiseGeneric(...)`
  Lowers supported elementwise generic expressions tile-by-tile.
- `analyzeElementwiseGeneric(...)` and `emitElementwiseExprToReg(...)`
  These are the main extension points when adding new supported expression
  shapes inside FlashAttention-style generics.
- `rewriteReduceGeneric(...)`
  Lowers reduction-style `linalg.generic` (including gather-fed sum) using
  TTKernel reduce ops.

## Supported IR Contracts

The current implementation assumes:

- most tensor/memref shapes are static
- tile math is based on `32 x 32` tiles
- `scf.parallel` has no results/reductions
- `linalg.batch_matmul` is only rewritten when batch size is exactly `1`
- compute kernels are identified by `ThreadTypeAttr == Compute` or `__compute`
  naming
- data movement kernels are identified by `ThreadTypeAttr == Noc` or the
  `__reader` / `__writer` naming convention
- host helpers rely on function suffixes like `__host_cpp` and `__host_pybind`
- broadcast logic expects mapped spatial dims to expose both `@x` and `@y`

If an input stops satisfying one of these contracts, failures usually appear in:

- `replaceFuncArgsWithCompileArgs`
- `ConvertSCFParallelOp`
- `ConvertLinalgMatmulOp`
- `ConvertLoomMemoryLoadOp`
- `PostEmitCHostSignaturePass`

## Cross-File Invariants That Must Stay In Sync

### 1. Kernel argument layout

If you add or reorder per-memref kernel arguments, update all of:

- `CompileArgTracker::processInputArgs`
- `TTMetalHostProgramEmitter::emitReaderRuntimeArgsForCore`
- any logic that assumes `kRuntimeArgsPerMemref == 11`
- any downstream kernel source that reads those runtime args

Current canonical runtime-arg order must follow `CompileArgTracker` allocation:

1. per-memref 11-field tuple(s)
2. optional reduce runtime semaphores (`ready`, `token`)
3. core coordinates (`x`, `y`) from `scf.parallel` lowering
4. remaining internal/runtime tail args

### 2. Core coordinate mapping

If you change how spatial loops map to hardware dimensions, keep these aligned:

- `ConvertSCFParallelOp::matchAndRewrite`
- `CompileArgTracker::setCoreCoordForDim`
- reader-side broadcast logic in `ConvertLoomMemoryLoadOp`
- host-side runtime arg emission in `TTMetalHostProgramEmitter`

### 3. Specialization policy

If you change which ops live in compute, reader, writer, or host functions,
revisit all of:

- `makeComputeFunc`
- `makeReaderFunc`
- `makeWriterFunc`
- `makeHostFunc`
- `isGatherTransportOp` / `shouldKeepGatherOpInKernel`
  (gather-transport classification)
- the dynamic legality rules in `TileLoomToTTKernelPass::runOnOperation`

### 4. Post-EmitC host signature rewrite

`PostEmitCHostSignaturePass` depends on the host emitter setting:

- `loom.host_memref_count`

If host helpers stop carrying that attribute, the signature rewrite pass will fail.

### 5. Gather transport protocol and split

`loom.gather` is the transport op. It is retained in `__writer` kernels and
erased in `__compute` / `__reader` kernels. Sum math is lowered from
`linalg.generic` in compute kernels.

The writer-side gather transport lowering is handled by
`ConvertLoomGatherTransportOp` in `MemoryOpToTTKernel.cpp`.
`reduce-protocol` controls worker/reducer synchronization (`single-slot` vs
`multi-slot`) during gather payload transport.

Legacy `loom.reduce_sum` is rejected with a migration diagnostic
(`use loom.gather + linalg.generic sum`).

## Common Change Recipes

### Add a new compute op lowering

1. Add or update a pattern in `src/ComputeOpToTTKernel.cpp`.
2. Register it in `populateComputeOpConversionPatterns(...)`.
3. Mark the op illegal or dynamically illegal in `TileLoomToTTKernelPass::runOnOperation()`.
4. Add a focused regression test.

### Change DRAM <-> L1 transfer behavior

1. Inspect `ConvertLoomMemoryLoadOp` and `ConvertLoomMemoryStoreOp`.
2. If argument layout changes, also update `CompileArgTracker`.
3. If host runtime args change, update `TTMetalHostProgramEmitter`.
4. Re-check multicast helpers if broadcast is involved.

### Add a new supported `linalg.generic` expression

1. Extend `analyzeElementwiseExpr(...)` and possibly `analyzeElementwiseGeneric(...)`.
2. Extend `emitElementwiseExprToReg(...)`.
3. Keep `isSupportedFlashAttentionGeneric(...)` consistent with what can really lower.

### Change host code generation

1. Start in `TTMetalHostProgramEmitter::run()`.
2. Update the corresponding helper methods (`emitPreamble`, `emitCompileArgs`,
   `emitKernelRoles`, `emitRuntimeEnqueueEpilogue`, and runtime-arg emission).
3. Re-check `PostEmitCHostSignaturePass`.
4. Re-run the `lower.sh` flow to ensure EmitC/C++ translation still succeeds.

### Tune gather transport protocol

1. Inspect `ConvertLoomGatherTransportOp` in `MemoryOpToTTKernel.cpp`.
2. Keep runtime-arg plumbing aligned with `CompileArgTracker::processInputArgs`.
3. Validate both `single-slot` and `multi-slot` behavior with focused tests.
4. Ensure reducer-only sum/store control flow remains outside gather transport.

### Change the matmul reader/writer partition option

1. Inspect `prepareMatmulBReaderMerge(...)`.
2. Inspect `annotateMatmulBReaderMerge(...)`.
3. Inspect `makeReaderFunc(...)` and `makeWriterFunc(...)`.
4. Validate against `test/Passes/tileloom_to_ttkernel_noc.mlir`.

## Known Gotchas

- `generate_multicast_address(...)` exists in `MemoryOpToTTKernel.cpp`, but the
  current broadcast lowering path in `ConvertLoomMemoryLoadOp` inlines similar
  logic and the direct helper call is commented out. Do not assume the helper is
  on the active path.
- Compute-side `loom.copy` store lowering currently erases the op. The real
  output materialization for matmul happens inside `ConvertLinalgMatmulOp`.
- The host emitter hardcodes kernel source names:
  - `reader.cpp`
  - `writer.cpp`
  - `compute.cpp`
  under `tt_metal/programming_examples/mlir_matmul_simple/kernels/`
- Descriptor dialect ops from `df` and `adl` are stripped only at the very end.
  If you expect them during debugging, inspect pre-final-cleanup IR.
- `replace.py` is a real part of the current local workflow, not just a debug
  helper.
- `host_ttnn.py` is generated from `host_pybind` with a mostly fixed template.
  Today it requires `start_core_x == 0` and `start_core_y == 0`, because
  `ProgramDescriptor.runtime_args[i][j]` is absolute-core indexed.

## Tests And Debugging

Reference regression:

```bash
build/tool/tileloom-to-ttkernel/tileloom_to_ttkernel_opt \
  --loom-tileloom-to-ttkernel \
  test/Passes/tileloom_to_ttkernel_noc.mlir | \
  FileCheck test/Passes/tileloom_to_ttkernel_noc.mlir --check-prefix=DEFAULT
```

Merge-option regression:

```bash
build/tool/tileloom-to-ttkernel/tileloom_to_ttkernel_opt \
  --loom-tileloom-to-ttkernel='matmul-merge-b-reader-into-writer=1' \
  test/Passes/tileloom_to_ttkernel_noc.mlir | \
  FileCheck test/Passes/tileloom_to_ttkernel_noc.mlir --check-prefix=MERGED
```

Useful grep entry points:

```bash
rg -n "loom-tileloom-to-ttkernel|loom-post-emitc-host-signature" .
rg -n "CompileArgTracker|kRuntimeArgsPerMemref|emitReaderRuntimeArgsForCore" \
  lib/passes/TileLoomToTTKernel/src/FuncOpToTTKernel.cpp
rg -n "ConvertLoomMemoryLoadOp|ConvertLoomMemoryStoreOp|multicast_" \
  lib/passes/TileLoomToTTKernel/src/MemoryOpToTTKernel.cpp
rg -n "ConvertLinalgMatmulOp|rewriteElementwiseGeneric|rewriteReduceGeneric" \
  lib/passes/TileLoomToTTKernel/src/ComputeOpToTTKernel.cpp
rg -n "ReduceProtocol|ConvertLoomGather" \
  lib/passes/TileLoomToTTKernel/src/ lib/passes/TileLoomToTTKernel/inc/
```

When debugging a failure, inspect IR at these checkpoints:

- before specialization
- after specialization and cleanup
- after argument replacement
- after partial conversion
- after alloc cleanup
- after post-EmitC host signature rewrite

## LLM Handoff Checklist

Before editing this subsystem, answer these questions:

1. Is the change for compute, reader, writer, host, or more than one of them?
2. Does it change kernel argument layout or host runtime-arg layout?
3. Does it rely on `scf.parallel` mapped dims or broadcast semantics?
4. Does the op need to be marked illegal in the main conversion target?
5. Is there already a regression test that covers the affected path?
6. Does it add a new reduce combine kind? (See the recipe above.)

If the answer to question 2 is yes, inspect `FuncOpToTTKernel.cpp` before touching anything else.
