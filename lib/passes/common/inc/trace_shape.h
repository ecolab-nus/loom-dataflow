/**
 * @file trace_shape.h
 * @brief Symbolic-shape tracing for tensor/memref SSA values.
 */

#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace loom {
namespace utils {

using SymbolicDim = mlir::OpFoldResult;

/**
 * @brief Trace a tensor/memref SSA value back to its symbolic per-dimension
 * shape, expressed as an `OpFoldResult` per axis.
 * @details Walks the same SSA def-use chain as `traceToRootAlloc` (loom
 * forwarding ops, BlockArguments of scf.for / affine.for / affine.parallel,
 * DPS init redirects, casts), but stops at shape-source ops (tensor.empty,
 * tensor.extract_slice, loom.init_tensor, loom.bufferize_to_tensor,
 * memref.subview behind bufferization.to_tensor) and emits their mixed sizes.
 * Fails loudly via `report_fatal_error` if no rule matches the encountered op.
 */
llvm::SmallVector<SymbolicDim, 4> traceShape(mlir::Value v);

} // namespace utils
} // namespace loom
