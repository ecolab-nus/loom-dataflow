#pragma once

#include "mlir/IR/Value.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace loom {
class AllocOp;
} // namespace loom

namespace loom::utils {

enum class CopyMemoryDirection { Load, Store, Other };

/**
 * @brief Check whether `value` depends (transitively) on `target`.
 * @details Walks the SSA def-use graph backward from `value` to determine if
 * `target` appears among its transitive operands. Block arguments stop the
 * walk.
 */
bool dependsOn(mlir::Value value, mlir::Value target);

/**
 * @brief Trace a tensor/memref value to its root `loom.alloc` value.
 * @details Walks backward through Loom buffer/tensor bridge ops, casts,
 * DestinationStyle op result->init links, and loop-carried values.
 * Returns the `loom.alloc` result value if found, otherwise null.
 */
mlir::Value traceToRootAlloc(mlir::Value value);

/**
 * @brief Typed wrapper over `traceToRootAlloc`.
 * @details Returns the backing `loom.alloc` op, or null if the trace fails.
 */
loom::AllocOp traceToRootAllocOp(mlir::Value value);

/**
 * @brief Classify a `loom.copy` direction using canonical memory-space names.
 * @details DRAM->L1 is Load, L1->DRAM is Store, everything else is Other.
 */
CopyMemoryDirection classifyCopyMemoryDirection(mlir::Operation *op);

/**
 * @brief Trace the L1 endpoint of a load/store `loom.copy` to its root alloc.
 * @details For Load this traces the destination. For Store this traces the
 * source. Other directions return null.
 */
loom::AllocOp traceCopyL1EndpointRootAlloc(mlir::Operation *op);

} // namespace loom::utils
