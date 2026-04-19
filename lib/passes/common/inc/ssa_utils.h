#pragma once

#include "mlir/IR/Value.h"

namespace loom::utils {

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

} // namespace loom::utils
