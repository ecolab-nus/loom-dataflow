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

} // namespace loom::utils
