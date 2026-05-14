#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace loom::utils {

/// Collect loop IVs that a root index SSA value transitively depends on.
/// Recognized IVs:
/// - affine.parallel induction variables (block arguments in IV positions)
/// - scf.for induction variable
llvm::SmallVector<mlir::Value, 8> collectDependentIVs(mlir::Value root);

/// Collect temporal loops whose IVs are transitively used by `root`.
/// Returned operations are deduplicated in first-seen order.
llvm::SmallVector<mlir::Operation *, 8>
collectTemporalDependentLoops(mlir::Value root);

} // namespace loom::utils

