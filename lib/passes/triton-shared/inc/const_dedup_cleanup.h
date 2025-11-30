//===- const_dedup_cleanup.h -------------------------------*- C++ -*-===//
// Deduplicate constants, erase unused ones, and fold constant operands
// into affine.apply maps where possible.
//===------------------------------------------------------------------===//

#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace tmd {
namespace passes {

/**
 * Create a pass that performs constant cleanup:
 * - Deduplicate identical arith.constant/constant_index per function
 * - Remove unused constants
 * - Fold constant operands into affine.apply by embedding them into maps
 */
std::unique_ptr<mlir::Pass> createConstDedupCleanupPass();

} // namespace passes
} // namespace tmd
