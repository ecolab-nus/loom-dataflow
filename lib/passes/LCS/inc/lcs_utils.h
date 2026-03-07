/**
 * @file lcs_utils.h
 * @brief Tracing utilities for LCS (Loom Compute Schedule) analysis.
 */

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include <string>

// Forward declaration - AllocOp is defined in generated LoomOps.h.inc
namespace loom {
class AllocOp;
}

namespace loom {
namespace lcs {

/**
 * @brief Trace a memref Value backward through SemaphoreTakeOp to its AllocOp.
 * Follows the SSA def-use chain: memref -> AllocOp or SemaphoreTakeOp.
 *
 * @param memrefVal The memref SSA value to trace
 * @return The AllocOp if found, nullptr otherwise
 */
::loom::AllocOp traceToAlloc(mlir::Value memrefVal);

/**
 * @brief Convert AllocOp mixed sizes (static + dynamic) to symbolic dims string.
 * Static dimensions become integers (e.g., "128").
 * Dynamic dimensions are traced to symbolic variables (e.g., "BB", "BM").
 * Result is joined with " * " (e.g., "BB * BM * 128").
 *
 * @param allocOp The allocation operation
 * @return Symbolic dims string, or empty string if no valid dims found
 */
std::string formatAllocDims(::loom::AllocOp allocOp);

/**
 * @brief Recursively trace a tensor-typed Value to its underlying AllocOp dims.
 * Handles complex IR patterns: loom ops (copy_to_tensor, init_tensor),
 * linalg operations (fill, copy, generic, matmul), affine loops with iter_args,
 * and block arguments from affine.for / affine.parallel.
 *
 * @param tensorVal The tensor SSA value to trace
 * @return Symbolic dims string, or empty string if tracing fails
 */
std::string traceAllocDimsFromTensor(mlir::Value tensorVal);

} // namespace lcs
} // namespace loom
