/**
 * @file lcs_utils.h
 * @brief Tracing utilities for LCS (Loom Compute Schedule) analysis.
 */

#pragma once

#include "expr.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
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
 * @brief Convert AllocOp mixed sizes (static + dynamic) to a symbolic Expr.
 * Static dimensions become Expr::con(). Dynamic dimensions are traced to
 * Expr::sym(). All dimensions are folded into a single product expression.
 * Returns Expr::none() if no valid dims are found.
 *
 * @param allocOp The allocation operation
 * @return Product Expr (e.g., Sym("BB") * Sym("BM") * Con(128)), or none()
 */
Expr formatAllocDims(::loom::AllocOp allocOp);

/**
 * @brief Recursively trace a tensor-typed Value to its underlying AllocOp Expr.
 * Handles complex IR patterns: loom ops (copy_to_tensor, init_tensor),
 * linalg operations (fill, copy, generic, matmul), affine loops with iter_args,
 * and block arguments from affine.for / affine.parallel.
 *
 * @param tensorVal The tensor SSA value to trace
 * @return Symbolic Expr, or Expr::none() if tracing fails
 */
Expr traceAllocDimsFromTensor(mlir::Value tensorVal);

/**
 * @brief Convert an MLIR element type to a readable string (e.g., "f32", "i32").
 * This is not an algebraic expression, so it remains a std::string.
 *
 * @param elemType The MLIR element type to format
 * @return String representation of the type
 */
std::string formatElementType(mlir::Type elemType);

/**
 * @brief Convert an AffineExpr to a symbolic Expr ADT.
 * Recursively walks the expression tree and substitutes symbol references
 * with Expr::sym() nodes. Returns Expr::none() for unhandled cases.
 *
 * @param expr The affine expression to convert
 * @param symbolNames Vector mapping symbol index (s0, s1, ...) to traced names
 * @return Expr representing the affine expression
 */
Expr affineExprToExpr(mlir::AffineExpr expr,
                      const llvm::SmallVector<std::string> &symbolNames);

/**
 * @brief Extract symbolic trip count from an AffineForOp's upper bound.
 * Assumes lower bound is 0. Extracts the upper bound map, traces its symbol
 * operands to symbolic names, and converts the result expression to an Expr.
 *
 * @param forOp The affine loop to analyze
 * @return Expr for the trip count, or Expr::none() if extraction fails
 */
Expr extractLoopTripCount(mlir::affine::AffineForOp forOp);

} // namespace lcs
} // namespace loom
