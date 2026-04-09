/**
 * @file lcs_utils.h
 * @brief Tracing utilities for LCS (Loom Compute Schedule) analysis.
 */

#pragma once

#include "expr.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

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
 * @brief Convert AllocOp mixed sizes to individual symbolic Expr per dimension.
 * Static dimensions become Expr::con(). Dynamic dimensions are traced to
 * Expr::sym(). Returns one Expr per dimension. Empty vector if no valid dims.
 *
 * @param allocOp The allocation operation
 * @return Vector of per-dimension Exprs (e.g., [Sym("BM"), Sym("BK")])
 */
std::vector<Expr> formatAllocDims(::loom::AllocOp allocOp);

/**
 * @brief Recursively trace a tensor-typed Value to its underlying AllocOp dims.
 * Handles complex IR patterns: loom ops (copy_to_tensor, init_tensor),
 * linalg operations (fill, copy, generic, matmul), affine loops with iter_args,
 * and block arguments from affine.for / affine.parallel.
 *
 * @param tensorVal The tensor SSA value to trace
 * @return Vector of per-dimension Exprs, or empty vector if tracing fails
 */
std::vector<Expr> traceAllocDimsFromTensor(mlir::Value tensorVal);

/**
 * @brief Fold a vector of dimension Exprs into a single product Expr.
 * Returns Expr::none() if dims is empty.
 *
 * @param dims Vector of per-dimension Exprs
 * @return Product expression, or Expr::none()
 */
Expr productOfDims(const std::vector<Expr> &dims);

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

/// Extract symbolic trip count from an scf::ForOp's upper bound SSA value.
/// Traces arith constants, loom.sym refs, and simple arithmetic (ceildivui,
/// muli, addi) into Expr nodes. Assumes lb=0 and step=1.
Expr extractLoopTripCount(mlir::scf::ForOp forOp);

// ==========================================
// Generic Op Classification & Shape Analysis
// ==========================================

/// Classification of linalg.generic by iterator_types.
enum class GenericClass {
  Parallel,  // all iterator_types are "parallel"
  Reduction, // all iterator_types are "reduction"
  Mixed      // both parallel and reduction
};

/// Classify an iterator types array into Parallel, Reduction, or Mixed.
GenericClass classifyIteratorTypes(
    llvm::ArrayRef<mlir::utils::IteratorType> iteratorTypes);

/// Result of analyzeGenericDims: folded symbolic products per iterator class.
struct GenericDimAnalysis {
  GenericClass generic_class;
  Expr parallel_product;  // Expr::none() if no parallel dims
  Expr reduction_product; // Expr::none() if no reduction dims
};

/**
 * @brief Analyze a linalg.generic's indexing_maps and iterator_types to
 * produce folded symbolic dimension products for parallel and reduction dims.
 *
 * For each loop dimension d_i:
 *   1. Classify as parallel or reduction from iterator_types.
 *   2. Scan all operands' indexing maps for a simple AffineDimExpr at d_i.
 *   3. Use the matched map result position to index into that operand's
 *      traced tensor dims (via traceAllocDimsFromTensor) → symbolic Expr.
 *   4. Fold parallel Exprs into parallel_product, reduction into
 *      reduction_product.
 *
 * @param genericOp The linalg.generic operation to analyze
 * @return GenericDimAnalysis with class and folded products
 */
GenericDimAnalysis analyzeGenericDims(mlir::linalg::LinalgOp genericOp);

} // namespace lcs
} // namespace loom
