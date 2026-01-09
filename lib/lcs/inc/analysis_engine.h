//===- analysis_engine.h - Loom Constraint Analysis Engine ------*- C++ -*-===//
//
// Defines the AnalysisEngine class that converts Loom constraint IR operations
// into mathematical ConstraintSet objects using the Presburger library.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_LCS_ANALYSIS_ENGINE_H
#define LOOM_LCS_ANALYSIS_ENGINE_H

#include "constraint_set.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

// Forward declarations for Loom operations
namespace loom {
class ConstraintSpaceOp;
class SymbolicVarOp;
class RangeOp;
class AlignOp;
class LinearConstraintOp;
} // namespace loom

namespace loom {
namespace lcs {

/// @brief AnalysisEngine converts Loom constraint IR into ConstraintSet.
///
/// This class uses a visitor pattern to traverse the operations within a
/// ConstraintSpaceOp and build up an IntegerPolyhedron representation.
/// It serves as the bridge between MLIR IR and the Presburger math library.
///
/// Usage:
/// @code
///   loom::ConstraintSpaceOp csOp = ...;
///   ConstraintSet cs = AnalysisEngine::buildConstraintSet(csOp);
///   bool valid = cs.contains({64, 128, 32});
/// @endcode
///
/// The engine processes operations in the following order:
/// 1. SymbolicVarOp: Register dimension variables
/// 2. RangeOp: Add lower/upper bound constraints
/// 3. AlignOp: Add alignment (modulo) constraints
/// 4. LinearConstraintOp: Add linear inequality constraints from AffineMap
class AnalysisEngine {
public:
  /// @brief Builds a ConstraintSet from a ConstraintSpaceOp.
  ///
  /// This is the main entry point. It traverses all operations in the
  /// constraint space's body block and populates a ConstraintSet.
  ///
  /// @param csOp The constraint space operation to analyze.
  /// @return A ConstraintSet representing all constraints.
  static ConstraintSet buildConstraintSet(ConstraintSpaceOp csOp);

  /// @brief Default constructor.
  AnalysisEngine() = default;

  /// @brief Processes all operations in a constraint space.
  /// @param csOp The constraint space operation.
  void processConstraintSpace(ConstraintSpaceOp csOp);

  /// @brief Gets the built constraint set.
  /// @return The constraint set built by processing operations.
  ConstraintSet &getConstraintSet() { return constraintSet_; }
  const ConstraintSet &getConstraintSet() const { return constraintSet_; }

private:
  /// @brief Visits a symbolic variable definition.
  ///
  /// Registers the variable name and maps it to a dimension index.
  /// Also records the mapping from SSA value to dimension index.
  ///
  /// @param op The symbolic variable operation.
  void visitSymbolicVar(SymbolicVarOp op);

  /// @brief Visits a range constraint.
  ///
  /// Adds lower and upper bound constraints for the variable.
  ///
  /// @param op The range operation.
  void visitRange(RangeOp op);

  /// @brief Visits an alignment constraint.
  ///
  /// Introduces a local variable and adds an equality constraint
  /// to enforce variable ≡ 0 (mod alignment).
  ///
  /// @param op The align operation.
  void visitAlign(AlignOp op);

  /// @brief Visits a linear constraint.
  ///
  /// Parses the AffineMap and adds inequality constraints.
  /// Each result of the map represents a constraint: result >= 0.
  ///
  /// @param op The linear constraint operation.
  void visitLinearConstraint(LinearConstraintOp op);

  /// @brief Resolves the dimension index for an SSA value.
  ///
  /// The value must be the result of a SymbolicVarOp.
  ///
  /// @param val The SSA value to resolve.
  /// @return The dimension index if found, std::nullopt otherwise.
  std::optional<unsigned> resolveDimIndex(mlir::Value val) const;

  /// @brief Extracts coefficients from an AffineExpr.
  ///
  /// The expression must be affine in the dimensions. Symbols are not
  /// supported and will cause an assertion failure.
  ///
  /// @param expr The affine expression to analyze.
  /// @param numDims The number of dimension variables.
  /// @param[out] coeffs Output coefficient vector (one per dimension).
  /// @param[out] constant Output constant term.
  /// @return true if extraction succeeded, false otherwise.
  bool extractCoefficients(mlir::AffineExpr expr, unsigned numDims,
                           llvm::SmallVectorImpl<int64_t> &coeffs,
                           int64_t &constant) const;

  /// The constraint set being built.
  ConstraintSet constraintSet_;

  /// Maps SSA values (results of SymbolicVarOp) to dimension indices.
  llvm::DenseMap<mlir::Value, unsigned> valueToDimIndex_;
};

} // namespace lcs
} // namespace loom

#endif // LOOM_LCS_ANALYSIS_ENGINE_H

