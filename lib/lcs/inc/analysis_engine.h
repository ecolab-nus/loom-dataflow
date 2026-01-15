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
#include <map>

// Forward declarations for Loom operations
namespace loom {
class ConstraintSpaceOp;
class SymbolicVarOp;
class RangeOp;
class AlignOp;
class LinearConstraintOp;
class PolynomialConstraintOp;
class ExpressionOp;
} // namespace loom

namespace mlir {
class Value;
class Operation;
} // namespace mlir

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

/// @brief Represents an interval [lower, upper] for bound inference.
struct Interval {
  int64_t lower;
  int64_t upper;

  bool isValid() const { return lower <= upper; }

  static Interval unbounded() {
    return {std::numeric_limits<int64_t>::min() / 2,
            std::numeric_limits<int64_t>::max() / 2};
  }

  static Interval constant(int64_t val) { return {val, val}; }
};

/// @brief ValueTracker manages the mapping between SSA values and matrix column
/// indices.
class ValueTracker {
public:
  ValueTracker() = default;

  /// @brief Tracks a symbolic variable as a dimension.
  unsigned trackDimension(mlir::Value val, llvm::StringRef name);

  /// @brief Tracks an expression result as a local (auxiliary) variable.
  unsigned trackLocalId(mlir::Value val);

  /// @brief Resolves the column index for an SSA value.
  std::optional<unsigned> getColumnIndex(mlir::Value val) const;

  /// @brief Checks if a value is tracked as a dimension.
  bool isDimension(mlir::Value val) const;

  /// @brief Gets the number of dimensions.
  unsigned getNumDims() const { return valueToDimIndex_.size(); }

  /// @brief Gets the number of local IDs.
  unsigned getNumLocals() const { return valueToLocalIndex_.size(); }

  /// @brief Gets the mapping of SSA values to dimension indices.
  const llvm::DenseMap<mlir::Value, unsigned> &getDimensions() const {
    return valueToDimIndex_;
  }

  /// @brief Gets the mapping of SSA values to local indices.
  const llvm::DenseMap<mlir::Value, unsigned> &getLocalIds() const {
    return valueToLocalIndex_;
  }

private:
  llvm::DenseMap<mlir::Value, unsigned> valueToDimIndex_;
  llvm::DenseMap<mlir::Value, unsigned> valueToLocalIndex_;
};

/// @brief BoundInferenceService provides interval arithmetic and range queries.
class BoundInferenceService {
public:
  explicit BoundInferenceService(loom::ConstraintSpaceOp csOp);

  /// @brief Initializes the service by scanning range operations.
  void initialize();

  /// @brief Gets the range for a given SSA value.
  Interval getRange(mlir::Value val);

  /// @brief Manual override or update for a value's range.
  void setRange(mlir::Value val, Interval range);

  // Interval arithmetic helpers
  static Interval add(Interval a, Interval b);
  static Interval multiply(Interval a, Interval b);
  static Interval scalarMultiply(int64_t scalar, Interval a);

private:
  mlir::Operation *csOp_;
  llvm::DenseMap<mlir::Value, Interval> boundsTable_;

  /// @brief Computes bounds for an expression operation recursively.
  Interval computeExpressionBounds(ExpressionOp op);
};

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

  /// @brief Visits a polynomial constraint.
  ///
  /// For now, this just logs the constraint as they are not yet supported
  /// by the Presburger-based feasibility check.
  ///
  /// @param op The polynomial constraint operation.
  void visitPolynomialConstraint(PolynomialConstraintOp op);

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
  ValueTracker valueTracker_;

  /// Service for bound inference.
  std::unique_ptr<BoundInferenceService> boundService_;

  /// @brief Visits an expression operation.
  void visitExpression(ExpressionOp op);
};

} // namespace lcs
} // namespace loom

#endif // LOOM_LCS_ANALYSIS_ENGINE_H
