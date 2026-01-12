//===- constraint_set.h - Loom Constraint Set ------------------*- C++ -*-===//
//
// Defines the ConstraintSet class, a wrapper around Presburger's
// IntegerPolyhedron for managing symbolic constraints from Loom IR.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_LCS_CONSTRAINT_SET_H
#define LOOM_LCS_CONSTRAINT_SET_H

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

namespace loom {
namespace lcs {

using mlir::presburger::BoundType;
using mlir::presburger::IntegerPolyhedron;
using mlir::presburger::PresburgerSpace;
using mlir::presburger::VarKind;

/// @brief ConstraintSet wraps an IntegerPolyhedron and maintains mappings
/// from symbolic variable names to dimension indices.
///
/// This class provides a high-level interface for querying constraint
/// satisfaction and computing variable bounds. It is built from Loom
/// constraint operations (SymbolicVarOp, RangeOp, AlignOp, LinearConstraintOp).
///
/// All symbolic variables (M, N, K, etc.) are treated as Dimensions in the
/// Presburger space to enable enumeration and sampling.
class ConstraintSet {
public:
  /// @brief Constructs an empty ConstraintSet.
  ConstraintSet();

  /// @brief Constructs a ConstraintSet with a specified number of dimensions.
  /// @param numDims The number of dimension variables.
  explicit ConstraintSet(unsigned numDims);

  /// @brief Copy constructor for deep copying a ConstraintSet.
  /// @param other The ConstraintSet to copy from.
  ConstraintSet(const ConstraintSet &other);

  /// @brief Move constructor.
  /// @param other The ConstraintSet to move from.
  ConstraintSet(ConstraintSet &&other) noexcept = default;

  /// @brief Copy assignment operator.
  ConstraintSet &operator=(const ConstraintSet &other);

  /// @brief Move assignment operator.
  ConstraintSet &operator=(ConstraintSet &&other) noexcept = default;

  /// @brief Creates a deep copy of this ConstraintSet.
  /// @return A new ConstraintSet that is a copy of this one.
  ConstraintSet clone() const;

  /// @brief Registers a symbolic variable and returns its dimension index.
  /// @param name The name of the symbolic variable.
  /// @return The dimension index assigned to this variable.
  unsigned registerVariable(llvm::StringRef name);

  /// @brief Gets the dimension index for a registered variable.
  /// @param name The name of the symbolic variable.
  /// @return The dimension index if found, std::nullopt otherwise.
  std::optional<unsigned> getDimIndex(llvm::StringRef name) const;

  /// @brief Gets the name of a variable by its dimension index.
  /// @param dimIdx The dimension index.
  /// @return The variable name if valid, std::nullopt otherwise.
  std::optional<llvm::StringRef> getVarName(unsigned dimIdx) const;

  /// @brief Adds a lower bound constraint: variable >= lowerBound.
  /// @param dimIdx The dimension index of the variable.
  /// @param lowerBound The lower bound value.
  void addLowerBound(unsigned dimIdx, int64_t lowerBound);

  /// @brief Adds an upper bound constraint: variable <= upperBound.
  /// @param dimIdx The dimension index of the variable.
  /// @param upperBound The upper bound value.
  void addUpperBound(unsigned dimIdx, int64_t upperBound);

  /// @brief Adds a range constraint: lowerBound <= variable <= upperBound.
  /// @param dimIdx The dimension index of the variable.
  /// @param lowerBound The lower bound value (inclusive).
  /// @param upperBound The upper bound value (inclusive).
  void addRange(unsigned dimIdx, int64_t lowerBound, int64_t upperBound);

  /// @brief Adds an alignment constraint: variable ≡ 0 (mod alignment).
  ///
  /// This is implemented by introducing a local variable q and adding the
  /// equality constraint: variable = alignment * q.
  ///
  /// @param dimIdx The dimension index of the variable.
  /// @param alignment The alignment value (must be positive).
  void addAlignment(unsigned dimIdx, int64_t alignment);

  /// @brief Adds a linear inequality constraint from coefficient vector.
  ///
  /// The constraint is: sum(coeffs[i] * var_i) + constant >= 0.
  ///
  /// @param coeffs Coefficients for each dimension variable.
  /// @param constant The constant term.
  void addInequality(llvm::ArrayRef<int64_t> coeffs, int64_t constant);

  /// @brief Adds a linear equality constraint from coefficient vector.
  ///
  /// The constraint is: sum(coeffs[i] * var_i) + constant == 0.
  ///
  /// @param coeffs Coefficients for each dimension variable.
  /// @param constant The constant term.
  void addEquality(llvm::ArrayRef<int64_t> coeffs, int64_t constant);

  /// @brief Checks if a point satisfies all constraints.
  /// @param point The point to check, one value per dimension variable.
  /// @return true if the point is in the constraint set, false otherwise.
  bool contains(llvm::ArrayRef<int64_t> point) const;

  /// @brief Computes the bounds for a single variable.
  ///
  /// This uses projection to eliminate other variables and find the
  /// absolute range of the specified variable.
  ///
  /// @param varName The name of the variable.
  /// @return A pair (lowerBound, upperBound) if computable, std::nullopt if
  ///         unbounded or the variable is not found.
  std::optional<std::pair<int64_t, int64_t>>
  getBounds(llvm::StringRef varName) const;

  /// @brief Checks if the constraint set is empty (no valid points).
  /// @return true if no points satisfy all constraints.
  bool isEmpty() const;

  /// @brief Gets the number of registered dimension variables.
  /// @return The number of dimensions.
  unsigned getNumDims() const;

  /// @brief Gets the number of local (existentially quantified) variables.
  /// @return The number of local variables.
  unsigned getNumLocals() const;

  /// @brief Gets the underlying IntegerPolyhedron.
  /// @return A const reference to the polyhedron.
  const IntegerPolyhedron &getPolyhedron() const { return polyhedron_; }

  /// @brief Gets the underlying IntegerPolyhedron (mutable).
  /// @return A mutable reference to the polyhedron.
  IntegerPolyhedron &getPolyhedron() { return polyhedron_; }

  /// @brief Dumps the constraint set to stderr for debugging.
  void dump() const;

private:
  /// The underlying Presburger polyhedron.
  IntegerPolyhedron polyhedron_;

  /// Maps variable names to dimension indices.
  llvm::StringMap<unsigned> varToDimIndex_;

  /// Stores variable names in order of dimension index.
  llvm::SmallVector<std::string, 8> dimNames_;
};

} // namespace lcs
} // namespace loom

#endif // LOOM_LCS_CONSTRAINT_SET_H

