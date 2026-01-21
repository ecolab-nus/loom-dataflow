//===- constraint_space_utils.h - Constraint Space Utilities ----*- C++ -*-===//
//
// Utilities for manipulating Loom ConstraintSpaceOp at the IR level,
// including cloning, adding constraints, and feasibility checking.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_LCS_CONSTRAINT_SPACE_UTILS_H
#define LOOM_LCS_CONSTRAINT_SPACE_UTILS_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class Operation;
}

namespace loom {
class ConstraintSpaceOp;
class LinearConstraintOp;
class PolynomialConstraintOp;
class RangeOp;
class SymbolicVarOp;
} // namespace loom

namespace loom {
namespace lcs {

/// Export constraint space to JSON string
std::string exportConstraintSpaceToJson(mlir::Operation *csOp,
                                        llvm::StringRef passName);

/// @brief Deep clone a ConstraintSpaceOp into a target location.
///
/// Creates a complete copy of the source constraint space including all
/// symbolic variables, range constraints, alignment constraints, and linear
/// constraints. The cloned constraint space is inserted at the builder's
/// current insertion point.
///
/// @param builder OpBuilder for creating the clone.
/// @param sourceSpace The constraint space to clone.
/// @param newName Optional new name for the cloned space (uses source name if
/// empty).
/// @return The cloned ConstraintSpaceOp.
ConstraintSpaceOp cloneConstraintSpace(mlir::OpBuilder &builder,
                                       ConstraintSpaceOp sourceSpace,
                                       llvm::StringRef newName = "");

/// @brief Find a ConstraintSpaceOp in a module.
///
/// Searches the module for a ConstraintSpaceOp. If multiple exist, returns
/// the first one found.
///
/// @param module The module to search.
/// @return The found ConstraintSpaceOp, or nullptr if not found.
ConstraintSpaceOp findConstraintSpace(mlir::ModuleOp module);

/// @brief Find a SymbolicVarOp by name within a ConstraintSpaceOp.
///
/// @param csOp The constraint space to search.
/// @param varName The name of the symbolic variable.
/// @return The SymbolicVarOp if found, nullptr otherwise.
SymbolicVarOp findSymbolicVar(ConstraintSpaceOp csOp, llvm::StringRef varName);

/// @brief Add a linear constraint to an existing ConstraintSpaceOp.
///
/// Creates a LinearConstraintOp using the specified variables and constraint
/// map. The variables must already exist in the constraint space.
///
/// @param csOp The constraint space to add the constraint to.
/// @param varNames Names of the symbolic variables used in the constraint.
/// @param constraintMap AffineMap representing the constraint (result >= 0).
/// @return The created LinearConstraintOp, or nullptr if variables not found.
LinearConstraintOp addLinearConstraint(ConstraintSpaceOp csOp,
                                       llvm::ArrayRef<llvm::StringRef> varNames,
                                       mlir::AffineMap constraintMap);

/// Monomial representation for polynomial constraints
struct Monomial {
  llvm::SmallVector<int64_t, 2> varIndices; // Indices into operand list
  int64_t coeff;

  size_t degree() const { return varIndices.size(); }

  bool hasVar(int64_t v) const {
    return std::find(varIndices.begin(), varIndices.end(), v) !=
           varIndices.end();
  }

  void removeVar(int64_t v) {
    varIndices.erase(std::remove(varIndices.begin(), varIndices.end(), v),
                     varIndices.end());
  }

  bool operator<(const Monomial &other) const {
    if (varIndices.size() != other.varIndices.size())
      return varIndices.size() < other.varIndices.size();
    return varIndices < other.varIndices;
  }

  bool sameVars(const Monomial &other) const {
    return varIndices == other.varIndices;
  }
};

/// @brief Parse monomials from an ArrayAttr.
llvm::SmallVector<Monomial> parseMonomials(mlir::ArrayAttr monomialsAttr);

/// @brief Build monomials attribute from parsed structure.
mlir::ArrayAttr buildMonomialsAttr(mlir::MLIRContext *ctx,
                                   llvm::ArrayRef<Monomial> monomials);

/// @brief Compute GCD of two integers.
int64_t gcd(int64_t a, int64_t b);

/// @brief Compute GCD of a vector of integers.
int64_t gcdVector(llvm::ArrayRef<int64_t> values);

/// @brief Add a polynomial constraint to an existing ConstraintSpaceOp.
///
/// Creates a PolynomialConstraintOp using the specified variables, monomials
/// and upper bound. The variables must already exist in the constraint space.
///
/// @param csOp The constraint space to add the constraint to.
/// @param varNames Names of the symbolic variables used in the constraint.
/// @param monomials List of monomials in the polynomial.
/// @param upperBound The upper bound for the polynomial (sum <= upperBound).
/// @return The created PolynomialConstraintOp, or nullptr if variables not
/// found.
PolynomialConstraintOp
addPolynomialConstraint(ConstraintSpaceOp csOp,
                        llvm::ArrayRef<llvm::StringRef> varNames,
                        llvm::ArrayRef<Monomial> monomials, int64_t upperBound);

/// @brief Add a range constraint to an existing ConstraintSpaceOp.
///
/// Creates a RangeOp for the specified variable with the given bounds.
/// The variable must already exist in the constraint space.
///
/// @param csOp The constraint space to add the constraint to.
/// @param varName Name of the symbolic variable.
/// @param lowerBound The lower bound (inclusive).
/// @param upperBound The upper bound (inclusive).
/// @return The created RangeOp, or nullptr if variable not found.
RangeOp addRangeConstraint(ConstraintSpaceOp csOp, llvm::StringRef varName,
                           int64_t lowerBound, int64_t upperBound);

/// @brief Update lower bounds of all range constraints in a space.
void updateRangeLowerBounds(ConstraintSpaceOp csOp, int64_t newLowerBound);

/// @brief Add alignment constraints for all symbolic variables in a space.
void addAlignConstraintsForAllVars(ConstraintSpaceOp csOp, int64_t alignment);

/// @brief Add a pipeline parallelism constraint (multi-variable product >=
/// minVal).
PolynomialConstraintOp
addPipelineParallelismConstraint(ConstraintSpaceOp csOp,
                                 llvm::ArrayRef<llvm::StringRef> varNames,
                                 int64_t minMatUnits, int64_t alignment);

/// @brief Check if the constraint space is feasible.
///
/// Uses AnalysisEngine to build a ConstraintSet from the IR and checks
/// if there are any valid points satisfying all constraints.
///
/// @param csOp The constraint space to check.
/// @return true if the constraint space is feasible (non-empty), false
/// otherwise.
bool isFeasible(ConstraintSpaceOp csOp);

/// @brief Set the pass name attribute on a module for provenance tracking.
///
/// @param module The module to set the attribute on.
/// @param passName The name of the pass that created/modified this module.
void setPassNameAttr(mlir::ModuleOp module, llvm::StringRef passName);

/// @brief Get the pass name attribute from a module.
///
/// @param module The module to get the attribute from.
/// @return The pass name if set, empty string otherwise.
llvm::StringRef getPassNameAttr(mlir::ModuleOp module);

} // namespace lcs
} // namespace loom

#endif // LOOM_LCS_CONSTRAINT_SPACE_UTILS_H
