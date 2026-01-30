//===- constraint_set.cpp - Loom Constraint Set Implementation -----------===//
//
// Implementation of the ConstraintSet class.
//
//===----------------------------------------------------------------------===//

#include "constraint_set.h"
#include "analysis_engine.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "loom-constraint-set"

namespace loom {
namespace lcs {

using mlir::presburger::BoundType;
using mlir::presburger::PresburgerSpace;
using mlir::presburger::VarKind;

ConstraintSet::ConstraintSet()
    : polyhedron_(PresburgerSpace::getSetSpace(/*numDims=*/0)) {}

ConstraintSet::ConstraintSet(unsigned numDims)
    : polyhedron_(PresburgerSpace::getSetSpace(numDims)) {}

ConstraintSet::ConstraintSet(const ConstraintSet &other)
    : polyhedron_(other.polyhedron_), varToDimIndex_(other.varToDimIndex_),
      dimNames_(other.dimNames_) {
  LLVM_DEBUG(llvm::dbgs() << "ConstraintSet copy constructed with "
                          << getNumDims() << " dims\n");
}

ConstraintSet &ConstraintSet::operator=(const ConstraintSet &other) {
  if (this != &other) {
    polyhedron_ = other.polyhedron_;
    varToDimIndex_ = other.varToDimIndex_;
    dimNames_ = other.dimNames_;
  }
  return *this;
}

ConstraintSet ConstraintSet::clone() const { return ConstraintSet(*this); }

unsigned ConstraintSet::registerVariable(llvm::StringRef name) {
  // Check if already registered
  auto it = varToDimIndex_.find(name);
  if (it != varToDimIndex_.end()) {
    return it->second;
  }

  // Append a new dimension variable
  unsigned newDimIdx = polyhedron_.getNumDimVars();
  polyhedron_.insertVar(VarKind::SetDim, newDimIdx);

  // Store the mapping
  dimNames_.push_back(name.str());
  varToDimIndex_[dimNames_.back()] = newDimIdx;

  LLVM_DEBUG(llvm::dbgs() << "Registered variable '" << name << "' at dim "
                          << newDimIdx << "\n");
  return newDimIdx;
}

unsigned ConstraintSet::registerLocalVariable() {
  unsigned localIdx = polyhedron_.getNumLocalVars();
  polyhedron_.appendVar(VarKind::Local, 1);
  return localIdx;
}

std::optional<unsigned> ConstraintSet::getDimIndex(llvm::StringRef name) const {
  auto it = varToDimIndex_.find(name);
  if (it != varToDimIndex_.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::optional<llvm::StringRef>
ConstraintSet::getVarName(unsigned dimIdx) const {
  if (dimIdx < dimNames_.size()) {
    return llvm::StringRef(dimNames_[dimIdx]);
  }
  return std::nullopt;
}

void ConstraintSet::addLowerBound(unsigned dimIdx, int64_t lowerBound) {
  // Constraint: variable >= lowerBound
  // In Presburger form: variable - lowerBound >= 0
  polyhedron_.addBound(BoundType::LB, dimIdx, lowerBound);
  LLVM_DEBUG(llvm::dbgs() << "Added lower bound: dim" << dimIdx
                          << " >= " << lowerBound << "\n");
}

void ConstraintSet::addUpperBound(unsigned dimIdx, int64_t upperBound) {
  // Constraint: variable <= upperBound
  // In Presburger form: -variable + upperBound >= 0
  polyhedron_.addBound(BoundType::UB, dimIdx, upperBound);
  LLVM_DEBUG(llvm::dbgs() << "Added upper bound: dim" << dimIdx
                          << " <= " << upperBound << "\n");
}

void ConstraintSet::addRange(unsigned dimIdx, int64_t lowerBound,
                             int64_t upperBound) {
  addLowerBound(dimIdx, lowerBound);
  addUpperBound(dimIdx, upperBound);
}

void ConstraintSet::addAlignment(unsigned dimIdx, int64_t alignment) {
  assert(alignment > 0 && "Alignment must be positive");

  // Alignment constraint: variable ≡ 0 (mod alignment)
  // Implemented as: variable = alignment * q, where q is a local variable.
  //
  // We add a local variable q and the equality: variable - alignment*q = 0
  //
  // Coefficient vector layout: [dims..., locals..., constant]
  // For equality: coeffs[dimIdx] = 1, coeffs[localIdx] = -alignment, const = 0

  unsigned numDims = polyhedron_.getNumDimVars();
  unsigned numLocals = polyhedron_.getNumLocalVars();

  // Insert a new local variable
  polyhedron_.insertVar(VarKind::Local, numLocals);
  unsigned localIdx = numDims + numLocals; // Position in the coefficient vector

  // Build the equality constraint: dim - alignment*local = 0
  // Layout: [d0, d1, ..., dn, l0, l1, ..., lm, const]
  unsigned numCols = polyhedron_.getNumCols();
  llvm::SmallVector<int64_t, 8> eq(numCols, 0);
  eq[dimIdx] = 1;
  eq[localIdx] = -alignment;
  // eq[numCols - 1] = 0; // constant term is already 0

  polyhedron_.addEquality(eq);

  LLVM_DEBUG(llvm::dbgs() << "Added alignment: dim" << dimIdx << " ≡ 0 (mod "
                          << alignment << ")\n");
}

void ConstraintSet::addInequality(llvm::ArrayRef<int64_t> coeffs,
                                  int64_t constant) {
  // Build the full coefficient vector including locals and constant
  unsigned numCols = polyhedron_.getNumCols();
  unsigned numDims = polyhedron_.getNumDimVars();

  assert(coeffs.size() == numDims &&
         "Coefficient count must match dimension count");

  llvm::SmallVector<int64_t, 8> fullCoeffs(numCols, 0);
  for (unsigned i = 0; i < numDims; ++i) {
    fullCoeffs[i] = coeffs[i];
  }
  fullCoeffs[numCols - 1] = constant;

  polyhedron_.addInequality(fullCoeffs);

  LLVM_DEBUG({
    llvm::dbgs() << "Added inequality: ";
    for (size_t i = 0; i < coeffs.size(); ++i) {
      if (i > 0 && coeffs[i] >= 0)
        llvm::dbgs() << "+ ";
      llvm::dbgs() << coeffs[i] << "*d" << i << " ";
    }
    llvm::dbgs() << "+ " << constant << " >= 0\n";
  });
}

void ConstraintSet::addEquality(llvm::ArrayRef<int64_t> coeffs,
                                int64_t constant) {
  unsigned numCols = polyhedron_.getNumCols();
  unsigned numDims = polyhedron_.getNumDimVars();

  assert(coeffs.size() == numDims &&
         "Coefficient count must match dimension count");

  llvm::SmallVector<int64_t, 8> fullCoeffs(numCols, 0);
  for (unsigned i = 0; i < numDims; ++i) {
    fullCoeffs[i] = coeffs[i];
  }
  fullCoeffs[numCols - 1] = constant;

  polyhedron_.addEquality(fullCoeffs);

  LLVM_DEBUG({
    llvm::dbgs() << "Added equality: ";
    for (size_t i = 0; i < coeffs.size(); ++i) {
      if (i > 0 && coeffs[i] >= 0)
        llvm::dbgs() << "+ ";
      llvm::dbgs() << coeffs[i] << "*d" << i << " ";
    }
    llvm::dbgs() << "+ " << constant << " == 0\n";
  });
}

bool ConstraintSet::contains(llvm::ArrayRef<int64_t> point) const {
  // The point should have one value per dimension variable
  assert(point.size() == polyhedron_.getNumDimVars() &&
         "Point size must match number of dimensions");

  // Use containsPointNoLocal to handle local variables (e.g., from alignment)
  // This function returns a satisfying assignment to locals if the point
  // satisfies all constraints, or std::nullopt otherwise.
  auto result = polyhedron_.containsPointNoLocal(point);
  return result.has_value();
}

std::optional<std::pair<int64_t, int64_t>>
ConstraintSet::getBounds(llvm::StringRef varName) const {
  auto dimIdxOpt = getDimIndex(varName);
  if (!dimIdxOpt) {
    return std::nullopt;
  }
  unsigned dimIdx = *dimIdxOpt;

  // Try to get direct constant bounds first (without projection)
  auto optLB = polyhedron_.getConstantBound64(BoundType::LB, dimIdx);
  auto optUB = polyhedron_.getConstantBound64(BoundType::UB, dimIdx);

  if (optLB && optUB) {
    return std::make_pair(*optLB, *optUB);
  }

  // If direct bounds don't work, try projection approach
  // Make a copy to project out other dimensions
  IntegerPolyhedron projected = polyhedron_;

  // Project out all dimensions except the one we're interested in.
  // We need to project out dimensions in reverse order to maintain indices.
  unsigned numDims = projected.getNumDimVars();
  for (unsigned i = numDims; i > 0; --i) {
    if (i - 1 != dimIdx) {
      projected.projectOut(i - 1);
    }
  }

  // After projection, our variable is at index 0 (if it was kept)
  // The remaining polyhedron should have 1 dimension.
  if (projected.getNumDimVars() != 1) {
    return std::nullopt;
  }

  // Now try to get bounds from the projected polyhedron
  auto projectedLB = projected.getConstantBound64(BoundType::LB, 0);
  auto projectedUB = projected.getConstantBound64(BoundType::UB, 0);

  if (!projectedLB || !projectedUB) {
    return std::nullopt;
  }

  return std::make_pair(*projectedLB, *projectedUB);
}

bool ConstraintSet::isEmpty() const { return polyhedron_.isEmpty(); }

unsigned ConstraintSet::getNumDims() const {
  return polyhedron_.getNumDimVars();
}

unsigned ConstraintSet::getNumLocals() const {
  return polyhedron_.getNumLocalVars();
}

void ConstraintSet::dump() const {
  llvm::errs() << "ConstraintSet with " << getNumDims() << " dimensions and "
               << getNumLocals() << " locals:\n";
  llvm::errs() << "  Variables: ";
  for (unsigned i = 0; i < dimNames_.size(); ++i) {
    if (i > 0)
      llvm::errs() << ", ";
    llvm::errs() << dimNames_[i] << " (d" << i << ")";
  }
  llvm::errs() << "\n";
  polyhedron_.dump();
}

//===----------------------------------------------------------------------===//
// PolyhedronBuilder Implementation
//===----------------------------------------------------------------------===//

PolyhedronBuilder::PolyhedronBuilder(ConstraintSet &cs,
                                     const ValueTracker &tracker)
    : cs_(cs), tracker_(tracker) {
  // Add local variables if they don't exist
  unsigned currentLocals = cs_.getPolyhedron().getNumLocalVars();
  if (currentLocals < tracker.getNumLocals()) {
    cs_.getPolyhedron().insertVar(VarKind::Local, currentLocals,
                                  tracker.getNumLocals() - currentLocals);
  }
}

void PolyhedronBuilder::addLinearConstraint(
    llvm::ArrayRef<mlir::Value> operands, llvm::ArrayRef<int64_t> coeffs,
    int64_t constant) {
  unsigned numCols = cs_.getPolyhedron().getNumCols();
  llvm::SmallVector<int64_t, 8> fullCoeffs(numCols, 0);

  for (unsigned i = 0; i < operands.size(); ++i) {
    auto colIdx = tracker_.getColumnIndex(operands[i]);
    if (colIdx) {
      fullCoeffs[*colIdx] += coeffs[i];
    }
  }
  fullCoeffs[numCols - 1] = constant;

  cs_.getPolyhedron().addInequality(fullCoeffs);
}

void PolyhedronBuilder::addExpressionEquality(
    mlir::Value exprValue, llvm::ArrayRef<mlir::Value> operands,
    llvm::ArrayRef<int64_t> coeffs) {
  unsigned numCols = cs_.getPolyhedron().getNumCols();
  llvm::SmallVector<int64_t, 8> fullCoeffs(numCols, 0);

  // exprValue = sum(coeffs[i] * operands[i])
  // exprValue - sum(coeffs[i] * operands[i]) = 0
  auto exprColIdx = tracker_.getColumnIndex(exprValue);
  if (exprColIdx) {
    fullCoeffs[*exprColIdx] = 1;
  }

  for (unsigned i = 0; i < operands.size(); ++i) {
    auto colIdx = tracker_.getColumnIndex(operands[i]);
    if (colIdx) {
      fullCoeffs[*colIdx] -= coeffs[i];
    }
  }

  cs_.getPolyhedron().addEquality(fullCoeffs);
}

} // namespace lcs
} // namespace loom
