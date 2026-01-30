//===- constraint_exporter.h - Loom Constraint JSON Exporter ----*- C++ -*-===//
//
// Defines the ConstraintExporter class that serializes Loom constraint IR
// operations into a unified JSON format for downstream Python analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_LCS_CONSTRAINT_EXPORTER_H
#define LOOM_LCS_CONSTRAINT_EXPORTER_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LogicalResult.h"
#include <optional>
#include <string>

namespace loom {
class ConstraintSpaceOp;
class PolynomialConstraintOp;
class LinearConstraintOp;
} // namespace loom

namespace mlir {
class Operation;
}

namespace loom {
namespace lcs {

/// Unified term representation for both polynomial and linear constraints
struct TermExport {
  int64_t coefficient;
  llvm::SmallVector<std::string, 4> variables; // Variable names
};

/// Variable metadata for JSON export
struct VariableExport {
  std::string name;
  unsigned index;
  bool isIntermediate;
};

/// Metadata for range/align constraints
struct MetadataExport {
  std::string variable;
  std::string type; // "range" or "align"
  std::optional<int64_t> lowerBound;
  std::optional<int64_t> upperBound;
  std::optional<int64_t> alignment;
};

/// Single constraint export
struct ConstraintExport {
  std::string type; // "polynomial" or "linear"
  llvm::SmallVector<TermExport, 8> terms;
  std::optional<int64_t> constant;
  std::optional<int64_t> upperBound;
  bool isEquality = false;
};

/// Full constraint space export
struct ConstraintSpaceExport {
  std::string passName;
  std::string functionName;
  llvm::SmallVector<VariableExport, 8> variables;
  llvm::SmallVector<MetadataExport, 16> metadata;
  llvm::SmallVector<ConstraintExport, 16> constraints;
};

/// Exporter class for Loom constraints
class ConstraintExporter {
public:
  explicit ConstraintExporter(mlir::Operation *csOp, llvm::StringRef passName);

  /// Export everything to structured data
  ConstraintSpaceExport exportToStruct();

  /// Serialize structured data to JSON
  llvm::json::Value exportToJson();

  /// Utility to get JSON string
  std::string toJsonString();

  /// Write JSON to a file
  mlir::LogicalResult writeToFile(llvm::StringRef path);

private:
  /// Maps all symbolic and intermediate variables to their names
  void buildVariableMapping();

  /// Processes individual constraint types
  ConstraintExport visitPolynomial(loom::PolynomialConstraintOp op);
  ConstraintExport visitLinear(loom::LinearConstraintOp op);

  /// Helper to convert Value to variable name
  std::string getValueName(mlir::Value val);

  mlir::Operation *csOp_;
  std::string passName_;
  llvm::DenseMap<mlir::Value, std::string> valueToName_;
  llvm::SmallVector<mlir::Value, 8> operands_; // Resolved operands of the space
};

/// Convenience function to export JSON for a given pass
std::string exportConstraintSpaceToJson(mlir::Operation *csOp,
                                        llvm::StringRef passName);

/// Convenience function to export JSON Value for a given pass
llvm::json::Value exportConstraintSpaceToValue(mlir::Operation *csOp,
                                               llvm::StringRef passName);

} // namespace lcs
} // namespace loom

#endif // LOOM_LCS_CONSTRAINT_EXPORTER_H
