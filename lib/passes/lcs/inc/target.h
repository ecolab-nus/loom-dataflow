#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace loom {
namespace lcs {

enum class Target { Generic, TT };

llvm::StringRef stringifyTarget(Target target);

/// LOOM_TARGET=tt selects TT. An unset or empty value selects generic.
mlir::FailureOr<Target> parseTargetEnvironment(std::string &error);

/// Record the target on a module, or reject a conflicting existing value.
mlir::LogicalResult resolveModuleTarget(mlir::ModuleOp module, Target requested,
                                        std::string &error);

} // namespace lcs
} // namespace loom
