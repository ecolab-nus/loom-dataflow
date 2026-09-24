#include "target.h"

#include "mlir/IR/BuiltinAttributes.h"
#include <cstdlib>

namespace loom {
namespace lcs {

llvm::StringRef stringifyTarget(Target target) {
  switch (target) {
  case Target::Generic:
    return "generic";
  case Target::TT:
    return "tt";
  }
  llvm_unreachable("unknown Loom target");
}

mlir::FailureOr<Target> parseTargetEnvironment(std::string &error) {
  const char *raw = std::getenv("LOOM_TARGET");
  if (!raw || raw[0] == '\0')
    return Target::Generic;
  llvm::StringRef value(raw);
  if (value == "tt")
    return Target::TT;
  error = "LOOM_TARGET must be unset, empty, or 'tt'; got '" + value.str() +
          "'";
  return mlir::failure();
}

mlir::LogicalResult resolveModuleTarget(mlir::ModuleOp module,
                                        Target requested,
                                        std::string &error) {
  llvm::StringRef requestedName = stringifyTarget(requested);
  if (auto existing =
          module->getAttrOfType<mlir::StringAttr>("loom.target")) {
    if (existing.getValue() != requestedName) {
      error = "artifact target '" + existing.getValue().str() +
              "' conflicts with requested target '" + requestedName.str() +
              "'";
      return mlir::failure();
    }
    return mlir::success();
  }
  if (module->hasAttr("loom.target")) {
    error = "loom.target must be a string attribute";
    return mlir::failure();
  }
  module->setAttr("loom.target",
                  mlir::StringAttr::get(module.getContext(), requestedName));
  return mlir::success();
}

} // namespace lcs
} // namespace loom
