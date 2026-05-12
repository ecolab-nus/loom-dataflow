#pragma once

#include "staged_etg_builder.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <map>
#include <string>

namespace loom {
namespace lcs {

/// Apply hardware-specific alignment metadata to registered constraint symbols.
void applyHardwareAlignments(mlir::func::FuncOp func_op,
                             std::map<std::string, SymbolInfo> &symbols);

} // namespace lcs
} // namespace loom
