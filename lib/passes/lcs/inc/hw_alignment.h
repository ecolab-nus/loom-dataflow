#pragma once

#include "staged_etg_builder.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <map>
#include <string>

namespace loom {
namespace lcs {

/// TT storage layout: tile symbols sizing an allocation's bottom-2 dims must
/// be multiples of 32. Applied only for Target::TT.
void applyHardwareAlignments(mlir::func::FuncOp func_op,
                             std::map<std::string, SymbolInfo> &symbols);

} // namespace lcs
} // namespace loom
