#include "hw_alignment.h"
#include "utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

namespace loom {
namespace lcs {

namespace {

constexpr int64_t kInnerDimAlignment = 32;

void applyAlignmentFromValue(mlir::Value value,
                             std::map<std::string, SymbolInfo> &symbols) {
  llvm::StringRef symbolName = loom::utils::traceToSymbolicVar(value);
  if (symbolName.empty())
    return;

  auto it = symbols.find(symbolName.str());
  if (it == symbols.end())
    return;

  it->second.alignment = std::max(it->second.alignment, kInnerDimAlignment);
}

} // namespace

void applyHardwareAlignments(mlir::func::FuncOp func_op,
                             std::map<std::string, SymbolInfo> &symbols) {
  func_op.walk([&](loom::AllocOp allocOp) {
    auto staticSizes = allocOp.getStaticSizes();
    auto dynamicSizes = allocOp.getSizes();
    if (staticSizes.empty())
      return;

    unsigned dynamicIdx = 0;
    for (auto indexedDim : llvm::enumerate(staticSizes)) {
      int64_t staticDim = indexedDim.value();
      bool isInnerDim = indexedDim.index() + 2 >= staticSizes.size();
      bool isDynamic = mlir::ShapedType::isDynamic(staticDim);

      mlir::Value dynamicValue;
      if (isDynamic && dynamicIdx < dynamicSizes.size())
        dynamicValue = dynamicSizes[dynamicIdx++];

      if (isInnerDim && dynamicValue)
        applyAlignmentFromValue(dynamicValue, symbols);
    }
  });
}

} // namespace lcs
} // namespace loom
