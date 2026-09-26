#pragma once

#include "hw_op_registry.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

namespace loom {
namespace lcs {
namespace detail {

inline bool isAllOnes(llvm::ArrayRef<int64_t> area) {
  return !area.empty() && llvm::all_of(area, [](int64_t value) {
           return value == 1;
         });
}

inline bool isSymbolicArea(llvm::ArrayRef<int64_t> area) {
  return llvm::any_of(area, [](int64_t value) {
    return mlir::ShapedType::isDynamic(value);
  });
}

} // namespace detail
} // namespace lcs
} // namespace loom

