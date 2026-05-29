#include "hw_op_registry.h"
#include "hw_op_registry_detail.h"
#include "utils.h"

#include "mlir/IR/BuiltinTypes.h"

#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

namespace loom {
namespace lcs {

std::optional<HWComputeFunc>
HWOpRegistry::extractDataMoverFromFunc(mlir::func::FuncOp func,
                                       llvm::StringRef hw_component) {
  auto bindingMap = collectBindingMap(func);

  loom::CopyOp copyOp = nullptr;
  loom::GatherOp gatherOp = nullptr;
  func.walk([&](loom::CopyOp op) {
    assert(!copyOp && "Expected at most one loom.copy per data mover func");
    copyOp = op;
  });
  func.walk([&](loom::GatherOp op) {
    assert(!gatherOp && "Expected at most one loom.gather per data mover func");
    gatherOp = op;
  });

  assert(!(copyOp && gatherOp) &&
         "Expected exactly one loom.copy or loom.gather per data mover func");
  if (!copyOp && !gatherOp)
    return std::nullopt;

  HWComputeFunc result;
  result.is_data_mover = true;
  result.data_mover_kind =
      copyOp ? DataMoverKind::Copy : DataMoverKind::Gather;
  result.hw_func_name = func.getName().str();
  result.hw_component = hw_component.str();

  mlir::Value source;
  mlir::Value destination;
  mlir::SmallVector<mlir::OpFoldResult, 4> mixedArea;

  if (copyOp) {
    source = copyOp.getSource();
    destination = copyOp.getDestination();
    mixedArea = copyOp.getMixedArea();
    if (auto attr = copyOp.getSrcMemSpaceAttr())
      result.src_mem_space = detail::canonicalMemSpace(attr.getLeafReference());
    if (auto attr = copyOp.getDstMemSpaceAttr())
      result.dst_mem_space = detail::canonicalMemSpace(attr.getLeafReference());
  } else {
    source = gatherOp.getSource();
    destination = gatherOp.getDestination();
    mixedArea = gatherOp.getMixedArea();
    if (auto attr = gatherOp.getSrcMemSpaceAttr())
      result.src_mem_space = detail::canonicalMemSpace(attr.getLeafReference());
    if (auto attr = gatherOp.getDstMemSpaceAttr())
      result.dst_mem_space = detail::canonicalMemSpace(attr.getLeafReference());
  }

  for (mlir::OpFoldResult area : mixedArea) {
    if (auto attr = area.dyn_cast<mlir::Attribute>()) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
        result.broadcast.push_back(intAttr.getInt());
        result.area_symbols.push_back("");
        continue;
      }
    }
    mlir::Value areaValue = area.dyn_cast<mlir::Value>();
    llvm::StringRef symName = loom::utils::traceToSymbolicVar(areaValue);
    result.broadcast.push_back(mlir::ShapedType::kDynamic);
    result.area_symbols.push_back(symName.empty() ? "?" : symName.str());
  }

  auto pushBinding = [&](mlir::Value value,
                         std::vector<HWTensorBinding> &bindings) {
    auto it = bindingMap.find(value);
    bindings.push_back(it != bindingMap.end() ? it->second : HWTensorBinding{});
  };
  pushBinding(source, result.input_bindings);
  pushBinding(destination, result.output_bindings);

  return result;
}

} // namespace lcs
} // namespace loom
