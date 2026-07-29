#include "hw_op_registry.h"
#include "utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

namespace loom {
namespace lcs {

namespace {

mlir::FailureOr<int64_t> getLocalMemKind(mlir::Type type,
                                        mlir::Operation *op,
                                        unsigned operandIndex) {
  mlir::Attribute attr;
  if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    mlir::Attribute encoding = tensorType.getEncoding();
    if (!encoding)
      return 0;
    auto dictionary = mlir::dyn_cast<mlir::DictionaryAttr>(encoding);
    if (!dictionary) {
      op->emitError() << "staged-etg: tensor operand " << operandIndex
                      << " has a non-dictionary encoding; expected "
                         "'local_mem_kind : i64'";
      return mlir::failure();
    }
    attr = dictionary.get("local_mem_kind");
    if (!attr)
      return 0;
  } else if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
    attr = memrefType.getMemorySpace();
    if (!attr)
      return 0;
  } else {
    return 0;
  }

  auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attr);
  if (!integer) {
    op->emitError() << "staged-etg: operand " << operandIndex
                    << " local memory kind must be an integer attribute";
    return mlir::failure();
  }

  return integer.getInt();
}

} // namespace

mlir::FailureOr<ComputeOpMatchInfo>
getComputeOpMatchInfo(mlir::linalg::LinalgOp linalgOp) {
  ComputeOpMatchInfo result;
  unsigned operandIndex = 0;
  auto appendKind = [&](mlir::Value value) -> mlir::LogicalResult {
    mlir::FailureOr<int64_t> kind =
        getLocalMemKind(value.getType(), linalgOp.getOperation(), operandIndex);
    ++operandIndex;
    if (mlir::failed(kind))
      return mlir::failure();
    result.operand_mem_kinds.push_back(*kind);
    return mlir::success();
  };

  for (mlir::Value input : linalgOp.getDpsInputs())
    if (mlir::failed(appendKind(input)))
      return mlir::failure();
  for (mlir::Value init : linalgOp.getDpsInits())
    if (mlir::failed(appendKind(init)))
      return mlir::failure();
  return result;
}

std::string formatComputeOpMatchInfo(const ComputeOpMatchInfo &info) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "(";
  for (size_t i = 0; i < info.operand_mem_kinds.size(); ++i) {
    if (i)
      os << ",";
    os << info.operand_mem_kinds[i];
  }
  os << ")";
  return os.str();
}

bool hasOnlyDefaultMemoryOperands(const ComputeOpMatchInfo &info) {
  return llvm::all_of(info.operand_mem_kinds, [](int64_t kind) {
    return kind == 0;
  });
}

// ============================================================
// Extraction helpers: compute ops (linalg-based)
// ============================================================

mlir::Operation *HWOpRegistry::findComputeOp(mlir::func::FuncOp func) {
  mlir::Operation *computeOp = nullptr;
  func.walk([&](mlir::linalg::LinalgOp linalgOp) {
    mlir::Operation *op = linalgOp.getOperation();
    if (llvm::isa<mlir::linalg::FillOp, mlir::linalg::CopyOp>(op))
      return;
    assert(!computeOp && "Expected exactly one compute linalg op per hw func");
    computeOp = op;
  });
  return computeOp;
}

void HWOpRegistry::fillInputOutputBindings(
    mlir::linalg::LinalgOp linalgOp,
    const llvm::DenseMap<mlir::Value, HWTensorBinding> &bindingMap,
    HWComputeFunc &result) {
  auto pushBinding = [&](mlir::Value val,
                         std::vector<HWTensorBinding> &bindings) {
    auto it = bindingMap.find(val);
    bindings.push_back(it != bindingMap.end() ? it->second : HWTensorBinding{});
  };
  for (mlir::Value input : linalgOp.getDpsInputs())
    pushBinding(input, result.input_bindings);
  for (mlir::Value output : linalgOp.getDpsInits())
    pushBinding(output, result.output_bindings);
}

bool HWOpRegistry::fillGenericDetails(
    mlir::Operation *computeOp,
    const llvm::DenseMap<mlir::Value, HWTensorBinding> &bindingMap,
    HWComputeFunc &result) {
  auto linalgOp = llvm::cast<mlir::linalg::LinalgOp>(computeOp);

  // Find the single arith/math body op (compound ops like cmpf+select skip).
  mlir::Operation *singleBodyOp = nullptr;
  unsigned bodyOpCount = 0;
  for (mlir::Operation &bodyOp : computeOp->getRegion(0).front()) {
    if (llvm::isa<mlir::linalg::YieldOp>(&bodyOp))
      continue;
    mlir::Dialect *dialect = bodyOp.getDialect();
    if (!dialect)
      continue;
    llvm::StringRef ns = dialect->getNamespace();
    if (ns != "arith" && ns != "math")
      continue;
    singleBodyOp = &bodyOp;
    bodyOpCount++;
  }
  if (bodyOpCount != 1)
    return false;

  result.body_op_name = singleBodyOp->getName().getStringRef().str();
  result.generic_class = classifyIteratorTypes(linalgOp.getIteratorTypesArray());

  auto indexingMaps = linalgOp.getIndexingMapsArray();
  llvm::SmallVector<mlir::Value> allOperands;
  for (auto v : linalgOp.getDpsInputs())
    allOperands.push_back(v);
  for (auto v : linalgOp.getDpsInits())
    allOperands.push_back(v);

  auto iteratorTypes = linalgOp.getIteratorTypesArray();
  for (unsigned di = 0; di < iteratorTypes.size(); ++di) {
    bool found = false;
    for (unsigned opIdx = 0; opIdx < indexingMaps.size() && !found; ++opIdx) {
      mlir::AffineMap map = indexingMaps[opIdx];
      auto bindIt = bindingMap.find(allOperands[opIdx]);
      if (bindIt == bindingMap.end())
        continue;
      const auto &binding = bindIt->second;
      for (unsigned r = 0; r < map.getNumResults() && !found; ++r) {
        auto affineDim = mlir::dyn_cast<mlir::AffineDimExpr>(map.getResult(r));
        if (!affineDim || affineDim.getPosition() != di)
          continue;
        if (r >= binding.dim_symbols.size())
          continue;
        const std::string &sym = binding.dim_symbols[r];
        if (iteratorTypes[di] == mlir::utils::IteratorType::parallel &&
            result.parallel_symbol.empty())
          result.parallel_symbol = sym;
        else if (iteratorTypes[di] == mlir::utils::IteratorType::reduction &&
                 result.reduction_symbol.empty())
          result.reduction_symbol = sym;
        found = true;
      }
    }
  }
  return true;
}

mlir::FailureOr<std::optional<HWComputeFunc>>
HWOpRegistry::extractFromFunc(mlir::func::FuncOp func,
                              llvm::StringRef hw_component) {
  auto bindingMap = collectBindingMap(func);

  mlir::Operation *computeOp = findComputeOp(func);
  if (!computeOp)
    return std::optional<HWComputeFunc>{};

  auto linalgOp = llvm::cast<mlir::linalg::LinalgOp>(computeOp);
  mlir::FailureOr<ComputeOpMatchInfo> matchInfo =
      getComputeOpMatchInfo(linalgOp);
  if (mlir::failed(matchInfo))
    return mlir::failure();

  HWComputeFunc result;
  result.linalg_op_name = computeOp->getName().getStringRef().str();
  result.hw_func_name = func.getName().str();
  result.hw_component = hw_component.str();
  result.compute_match = std::move(*matchInfo);
  fillInputOutputBindings(linalgOp, bindingMap, result);

  if (llvm::isa<mlir::linalg::GenericOp>(computeOp))
    if (!fillGenericDetails(computeOp, bindingMap, result))
      return std::optional<HWComputeFunc>{};

  return std::optional<HWComputeFunc>{std::move(result)};
}

} // namespace lcs
} // namespace loom
