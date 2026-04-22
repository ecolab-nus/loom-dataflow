#include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"

// Include Loom dialect headers
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace {

/// Represents a single materialized block size configuration.
struct BlockSizeBinding {
  llvm::StringMap<int64_t> varValues;

  std::optional<int64_t> getValue(StringRef name) const {
    auto it = varValues.find(name);
    if (it != varValues.end())
      return it->second;
    return std::nullopt;
  }

  std::string getSuffix() const {
    std::string suffix = "";
    llvm::SmallVector<llvm::StringRef, 4> keys;
    for (auto &entry : varValues) {
      keys.push_back(entry.first());
    }
    std::sort(keys.begin(), keys.end());

    for (auto key : keys) {
      suffix += "__" + key.str() + std::to_string(varValues.lookup(key));
    }
    return suffix;
  }
};

/// Placeholder solver: returns hardcoded candidate tuples.
/// Used only when no external BlockSizeMap is provided (backward compat).
SmallVector<SmallVector<int64_t>> solveCandidateBlockSizes(unsigned numVars) {
  if (numVars == 3) {
    return {{1, 1024, 64}};
  }
  else if (numVars == 6) {
    return {{64, 32, 64, 8, 1, 4}};
  }
  SmallVector<int64_t> fallback(numVars, 64);
  return {fallback};
}

/// Build bindings by zipping variable names with candidate tuples.
SmallVector<BlockSizeBinding>
buildBindings(ArrayRef<StringRef> varNames,
              ArrayRef<SmallVector<int64_t>> candidates) {
  SmallVector<BlockSizeBinding> bindings;
  for (const auto &tuple : candidates) {
    BlockSizeBinding b;
    for (auto [name, val] : llvm::zip(varNames, tuple))
      b.varValues[name] = val;
    bindings.push_back(b);
  }
  return bindings;
}

/// Materialize a function with a specific block size binding.
LogicalResult materializeFunction(func::FuncOp func,
                                  const BlockSizeBinding &binding) {
  OpBuilder builder(func->getContext());
  SmallVector<Operation *, 16> opsToErase;

  func->walk([&](loom::SymOp op) {
    SymbolRefAttr ref = op.getSymbolRef();
    StringRef varName = ref.getLeafReference();

    auto valueOpt = binding.getValue(varName);
    if (!valueOpt) {
      // A missing concrete value here means the solver did not assign this
      // symbolic variable in the current binding (e.g., due to partial or
      // UNSAT results). Warn to aid debugging but continue — the caller is
      // responsible for filtering out incomplete variants.
      op.emitWarning() << "No concrete value found for symbolic variable '"
                       << varName << "'";
      return;
    }

    builder.setInsertionPoint(op);
    auto constant =
        arith::ConstantIndexOp::create(builder, op.getLoc(), *valueOpt);
    op.getResult().replaceAllUsesWith(constant.getResult());
    opsToErase.push_back(op);
  });

  for (auto *op : opsToErase) {
    op->erase();
  }

  return success();
}

class MaterializePass
    : public PassWrapper<MaterializePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializePass)

  /// Default constructor: uses hardcoded placeholder solver.
  MaterializePass() = default;

  /// Constructor with external block sizes from the SMT solver.
  /// blockSizes maps each variant function name to its symbol assignments.
  explicit MaterializePass(const loom::passes::BlockSizeMap &blockSizes)
      : externalBlockSizes(&blockSizes) {}

  StringRef getArgument() const override { return "loom-materialize"; }

  StringRef getDescription() const override {
    return "Materialize symbolic block sizes into concrete constants";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<affine::AffineDialect, arith::ArithDialect, func::FuncDialect,
                memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module->getContext());

    SmallVector<ModuleOp, 4> nestedModules;
    for (auto nestedModule : module.getOps<ModuleOp>()) {
      nestedModules.push_back(nestedModule);
    }

    for (auto nestedModule : nestedModules) {
      // Collect functions to determine bindings
      SmallVector<func::FuncOp, 4> funcs;
      for (auto func : nestedModule.getOps<func::FuncOp>()) {
        funcs.push_back(func);
      }

      SmallVector<BlockSizeBinding> bindings;

      if (externalBlockSizes) {
        // --- External path: one binding per function from solver results ---
        // Each nested module contains exactly one func.func whose name is
        // the variant name (e.g., "matmul__d0i0_d1i0__f01__d_d").
        for (auto func : funcs) {
          StringRef funcName = func.getName();
          auto it = externalBlockSizes->find(funcName);
          if (it == externalBlockSizes->end()) {
            // An empty or absent solver result is a valid outcome — it means no
            // feasible block-size binding was found for this variant under the
            // current constraints (UNSAT). Emit a diagnostic to aid debugging
            // and skip emission of this function variant from the output IR.
            func.emitWarning()
                << "No SMT solver result for function '" << funcName
                << "'; skipping materialization for this variant";
            continue;
          }
          BlockSizeBinding b;
          for (auto &entry : it->second)
            b.varValues[entry.first()] = entry.second;
          bindings.push_back(std::move(b));
        }
      } else {
        // --- Fallback path: use hardcoded placeholder solver ---
        SmallVector<StringRef> varNames;
        llvm::SmallPtrSet<StringAttr, 4> seenVars;

        nestedModule->walk([&](loom::SymOp op) {
          SymbolRefAttr ref = op.getSymbolRef();
          StringAttr varNameAttr = ref.getLeafReference();
          if (seenVars.insert(varNameAttr).second) {
            varNames.push_back(varNameAttr.getValue());
          }
        });

        if (varNames.empty())
          continue;

        auto candidates = solveCandidateBlockSizes(varNames.size());
        bindings = buildBindings(varNames, candidates);
      }

      if (bindings.empty()) {
        if (externalBlockSizes) {
          // Variant is UNSAT or has no solver results — remove from output IR
          nestedModule.erase();
        }
        continue;
      }

      // Create a single nested module in the output to contain all variants
      builder.setInsertionPoint(nestedModule);
      auto variantsModule = ModuleOp::create(builder, nestedModule->getLoc());

      // Copy attributes from the original nested module
      if (!nestedModule->getAttrs().empty()) {
        variantsModule->setAttrs(nestedModule->getAttrs());
      }
      variantsModule->setAttr("loom.pass_name",
                              builder.getStringAttr("Materialize"));

      OpBuilder variantsBuilder(variantsModule.getBodyRegion());

      for (const auto &binding : bindings) {
        for (auto func : funcs) {
          IRMapping funcMapping;
          auto clonedFunc =
              cast<func::FuncOp>(variantsBuilder.clone(*func, funcMapping));

          // Rename with suffix
          std::string newName = func.getName().str() + binding.getSuffix();
          clonedFunc.setName(newName);

          // Materialize
          (void)materializeFunction(clonedFunc, binding);
        }
      }

      // Remove the original nested module
      nestedModule.erase();
    }
  }

private:
  // Non-owning pointer to an external block size map (from SMT solver).
  // Null when using the hardcoded placeholder solver.
  const loom::passes::BlockSizeMap *externalBlockSizes = nullptr;
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createMaterializePass() {
  return std::make_unique<MaterializePass>();
}

std::unique_ptr<mlir::Pass> loom::passes::createMaterializePass(
    const loom::passes::BlockSizeMap &blockSizes) {
  return std::make_unique<MaterializePass>(blockSizes);
}
