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

/// Placeholder for a future solver. Returns hardcoded candidate tuples.
/// Each tuple has one value per symbolic variable, in definition order.
SmallVector<SmallVector<int64_t>> solveCandidateBlockSizes(unsigned numVars) {
  // Future: call the constraint solver here.
  // For now, return one hardcoded tuple if we have 3 variables.
  if (numVars == 3) {
    return {{1, 64, 128}};
  }
  // Fallback: all-64 tuple
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

  func->walk([&](loom::GetSymbolicBlockSizeOp op) {
    SymbolRefAttr ref = op.getSymbolRef();
    StringRef varName;
    if (ref.getNestedReferences().size() > 0) {
      varName = ref.getNestedReferences().back().getLeafReference();
    } else {
      varName = ref.getLeafReference();
    }

    auto valueOpt = binding.getValue(varName);
    if (!valueOpt) {
      op.emitWarning() << "No concrete value found for symbolic variable '"
                       << varName << "'";
      return;
    }

    builder.setInsertionPoint(op);
    auto constant =
        builder.create<arith::ConstantIndexOp>(op.getLoc(), *valueOpt);
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
      // Collect all symbolic variable names from get_symbolic_block_size ops
      SmallVector<StringRef> varNames;
      llvm::SmallPtrSet<StringAttr, 4> seenVars;

      nestedModule->walk([&](loom::GetSymbolicBlockSizeOp op) {
        SymbolRefAttr ref = op.getSymbolRef();
        StringAttr varNameAttr;
        if (ref.getNestedReferences().size() > 0) {
          varNameAttr = ref.getNestedReferences().back().getLeafReference();
        } else {
          varNameAttr = ref.getLeafReference();
        }

        if (seenVars.insert(varNameAttr).second) {
          varNames.push_back(varNameAttr.getValue());
        }
      });

      if (varNames.empty())
        continue;

      auto candidates = solveCandidateBlockSizes(varNames.size());
      auto bindings = buildBindings(varNames, candidates);

      // Collect functions to clone
      SmallVector<func::FuncOp, 4> funcs;
      for (auto func : nestedModule.getOps<func::FuncOp>()) {
        funcs.push_back(func);
      }

      // Create a single nested module in the output to contain all variants
      builder.setInsertionPoint(nestedModule);
      auto variantsModule = builder.create<ModuleOp>(nestedModule->getLoc());

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
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createMaterializePass() {
  return std::make_unique<MaterializePass>();
}
