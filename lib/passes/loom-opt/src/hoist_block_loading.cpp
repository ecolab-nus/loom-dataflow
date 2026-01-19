#include "Passes.h"
#include "block_loading_pattern.h"
#include "constraint_space_utils.h"
#include "hardware_info.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "utils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include <cstddef>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

// Include Loom dialect headers for CopyOp
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#define DEBUG_TYPE "hoist-block-loading"

namespace loom {
namespace passes {

class HoistBlockLoadingPass
    : public mlir::PassWrapper<HoistBlockLoadingPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  /**
   * @brief Hoist block loading.
   *
   * @details Hoist block loading is a pass that hoists block loading operations
   * to the top of the function.
   */
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistBlockLoadingPass)

  /// @brief Find the innermost scf.for loop in a function.
  /// @param funcOp The function to search in.
  /// @return The innermost scf.for loop, or nullptr if not found.
  mlir::scf::ForOp findInnermostForOp(mlir::func::FuncOp funcOp) {
    mlir::scf::ForOp innermostForOp = nullptr;

    funcOp.walk([&](mlir::scf::ForOp forOp) {
      // Check if this forOp is the innermost (no nested loops in its body)
      bool hasNestedLoop = false;
      forOp.getBody()->walk([&](mlir::Operation *op) {
        if (loom::affine::isInWhitelist(op) && op != forOp.getOperation()) {
          hasNestedLoop = true;
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });

      if (!hasNestedLoop) {
        innermostForOp = forOp;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });

    return innermostForOp;
  }

  /// @brief Run the hoist block loading pass on the module.
  /// Traverses all functions in the module and creates explore variants with
  /// hoisted blocks.
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder moduleBuilder(module.getBodyRegion());

    // Collect hardware info to get L1 size
    loom::HardwareInfo hardwareInfo;
    if (failed(loom::GetHardwareInfoForExploration(module, hardwareInfo))) {
      hardwareInfo.l1Size = 0;
    }

    // Collect all functions first to avoid iterator invalidation
    llvm::SmallVector<mlir::func::FuncOp> funcs =
        loom::utils::collectFunctions(module);

    // Process each function and insert clones immediately after it
    for (mlir::func::FuncOp originalFunc : funcs) {
      // Get the attributes from the parent module of the function
      mlir::ModuleOp parentModule = loom::utils::getParentModule(originalFunc);
      mlir::DictionaryAttr moduleAttrs = nullptr;
      if (parentModule) {
        moduleAttrs = parentModule->getAttrDictionary();
      }
      // Find the innermost scf.for loop
      mlir::scf::ForOp innermostForOp = findInnermostForOp(originalFunc);

      if (!innermostForOp) {
        // No innermost loop found, skip this function
        continue;
      }

      // Build loading blocks to determine how many variants we need
      llvm::SmallVector<loom::affine::LoadingBlock, 2> loading_blocks;
      if (failed(loom::affine::BuildLoadingBlocks(innermostForOp,
                                                  loading_blocks))) {
        continue;
      }

      // Track insertion point for consecutive clones (at module level)
      mlir::Operation *insertAfter = parentModule;

      // Create clones for each block index and insert them immediately after
      // the previous one
      for (size_t block_idx = 0; block_idx < loading_blocks.size();
           ++block_idx) {
        std::string newName = originalFunc.getSymName().str() +
                              "__hoist_block_" + std::to_string(block_idx);

        // Clone the function and apply hoisting modification with module
        // wrapper
        mlir::func::FuncOp clonedFunc = loom::utils::cloneFuncWithConstraints(
            moduleBuilder, originalFunc, newName, moduleAttrs,
            "HoistBlockLoading",
            [&](mlir::func::FuncOp func,
                loom::ConstraintSpaceOp csOp) -> mlir::LogicalResult {
              // Find the innermost loop in the cloned function and hoist the
              // block
              mlir::scf::ForOp clonedInnermostForOp = findInnermostForOp(func);
              if (!clonedInnermostForOp) {
                return mlir::failure();
              }

              // Attempt to hoist the block
              if (failed(loom::affine::HoistSingleBlock(clonedInnermostForOp,
                                                        block_idx))) {
                return mlir::failure();
              }

              // Add L1 cache constraints if applicable
              if (csOp && hardwareInfo.l1Size > 0) {
                auto allocInfos = loom::utils::collectL1AllocInfos(func);
                if (!allocInfos.empty()) {
                  llvm::SmallVector<llvm::StringRef> symVarNames;
                  llvm::StringMap<unsigned> symVarToIndex;

                  // Build unique variable list
                  for (const auto &info : allocInfos) {
                    for (llvm::StringRef dimName : info.dims) {
                      if (symVarToIndex.find(dimName) == symVarToIndex.end()) {
                        symVarToIndex[dimName] = symVarNames.size();
                        symVarNames.push_back(dimName);
                      }
                    }
                  }

                  // Build monomials
                  llvm::SmallVector<loom::lcs::Monomial> monomials;
                  for (const auto &info : allocInfos) {
                    loom::lcs::Monomial m;
                    for (llvm::StringRef dimName : info.dims) {
                      m.varIndices.push_back(symVarToIndex[dimName]);
                    }
                    m.coeff = info.elemSize;
                    monomials.push_back(m);
                  }

                  // Add the polynomial constraint to csOp
                  loom::lcs::addPolynomialConstraint(
                      csOp, symVarNames, monomials, hardwareInfo.l1Size);
                }
              }

              return mlir::success();
            },
            insertAfter);

        // Update insertion point for next clone (should point to the wrapper
        // module)
        if (clonedFunc) {
          mlir::ModuleOp clonedParentModule =
              loom::utils::getParentModule(clonedFunc);
          if (clonedParentModule) {
            insertAfter = clonedParentModule;
          }
        }
        // If clonedFunc is nullptr, keep insertAfter as-is for next attempt
      }
    }
  }
};

std::unique_ptr<mlir::Pass> createHoistBlockLoadingPass() {
  return std::make_unique<HoistBlockLoadingPass>();
}

} // namespace passes
} // namespace loom

/// @brief Register the hoist block loading pass with MLIR.
