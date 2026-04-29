#include "Passes.h"
#include "block_loading_pattern.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "utils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include <cstddef>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

// Include Loom dialect headers
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
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

  /// @brief Find all innermost affine.for loops in a function.
  /// @param funcOp The function to search in.
  /// @return A vector of innermost affine.for loops.
  llvm::SmallVector<mlir::affine::AffineForOp>
  findInnermostAffineForOps(mlir::func::FuncOp funcOp) {
    llvm::SmallVector<mlir::affine::AffineForOp> innermostOps;

    funcOp.walk([&](mlir::affine::AffineForOp forOp) {
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
        innermostOps.push_back(forOp);
      }
      return mlir::WalkResult::advance();
    });

    return innermostOps;
  }

  /// @brief Run the hoist block loading pass on the module.
  /// Traverses all functions in the module and creates explore variants with
  /// hoisted blocks.
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder moduleBuilder(module.getBodyRegion());

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

      // Find all innermost loops
      auto innermostOps = findInnermostAffineForOps(originalFunc);

      if (innermostOps.empty()) {
        // No innermost loop found, skip this function
        continue;
      }

      // We need to collect ALL loading blocks from ALL innermost loops
      // for this function variant generation logic.
      // But typically we generate one variant per hoisted block.

      struct BlockInstance {
        mlir::affine::AffineForOp loop;
        size_t index;
      };
      llvm::SmallVector<BlockInstance> all_loading_blocks;

      for (auto loop : innermostOps) {
        llvm::SmallVector<loom::affine::LoadingBlock, 2> loop_blocks;
        if (succeeded(loom::affine::BuildLoadingBlocks(loop, loop_blocks))) {
          for (size_t i = 0; i < loop_blocks.size(); ++i) {
            all_loading_blocks.push_back({loop, i});
          }
        }
      }

      if (all_loading_blocks.empty())
        continue;

      // Track insertion point for consecutive clones (at module level)
      mlir::Operation *insertAfter = parentModule;

      // Create clones for each block instance
      for (size_t block_instance_idx = 0;
           block_instance_idx < all_loading_blocks.size();
           ++block_instance_idx) {

        auto instance = all_loading_blocks[block_instance_idx];
        std::string newName = originalFunc.getSymName().str() +
                              "__hoist_block_" +
                              std::to_string(block_instance_idx);

        // Clone the function and apply hoisting modification with module
        // wrapper
        mlir::func::FuncOp clonedFunc = loom::utils::cloneFunc(
            moduleBuilder, originalFunc, newName, moduleAttrs,
            [&](mlir::func::FuncOp func) -> mlir::LogicalResult {
              // We need to find the EQUIVALENT loop in the cloned function.
              // We can use the index of the loop in the original list.
              // Wait, indices might change if we modify the function.
              // Better: find all innermost loops again and use the same index.

              auto clonedInnermostOps = findInnermostAffineForOps(func);

              // Find which loop corresponds to the one we want to hoist.
              // We can find it by matching the location or just using the same
              // index in the walk. Since clone preserves order, the n-th
              // innermost loop should correspond.

              // Let's find the index of 'instance.loop' in 'innermostOps'
              size_t loop_idx = 0;
              for (; loop_idx < innermostOps.size(); ++loop_idx) {
                if (innermostOps[loop_idx] == instance.loop)
                  break;
              }

              if (loop_idx >= clonedInnermostOps.size()) {
                return mlir::failure();
              }

              mlir::affine::AffineForOp targetLoop =
                  clonedInnermostOps[loop_idx];

              // Attempt to hoist the block
              if (failed(loom::affine::HoistSingleBlock(targetLoop,
                                                        instance.index))) {
                return mlir::failure();
              }

              return mlir::success();
            },
            insertAfter);

        // Update insertion point for next clone
        if (clonedFunc) {
          mlir::ModuleOp clonedParentModule =
              loom::utils::getParentModule(clonedFunc);
          if (clonedParentModule) {
            insertAfter = clonedParentModule;
          }
        }
      }
    }
  }
};

std::unique_ptr<mlir::Pass> createHoistBlockLoadingPass() {
  return std::make_unique<HoistBlockLoadingPass>();
}

} // namespace passes
} // namespace loom
