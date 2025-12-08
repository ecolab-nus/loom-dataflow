#include "hoist_block_loading.h"
#include "block_loading_pattern.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Builders.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include "llvm/ADT/SmallVector.h"
#include <cstddef>


namespace loom {
namespace passes {

class HoistBlockLoadingPass : public mlir::PassWrapper<HoistBlockLoadingPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    /**
     * @brief Hoist block loading.
     *
     * @details Hoist block loading is a pass that hoists block loading operations to the top of the function.
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
    /// Traverses all functions in the module and creates explore variants with hoisted blocks.
    void runOnOperation() override {
        mlir::ModuleOp module = getOperation();
        mlir::OpBuilder moduleBuilder(module.getBodyRegion());
        
        // Collect all functions first to avoid iterator invalidation
        llvm::SmallVector<mlir::func::FuncOp> funcs(module.getOps<mlir::func::FuncOp>().begin(),
                                                     module.getOps<mlir::func::FuncOp>().end());
        
        // Process each function and insert clones immediately after it
        for (mlir::func::FuncOp originalFunc : funcs) {
            // Find the innermost scf.for loop
            mlir::scf::ForOp innermostForOp = findInnermostForOp(originalFunc);
            
            if (!innermostForOp) {
                // No innermost loop found, skip this function
                continue;
            }
            
            // Build loading blocks to determine how many variants we need
            llvm::SmallVector<loom::affine::LoadingBlock, 2> loading_blocks;
            if (failed(loom::affine::BuildLoadingBlocks(innermostForOp, loading_blocks))) {
                continue;
            }
            
            // Set insertion point right after the original function
            moduleBuilder.setInsertionPointAfter(originalFunc);
            
            // Create clones for each block index and insert them immediately after the original function
            for (size_t block_idx = 0; block_idx < loading_blocks.size(); ++block_idx) {
                // Clone the function
                mlir::IRMapping map;
                mlir::func::FuncOp clonedFunc = 
                    mlir::cast<mlir::func::FuncOp>(moduleBuilder.clone(*originalFunc, map));
                
                // Rename the cloned function
                std::string newName = originalFunc.getSymName().str() + "__hoist_block_" + 
                                     std::to_string(block_idx);
                clonedFunc.setName(newName);
                
                // Find the innermost loop in the cloned function and hoist the block
                mlir::scf::ForOp clonedInnermostForOp = findInnermostForOp(clonedFunc);
                bool isValid = clonedInnermostForOp && 
                              succeeded(loom::affine::HoistSingleBlock(clonedInnermostForOp, block_idx));
                
                if (!isValid) {
                    // Block is invalid (never used CreateHoistedOpsSimple), erase the cloned function
                    clonedFunc.erase();
                    moduleBuilder.setInsertionPointAfter(originalFunc);
                } else {
                    // Block is valid, keep the cloned function
                    moduleBuilder.setInsertionPointAfter(clonedFunc);
                }
            }
        }
    }

};

} // namespace passes
} // namespace loom


/// @brief Create the hoist block loading pass.
/// @return A unique pointer to the created pass.
std::unique_ptr<mlir::Pass>
loom::passes::createHoistBlockLoadingPass() {
  return std::make_unique<HoistBlockLoadingPass>();
}

/// @brief Register the hoist block loading pass with MLIR.
void loom::passes::registerHoistBlockLoadingPass() {
  mlir::PassRegistration<HoistBlockLoadingPass>();
}
