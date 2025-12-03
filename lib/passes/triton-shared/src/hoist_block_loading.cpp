#include "hoist_block_loading.h"
#include "block_loading_pattern.h"
#include "mlir/IR/BuiltinOps.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>


namespace tmd {
namespace passes {

class HoistBlockLoadingPass : public mlir::PassWrapper<HoistBlockLoadingPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    /**
     * @brief Hoist block loading.
     *
     * @details Hoist block loading is a pass that hoists block loading operations to the top of the function.
     */
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistBlockLoadingPass)

    /// @brief Run the hoist block loading pass on the module.
    /// Traverses all functions in the module and hoists loading blocks from innermost loops.
    void runOnOperation() override {
        mlir::ModuleOp module = getOperation();
        
        // Traverse each function in the module
        module.walk([&](mlir::func::FuncOp funcOp) {
            // Find the innermost scf.for loop
            mlir::scf::ForOp innermostForOp = nullptr;
            
            funcOp.walk([&](mlir::scf::ForOp forOp) {
                // Check if this forOp is the innermost (no nested forOp in its body)
                bool hasNestedForOp = false;
                forOp.getBody()->walk([&](mlir::Operation *op) {
                    // Check if there are nested scf.for or affine.for loops
                    if (tmd::affine::isInWhitelist(op) && op != forOp.getOperation()) {
                        hasNestedForOp = true;
                        return mlir::WalkResult::interrupt();
                    }
                    return mlir::WalkResult::advance();
                });
                
                if (!hasNestedForOp) {
                    innermostForOp = forOp;
                    return mlir::WalkResult::interrupt(); // Found innermost, stop traversal
                }
                return mlir::WalkResult::advance();
            });
            
            if (innermostForOp) {
                (void)tmd::affine::MatchAndHoist(innermostForOp);
            }
        });
    }

};

} // namespace passes
} // namespace tmd


/// @brief Create the hoist block loading pass.
/// @return A unique pointer to the created pass.
std::unique_ptr<mlir::Pass>
tmd::passes::createHoistBlockLoadingPass() {
  return std::make_unique<HoistBlockLoadingPass>();
}

/// @brief Register the hoist block loading pass with MLIR.
void tmd::passes::registerHoistBlockLoadingPass() {
  mlir::PassRegistration<HoistBlockLoadingPass>();
}
