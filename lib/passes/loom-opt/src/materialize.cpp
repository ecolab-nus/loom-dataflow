/**
 * @file materialize.cpp
 * @brief Implementation of materialize pass for canonicalizing IR.
 * @details
 * This pass materializes all loom.get_module_attribute operations by replacing
 * them with arith.constant operations. For each loom.get_module_attribute operation,
 * it reads the attribute name from the operation, looks up the corresponding value
 * in the module's attributes, and creates an arith.constant with that value.
 */

#include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Casting.h"

// Include Loom dialect headers for GetBlockSizeOp
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace {

class MaterializePass
    : public PassWrapper<MaterializePass, OperationPass<ModuleOp>> {
public:
  /**
   * @brief Materialize and canonicalize IR operations.
   *
   * @details This pass materializes all loom.get_module_attribute operations
   * by replacing them with arith.constant operations. This simplifies the IR
   * by converting symbolic attribute references into concrete constant values.
   */
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializePass)

  /// Command-line flag name.
  StringRef getArgument() const override {
    return "loom-materialize";
  }

  /// Short pass description for diagnostics and help.
  StringRef getDescription() const override {
    return "Materialize and canonicalize IR operations";
  }

  /// Declare dialect dependencies used by this pass implementation.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  /**
   * @brief Execute the pass over the module.
   *
   * @details This pass materializes all loom.get_module_attribute operations
   * by replacing them with arith.constant operations. For each
   * loom.get_module_attribute operation, it reads the attribute name from the
   * operation, looks up the corresponding value in the module's attributes, and
   * creates an arith.constant with that value. After all operations are
   * replaced, the corresponding module attributes are removed since they are
   * no longer needed.
   */
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *, 16> opsToErase;
    llvm::DenseSet<StringRef> usedAttrNames;

    // Walk through all operations in the module (including functions)
    module.walk([&](loom::GetBlockSizeOp getAttrOp) {
      // Get the attribute name from the operation
      StringRef attrName = getAttrOp.getAttr();

      // Find the nearest parent ModuleOp (could be nested module containing the func)
      ModuleOp parentModule = getAttrOp->getParentOfType<ModuleOp>();
      if (!parentModule) {
        getAttrOp->emitWarning() << "No parent module found for attribute '" 
                                  << attrName << "', skipping materialization";
        return;
      }

      // Look up the attribute in the nearest parent module
      Attribute moduleAttr = parentModule->getAttr(attrName);
      if (!moduleAttr) {
        // If attribute not found, skip this operation
        getAttrOp->emitWarning() << "Module attribute '" << attrName
                                  << "' not found in parent module, skipping materialization";
        return;
      }

      // Verify that the attribute is an IntegerAttr with index type
      auto intAttr = llvm::dyn_cast<IntegerAttr>(moduleAttr);
      if (!intAttr || !intAttr.getType().isIndex()) {
        getAttrOp->emitWarning() << "Module attribute '" << attrName
                                  << "' is not an index integer attribute, skipping";
        return;
      }

      // Create arith.constant operation to replace the get_module_attribute
      OpBuilder builder(getAttrOp);
      auto constant = builder.create<arith::ConstantIndexOp>(
          getAttrOp->getLoc(), intAttr.getInt());

      // Replace all uses of the original operation with the constant
      getAttrOp.getResult().replaceAllUsesWith(constant.getResult());

      // Mark the original operation for deletion
      opsToErase.push_back(getAttrOp);

      // Record the attribute name for later removal from the parent module
      // We store a pair of (module, attrName) to handle nested modules
      usedAttrNames.insert(attrName);
    });

    // Erase all replaced operations
    for (Operation *op : opsToErase) {
      op->erase();
    }

    // Remove used attributes from all nested modules
    // Walk all nested modules and remove the used attributes
    module.walk([&](ModuleOp nestedModule) {
      for (StringRef attrName : usedAttrNames) {
        if (nestedModule->getAttr(attrName)) {
          nestedModule->removeAttr(attrName);
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createMaterializePass() {
  return std::make_unique<MaterializePass>();
}



