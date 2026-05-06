// Standalone driver to run the LOOM tensor-canonicalize passes.
//
// Usage:
//   tensor_canonicalize --input <input.mlir>

#include "Passes.h"
#include "driver_utils.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

static llvm::cl::opt<std::string>
    clInput("input", llvm::cl::desc("Path to input MLIR file"),
            llvm::cl::value_desc("filename"), llvm::cl::Required);

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                     "LOOM tensor-canonicalize pass driver\n");

  MLIRContext context;
  loom::driver::registerLoomDialects(context);
  context.loadDialect<mlir::cf::ControlFlowDialect>();

  auto module = loom::driver::parseMLIRFile(clInput, context);
  if (!module) return 1;

  PassManager pm(&context);

  // Preprocessing: fuse elementwise generics to make patterns visible and
  // clean, but skip fusions that would produce a leading-reduction generic
  // (unsupported on target hardware).
  pm.addPass(loom::passes::createLinalgGuardedElementwiseOpFusionPass());
  // pm.addPass(mlir::createLinalgFoldUnitExtentDimsPass());

  // Destination specialization (with guards for post matmul accumulations).
  pm.addPass(loom::passes::createLinalgDestinationSpecializationPass());

  // Eliminate redundant tensor.extract_slice surviving fusion/canonicalization
  pm.addPass(loom::passes::createFoldRedundantExtractSlicePass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // De-CSE: clone and sink all linalg.fill ops to ensure unique SSA chains
  // for initialized tensors, eliminating cross-scope fill sharing.

  pm.addPass(loom::passes::createSinkFillOpsPass());
  pm.addPass(loom::passes::createLoopHandoffProxyCopyInsertionPass());
  pm.addPass(loom::passes::createCanonicalBufferizationToLoomPass());
  // pm.addPass(loom::passes::createHandoffSyncInsertionPass());

  if (failed(pm.run(*module))) {
    llvm::errs() << "LOOM tensor canonicalization pipeline failed\n";
    return 2;
  }

  // Step-2 (memory_binding) intentionally does not register the cf dialect.
  // Strip cf.assert guards here so Step-1 output is cf-free.
  SmallVector<Operation *> assertsToErase;
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "cf.assert")
      assertsToErase.push_back(op);
  });
  for (Operation *op : assertsToErase)
    op->erase();

  loom::driver::printModule(*module);
  return 0;
}
