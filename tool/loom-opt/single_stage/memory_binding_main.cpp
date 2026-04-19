// Standalone driver to run the LOOM memory-binding pass.
//
// Usage:
//   memory_binding --input <input.mlir>

#include "Passes.h"
#include "driver_utils.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

static llvm::cl::opt<std::string>
    clInput("input", llvm::cl::desc("Path to input MLIR file"),
            llvm::cl::value_desc("filename"), llvm::cl::Required);

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                     "LOOM memory-binding pass driver\n");

  MLIRContext context;
  // Step 1 (tensor_canonicalize) is guaranteed to eliminate all ControlFlow
  // dialect ops (e.g., cf.br, cf.cond_br) from the IR. Thus, we don't need
  // to load or register the ControlFlow dialect here.
  loom::driver::registerLoomDialects(context);

  auto module = loom::driver::parseMLIRFile(clInput, context);
  if (!module) return 1;

  PassManager pm(&context);
  pm.addPass(loom::passes::createMemoryBindingPass());
  pm.addPass(loom::passes::createGatherSyncInsertionPass());
  if (failed(pm.run(*module))) {
    llvm::errs() << "LOOM memory-binding pass failed\n";
    return 2;
  }

  loom::driver::printModule(*module);
  return 0;
}
