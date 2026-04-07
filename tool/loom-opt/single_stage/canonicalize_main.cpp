// Standalone driver to run canonicalization on LOOM MLIR.
//
// Usage:
//   canonicalize --input <input.mlir>

#include "Passes.h"
#include "driver_utils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

static llvm::cl::opt<std::string>
    clInput("input", llvm::cl::desc("Path to input MLIR file"),
            llvm::cl::value_desc("filename"), llvm::cl::init("-"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                     "LOOM canonicalization\n");

  MLIRContext context;
  loom::driver::registerLoomAndADLDialects(context);

  auto module = loom::driver::parseMLIRFile(clInput, context);
  if (!module) return 1;

  PassManager pm(&context);
  pm.addPass(mlir::createCanonicalizerPass());
  if (failed(pm.run(*module))) {
    llvm::errs() << "Canonicalization failed\n";
    return 2;
  }

  loom::driver::printModule(*module);
  return 0;
}
