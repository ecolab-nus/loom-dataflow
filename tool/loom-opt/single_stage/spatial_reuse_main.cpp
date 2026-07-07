// Standalone driver to run the LOOM spatial reuse pass.
//
// Usage:
//   spatial_reuse --input <input.mlir>

#include "Passes.h"
#include "driver_utils.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

static llvm::cl::opt<std::string>
    clInput("input",
            llvm::cl::desc("Path to input MLIR file (use '-' for stdin)"),
            llvm::cl::value_desc("filename"), llvm::cl::init("-"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "LOOM spatial reuse annotation\n");

  MLIRContext context;
  loom::driver::registerLoomAndADLDialects(context);

  auto module = loom::driver::parseMLIRFile(clInput, context);
  if (!module)
    return 1;

  PassManager pm(&context);
  pm.addPass(loom::passes::createSpatialReusePass());
  if (failed(pm.run(*module))) {
    llvm::errs() << "LOOM spatial reuse pass failed\n";
    return 2;
  }

  loom::driver::printModule(*module);
  return 0;
}
