// Standalone driver that annotates `memref.reinterpret_cast` ops with reuse
// information relative to surrounding affine/scf iterators.
//
// Usage:
//   analyze_reuse --input <input.mlir>

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
                                    "LOOM Triton-shared reuse annotation\n");

  MLIRContext context;
  loom::driver::registerLoomAndADLDialects(context);

  auto module = loom::driver::parseMLIRFile(clInput, context);
  if (!module) return 1;

  PassManager pm(&context);
  pm.addPass(loom::passes::createAnnotateSubviewReusePass());
  if (failed(pm.run(*module))) {
    llvm::errs() << "Reuse annotation pass failed\n";
    return 2;
  }

  loom::driver::printModule(*module);
  return 0;
}
