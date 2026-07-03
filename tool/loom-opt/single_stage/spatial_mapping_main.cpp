// Driver for merging ADL hardware declarations into a Triton-shared module.
//
// Usage:
//   spatial_mapping --input <input.mlir> --hw_spec <adl.mlir>

#include "driver_utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

static llvm::cl::opt<std::string>
    clTTSharedInput("input", llvm::cl::desc("Path to input MLIR file"),
                    llvm::cl::value_desc("filename"), llvm::cl::Required);

static llvm::cl::opt<std::string>
    clHwSpecInput("hw_spec",
                  llvm::cl::desc("Path to ADL MLIR file (hardware specification)"),
                  llvm::cl::value_desc("filename"), llvm::cl::Required);

static llvm::cl::opt<bool>
    clFullOccupancy("full_occ",
                    llvm::cl::desc("Use only full hardware occupancy instead "
                                   "of enumerating partial occupancies"),
                    llvm::cl::init(false));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                     "LOOM Triton-shared ADL declaration merger\n");

  MLIRContext context;
  loom::driver::registerLoomAndADLDialects(context);

  // Parse ADL module containing hardware specification.
  auto hwModule = loom::driver::parseMLIRFile(clHwSpecInput, context);
  if (!hwModule) return 1;

  // Parse the input module.
  auto tsModule = loom::driver::parseMLIRFile(clTTSharedInput, context);
  if (!tsModule) return 1;

  // Merge ADL hardware declarations and the input IR into a single module.
  OwningOpRef<ModuleOp> merged = ModuleOp::create(UnknownLoc::get(&context));
  if (!(*tsModule)->getAttrs().empty()) {
    (*merged)->setAttrs((*tsModule)->getAttrs());
  }
  OpBuilder builder(merged->getBodyRegion());

  if (failed(loom::driver::cloneArchSystemDeclarations(*hwModule, builder)))
    return 1;

  IRMapping mapping;
  for (Operation &op : *tsModule->getBody())
    builder.clone(op, mapping);

  // Clean up the generated code with CSE and DCE (Canonicalizer)
  PassManager pm(&context);
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  if (failed(pm.run(*merged))) {
    llvm::errs() << "Cleanup passes failed\n";
    return 1;
  }

  loom::driver::printModule(*merged);
  return 0;
}
