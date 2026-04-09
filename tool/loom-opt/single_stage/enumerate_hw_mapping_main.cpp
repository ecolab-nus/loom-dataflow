// Driver for enumerating Triton-shared grid-to-spatial mappings and printing a
// combined module.
//
// Usage:
//   enumerate_hw_mapping --input <input.mlir> --hw_spec <adl.mlir>

#include "Passes.h"
#include "driver_utils.h"
#include "hardware_info.h"

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

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                     "LOOM Triton-shared spatial explorer\n");

  MLIRContext context;
  loom::driver::registerLoomAndADLDialects(context);

  // Parse ADL module containing hardware specification.
  auto hwModule = loom::driver::parseMLIRFile(clHwSpecInput, context);
  if (!hwModule) return 1;

  // Collect spatial dimensions from ADL ops.
  loom::HardwareInfo hardwareInfo;
  if (failed(loom::GetHardwareInfoForExploration(*hwModule, hardwareInfo))) {
    llvm::errs() << "Failed to collect hardware information from ADL module\n";
    return 1;
  }

  // Parse the input module to explore.
  auto tsModule = loom::driver::parseMLIRFile(clTTSharedInput, context);
  if (!tsModule) return 1;

  // Enumerate all grid-to-spatial assignments for the input module.
  OwningOpRef<ModuleOp> out =
      loom::EnumerateSpatialMappings(*tsModule, hardwareInfo);

  // Merge ADL hardware declarations and generated clones into a single module.
  OwningOpRef<ModuleOp> merged = ModuleOp::create(UnknownLoc::get(&context));
  if (!(*out)->getAttrs().empty()) {
    (*merged)->setAttrs((*out)->getAttrs());
  }
  OpBuilder builder(merged->getBodyRegion());
  IRMapping mapping;

  ModuleOp systemModule = loom::driver::findArchSystemModule(*hwModule);
  if (!systemModule) {
    llvm::errs() << "Could not find module @arch_system in hw_spec file\n";
    return 1;
  }
  for (Operation &op : *systemModule.getBody()) {
    // Skip child modules (e.g., @matrix_lane, @vector_lane, @data_movers)
    if (isa<ModuleOp>(&op))
      continue;
    // Skip the module terminator
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    builder.clone(op, mapping);
  }

  // Then, insert all the nested modules containing function variants
  for (Operation &op : *out->getBody())
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
