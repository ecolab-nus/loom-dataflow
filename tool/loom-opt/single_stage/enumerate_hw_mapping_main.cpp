// Driver for enumerating Triton-shared grid-to-spatial mappings and printing a
// combined module.
//
// Usage:
//   enumerate_hw_mapping --input <input.mlir> --hw_spec <adl.mlir>
//   [--grid-dims N]
//
// The driver loads both modules, collects spatial dimensions from the ADL
// module (via adl.arch.scale), enumerates all unique assignments of grid
// dimensions to the hardware spatial dimensions, and prints a new module
// containing the ADL hardware description at the top followed by one clone of
// each function per mapping with attributes encoding the binding.

#include "Passes.h"
#include "hardware_info.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"

#include "ADLDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "ADLTypes.h.inc"
#define GET_OP_CLASSES
#include "ADLOps.h.inc"
#include "LoomDialect.h.inc"

using namespace mlir;

static llvm::cl::opt<std::string>
    clTTSharedInput("input", llvm::cl::desc("Path to input MLIR file"),
                    llvm::cl::value_desc("filename"), llvm::cl::Required);

static llvm::cl::opt<std::string>
    clHwSpecInput("hw_spec",
                  llvm::cl::desc("Path to ADL MLIR file (hardware specification)"),
                  llvm::cl::value_desc("filename"), llvm::cl::Required);

static llvm::cl::opt<unsigned> clNumGridDims(
    "grid-dims", llvm::cl::desc("Number of grid dimensions to consider (1..3)"),
    llvm::cl::init(3));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "LOOM Triton-shared spatial explorer\n");

  // Legacy option retained for compatibility of CLI; ignored in the new format.
  (void)clNumGridDims;

  mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                  mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                  mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                  mlir::bufferization::BufferizationDialect,
                  adl::ADLDialect, loom::LoomDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Parse ADL module containing hardware specification.
  llvm::SourceMgr hwSm;
  auto hwFile = mlir::openInputFile(clHwSpecInput);
  if (!hwFile) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to open hw_spec input file: " << clHwSpecInput << "\n";
    return 1;
  }
  hwSm.AddNewSourceBuffer(std::move(hwFile), llvm::SMLoc());
  OwningOpRef<ModuleOp> hwModule = parseSourceFile<ModuleOp>(hwSm, &context);
  if (!hwModule) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to parse ADL MLIR module\n";
    return 1;
  }

  // Collect spatial dimensions from ADL ops.
  loom::HardwareInfo hardwareInfo;
  if (failed(loom::GetHardwareInfoForExploration(*hwModule, hardwareInfo))) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to collect hardware information from ADL module\n";
    return 1;
  }

  // Parse the input module to explore.
  llvm::SourceMgr tsSm;
  auto tsFile = mlir::openInputFile(clTTSharedInput);
  if (!tsFile) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to open input file: " << clTTSharedInput << "\n";
    return 1;
  }
  tsSm.AddNewSourceBuffer(std::move(tsFile), llvm::SMLoc());
  OwningOpRef<ModuleOp> tsModule = parseSourceFile<ModuleOp>(tsSm, &context);
  if (!tsModule) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to parse input MLIR module\n";
    return 1;
  }

  // Enumerate all grid-to-spatial assignments for the input module.
  // Enumerate mappings directly over the outermost affine.parallel.
  // We ignore numGridDims and rely purely on the number of iterators in the
  // parallel op. Also explore outer-for loop orderings.
  OwningOpRef<ModuleOp> out =
      loom::EnumerateSpatialMappings(*tsModule, hardwareInfo);

  // Merge ADL hardware declarations and generated clones into a single module.
  // Structure: outer module -> ADL ops at top -> nested modules (each containing
  // a func variant)
  OwningOpRef<ModuleOp> merged = ModuleOp::create(UnknownLoc::get(&context));
  if (!(*out)->getAttrs().empty()) {
    (*merged)->setAttrs((*out)->getAttrs());
  }
  OpBuilder builder(merged->getBodyRegion());
  IRMapping mapping;

  // First, insert ADL hardware declarations at the top of the outer module.
  // The ADL ops live directly inside `module @system` (not inside child modules
  // like @matrix_lane, @vector_lane, @data_movers). We clone only non-module
  // ops from the @system module body.
  //
  // The parsed hw_spec file may be `module @system { ... }` at the top level,
  // or it may contain `module @system` as a nested child. Handle both cases.
  ModuleOp systemModule = nullptr;
  if (hwModule->getSymName() && *hwModule->getSymName() == "system") {
    systemModule = *hwModule;
  } else {
    for (Operation &op : *hwModule->getBody()) {
      if (auto mod = dyn_cast<ModuleOp>(&op)) {
        if (mod.getSymName() && *mod.getSymName() == "system") {
          systemModule = mod;
          break;
        }
      }
    }
  }
  if (!systemModule) {
    llvm::WithColor::error(llvm::errs())
        << "Could not find module @system in hw_spec file\n";
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

  mlir::OpPrintingFlags flags;
  flags.useLocalScope();
  merged->print(llvm::outs(), flags);
  llvm::outs() << "\n";
  return 0;
}
