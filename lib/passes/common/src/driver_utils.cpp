#include "driver_utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

// Dialect headers
#include "ADLDialect.h.inc"
#include "LoomDialect.h.inc"

using namespace mlir;

namespace loom::driver {

void registerLoomDialects(MLIRContext &context) {
  DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                  mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect, mlir::tensor::TensorDialect,
                  mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                  mlir::bufferization::BufferizationDialect,
                  loom::LoomDialect>();

  // Register external models for the tensor dialect interfaces.
  mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);

  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

void registerLoomAndADLDialects(MLIRContext &context) {
  registerLoomDialects(context);
  context.loadDialect<adl::ADLDialect>();
}

OwningOpRef<ModuleOp> parseMLIRFile(llvm::StringRef path, MLIRContext &context) {
  llvm::SourceMgr sm;
  auto file = mlir::openInputFile(path);
  if (!file) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to open input file: " << path << "\n";
    return nullptr;
  }
  sm.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sm, &context);
  if (!module) {
    llvm::WithColor::error(llvm::errs()) << "Failed to parse MLIR module\n";
    return nullptr;
  }
  return module;
}

void printModule(ModuleOp module) {
  mlir::OpPrintingFlags flags;
  flags.useLocalScope();
  module.print(llvm::outs(), flags);
  llvm::outs() << "\n";
}

ModuleOp findArchSystemModule(ModuleOp hwModule) {
  if (hwModule.getSymName() && *hwModule.getSymName() == "arch_system") {
    return hwModule;
  }
  for (Operation &op : *hwModule.getBody()) {
    if (auto mod = dyn_cast<ModuleOp>(&op)) {
      if (mod.getSymName() && *mod.getSymName() == "arch_system") {
        return mod;
      }
    }
  }
  return nullptr;
}

LogicalResult cloneArchSystemDeclarations(ModuleOp hwModule,
                                          OpBuilder &builder) {
  ModuleOp systemModule = findArchSystemModule(hwModule);
  if (!systemModule) {
    llvm::errs() << "Could not find module @arch_system in hw_spec file\n";
    return failure();
  }

  IRMapping mapping;
  for (Operation &op : *systemModule.getBody()) {
    if (isa<ModuleOp>(&op))
      continue;
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    builder.clone(op, mapping);
  }

  return success();
}

} // namespace loom::driver
