/**
 * @file tileloom_to_ttkernel_opt.cpp
 * @brief mlir-opt wrapper that registers TileLoom to TTKernel pass.
 * 
 * Usage:
 *   tileloom_to_ttkernel_opt --loom-tileloom-to-ttkernel input.mlir
 *   or simply:
 *   tileloom_to_ttkernel_opt input.mlir
 */

#include "TileLoomToTTKernel.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"

// ADL dialect for hardware/resource descriptions.
#include "ADLDialect.h.inc"
// Loom dialect (loom.*) for loom.alloc, loom.copy, etc.
#include "LoomDialect.h.inc"
// TTKernel dialect from tt-mlir (types like DataFormatType, ops, etc.).
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"

using namespace mlir;
using namespace mlir::loom;

int main(int argc, char **argv) {
  // Register the pass before calling mlir-opt's main
  registerTileLoomToTTKernelPass();

  // Register all MLIR dialects and passes
  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllPasses();
  // Register ADL dialect for hardware/resource descriptions.
  registry.insert<adl::ADLDialect>();
  // Register Loom dialect so loom.alloc, loom.copy, etc. are available.
  registry.insert<::loom::LoomDialect>();
  // Register TTKernel dialect so its types/ops are available.
  registry.insert<mlir::tt::ttkernel::TTKernelDialect>();
  // Register EmitC dialect for verbatim operations in host functions.
  registry.insert<mlir::emitc::EmitCDialect>();

  // Call mlir-opt's main function
  return failed(mlir::MlirOptMain(
      argc, argv, "TileLoom to TTKernel optimizer driver\n", registry));
}
