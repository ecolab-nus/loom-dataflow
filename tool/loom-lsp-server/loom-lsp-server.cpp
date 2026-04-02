#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

// The .h.inc files are generated into the build directory and 
// contain the classes in the namespaces defined in TableGen.
#include "ADLDialect.h.inc"
#include "LoomDialect.h.inc"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  
  // Register all standard MLIR dialects.
  registerAllDialects(registry);
  
  // Register custom dialects for LOOM.
  registry.insert<adl::ADLDialect>();
  registry.insert<loom::LoomDialect>();

  return failed(MlirLspServerMain(argc, argv, registry));
}
