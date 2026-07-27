#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "ADL/IR/ADLDialect.h"
#include "LoomDialect.h.inc"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  registry.insert<mlir::adl::ADLDialect, loom::LoomDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "LOOM modular optimizer driver\n",
                        registry));
}
