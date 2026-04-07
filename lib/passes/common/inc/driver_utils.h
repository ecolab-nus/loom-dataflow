#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"

namespace loom::driver {

/// Standard set of dialects for the loom pipeline (excluding ADL).
/// Registers: Builtin, Func, Arith, Affine, Tensor, Linalg, MemRef, SCF,
///            Bufferization, Loom.
void registerLoomDialects(mlir::MLIRContext &context);

/// Standard set of dialects including ADL (Steps 3-6).
void registerLoomAndADLDialects(mlir::MLIRContext &context);

/// Parse a single MLIR file into a ModuleOp.
/// Prints error and returns nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp>
parseMLIRFile(llvm::StringRef path, mlir::MLIRContext &context);

/// Print a ModuleOp to stdout with local scope.
void printModule(mlir::ModuleOp module);

/// Find the @arch_system module within a hardware specification module.
/// Can be the top-level module itself or a nested child.
mlir::ModuleOp findArchSystemModule(mlir::ModuleOp hwModule);

} // namespace loom::driver
