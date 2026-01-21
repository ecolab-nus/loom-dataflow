/**
 * @file MemoryOpToTTKernel.h
 * @brief Header for memory operation to TT kernel conversion pass.
 */

#ifndef LOOM_PASSES_TILELOOMTOTTKERNEL_MEMORYOPTOTTKERNEL_H
#define LOOM_PASSES_TILELOOMTOTTKERNEL_MEMORYOPTOTTKERNEL_H

#include "FuncOpToTTKernel.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace mlir {
namespace loom {
/**
 * @brief Populate conversion patterns for memory operations to TTKernel.
 * 
 * @details This function adds conversion patterns for memref.copy operations
 *          with {loom.copy.choice...} attributes to the provided pattern set.
 * 
 * @param patterns The pattern set to populate.
 * @param typeConverter The type converter for the conversion pipeline.
 * @param context The MLIR context.
 * @param tracker Shared tracker for compile-arg index assignment.
 */
void populateMemoryOpConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *context, std::shared_ptr<CompileArgTracker> tracker);

} // namespace loom
} // namespace mlir

#endif // LOOM_PASSES_TILELOOMTOTTKERNEL_MEMORYOPTOTTKERNEL_H

