/**
 * @file MemoryOpToTTKernel.h
 * @brief Header for memory operation to TT kernel conversion pass.
 */

#ifndef LOOM_PASSES_TILELOOMTOTTKERNEL_MEMORYOPTOTTKERNEL_H
#define LOOM_PASSES_TILELOOMTOTTKERNEL_MEMORYOPTOTTKERNEL_H

#include "ComputeOpToTTKernel.h"
#include "FuncOpToTTKernel.h"
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
 * @details This function adds conversion patterns for loom.semaphore and
 *          loom.copy operations (plus reinterpret-cast cleanup) to the
 *          provided pattern set.
 * 
 * @param patterns The pattern set to populate.
 * @param typeConverter The type converter for the conversion pipeline.
 * @param context The MLIR context.
 * @param tracker Shared tracker for compile-arg index assignment.
 */
void populateMemoryOpConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *context, std::shared_ptr<CompileArgTracker> tracker,
    ReduceProtocol reduceProtocol);

/**
 * @brief Populate cleanup patterns that erase dead loom.alloc operations.
 *
 * @details This is intended to run after other conversion patterns that still
 *          need loom.alloc for CB linkage discovery.
 */
void populateLoomAllocCleanupPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *context);

} // namespace loom
} // namespace mlir

#endif // LOOM_PASSES_TILELOOMTOTTKERNEL_MEMORYOPTOTTKERNEL_H
