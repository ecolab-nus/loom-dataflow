/**
 * @file ComputeOpToTTKernel.h
 * @brief Declarations for compute op conversion patterns to TTKernel.
 */

#pragma once

#include "FuncOpToTTKernel.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::loom {

enum class ReduceSumProtocol {
  MultiSlot,
  SingleSlot
};

/**
 * @brief Populate conversion patterns for compute ops (e.g., linalg.matmul).
 *
 * @param patterns Pattern set to populate.
 * @param typeConverter Type converter used for the conversion pipeline.
 * @param context MLIR context.
 */
void populateComputeOpConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::TypeConverter &typeConverter,
                                         mlir::MLIRContext *context,
                                         std::shared_ptr<CompileArgTracker> tracker,
                                         ReduceSumProtocol reduceSumProtocol);

/// Returns true when this `linalg.generic` matches one of the supported
/// FlashAttention compute forms handled by this pass.
bool isSupportedFlashAttentionGeneric(mlir::linalg::GenericOp op);

/// Returns true when this `linalg.copy` is a compute-kernel copy handled by
/// this pass.
bool shouldConvertComputeLinalgCopy(mlir::linalg::CopyOp op);

} // namespace mlir::loom
