/**
 * @file ComputeOpToTTKernel.h
 * @brief Declarations for compute op conversion patterns to TTKernel.
 */

#pragma once

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::loom {

/**
 * @brief Populate conversion patterns for compute ops (e.g., linalg.matmul).
 *
 * @param patterns Pattern set to populate.
 * @param typeConverter Type converter used for the conversion pipeline.
 * @param context MLIR context.
 */
void populateComputeOpConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::TypeConverter &typeConverter,
                                         mlir::MLIRContext *context);

} // namespace mlir::loom

