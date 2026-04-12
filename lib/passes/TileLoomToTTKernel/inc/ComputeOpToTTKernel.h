/**
 * @file ComputeOpToTTKernel.h
 * @brief Declarations for compute op conversion patterns to TTKernel.
 */

#pragma once

#include "FuncOpToTTKernel.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::loom {

/// Transport synchronization protocol for cross-core reduce operations.
/// @deprecated Use ReduceProtocol instead.
enum class ReduceProtocol {
  MultiSlot,
  SingleSlot
};

using ReduceSumProtocol = ReduceProtocol;

/**
 * @brief Supported tile-level combine operations for reduce lowering.
 *
 * @details Transport lowering is protocol-specific (single-slot vs multi-slot),
 *          while the tile combine operation is selected independently via this
 *          enum. Only `Sum` is currently implemented; `Max` and `Exp` are
 *          reserved for future extension and will produce compile-time
 *          diagnostics if used.
 */
enum class ReduceCombineOp {
  Sum,
  Max,
  Exp
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
                                         ReduceProtocol reduceProtocol);

/// Returns true when this `linalg.generic` matches one of the supported
/// FlashAttention compute forms handled by this pass.
bool isSupportedFlashAttentionGeneric(mlir::linalg::GenericOp op);

/// Returns true when this `linalg.copy` is a compute-kernel copy handled by
/// this pass.
bool shouldConvertComputeLinalgCopy(mlir::linalg::CopyOp op);

} // namespace mlir::loom
