/**
 * @file ComputeOpToTTKernel.h
 * @brief Declarations for compute op conversion patterns to TTKernel.
 */

#pragma once

#include "FuncOpToTTKernel.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"

namespace loom {
class SyncOp;
class BroadcastOp;
} // namespace loom

namespace mlir::loom {

/// Transport synchronization protocol for cross-core reduce operations.
enum class ReduceProtocol {
  MultiSlot,
  SingleSlot
};

/// @deprecated Use ReduceProtocol instead.
using ReduceSumProtocol = ReduceProtocol;

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
                                         std::shared_ptr<CompileArgTracker>
                                             tracker);

/// Returns true when this `linalg.generic` matches one of the supported
/// FlashAttention compute forms handled by this pass.
bool isSupportedFlashAttentionGeneric(mlir::linalg::GenericOp op);

/// Returns true when this `linalg.copy` is a compute-kernel copy handled by
/// this pass.
bool shouldConvertComputeLinalgCopy(mlir::linalg::CopyOp op);

/// Returns true when this `linalg.transpose` is a compute-kernel transpose
/// handled by this pass.
bool shouldConvertComputeLinalgTranspose(mlir::linalg::TransposeOp op);

/// Returns true when this `loom.sync` should be lowered in compute kernels.
bool shouldConvertComputeLoomSync(::loom::SyncOp op);

/// Returns true when this `loom.broadcast` should be lowered in compute kernels.
bool shouldConvertComputeLoomBroadcast(::loom::BroadcastOp op);

} // namespace mlir::loom
