/**
 * @file SCFOpToTTKernel.h
 * @brief Declarations for SCF dialect to TTKernel conversion helpers.
 */

#ifndef LOOM_PASSES_TILELOOMTOTTKERNEL_SCFOPTOTTKERNEL_H
#define LOOM_PASSES_TILELOOMTOTTKERNEL_SCFOPTOTTKERNEL_H

#include "FuncOpToTTKernel.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {
namespace loom {

/**
 * @brief Populate conversion patterns for SCF operations to TTKernel.
 *
 * @details
 * This helper registers patterns that:
 * - Replace `scf.parallel` induction variables with values derived from two
 *   TTKernel `GetArgValOp` physical core coordinates (`x` and `y`).
 * - Reject `scf.parallel` induction variables that are not mapped through
 *   `loom.physical_dims` / `loom.mapped_to_dims` x/y metadata.
 * - Inline the bodies of such `scf.parallel` operations and erase the
 *   loop ops, effectively treating the parallel iterators as compile-time
 *   constants rather than dynamic loop indices.
 *
 * Currently only `scf.parallel` without reductions/results is supported.
 *
 * @param patterns Pattern set to populate.
 * @param typeConverter Type converter used for the conversion pipeline.
 * @param context MLIR context.
 * @param tracker Shared compile-arg tracker used to allocate indices.
 */
void populateSCFOpConversionPatterns(RewritePatternSet &patterns,
                                     TypeConverter &typeConverter,
                                     MLIRContext *context,
                                     std::shared_ptr<CompileArgTracker> tracker);

} // namespace loom
} // namespace mlir

#endif // LOOM_PASSES_TILELOOMTOTTKERNEL_SCFOPTOTTKERNEL_H
