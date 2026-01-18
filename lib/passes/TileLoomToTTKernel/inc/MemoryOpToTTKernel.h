//===- MemoryOpToTTKernel.h - Memory op to TTKernel lowering ----*- C++ -*-===//
//
// This header declares the pattern population functions for converting
// memref operations to TTKernel dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_PASSES_TILELOOMTOTTKERNEL_MEMORYOPTOTTKERNEL_H
#define LOOM_PASSES_TILELOOMTOTTKERNEL_MEMORYOPTOTTKERNEL_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace loom {

/// Populates the pattern set with memref.copy to TTKernel conversion patterns.
///
/// These patterns lower memref.copy operations to TTKernel NOC (Network on Chip)
/// operations for efficient data movement between DRAM and L1 memory.
///
/// The patterns handle:
/// - DRAM to L1 copies (load path): Identified by the attribute
///   `{loom.copy.choice = {kind = "mem", memory_name = "L1"}}`
/// - L1 to DRAM copies (store path): Identified by destination being
///   a memref.reinterpret_cast (indicating DRAM access)
///
/// @param typeConverter The type converter for handling type conversions.
/// @param patterns The RewritePatternSet to populate with conversion patterns.
/// @param benefit The pattern benefit (priority) for the conversion patterns.
void populateMemoryOpToTTKernelPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit);

/// Populates patterns with default benefit of 1.
///
/// @param typeConverter The type converter for handling type conversions.
/// @param patterns The RewritePatternSet to populate with conversion patterns.
void populateMemoryOpToTTKernelPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns);

} // namespace loom
} // namespace mlir

#endif // LOOM_PASSES_TILELOOMTOTTKERNEL_MEMORYOPTOTTKERNEL_H
