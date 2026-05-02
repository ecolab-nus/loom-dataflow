/**
 * @file TTKernelAttrs.h
 * @brief Shared TileLoomToTTKernel metadata keys.
 */

#pragma once

#include "llvm/ADT/StringRef.h"

namespace mlir::loom {

inline constexpr llvm::StringLiteral kReductionScaleCbAttrName =
    "loom.reduction_scale_cb";
inline constexpr llvm::StringLiteral kPartitionStrategyAttrName =
    "loom.ttkernel.partition_strategy";
inline constexpr llvm::StringLiteral kCopyDataMovementRoleAttrName =
    "loom.ttkernel.dm_role";
inline constexpr llvm::StringLiteral kDMProcessorAttrName =
    "loom.ttkernel.dm_processor";
inline constexpr llvm::StringLiteral kHasReduceAttrName =
    "loom.ttkernel.has_reduce";
inline constexpr llvm::StringLiteral kScalarPreprocessTmpAttrName =
    "loom.ttkernel.scalar_preprocess_tmp";
inline constexpr llvm::StringLiteral kScalarSiteIdAttrName =
    "loom.ttkernel.scalar_site_id";
inline constexpr llvm::StringLiteral kScalarSiteCountAttrName =
    "loom.ttkernel.scalar_site_count";
inline constexpr llvm::StringLiteral kScalarSourceOnlyAttrName =
    "loom.ttkernel.scalar_source_only";
inline constexpr llvm::StringLiteral kVecKindAttrName =
    "loom.ttkernel.vec_kind";
inline constexpr llvm::StringLiteral kVecTilesAttrName =
    "loom.ttkernel.vec_tiles";
inline constexpr llvm::StringLiteral kBroadcastDimAttrName =
    "loom.ttkernel.broadcast_dim";
inline constexpr llvm::StringLiteral kSemaphoreSlotAttrName =
    "loom.ttkernel.semaphore_slot";
inline constexpr llvm::StringLiteral kCopyBindingSlotAttrName =
    "loom.ttkernel.copy_binding_slot";
inline constexpr llvm::StringLiteral kCopyBindingCountAttrName =
    "loom.ttkernel.copy_binding_count";
inline constexpr llvm::StringLiteral kCBIndexAttrName =
    "loom.ttkernel.cb_index";
inline constexpr llvm::StringLiteral kCBConstNameAttrName =
    "loom.ttkernel.cb_const_name";
inline constexpr llvm::StringLiteral kCBConstValueAttrName =
    "loom.ttkernel.cb_const_value";
inline constexpr llvm::StringLiteral kCBConstBindingSlotAttrName =
    "loom.ttkernel.cb_const_binding_slot";
inline constexpr llvm::StringLiteral kCBConstInternalSlotAttrName =
    "loom.ttkernel.cb_const_internal_slot";
inline constexpr llvm::StringLiteral kCBConstDeclAttrName =
    "loom.ttkernel.cb_const_decl";

inline constexpr llvm::StringLiteral kMergedBIntoWriterStrategy =
    "matmul_merge_b_reader_into_writer";
inline constexpr llvm::StringLiteral kReaderDataMovementRole = "reader";
inline constexpr llvm::StringLiteral kWriterDataMovementRole = "writer";
inline constexpr llvm::StringLiteral kDMProcessorRISCV0 = "riscv0";
inline constexpr llvm::StringLiteral kDMProcessorRISCV1 = "riscv1";

} // namespace mlir::loom
