/**
 * @file TTKernelUtils.h
 * @brief Shared helper APIs for TileLoomToTTKernel stages.
 */

#pragma once

#include "FuncOpToTTKernel.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace loom {
class CopyOp;
} // namespace loom

namespace mlir::loom {

std::optional<DataMovementKernelRole>
getDataMovementKernelRoleForFuncName(StringRef name);

std::optional<DataMovementKernelSpec>
getDataMovementKernelSpecForProcessorAttr(StringRef processorAttrValue);

std::optional<DataMovementKernelSpec>
resolveDataMovementKernelSpec(func::FuncOp func);

Value stripMemrefCasts(Value value);
Value stripLoomSemaphores(Value value);
Value stripViewLikeWrappers(Value value);
Value stripBindingLookupWrappers(Value value);

bool isDramToL1Copy(::loom::CopyOp copyOp);
bool isL1ToDramCopy(::loom::CopyOp copyOp);
Value getCopyBindingDramMemref(::loom::CopyOp copyOp);
Value getCopyBindingL1Endpoint(::loom::CopyOp copyOp);
MemRefType getCopyBindingCBMemrefType(::loom::CopyOp copyOp);

std::optional<int64_t> getCopyBindingSlot(Operation *op);
int64_t getAnnotatedCopyBindingCount(func::FuncOp func);

std::optional<int64_t> evaluateConstInt(Value value);
std::optional<int64_t> getStaticParallelCoreCount(Operation *op);
LogicalResult collectCopyBindingOps(func::FuncOp func,
                                    SmallVectorImpl<::loom::CopyOp> &ops);

bool hasRuntimeBroadcast(::loom::CopyOp copyOp);
bool hasMappedParallel(func::FuncOp func);
SmallVector<int64_t, 8> collectInternalSemaphoreSlots(func::FuncOp func);

bool isComputeKernel(Operation *op);
bool isWriterKernel(Operation *op);

std::optional<int64_t> ceilDiv32(int64_t value);
std::optional<int64_t> getNumTilesFromShapedType(Type type);
std::optional<std::pair<int64_t, int64_t>>
parseBroadcastAttr(ArrayAttr broadcastAttr);

Value i32Const(OpBuilder &rewriter, Location loc, int64_t value);
Value toI32(OpBuilder &rewriter, Location loc, Value value);

FailureOr<Value>
getOrCreateCBConst(Location loc, OpBuilder &rewriter, func::FuncOp func,
                   int64_t cbIndex, StringRef name, Type resultType,
                   std::optional<int64_t> copyBindingSlot = std::nullopt,
                   std::optional<int64_t> internalSlot = std::nullopt);

void rewriteNamedCBCompileTimeArgLiterals(ModuleOp module);

struct ReduceCoreRegionAnalysis {
  Value ulX;
  Value ulY;
  Value lrX;
  Value lrY;
  Value participants;
  Value workerCount;
  Value rank;
  Value inRegion;
  Value isReducer;
};

FailureOr<ReduceCoreRegionAnalysis>
analyzeReduceCoreRegion(OpBuilder &rewriter, Location loc, Value coreX,
                        Value coreY, Value ulX, Value ulY, Value lrX,
                        Value lrY);

} // namespace mlir::loom
