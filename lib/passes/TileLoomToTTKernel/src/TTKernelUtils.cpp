/**
 * @file TTKernelUtils.cpp
 * @brief Shared helper implementations for TileLoomToTTKernel stages.
 */

#include "TTKernelUtils.h"
#include "TTKernelAttrs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#define GET_OP_CLASSES
#include "LoomEnums.h.inc"
#include "LoomAttributes.h.inc"
#include "LoomOps.h.inc"

using namespace mlir;
using namespace mlir::loom;
using namespace tt::ttkernel;

std::optional<DataMovementKernelRole>
mlir::loom::getDataMovementKernelRoleForFuncName(StringRef name) {
  if (name.ends_with("__reader"))
    return DataMovementKernelRole::Reader;
  if (name.ends_with("__writer"))
    return DataMovementKernelRole::Writer;
  return std::nullopt;
}

std::optional<DataMovementKernelSpec>
mlir::loom::getDataMovementKernelSpecForProcessorAttr(
    StringRef processorAttrValue) {
  if (processorAttrValue == kDMProcessorRISCV1)
    return getDataMovementKernelSpec(DataMovementKernelRole::Reader);
  if (processorAttrValue == kDMProcessorRISCV0)
    return getDataMovementKernelSpec(DataMovementKernelRole::Writer);
  return std::nullopt;
}

std::optional<DataMovementKernelSpec>
mlir::loom::resolveDataMovementKernelSpec(func::FuncOp func) {
  if (auto processorAttr =
          func->getAttrOfType<StringAttr>(kDMProcessorAttrName)) {
    if (auto spec =
            getDataMovementKernelSpecForProcessorAttr(processorAttr.getValue()))
      return spec;
  }

  if (auto role = getDataMovementKernelRoleForFuncName(func.getName()))
    return getDataMovementKernelSpec(*role);
  return std::nullopt;
}

Value mlir::loom::stripMemrefCasts(Value value) {
  Value current = value;
  while (current) {
    auto cast = current.getDefiningOp<memref::CastOp>();
    if (!cast)
      break;
    current = cast.getSource();
  }
  return current;
}

Value mlir::loom::stripLoomSemaphores(Value value) {
  Value current = value;
  while (current) {
    auto sem = current.getDefiningOp<::loom::SemaphoreTakeOp>();
    if (!sem)
      break;
    current = sem.getSource();
  }
  return current;
}

Value mlir::loom::stripViewLikeWrappers(Value value) {
  Value current = value;
  while (current) {
    if (auto sem = current.getDefiningOp<::loom::SemaphoreTakeOp>()) {
      current = sem.getSource();
      continue;
    }
    if (auto viewLike = current.getDefiningOp<ViewLikeOpInterface>()) {
      current = viewLike.getViewSource();
      continue;
    }
    break;
  }
  return current;
}

Value mlir::loom::stripBindingLookupWrappers(Value value) {
  Value current = value;
  while (current) {
    if (auto cast = current.getDefiningOp<memref::CastOp>()) {
      current = cast.getSource();
      continue;
    }
    if (auto viewLike = current.getDefiningOp<ViewLikeOpInterface>()) {
      current = viewLike.getViewSource();
      continue;
    }
    break;
  }
  return current;
}

bool mlir::loom::isDramToL1Copy(::loom::CopyOp copyOp) {
  return copyOp.getSource().getDefiningOp<memref::ReinterpretCastOp>() !=
         nullptr;
}

bool mlir::loom::isL1ToDramCopy(::loom::CopyOp copyOp) {
  return copyOp.getDestination().getDefiningOp<memref::ReinterpretCastOp>() !=
         nullptr;
}

std::optional<int64_t> mlir::loom::getCopyBindingSlot(Operation *op) {
  if (!op)
    return std::nullopt;
  auto slotAttr = op->getAttrOfType<IntegerAttr>(kCopyBindingSlotAttrName);
  if (!slotAttr)
    return std::nullopt;
  return slotAttr.getInt();
}

int64_t mlir::loom::getAnnotatedCopyBindingCount(func::FuncOp func) {
  if (!func)
    return 0;
  auto countAttr = func->getAttrOfType<IntegerAttr>(kCopyBindingCountAttrName);
  return countAttr ? countAttr.getInt() : 0;
}

Value mlir::loom::getCopyBindingDramMemref(::loom::CopyOp copyOp) {
  if (!copyOp)
    return {};

  if (auto sourceRC =
          copyOp.getSource().getDefiningOp<memref::ReinterpretCastOp>())
    return stripMemrefCasts(sourceRC.getSource());

  if (auto destRC =
          copyOp.getDestination().getDefiningOp<memref::ReinterpretCastOp>())
    return stripMemrefCasts(destRC.getSource());

  return {};
}

Value mlir::loom::getCopyBindingL1Endpoint(::loom::CopyOp copyOp) {
  if (!copyOp)
    return {};

  if (isDramToL1Copy(copyOp))
    return stripMemrefCasts(copyOp.getDestination());
  if (isL1ToDramCopy(copyOp))
    return stripMemrefCasts(copyOp.getSource());
  return {};
}

MemRefType mlir::loom::getCopyBindingCBMemrefType(::loom::CopyOp copyOp) {
  Value l1Endpoint = getCopyBindingL1Endpoint(copyOp);
  if (!l1Endpoint)
    return {};

  Value base = stripLoomSemaphores(stripMemrefCasts(l1Endpoint));
  if (auto allocOp = base.getDefiningOp<::loom::AllocOp>())
    return dyn_cast<MemRefType>(allocOp.getType());

  return dyn_cast<MemRefType>(l1Endpoint.getType());
}

std::optional<int64_t> mlir::loom::evaluateConstInt(Value value) {
  if (!value)
    return std::nullopt;

  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();

  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();

  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return intAttr.getInt();
  }

  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return evaluateConstInt(cast.getIn());

  return std::nullopt;
}

std::optional<int64_t> mlir::loom::getStaticParallelCoreCount(Operation *op) {
  if (!op)
    return std::nullopt;

  int64_t coreCount = 1;
  bool foundParallel = false;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto parallelOp = dyn_cast<scf::ParallelOp>(parent);
    if (!parallelOp)
      continue;

    foundParallel = true;
    ValueRange lbs = parallelOp.getLowerBound();
    ValueRange ubs = parallelOp.getUpperBound();
    ValueRange steps = parallelOp.getStep();
    size_t rank = parallelOp.getInductionVars().size();
    if (lbs.size() < rank || ubs.size() < rank || steps.size() < rank)
      return std::nullopt;

    for (size_t idx = 0; idx < rank; ++idx) {
      auto lb = evaluateConstInt(lbs[idx]);
      auto ub = evaluateConstInt(ubs[idx]);
      auto step = evaluateConstInt(steps[idx]);
      if (!lb || !ub || !step || *step <= 0)
        return std::nullopt;

      int64_t span = *ub - *lb;
      if (span <= 0)
        return std::nullopt;
      coreCount *= (span + *step - 1) / *step;
    }
  }

  if (!foundParallel)
    return std::nullopt;
  return coreCount;
}

LogicalResult
mlir::loom::collectCopyBindingOps(func::FuncOp func,
                                  SmallVectorImpl<::loom::CopyOp> &ops) {
  ops.clear();
  int64_t bindingCount = getAnnotatedCopyBindingCount(func);
  if (bindingCount < 0)
    return func.emitError() << "invalid negative copy binding count";
  if (bindingCount == 0)
    return success();

  llvm::DenseMap<int64_t, ::loom::CopyOp> copyBySlot;
  SmallVector<int64_t, 8> slots;
  LogicalResult status = success();

  func.walk([&](::loom::CopyOp copyOp) {
    if (failed(status))
      return;
    auto slot = getCopyBindingSlot(copyOp.getOperation());
    if (!slot)
      return;
    if (*slot < 0 || *slot >= bindingCount) {
      status = copyOp.emitOpError("has out-of-range copy binding slot ")
               << *slot << " for binding count " << bindingCount;
      return;
    }
    if (copyBySlot.count(*slot)) {
      status = copyOp.emitOpError("duplicate copy binding slot ") << *slot;
      return;
    }
    copyBySlot.try_emplace(*slot, copyOp);
    slots.push_back(*slot);
  });

  if (failed(status))
    return failure();

  llvm::sort(slots);
  for (int64_t slot : slots)
    ops.push_back(copyBySlot.lookup(slot));

  return success();
}

bool mlir::loom::hasRuntimeBroadcast(::loom::CopyOp copyOp) {
  auto parsed = parseBroadcastAttr(copyOp.getBroadcastAttr());
  if (!parsed)
    return false;
  return parsed->first > 1 || parsed->second > 1;
}

bool mlir::loom::hasMappedParallel(func::FuncOp func) {
  bool found = false;
  func.walk([&](scf::ParallelOp op) {
    if (found)
      return;
    if (op->hasAttr("loom.physical_dims") ||
        op->hasAttr("loom.mapped_to_dims"))
      found = true;
  });
  return found;
}

SmallVector<int64_t, 8>
mlir::loom::collectInternalSemaphoreSlots(func::FuncOp func) {
  SmallVector<int64_t, 8> slots;
  func.walk([&](::loom::SemaphoreTakeOp sem) {
    auto slotAttr = sem->getAttrOfType<IntegerAttr>(kSemaphoreSlotAttrName);
    if (!slotAttr)
      return;
    int64_t slot = slotAttr.getInt();
    if (!llvm::is_contained(slots, slot))
      slots.push_back(slot);
  });
  llvm::sort(slots);
  return slots;
}

bool mlir::loom::isComputeKernel(Operation *op) {
  auto parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc)
    return false;

  if (auto threadAttr =
          parentFunc->getAttrOfType<ThreadTypeAttr>(ThreadTypeAttr::name))
    return threadAttr.getValue() == ThreadType::Compute;

  return parentFunc.getName().ends_with("__compute");
}

bool mlir::loom::isWriterKernel(Operation *op) {
  auto parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc)
    return false;
  return parentFunc.getName().ends_with("__writer");
}

std::optional<int64_t> mlir::loom::ceilDiv32(int64_t value) {
  if (value <= 0)
    return std::nullopt;
  return (value + 31) / 32;
}

std::optional<int64_t> mlir::loom::getNumTilesFromShapedType(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return std::nullopt;

  ArrayRef<int64_t> shape = shaped.getShape();
  unsigned rank = shape.size();
  if (rank == 0)
    return std::nullopt;

  int64_t tiles = 1;
  for (unsigned i = 0; i + 2 < rank; ++i) {
    if (shape[i] <= 0)
      return std::nullopt;
    tiles *= shape[i];
  }

  unsigned tiledStart = rank > 1 ? rank - 2 : 0;
  for (unsigned i = tiledStart; i < rank; ++i) {
    auto dimTiles = ceilDiv32(shape[i]);
    if (!dimTiles)
      return std::nullopt;
    tiles *= *dimTiles;
  }
  return tiles;
}

std::optional<std::pair<int64_t, int64_t>>
mlir::loom::parseBroadcastAttr(ArrayAttr broadcastAttr) {
  if (!broadcastAttr || broadcastAttr.size() < 2)
    return std::nullopt;

  auto xAttr = dyn_cast<IntegerAttr>(broadcastAttr[0]);
  auto yAttr = dyn_cast<IntegerAttr>(broadcastAttr[1]);
  if (!xAttr || !yAttr)
    return std::nullopt;

  return std::make_pair(xAttr.getInt(), yAttr.getInt());
}

Value mlir::loom::i32Const(OpBuilder &rewriter, Location loc, int64_t value) {
  return rewriter.create<arith::ConstantIntOp>(loc, value, 32);
}

Value mlir::loom::toI32(OpBuilder &rewriter, Location loc, Value value) {
  if (!value)
    return {};
  if (value.getType().isInteger(32))
    return value;
  if (value.getType().isIndex())
    return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), value);
  if (auto intType = dyn_cast<IntegerType>(value.getType())) {
    if (intType.getWidth() < 32)
      return rewriter.create<arith::ExtSIOp>(loc, rewriter.getI32Type(), value);
    if (intType.getWidth() > 32)
      return rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), value);
  }
  return {};
}

FailureOr<ReduceCoreRegionAnalysis>
mlir::loom::analyzeReduceCoreRegion(OpBuilder &rewriter, Location loc,
                                    Value coreX, Value coreY, Value ulX,
                                    Value ulY, Value lrX, Value lrY) {
  llvm::DenseMap<Value, Value> i32Cache;
  auto toI32Cached = [&](Value value) -> Value {
    if (!value)
      return {};
    auto it = i32Cache.find(value);
    if (it != i32Cache.end())
      return it->second;
    Value converted = toI32(rewriter, loc, value);
    if (converted)
      i32Cache.try_emplace(value, converted);
    return converted;
  };

  coreX = toI32Cached(coreX);
  coreY = toI32Cached(coreY);
  ulX = toI32Cached(ulX);
  ulY = toI32Cached(ulY);
  lrX = toI32Cached(lrX);
  lrY = toI32Cached(lrY);
  if (!coreX || !coreY || !ulX || !ulY || !lrX || !lrY)
    return failure();

  Value zero = i32Const(rewriter, loc, 0);
  Value one = i32Const(rewriter, loc, 1);
  auto subI32 = [&](Value lhs, Value rhs) -> Value {
    if (lhs == rhs)
      return zero;
    return arith::SubIOp::create(rewriter, loc, lhs, rhs);
  };
  auto addI32 = [&](Value lhs, Value rhs) -> Value {
    return arith::AddIOp::create(rewriter, loc, lhs, rhs);
  };

  Value width = addI32(subI32(lrX, ulX), one);
  Value height = addI32(subI32(lrY, ulY), one);
  Value participants = arith::MulIOp::create(rewriter, loc, width, height);
  Value workerCount = arith::SubIOp::create(rewriter, loc, participants, one);

  Value relX = subI32(coreX, ulX);
  Value relY = subI32(coreY, ulY);
  Value rank = arith::AddIOp::create(
      rewriter, loc, arith::MulIOp::create(rewriter, loc, relY, width), relX);

  Value geUlX =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge, coreX, ulX);
  Value leLrX =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sle, coreX, lrX);
  Value geUlY =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge, coreY, ulY);
  Value leLrY =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sle, coreY, lrY);
  Value inRegion = arith::AndIOp::create(
      rewriter, loc, arith::AndIOp::create(rewriter, loc, geUlX, leLrX),
      arith::AndIOp::create(rewriter, loc, geUlY, leLrY));

  Value isReducer = arith::AndIOp::create(
      rewriter, loc,
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, coreX, ulX),
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, coreY, ulY));
  isReducer = arith::AndIOp::create(rewriter, loc, inRegion, isReducer);

  return ReduceCoreRegionAnalysis{ulX,        ulY,      lrX,
                                  lrY,        participants,
                                  workerCount, rank,    inRegion,
                                  isReducer};
}
