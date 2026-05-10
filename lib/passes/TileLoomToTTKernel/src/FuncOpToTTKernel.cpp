/**
 * @file FuncOpToTTKernel.cpp
 * @brief Implementation for function specialization pass.
 *
 * @details This pass clones each func::FuncOp into five specialized versions:
 *          - `__compute`: retains compute ops, erases store operations
 *          - `__reader` : retains memory load ops, erases stores & compute ops
 *          - `__writer` : retains memory store ops, erases loads & compute ops
 *          - `__host_cpp`   : emits a host helper with DRAM allocation/I/O
 *          - `__host_pybind`: emits a host helper over pre-bound buffers
 *          This loosely mimics the CoreSpecialize pattern from
 *          triton-tenstorrent while remaining TileLoom-specific.
 */

#include "FuncOpToTTKernel.h"
#include "TTKernelAttrs.h"
#include "TTKernelUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <limits>
#include <optional>

// Loom dialect headers for ::::loom::CopyOp, ::loom::AllocOp
#include "mlir/Interfaces/ViewLikeInterface.h"
#define GET_OP_CLASSES
#include "LoomEnums.h.inc"
#include "LoomAttributes.h.inc"
#include "LoomOps.h.inc"

// TTKernel thread type attribute and enum.
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

using namespace mlir;
using namespace mlir::loom;
using namespace tt::ttkernel;

using mlir::loom::DataMovementKernelRole;
using mlir::loom::DataMovementKernelSpec;
using mlir::loom::CopyBindingRuntimeField;
using mlir::loom::CoreCoordRuntimeField;
using mlir::loom::KernelRuntimeRole;
using mlir::loom::ReduceRuntimeField;
using mlir::loom::RuntimeArgKey;
using mlir::loom::RuntimeArgKind;
using mlir::loom::RuntimeArgLayout;

//===----------------------------------------------------------------------===//
// CompileArgTracker Implementation
//===----------------------------------------------------------------------===//

namespace {

enum class HostProgramKind { Cpp, Pybind };
enum class HostArgKind { Memref, Scalar };

static std::string
buildHostDataMovementConfigExpr(const DataMovementKernelSpec &spec) {
  return "tt_metal::DataMovementConfig{.processor = " +
         spec.hostProcessorExpr.str() + ", .noc = " + spec.hostNocExpr.str() +
         ", .compile_args = compile_args}";
}

// Infer per-argument CB memref shape from loom.copy links to loom.alloc.
// This decouples DRAM tensor shape (function args / host buffers) from
// on-core CB tile shape (loom.alloc).
[[maybe_unused]] static LogicalResult inferArgToCBMemrefType(
    func::FuncOp func, llvm::DenseMap<Value, MemRefType> &argToCBMemrefType) {
  Block &entry = func.front();
  LogicalResult status = success();

  auto recordLinkedType = [&](Value linkedArg, MemRefType cbMemrefType) {
    auto blockArg = dyn_cast<BlockArgument>(linkedArg);
    if (!blockArg || blockArg.getOwner() != &entry)
      return success();

    Type blockArgType = blockArg.getType();
    if (!isa<MemRefType, UnrankedMemRefType>(blockArgType))
      return success();

    auto [it, inserted] = argToCBMemrefType.try_emplace(blockArg, cbMemrefType);
    if (inserted || it->second == cbMemrefType)
      return success();

    func.emitError() << "inconsistent CB memref shapes inferred for argument #"
                     << blockArg.getArgNumber() << ": " << it->second << " vs "
                     << cbMemrefType;
    return failure();
  };

  func.walk([&](::loom::CopyOp copyOp) {
    if (failed(status))
      return;

    // DRAM -> L1: source is reinterpret_cast(arg), destination is loom.alloc.
    if (auto sourceRC = copyOp.getSource().getDefiningOp<memref::ReinterpretCastOp>()) {
      Value linkedArg = stripMemrefCasts(sourceRC.getSource());
      Value destination =
          stripLoomSemaphores(stripMemrefCasts(copyOp.getDestination()));
      if (auto allocOp = destination.getDefiningOp<::loom::AllocOp>()) {
        if (auto allocMemrefType = dyn_cast<MemRefType>(allocOp.getType()))
          status = recordLinkedType(linkedArg, allocMemrefType);
      }
    }

    // L1 -> DRAM: source is loom.alloc, destination is reinterpret_cast(arg).
    if (auto destinationRC =
            copyOp.getDestination().getDefiningOp<memref::ReinterpretCastOp>()) {
      Value linkedArg = stripMemrefCasts(destinationRC.getSource());
      Value source = stripLoomSemaphores(stripMemrefCasts(copyOp.getSource()));
      if (auto allocOp = source.getDefiningOp<::loom::AllocOp>()) {
        if (auto allocMemrefType = dyn_cast<MemRefType>(allocOp.getType()))
          status = recordLinkedType(linkedArg, allocMemrefType);
      }
    }
  });

  return status;
}

static bool isReductionGenericWithoutScaleInput(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
    return false;
  auto iteratorTypes = op.getIteratorTypesArray();
  if (iteratorTypes.size() < 2)
    return false;
  return iteratorTypes[iteratorTypes.size() - 1] ==
             utils::IteratorType::reduction;
   // && iteratorTypes[iteratorTypes.size() - 2] ==  utils::IteratorType::reduction;
}

static SymbolRefAttr findL1MemorySymbol(func::FuncOp func) {
  SymbolRefAttr l1Memory;
  func.walk([&](::loom::AllocOp alloc) {
    if (l1Memory)
      return;
    l1Memory = alloc.getMemoryAttr();
  });
  if (!l1Memory)
    l1Memory = SymbolRefAttr::get(func.getContext(), "L1");
  return l1Memory;
}

static Block *findReductionScaleInsertionBlock(func::FuncOp func) {
  scf::ParallelOp targetParallel;
  func.walk([&](scf::ParallelOp parallelOp) {
    if (targetParallel)
      return;
    targetParallel = parallelOp;
  });
  if (targetParallel)
    return targetParallel.getBody();
  return &func.front();
}

static Value getOrCreateReductionScaleSemaphore(
    func::FuncOp func, MemRefType scaleType, SymbolRefAttr l1Memory,
    llvm::DenseMap<Type, Value> &cache) {
  auto cached = cache.find(scaleType);
  if (cached != cache.end())
    return cached->second;

  Value existing;
  func.walk([&](::loom::SemaphoreTakeOp sem) {
    if (existing || !sem->hasAttr(kReductionScaleCbAttrName))
      return;
    auto semType = dyn_cast<MemRefType>(sem.getResult().getType());
    if (!semType || semType != scaleType)
      return;
    existing = sem.getResult();
  });
  if (existing) {
    cache.try_emplace(scaleType, existing);
    return existing;
  }

  OpBuilder builder(func.getContext());
  builder.setInsertionPointToStart(findReductionScaleInsertionBlock(func));
  auto loc = func.getLoc();
  SmallVector<int64_t, 4> scaleShape(scaleType.getShape().begin(),
                                     scaleType.getShape().end());
  auto alloc = builder.create<::loom::AllocOp>(
      loc, scaleType, ValueRange{}, builder.getDenseI64ArrayAttr(scaleShape),
      IntegerAttr{}, builder.getI64IntegerAttr(1), l1Memory);
  auto sem = builder.create<::loom::SemaphoreTakeOp>(loc, scaleType,
                                                      alloc.getResult());
  sem->setAttr(kReductionScaleCbAttrName, builder.getUnitAttr());
  cache.try_emplace(scaleType, sem.getResult());
  return sem.getResult();
}

static LogicalResult rewriteReductionGenericWithScale(linalg::GenericOp op,
                                                      Value scaleInput) {
  auto scaleType = dyn_cast<ShapedType>(scaleInput.getType());
  if (!scaleType)
    return failure();

  unsigned oldNumInputs = op.getNumDpsInputs();
  unsigned oldNumOutputs = op.getNumDpsInits();

  SmallVector<AffineMap, 6> newIndexingMaps;
  auto oldMaps = op.getIndexingMapsArray();
  if (oldMaps.size() != oldNumInputs + oldNumOutputs)
    return failure();
  newIndexingMaps.append(oldMaps.begin(), oldMaps.begin() + oldNumInputs);
  newIndexingMaps.push_back(oldMaps[oldNumInputs]);
  newIndexingMaps.append(oldMaps.begin() + oldNumInputs, oldMaps.end());

  Block &body = op.getRegion().front();
  body.insertArgument(oldNumInputs, scaleType.getElementType(), op.getLoc());
  op->insertOperands(oldNumInputs, scaleInput);

  OpBuilder builder(op);
  op.setIndexingMapsAttr(builder.getAffineMapArrayAttr(newIndexingMaps));
  op->setAttr(
      "operandSegmentSizes",
      builder.getDenseI32ArrayAttr(
          {static_cast<int32_t>(oldNumInputs + 1),
           static_cast<int32_t>(oldNumOutputs)}));
  return success();
}

static void ensureReductionScaleInputs(func::FuncOp func) {
  SmallVector<linalg::GenericOp, 8> targets;
  func.walk([&](linalg::GenericOp genericOp) {
    if (isReductionGenericWithoutScaleInput(genericOp))
      targets.push_back(genericOp);
  });
  if (targets.empty())
    return;

  SymbolRefAttr l1Memory = findL1MemorySymbol(func);
  llvm::DenseMap<Type, Value> scaleSemaphoresByType;

  for (linalg::GenericOp genericOp : targets) {
    auto outputType = dyn_cast<MemRefType>(genericOp.getDpsInits()[0].getType());
    if (!outputType)
      continue;

    Value scaleSemaphore = getOrCreateReductionScaleSemaphore(
        func, outputType, l1Memory, scaleSemaphoresByType);
    (void)rewriteReductionGenericWithScale(genericOp, scaleSemaphore);
  }
}

static bool isScalarOnlyRuntimeSourceArg(func::FuncOp func, BlockArgument arg);

struct ScalarRuntimeListDimInfo {
  scf::ForOp forOp;
  int64_t dimOrdinal = 0;
  int64_t lb = 0;
  int64_t step = 1;
  int64_t extent = 1;
};

static bool isScfForInductionVar(BlockArgument arg) {
  if (!arg || !arg.getOwner())
    return false;
  auto forOp = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
  return forOp && forOp.getInductionVar() == arg;
}

static bool isScfParallelInductionVar(BlockArgument arg) {
  if (!arg || !arg.getOwner())
    return false;
  auto parallelOp =
      dyn_cast_or_null<scf::ParallelOp>(arg.getOwner()->getParentOp());
  return parallelOp &&
         llvm::is_contained(parallelOp.getInductionVars(), Value(arg));
}

static LogicalResult collectScalarRuntimeTemporalIvs(
    Value value, llvm::SmallPtrSetImpl<Value> &temporalIvs,
    Operation *diagnosticOp) {
  if (!value)
    return failure();

  if (evaluateConstInt(value))
    return success();

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (isScfForInductionVar(arg)) {
      temporalIvs.insert(arg);
      return success();
    }
    if (isScfParallelInductionVar(arg))
      return success();

    diagnosticOp->emitError()
        << "scalar runtime offset depends on unsupported block argument";
    return failure();
  }

  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return collectScalarRuntimeTemporalIvs(cast.getIn(), temporalIvs,
                                           diagnosticOp);

  auto collectBinary = [&](Value lhs, Value rhs) -> LogicalResult {
    if (failed(collectScalarRuntimeTemporalIvs(lhs, temporalIvs, diagnosticOp)))
      return failure();
    return collectScalarRuntimeTemporalIvs(rhs, temporalIvs, diagnosticOp);
  };

  if (auto op = value.getDefiningOp<arith::AddIOp>())
    return collectBinary(op.getLhs(), op.getRhs());
  if (auto op = value.getDefiningOp<arith::SubIOp>())
    return collectBinary(op.getLhs(), op.getRhs());
  if (auto op = value.getDefiningOp<arith::MulIOp>())
    return collectBinary(op.getLhs(), op.getRhs());
  if (auto op = value.getDefiningOp<arith::DivSIOp>())
    return collectBinary(op.getLhs(), op.getRhs());
  if (auto op = value.getDefiningOp<arith::DivUIOp>())
    return collectBinary(op.getLhs(), op.getRhs());
  if (auto op = value.getDefiningOp<arith::RemSIOp>())
    return collectBinary(op.getLhs(), op.getRhs());
  if (auto op = value.getDefiningOp<arith::RemUIOp>())
    return collectBinary(op.getLhs(), op.getRhs());
  if (auto op = value.getDefiningOp<arith::CeilDivSIOp>())
    return collectBinary(op.getLhs(), op.getRhs());
  if (auto op = value.getDefiningOp<arith::CeilDivUIOp>())
    return collectBinary(op.getLhs(), op.getRhs());

  Operation *defOp = value.getDefiningOp();
  diagnosticOp->emitError()
      << "unsupported operation in scalar runtime offset: "
      << (defOp ? defOp->getName().getStringRef() : StringRef("<unknown>"));
  return failure();
}

static LogicalResult getStaticLoopListDim(scf::ForOp forOp,
                                          int64_t dimOrdinal,
                                          ScalarRuntimeListDimInfo &dim,
                                          Operation *diagnosticOp) {
  auto lb = evaluateConstInt(forOp.getLowerBound());
  auto ub = evaluateConstInt(forOp.getUpperBound());
  auto step = evaluateConstInt(forOp.getStep());
  if (!lb || !ub || !step || *step <= 0) {
    diagnosticOp->emitError()
        << "scalar runtime list requires static positive scf.for bounds";
    return failure();
  }

  int64_t span = *ub - *lb;
  if (span < 0) {
    diagnosticOp->emitError()
        << "scalar runtime list requires scf.for upper bound >= lower bound";
    return failure();
  }

  int64_t extent = (span + *step - 1) / *step;
  if (extent <= 0) {
    diagnosticOp->emitError()
        << "scalar runtime list requires non-empty scf.for iteration space";
    return failure();
  }

  dim = ScalarRuntimeListDimInfo{forOp, dimOrdinal, *lb, *step, extent};
  return success();
}

static LogicalResult analyzeScalarRuntimeListDims(
    ::loom::CopyOp copyOp, memref::ReinterpretCastOp sourceRC, int64_t siteId,
    SmallVectorImpl<ScalarRuntimeListDimInfo> &dims, int64_t &siteSize) {
  siteSize = 1;

  llvm::SmallPtrSet<Value, 8> temporalIvs;
  auto offsets = sourceRC.getOffsets();
  if (!offsets.empty()) {
    if (failed(collectScalarRuntimeTemporalIvs(offsets.front(), temporalIvs,
                                              copyOp.getOperation())))
      return failure();
  }

  if (temporalIvs.empty())
    return success();

  SmallVector<scf::ForOp, 4> enclosingLoops;
  for (Operation *parent = copyOp->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent))
      enclosingLoops.push_back(forOp);
  }
  std::reverse(enclosingLoops.begin(), enclosingLoops.end());

  llvm::SmallPtrSet<Value, 8> matchedIvs;
  for (scf::ForOp forOp : enclosingLoops) {
    Value iv = forOp.getInductionVar();
    if (!temporalIvs.contains(iv))
      continue;

    ScalarRuntimeListDimInfo dim;
    if (failed(getStaticLoopListDim(forOp, static_cast<int64_t>(dims.size()),
                                    dim, copyOp.getOperation())))
      return failure();

    if (siteSize > std::numeric_limits<int64_t>::max() / dim.extent) {
      copyOp.emitOpError()
          << "scalar runtime list size overflow for site " << siteId;
      return failure();
    }
    siteSize *= dim.extent;
    dims.push_back(dim);
    matchedIvs.insert(iv);
  }

  if (matchedIvs.size() != temporalIvs.size()) {
    copyOp.emitOpError()
        << "scalar runtime offset depends on a non-enclosing scf.for IV";
    return failure();
  }

  return success();
}

static void appendScalarRuntimeListDimAttr(MLIRContext *context,
                                           const ScalarRuntimeListDimInfo &dim,
                                           int64_t siteId) {
  SmallVector<int64_t, 16> values;
  if (auto existing =
          dim.forOp->getAttrOfType<DenseI64ArrayAttr>(
              kScalarSiteListDimsAttrName)) {
    values.append(existing.asArrayRef().begin(), existing.asArrayRef().end());
  }
  values.append({siteId, dim.dimOrdinal, dim.lb, dim.step, dim.extent});
  dim.forOp->setAttr(kScalarSiteListDimsAttrName,
                     DenseI64ArrayAttr::get(context, values));
}

static LogicalResult annotateScalarRuntimeSites(func::FuncOp func) {
  MLIRContext *context = func.getContext();
  func.walk([](::loom::CopyOp copyOp) {
    copyOp->removeAttr(kScalarSiteIdAttrName);
  });
  func.walk([](::loom::SemaphoreTakeOp sem) {
    sem->removeAttr(kScalarSiteIdAttrName);
  });
  func.walk([](scf::ForOp forOp) {
    forOp->removeAttr(kScalarSiteListDimsAttrName);
  });

  int64_t nextSiteId = 0;
  SmallVector<int64_t, 8> scalarSiteSizes;
  LogicalResult status = success();
  func.walk([&](::loom::CopyOp copyOp) {
    if (failed(status))
      return;

    auto sourceRC = copyOp.getSource().getDefiningOp<memref::ReinterpretCastOp>();
    if (!sourceRC)
      return;

    auto dstType = dyn_cast<MemRefType>(copyOp.getDestination().getType());
    if (!dstType || !dstType.hasStaticShape() || dstType.getNumElements() != 1)
      return;

    Value dstBase = stripLoomSemaphores(stripMemrefCasts(copyOp.getDestination()));
    if (!dstBase.getDefiningOp<::loom::AllocOp>())
      return;

    SmallVector<ScalarRuntimeListDimInfo, 4> dims;
    int64_t siteSize = 1;
    status = analyzeScalarRuntimeListDims(copyOp, sourceRC, nextSiteId, dims,
                                          siteSize);
    if (failed(status))
      return;

    IntegerAttr siteAttr =
        IntegerAttr::get(IntegerType::get(context, 64), nextSiteId);
    copyOp->setAttr(kScalarSiteIdAttrName, siteAttr);

    Value dst = stripMemrefCasts(copyOp.getDestination());
    if (auto sem = dst.getDefiningOp<::loom::SemaphoreTakeOp>())
      sem->setAttr(kScalarSiteIdAttrName, siteAttr);

    for (const ScalarRuntimeListDimInfo &dim : dims)
      appendScalarRuntimeListDimAttr(context, dim, nextSiteId);
    scalarSiteSizes.push_back(siteSize);
    ++nextSiteId;
  });

  if (failed(status))
    return failure();

  if (nextSiteId > 0) {
    func->setAttr(kScalarSiteCountAttrName,
                  IntegerAttr::get(IntegerType::get(context, 64), nextSiteId));
    func->setAttr(kScalarSiteSizesAttrName,
                  DenseI64ArrayAttr::get(context, scalarSiteSizes));
  } else {
    func->removeAttr(kScalarSiteCountAttrName);
    func->removeAttr(kScalarSiteSizesAttrName);
  }

  for (BlockArgument arg : func.front().getArguments()) {
    if (isScalarOnlyRuntimeSourceArg(func, arg))
      func.setArgAttr(arg.getArgNumber(), kScalarSourceOnlyAttrName,
                      UnitAttr::get(func.getContext()));
    else
      func.removeArgAttr(arg.getArgNumber(), kScalarSourceOnlyAttrName);
  }

  return success();
}

enum class VecBroadcastKind { None, RowBcast, ColBcast };

static bool isIdentityMapForRank(AffineMap map, unsigned rank) {
  if (!map || map.getNumDims() != rank || map.getNumSymbols() != 0 ||
      map.getNumResults() != rank)
    return false;
  for (unsigned i = 0; i < rank; ++i) {
    auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(i));
    if (!dimExpr || dimExpr.getPosition() != i)
      return false;
  }
  return true;
}

static std::optional<unsigned> getDroppedDimFromBroadcastMap(AffineMap map,
                                                             unsigned rank) {
  if (!map || map.getNumDims() != rank || map.getNumSymbols() != 0 ||
      map.getNumResults() != rank - 1 || rank == 0)
    return std::nullopt;

  llvm::SmallBitVector seenDims(rank);
  unsigned prevPos = 0;
  bool hasPrev = false;
  for (AffineExpr expr : map.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      return std::nullopt;
    unsigned pos = dimExpr.getPosition();
    if (pos >= rank || seenDims.test(pos))
      return std::nullopt;
    if (hasPrev && pos <= prevPos)
      return std::nullopt;
    seenDims.set(pos);
    prevPos = pos;
    hasPrev = true;
  }

  for (unsigned dim = 0; dim < rank; ++dim) {
    if (!seenDims.test(dim))
      return dim;
  }
  return std::nullopt;
}

static std::optional<SmallVector<int64_t, 4>>
getElementwiseOutTileShape(linalg::GenericOp op) {
  if (op.getNumDpsInits() != 1)
    return std::nullopt;

  auto outType = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());
  if (!outType || !outType.hasStaticShape())
    return std::nullopt;

  unsigned rank = op.getNumLoops();
  if (outType.getRank() != rank)
    return std::nullopt;

  // Mirror compute lowering semantics: only innermost 2 dims are tiled.
  unsigned tiledStartDim = rank > 2 ? rank - 2 : 0;

  SmallVector<int64_t, 4> extents;
  extents.reserve(rank);
  for (unsigned dim = 0; dim < rank; ++dim) {
    int64_t dimSize = outType.getShape()[dim];
    if (dimSize <= 0)
      return std::nullopt;

    if (dim >= tiledStartDim) {
      auto dimTiles = ceilDiv32(dimSize);
      if (!dimTiles || *dimTiles <= 0)
        return std::nullopt;
      extents.push_back(*dimTiles);
    } else {
      extents.push_back(dimSize);
    }
  }

  return extents;
}

static std::optional<int64_t> getOutTilesFromTileShape(ArrayRef<int64_t> shape) {
  int64_t outTiles = 1;
  for (int64_t extent : shape) {
    if (extent <= 0)
      return std::nullopt;
    outTiles *= extent;
  }
  return outTiles;
}

static Value stripVecUseWrappers(Value value) {
  Value current = value;
  while (true) {
    if (auto cast = current.getDefiningOp<memref::CastOp>()) {
      current = cast.getSource();
      continue;
    }
    if (auto sem = current.getDefiningOp<::loom::SemaphoreTakeOp>()) {
      current = sem.getSource();
      continue;
    }
    if (auto buf = current.getDefiningOp<::loom::BufferizeToTensorOp>()) {
      current = buf.getSource();
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

static StringRef stringifyVecBroadcastKind(VecBroadcastKind kind) {
  switch (kind) {
  case VecBroadcastKind::None:
    return "none";
  case VecBroadcastKind::RowBcast:
    return "row_bcast";
  case VecBroadcastKind::ColBcast:
    return "col_bcast";
  }
  llvm_unreachable("unknown VecBroadcastKind");
}

static FailureOr<std::pair<VecBroadcastKind, int64_t>>
inferVecUseForGenericInput(linalg::GenericOp genericOp, unsigned inputIdx) {
  for (utils::IteratorType iterType : genericOp.getIteratorTypesArray()) {
    if (iterType == utils::IteratorType::reduction) {
      genericOp.emitOpError(
          "vector DRAM load annotation only supports elementwise "
          "linalg.generic (no reduction iterators)")
          << ", input #" << inputIdx;
      return failure();
    }
  }

  auto tileShape = getElementwiseOutTileShape(genericOp);
  if (!tileShape) {
    genericOp.emitOpError("failed to infer static output tile shape for "
                          "vector DRAM load annotation");
    return failure();
  }
  auto outTiles = getOutTilesFromTileShape(*tileShape);
  if (!outTiles || *outTiles <= 0) {
    genericOp.emitOpError(
        "invalid output tile count while annotating vector DRAM load");
    return failure();
  }

  auto maps = genericOp.getIndexingMapsArray();
  size_t expectedMapCount = static_cast<size_t>(
      genericOp.getNumDpsInputs() + genericOp.getNumDpsInits());
  if (maps.size() != expectedMapCount) {
    genericOp.emitOpError("unexpected indexing map arity");
    return failure();
  }
  if (inputIdx >= genericOp.getNumDpsInputs()) {
    genericOp.emitOpError("invalid generic input index for annotation");
    return failure();
  }

  unsigned rank = genericOp.getNumLoops();
  AffineMap inputMap = maps[inputIdx];
  if (isIdentityMapForRank(inputMap, rank))
    return std::make_pair(VecBroadcastKind::None, *outTiles);

  auto droppedDim = getDroppedDimFromBroadcastMap(inputMap, rank);
  if (!droppedDim) {
    genericOp.emitOpError("unsupported broadcast indexing map for "
                          "vector DRAM load annotation")
        << ", input #" << inputIdx;
    return failure();
  }

  if (*droppedDim != rank - 1 && (rank < 2 || *droppedDim != rank - 2)) {
    genericOp.emitOpError("vector DRAM load supports only row/col "
                          "broadcast generic inputs")
        << ", dropped dim=" << *droppedDim << ", rank=" << rank;
    return failure();
  }

  int64_t droppedDimTiles = (*tileShape)[*droppedDim];
  if (droppedDimTiles <= 0 || (*outTiles % droppedDimTiles) != 0) {
    genericOp.emitOpError("failed to derive input tile count for vector "
                          "DRAM load broadcast");
    return failure();
  }

  int64_t requiredTiles = *outTiles / droppedDimTiles;
  if (requiredTiles <= 0) {
    genericOp.emitOpError(
        "derived non-positive input tile count for vector DRAM load");
    return failure();
  }

  VecBroadcastKind kind =
      (*droppedDim == rank - 1) ? VecBroadcastKind::RowBcast
                                : VecBroadcastKind::ColBcast;
  return std::make_pair(kind, requiredTiles);
}

static FailureOr<bool> annotateVecLoadUsageInFunc(func::FuncOp func) {
  OpBuilder builder(func.getContext());
  bool changed = false;

  SmallVector<::loom::CopyOp, 8> vectorLoadCopies;
  func.walk([&](::loom::CopyOp copyOp) {
    auto sourceRC = copyOp.getSource().getDefiningOp<memref::ReinterpretCastOp>();
    if (!sourceRC)
      return;

    auto dstType = dyn_cast<MemRefType>(copyOp.getDestination().getType());
    if (!dstType || !dstType.hasStaticShape() || dstType.getRank() != 1)
      return;

    vectorLoadCopies.push_back(copyOp);
  });

  for (::loom::CopyOp copyOp : vectorLoadCopies) {
    Value destinationRoot = stripVecUseWrappers(copyOp.getDestination());

    std::optional<VecBroadcastKind> resolvedKind;
    std::optional<int64_t> resolvedTiles;
    LogicalResult status = success();

    func.walk([&](linalg::GenericOp genericOp) {
      if (failed(status))
        return;

      for (auto [inputIdx, inputValue] : llvm::enumerate(genericOp.getDpsInputs())) {
        if (stripVecUseWrappers(inputValue) != destinationRoot)
          continue;

        auto inferred = inferVecUseForGenericInput(genericOp, inputIdx);
        if (failed(inferred)) {
          status = failure();
          return;
        }

        VecBroadcastKind kind = inferred->first;
        int64_t tiles = inferred->second;
        if (!resolvedKind) {
          resolvedKind = kind;
          resolvedTiles = tiles;
          continue;
        }

        if (*resolvedTiles != tiles) {
          copyOp.emitOpError(
              "conflicting vector DRAM load broadcast usage across "
              "linalg.generic consumers")
              << " (existing kind=" << stringifyVecBroadcastKind(*resolvedKind)
              << ", tiles=" << *resolvedTiles
              << "; new kind=" << stringifyVecBroadcastKind(kind)
              << ", tiles=" << tiles << ")";
          status = failure();
          return;
        }

        if (*resolvedKind == kind)
          continue;

        // Treat `none` as neutral: prefer a concrete row/col kind when present.
        if (*resolvedKind == VecBroadcastKind::None) {
          resolvedKind = kind;
          continue;
        }
        if (kind == VecBroadcastKind::None)
          continue;

        copyOp.emitOpError(
            "conflicting vector DRAM load broadcast usage across "
            "linalg.generic consumers")
            << " (existing kind=" << stringifyVecBroadcastKind(*resolvedKind)
            << ", tiles=" << *resolvedTiles
            << "; new kind=" << stringifyVecBroadcastKind(kind)
            << ", tiles=" << tiles << ")";
        status = failure();
        return;
      }
    });

    if (failed(status))
      return failure();

    if (!resolvedKind || !resolvedTiles)
      continue;

    copyOp->setAttr(kVecKindAttrName,
                    builder.getStringAttr(stringifyVecBroadcastKind(*resolvedKind)));
    copyOp->setAttr(kVecTilesAttrName, builder.getI64IntegerAttr(*resolvedTiles));

    Value destination = stripMemrefCasts(copyOp.getDestination());
    if (auto sem = destination.getDefiningOp<::loom::SemaphoreTakeOp>()) {
      sem->setAttr(
          kVecKindAttrName,
          builder.getStringAttr(stringifyVecBroadcastKind(*resolvedKind)));
      sem->setAttr(kVecTilesAttrName, builder.getI64IntegerAttr(*resolvedTiles));
    }

    changed = true;
  }

  return changed;
}

static bool isScalarMemRefType(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType || !memrefType.hasStaticShape())
    return false;
  return memrefType.getNumElements() == 1;
}

static bool isScalarRuntimeSiteArg(func::FuncOp func, BlockArgument arg) {
  if (!arg || arg.getOwner() != &func.front())
    return false;
  return isa_and_nonnull<IntegerAttr>(
      func.getArgAttr(arg.getArgNumber(), kScalarSiteIdAttrName));
}

static bool isScalarOnlyRuntimeSourceArg(func::FuncOp func, BlockArgument arg) {
  if (!arg || arg.getOwner() != &func.front())
    return false;
  if (func.getArgAttr(arg.getArgNumber(), kScalarSourceOnlyAttrName))
    return true;

  bool hasScalarRuntimeSource = false;
  bool hasNonScalarDramUse = false;
  func.walk([&](::loom::CopyOp copyOp) {
    if (auto sourceRC =
            copyOp.getSource().getDefiningOp<memref::ReinterpretCastOp>()) {
      Value source = stripMemrefCasts(sourceRC.getSource());
      if (source == arg) {
        if (copyOp->hasAttr(kScalarSiteIdAttrName))
          hasScalarRuntimeSource = true;
        else
          hasNonScalarDramUse = true;
      }
    }

    if (auto destRC =
            copyOp.getDestination().getDefiningOp<memref::ReinterpretCastOp>()) {
      Value dest = stripMemrefCasts(destRC.getSource());
      if (dest == arg)
        hasNonScalarDramUse = true;
    }
  });

  return hasScalarRuntimeSource && !hasNonScalarDramUse;
}

// TODO(tmp): This is a temporary scalar-only preprocessing rewrite.
// It removes scalar alloc/semaphore memory scaffolding and forwards scalar
// sites via function arguments. We may replace this with a dedicated scalar
// ABI/lowering path in follow-up work.
static bool preprocessScalarMemoryOpsTmp(func::FuncOp func) {
  SmallVector<Value, 8> scalarAllocs;
  llvm::DenseMap<Value, SmallVector<::loom::SemaphoreTakeOp, 4>>
      semaphoresByAlloc;
  func.walk([&](::loom::SemaphoreTakeOp sem) {
    auto semType = dyn_cast<MemRefType>(sem.getResult().getType());
    auto allocOp = sem.getSource().getDefiningOp<::loom::AllocOp>();
    if (!semType || !allocOp || !isScalarMemRefType(semType) ||
        !isScalarMemRefType(allocOp.getType()))
      return;
    Value allocValue = allocOp.getResult();
    auto &semaphores = semaphoresByAlloc[allocValue];
    if (semaphores.empty())
      scalarAllocs.push_back(allocValue);
    semaphores.push_back(sem);
  });

  if (scalarAllocs.empty()) {
    func->removeAttr(kScalarPreprocessTmpAttrName);
    return false;
  }

  Block &entry = func.front();
  auto oldType = func.getFunctionType();
  SmallVector<Type, 8> newInputs(oldType.getInputs().begin(),
                                 oldType.getInputs().end());
  SmallVector<MemRefType, 8> scalarTypes;
  scalarTypes.reserve(scalarAllocs.size());
  for (Value allocValue : scalarAllocs) {
    auto &semaphores = semaphoresByAlloc[allocValue];
    auto semType = cast<MemRefType>(semaphores.front().getResult().getType());
    scalarTypes.push_back(semType);
    newInputs.push_back(semType);
  }

  func.setType(
      FunctionType::get(func.getContext(), newInputs, oldType.getResults()));

  SmallVector<BlockArgument, 8> scalarArgs;
  scalarArgs.reserve(scalarTypes.size());
  for (MemRefType semType : scalarTypes)
    scalarArgs.push_back(entry.addArgument(semType, func.getLoc()));

  for (auto [allocValue, scalarArg] : llvm::zip(scalarAllocs, scalarArgs)) {
    auto &semaphores = semaphoresByAlloc[allocValue];
    IntegerAttr siteAttr;
    SmallVector<::loom::SemaphoreGiveOp, 4> giveOps;

    for (::loom::SemaphoreTakeOp sem : semaphores) {
      if (!siteAttr)
        siteAttr = sem->getAttrOfType<IntegerAttr>(kScalarSiteIdAttrName);

      for (Operation *user : sem.getResult().getUsers()) {
        if (auto giveOp = dyn_cast<::loom::SemaphoreGiveOp>(user)) {
          giveOps.push_back(giveOp);
          continue;
        }
      }
    }

    if (siteAttr)
      func.setArgAttr(static_cast<unsigned>(scalarArg.getArgNumber()),
                      kScalarSiteIdAttrName, siteAttr);

    for (::loom::SemaphoreTakeOp sem : semaphores)
      sem.getResult().replaceAllUsesWith(scalarArg);

    for (::loom::SemaphoreGiveOp giveOp : giveOps)
      if (giveOp->getBlock())
        giveOp.erase();

    for (::loom::SemaphoreTakeOp sem : llvm::reverse(semaphores))
      if (sem->getBlock())
        sem.erase();

    auto allocOp = allocValue.getDefiningOp<::loom::AllocOp>();
    if (allocOp && allocOp.getResult().use_empty())
      allocOp.erase();
  }

  func->setAttr(kScalarPreprocessTmpAttrName,
                UnitAttr::get(func.getContext()));
  return true;
}

static LogicalResult annotateCopyBindingSlots(func::FuncOp func) {
  Builder builder(func.getContext());
  llvm::DenseMap<Operation *, int64_t> semaphoreToSlot;
  int64_t nextSlot = 0;

  LogicalResult status = success();
  func.walk([&](::loom::CopyOp copyOp) {
    if (failed(status))
      return;
    if (copyOp->hasAttr(kScalarSiteIdAttrName))
      return;
    if (!isDramToL1Copy(copyOp) && !isL1ToDramCopy(copyOp))
      return;

    Value l1Endpoint = getCopyBindingL1Endpoint(copyOp);
    auto sem = l1Endpoint.getDefiningOp<::loom::SemaphoreTakeOp>();
    if (!sem) {
      status = copyOp.emitOpError()
               << "DRAM/L1 copy binding requires an explicit "
                  "loom.semaphore_take endpoint";
      return;
    }
    auto [it, inserted] =
        semaphoreToSlot.try_emplace(sem.getOperation(), nextSlot);
    if (!inserted) {
      status = copyOp.emitOpError(
          "destination/source semaphore participates in multiple DRAM/L1 "
          "loom.copy bindings; one semaphore must map to one binding");
      return;
    }
    sem->setAttr(kCopyBindingSlotAttrName,
                 builder.getI64IntegerAttr(nextSlot));

    copyOp->setAttr(kCopyBindingSlotAttrName,
                    builder.getI64IntegerAttr(nextSlot));
    ++nextSlot;
  });

  if (failed(status))
    return failure();

  func->setAttr(kCopyBindingCountAttrName, builder.getI64IntegerAttr(nextSlot));
  return success();
}

static void annotateSemaphoreSlots(func::FuncOp func) {
  llvm::DenseMap<Value, SmallVector<::loom::SemaphoreTakeOp, 4>> semaphoresByAlloc;
  func.walk([&](::loom::SemaphoreTakeOp sem) {
    Value base = stripLoomSemaphores(stripMemrefCasts(sem.getSource()));
    if (!base.getDefiningOp<::loom::AllocOp>())
      return;
    semaphoresByAlloc[base].push_back(sem);
  });

  llvm::SmallPtrSet<Operation *, 16> memrefLinkedSemaphores;
  auto markMemrefLinkedSemaphoreFromEndpoint = [&](Value endpoint) {
    Value current = stripMemrefCasts(endpoint);
    if (auto sem = current.getDefiningOp<::loom::SemaphoreTakeOp>()) {
      memrefLinkedSemaphores.insert(sem.getOperation());
      return;
    }

  };

  func.walk([&](::loom::CopyOp copyOp) {
    if (copyOp.getSource().getDefiningOp<memref::ReinterpretCastOp>())
      markMemrefLinkedSemaphoreFromEndpoint(copyOp.getDestination());
    if (copyOp.getDestination().getDefiningOp<memref::ReinterpretCastOp>())
      markMemrefLinkedSemaphoreFromEndpoint(copyOp.getSource());
  });

  int64_t nextSlot = 0;
  func.walk([&](::loom::AllocOp alloc) {
    auto it = semaphoresByAlloc.find(alloc.getResult());
    if (it == semaphoresByAlloc.end())
      return;

    for (::loom::SemaphoreTakeOp sem : it->second) {
      if (memrefLinkedSemaphores.contains(sem.getOperation()))
        continue;
      sem->setAttr(
          kSemaphoreSlotAttrName,
          IntegerAttr::get(IntegerType::get(func.getContext(), 64), nextSlot));
      ++nextSlot;
    }
  });
}

static LogicalResult annotateStaticCbIndices(func::FuncOp func) {
  Builder builder(func.getContext());
  unsigned nextCbIndex = 0;
  llvm::DenseMap<Value, unsigned> semaphoreToCbIndex;

  llvm::DenseMap<Value, SmallVector<Value, 4>> semaphoresByAlloc;
  func.walk([&](::loom::SemaphoreTakeOp sem) {
    Value base = stripLoomSemaphores(stripMemrefCasts(sem.getSource()));
    if (!base.getDefiningOp<::loom::AllocOp>())
      return;
    semaphoresByAlloc[base].push_back(sem.getResult());
  });

  func.walk([&](::loom::AllocOp alloc) {
    auto it = semaphoresByAlloc.find(alloc.getResult());
    if (it == semaphoresByAlloc.end())
      return;

    for (Value semValue : it->second) {
      unsigned cbIndex = nextCbIndex++;
      semaphoreToCbIndex[semValue] = cbIndex;
      if (auto sem = semValue.getDefiningOp<::loom::SemaphoreTakeOp>()) {
        sem->setAttr(kCBIndexAttrName, builder.getI64IntegerAttr(cbIndex));
      }
    }
  });

  LogicalResult status = success();
  func.walk([&](::loom::CopyOp copyOp) {
    if (failed(status))
      return;
    auto slot = getCopyBindingSlot(copyOp.getOperation());
    if (!slot)
      return;
    Value endpoint = stripMemrefCasts(getCopyBindingL1Endpoint(copyOp));
    auto sem = endpoint.getDefiningOp<::loom::SemaphoreTakeOp>();
    if (!sem) {
      status = copyOp.emitOpError()
               << "failed to resolve semaphore endpoint while assigning CB "
                  "index for copy binding slot "
               << *slot;
      return;
    }
    auto it = semaphoreToCbIndex.find(sem.getResult());
    if (it == semaphoreToCbIndex.end()) {
      status = copyOp.emitOpError()
               << "missing static CB index for copy binding slot " << *slot;
      return;
    }
    copyOp->setAttr(kCBIndexAttrName, builder.getI64IntegerAttr(it->second));
  });

  return status;
}

} // namespace

LogicalResult mlir::loom::CompileArgTracker::processInputArgs(
    func::FuncOp func, TypeConverter &typeConverter, OpBuilder &rewriter) {
  (void)typeConverter;
  if (func.getName().contains("__host"))
    return success();

  FailureOr<RuntimeArgLayout> runtimeLayoutOr = buildRuntimeArgLayout(func);
  if (failed(runtimeLayoutOr))
    return failure();
  RuntimeArgLayout runtimeLayout = *runtimeLayoutOr;
  funcToRuntimeArgLayout[func.getOperation()] = runtimeLayout;

  Block &entry = func.front();
  Location loc = func.getLoc();

  auto runtimeRole = getKernelRuntimeRole(func);
  bool isComputeKernel =
      runtimeRole && *runtimeRole == KernelRuntimeRole::Compute;

  int8_t resolvedDmNocId = 0;
  if (!isComputeKernel) {
    auto dmSpec = resolveDataMovementKernelSpec(func);
    if (!dmSpec) {
      return func.emitError()
             << "failed to resolve data movement noc id for function "
             << func.getName();
    }
    resolvedDmNocId = dmSpec->nocId;
  }

  // Save insertion point and set to start of function body.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&entry);

  SmallVector<::loom::CopyOp, 8> copyBindings;
  if (failed(collectCopyBindingOps(func, copyBindings)))
    return failure();

  // First reserve shared TensorAccessorArgs indices for memref arguments.
  for (BlockArgument arg : entry.getArguments()) {
    if (isScalarRuntimeSiteArg(func, arg))
      continue;
    if (isScalarOnlyRuntimeSourceArg(func, arg))
      continue;

    Type argType = arg.getType();
    if (auto memrefType = dyn_cast<MemRefType>(argType)) {
      if (isScalarMemRefType(memrefType) && memrefType.getRank() == 0)
        continue;
    }

    if (isa<MemRefType, UnrankedMemRefType>(argType)) {
      memrefArgToTensorAccessorArgsIndex.try_emplace(
          arg, getNextTensorAccessorArgsIndex(func));
      continue;
    }

    if (!argType.isIndex())
      return func.emitError() << "unsupported argument type: " << argType;
  }

  funcToCopyBindingData.erase(func.getOperation());

  for (::loom::CopyOp copyOp : copyBindings) {
    auto slot = getCopyBindingSlot(copyOp.getOperation());
    if (!slot)
      return copyOp.emitOpError("missing required copy binding slot");

    Value linkedMemrefArg = getCopyBindingDramMemref(copyOp);
    auto blockArg = dyn_cast<BlockArgument>(linkedMemrefArg);
    if (!blockArg || blockArg.getOwner() != &entry) {
      return copyOp.emitOpError("expected DRAM side of copy binding to resolve "
                                "to a function memref argument");
    }

    auto cbMemrefType = getCopyBindingCBMemrefType(copyOp);
    if (!cbMemrefType) {
      return copyOp.emitOpError(
          "failed to infer L1 circular-buffer memref type for binding");
    }
    if (!cbMemrefType.hasStaticShape()) {
      return copyOp.emitOpError()
             << "has dynamic L1 circular-buffer memref type " << cbMemrefType
             << "; ttkernel CB types require a static element count";
    }

    auto accessorIndexIt = memrefArgToTensorAccessorArgsIndex.find(linkedMemrefArg);
    if (accessorIndexIt == memrefArgToTensorAccessorArgsIndex.end()) {
      return copyOp.emitOpError(
          "missing TensorAccessorArgs index for DRAM memref argument");
    }

    Location bindingLoc = copyOp.getLoc();

    auto copyKey = [&](CopyBindingRuntimeField field) {
      return RuntimeArgKey::copyBinding(*slot, field);
    };
    auto createCopyRuntimeArg = [&](CopyBindingRuntimeField field,
                                    Type resultType) -> Value {
      RuntimeArgKey key = copyKey(field);
      if (!runtimeLayout.contains(key))
        return {};
      return createRuntimeArg(bindingLoc, rewriter, func, key, resultType);
    };

    auto cbIndexAttr = copyOp->getAttrOfType<IntegerAttr>(kCBIndexAttrName);
    if (!cbIndexAttr) {
      return copyOp.emitOpError("missing static CB index for copy binding slot ")
             << *slot;
    }
    Value cbOp =
        createCBConst(bindingLoc, rewriter, func, cbIndexAttr.getInt(),
                      "cb_id_binding" + std::to_string(*slot),
                      CBType::get(cbMemrefType), *slot, std::nullopt);
    if (!cbOp)
      return copyOp.emitOpError("failed to materialize static copy-binding CB "
                                "for slot ")
             << *slot;

    Value baseAddrOp = createCopyRuntimeArg(CopyBindingRuntimeField::BaseAddr,
                                            rewriter.getI32Type());
    if (!isComputeKernel && !baseAddrOp)
      return copyOp.emitOpError("failed to materialize copy-binding base "
                                "address runtime arg for slot ")
             << *slot;

    Value mcastDestStartX = createCopyRuntimeArg(
        CopyBindingRuntimeField::McastDestStartX, rewriter.getI32Type());
    Value mcastDestStartY = createCopyRuntimeArg(
        CopyBindingRuntimeField::McastDestStartY, rewriter.getI32Type());
    Value mcastDestEndX = createCopyRuntimeArg(
        CopyBindingRuntimeField::McastDestEndX, rewriter.getI32Type());
    Value mcastDestEndY = createCopyRuntimeArg(
        CopyBindingRuntimeField::McastDestEndY, rewriter.getI32Type());
    Value mcastDestNum = createCopyRuntimeArg(
        CopyBindingRuntimeField::McastDestNum, rewriter.getI32Type());
    Value mcastSenderNocX = createCopyRuntimeArg(
        CopyBindingRuntimeField::McastSenderNocX, rewriter.getI32Type());
    Value mcastSenderNocY = createCopyRuntimeArg(
        CopyBindingRuntimeField::McastSenderNocY, rewriter.getI32Type());
    Value mcastSenderSemaphoreAddrArg = createCopyRuntimeArg(
        CopyBindingRuntimeField::McastSenderSemaphore, rewriter.getI32Type());
    Value mcastReceiverSemaphoreAddrArg = createCopyRuntimeArg(
        CopyBindingRuntimeField::McastReceiverSemaphore, rewriter.getI32Type());

    bool needsMcastRuntimeArgs =
        !isComputeKernel && isDramToL1Copy(copyOp) && hasRuntimeBroadcast(copyOp);
    if (needsMcastRuntimeArgs &&
        (!mcastDestStartX || !mcastDestStartY || !mcastDestEndX ||
         !mcastDestEndY || !mcastDestNum || !mcastSenderNocX ||
         !mcastSenderNocY || !mcastSenderSemaphoreAddrArg ||
         !mcastReceiverSemaphoreAddrArg)) {
      return copyOp.emitOpError("failed to materialize multicast runtime args "
                                "for copy binding slot ")
             << *slot;
    }

    Value tensorAccessor;
    Value mcastSenderSemaphoreAddrOp;
    Value mcastReceiverSemaphoreAddrOp;
    Value mcastReceiverSemaphoreAddrPtr;
    Value mcastSenderSemaphoreAddrPtr;
    Value mcastSenderSemaphoreNocAddrOp;
    Value mcastReceiverSemaphoreNocAddrOp;
    if (!isComputeKernel && baseAddrOp) {
      Value nocIdVal = rewriter.create<arith::ConstantIntOp>(
          bindingLoc, rewriter.getI8Type(), resolvedDmNocId);
      Value pageSize = GetTileSizeOp::create(rewriter, bindingLoc, cbOp);
      Value tensorAccessorArgsIdxVal = rewriter.create<arith::ConstantIntOp>(
          bindingLoc, rewriter.getI32Type(), accessorIndexIt->second);
      auto baseAddrArgs = rewriter.create<TensorAccessorArgsOp>(
          bindingLoc, tensorAccessorArgsIdxVal, tensorAccessorArgsIdxVal);
      tensorAccessor = rewriter
                           .create<TensorAccessorOp>(bindingLoc,
                                                     baseAddrArgs.getResult(),
                                                     baseAddrOp, pageSize)
                           .getResult();
      if (needsMcastRuntimeArgs) {
        mcastSenderSemaphoreAddrOp =
            GetSemaphoreOp::create(rewriter, bindingLoc,
                                   mcastSenderSemaphoreAddrArg);
        mcastReceiverSemaphoreAddrOp =
            GetSemaphoreOp::create(rewriter, bindingLoc,
                                   mcastReceiverSemaphoreAddrArg);
        mcastReceiverSemaphoreAddrPtr =
            CastToL1PtrOp::create(rewriter, bindingLoc,
                                  mcastReceiverSemaphoreAddrOp);
        mcastSenderSemaphoreAddrPtr =
            CastToL1PtrOp::create(rewriter, bindingLoc,
                                  mcastSenderSemaphoreAddrOp);
        mcastSenderSemaphoreNocAddrOp = GetNocAddrOp::create(
            rewriter, bindingLoc, mcastSenderNocX, mcastSenderNocY,
            mcastSenderSemaphoreAddrOp);
        mcastReceiverSemaphoreNocAddrOp =
            rewriter
                .create<ExperimentalGetNocMulticastAddrOp>(
                    bindingLoc, mcastDestStartX, mcastDestStartY, mcastDestEndX,
                    mcastDestEndY, mcastReceiverSemaphoreAddrOp, nocIdVal)
                .getResult();
      }
    }

    CopyBindingData bindingData;
    bindingData.slot = *slot;
    bindingData.linkedMemrefArg = linkedMemrefArg;
    bindingData.l1Endpoint = getCopyBindingL1Endpoint(copyOp);
    bindingData.isLoad = isDramToL1Copy(copyOp);
    bindingData.isStore = isL1ToDramCopy(copyOp);
    if (auto coreCount = getStaticParallelCoreCount(copyOp.getOperation()))
      bindingData.parallelCoreCount = *coreCount;
    bindingData.cb = cbOp;
    bindingData.baseAddr = baseAddrOp;
    bindingData.tensorAccessor = tensorAccessor;
    bindingData.mcast_dest_noc_start_x = mcastDestStartX;
    bindingData.mcast_dest_noc_start_y = mcastDestStartY;
    bindingData.mcast_dest_noc_end_x = mcastDestEndX;
    bindingData.mcast_dest_noc_end_y = mcastDestEndY;
    bindingData.mcast_dest_num = mcastDestNum;
    bindingData.mcast_sender_noc_x = mcastSenderNocX;
    bindingData.mcast_sender_noc_y = mcastSenderNocY;
    bindingData.mcast_sender_semaphore_addr = mcastSenderSemaphoreAddrOp;
    bindingData.mcast_receiver_semaphore_addr = mcastReceiverSemaphoreAddrOp;
    bindingData.mcast_sender_semaphore_addr_ptr = mcastSenderSemaphoreAddrPtr;
    bindingData.mcast_receiver_semaphore_addr_ptr =
        mcastReceiverSemaphoreAddrPtr;
    bindingData.mcast_sender_semaphore_noc_addr =
        mcastSenderSemaphoreNocAddrOp;
    bindingData.mcast_receiver_semaphore_noc_addr =
        mcastReceiverSemaphoreNocAddrOp;
    bindingData.noc_id = resolvedDmNocId;
    bindingData.initLoc = bindingLoc;
    funcToCopyBindingData[func.getOperation()][*slot] = bindingData;
  }

  // Process non-memref function arguments that still need runtime slots.
  for (BlockArgument arg : entry.getArguments()) {
    if (isScalarRuntimeSiteArg(func, arg))
      continue;
    if (isScalarOnlyRuntimeSourceArg(func, arg))
      continue;

    Type argType = arg.getType();
    if (auto memrefType = dyn_cast<MemRefType>(argType)) {
      if (isScalarMemRefType(memrefType) && memrefType.getRank() == 0)
        continue;
    }

    if (isa<MemRefType, UnrankedMemRefType>(argType)) {
      continue;
    } else if (argType.isIndex()) {
      return func.emitError()
             << "index function arguments are not part of the compact "
                "per-kernel runtime layout yet";
    } else {
      // Other types: not supported yet.
      return func.emitError()
             << "unsupported argument type: " << argType;
    }
  }

  funcToReduceRuntimeArgs.erase(func.getOperation());
  if (runtimeLayout.contains(
          RuntimeArgKey::reduce(ReduceRuntimeField::ReadySemaphore))) {
    auto createReduceArg = [&](ReduceRuntimeField field) -> Value {
      return createRuntimeArg(loc, rewriter, func, RuntimeArgKey::reduce(field),
                              rewriter.getI32Type());
    };
    Value readySemaphore = createReduceArg(ReduceRuntimeField::ReadySemaphore);
    Value tokenSemaphore = createReduceArg(ReduceRuntimeField::TokenSemaphore);
    Value tokenSemaphoreMcastDestStartX =
        createReduceArg(ReduceRuntimeField::TokenMcastDestStartX);
    Value tokenSemaphoreMcastDestStartY =
        createReduceArg(ReduceRuntimeField::TokenMcastDestStartY);
    Value tokenSemaphoreMcastDestEndX =
        createReduceArg(ReduceRuntimeField::TokenMcastDestEndX);
    Value tokenSemaphoreMcastDestEndY =
        createReduceArg(ReduceRuntimeField::TokenMcastDestEndY);
    if (!readySemaphore || !tokenSemaphore || !tokenSemaphoreMcastDestStartX ||
        !tokenSemaphoreMcastDestStartY || !tokenSemaphoreMcastDestEndX ||
        !tokenSemaphoreMcastDestEndY)
      return func.emitError()
             << "failed to materialize writer reduce runtime args";
    funcToReduceRuntimeArgs[func.getOperation()] = ReduceRuntimeArgs{
        readySemaphore, tokenSemaphore, tokenSemaphoreMcastDestStartX,
        tokenSemaphoreMcastDestStartY, tokenSemaphoreMcastDestEndX,
        tokenSemaphoreMcastDestEndY};
  }

  funcToScalarRuntimeArgs.erase(func.getOperation());
  if (auto scalarSiteCountAttr =
          func->getAttrOfType<IntegerAttr>(kScalarSiteCountAttrName)) {
    int64_t scalarSiteCount = scalarSiteCountAttr.getInt();
    if (scalarSiteCount < 0) {
      return func.emitError() << "invalid negative scalar site count: "
                              << scalarSiteCount;
    }
    DenseI64ArrayAttr scalarSiteSizesAttr =
        func->getAttrOfType<DenseI64ArrayAttr>(kScalarSiteSizesAttrName);
    ArrayRef<int64_t> scalarSiteSizes =
        scalarSiteSizesAttr ? scalarSiteSizesAttr.asArrayRef()
                            : ArrayRef<int64_t>();
    auto &runtimeArgsBySite = funcToScalarRuntimeArgs[func.getOperation()];
    runtimeArgsBySite.resize(static_cast<size_t>(scalarSiteCount));
    for (int64_t siteId = 0; siteId < scalarSiteCount; ++siteId) {
      int64_t siteSize =
          (siteId < static_cast<int64_t>(scalarSiteSizes.size()))
              ? scalarSiteSizes[siteId]
              : 1;
      if (siteSize <= 0)
        return func.emitError()
               << "invalid scalar runtime list size for site " << siteId
               << ": " << siteSize;

      auto &runtimeArgs = runtimeArgsBySite[siteId];
      runtimeArgs.reserve(static_cast<size_t>(siteSize));
      for (int64_t elementIndex = 0; elementIndex < siteSize; ++elementIndex) {
        RuntimeArgKey key = RuntimeArgKey::scalarSite(siteId, elementIndex);
        if (!runtimeLayout.contains(key))
          continue;
        Value scalarArg =
            createRuntimeArg(loc, rewriter, func, key, rewriter.getI32Type());
        if (!scalarArg)
          return func.emitError()
                 << "failed to materialize scalar runtime arg for site "
                 << siteId << ", element " << elementIndex;
        runtimeArgs.push_back(scalarArg);
      }
    }
  }

  return success();
}

mlir::loom::MemrefArgData *mlir::loom::CompileArgTracker::getMemrefData(Value arg) {
  auto it = memrefArgToData.find(arg);
  if (it != memrefArgToData.end())
    return &it->second;
  return nullptr;
}

mlir::loom::CopyBindingData *
mlir::loom::CompileArgTracker::getCopyBindingData(Operation *funcOp,
                                                  int64_t slot) {
  if (!funcOp || slot < 0)
    return nullptr;
  auto funcIt = funcToCopyBindingData.find(funcOp);
  if (funcIt == funcToCopyBindingData.end())
    return nullptr;
  auto slotIt = funcIt->second.find(slot);
  if (slotIt == funcIt->second.end())
    return nullptr;
  return &slotIt->second;
}

int64_t
mlir::loom::CompileArgTracker::getCopyBindingCount(Operation *funcOp) const {
  auto func = dyn_cast_or_null<func::FuncOp>(funcOp);
  if (!func)
    return 0;
  return getAnnotatedCopyBindingCount(func);
}

mlir::loom::IndexArgData *mlir::loom::CompileArgTracker::getIndexData(Value arg) {
  auto it = indexArgToData.find(arg);
  if (it != indexArgToData.end())
    return &it->second;
  return nullptr;
}

Value mlir::loom::CompileArgTracker::getCB(Value arg) {
  if (auto *data = getMemrefData(arg))
    return data->cb;
  return nullptr;
}

Value mlir::loom::CompileArgTracker::getBaseAddr(Value arg) {
  if (auto *data = getMemrefData(arg))
    return data->baseAddr;
  return nullptr;
}

Value mlir::loom::CompileArgTracker::getTensorAccessor(Value arg) {
  if (auto *data = getMemrefData(arg))
    return data->tensorAccessor;
  return nullptr;
}

Value mlir::loom::CompileArgTracker::getIndexValue(Value arg) {
  if (auto *data = getIndexData(arg))
    return data->indexValue;
  return nullptr;
}

Value mlir::loom::CompileArgTracker::createIndexCompileArg(Value value, Location loc,
                                                      OpBuilder &rewriter) {
  // Check if already created.
  if (auto *data = getIndexData(value))
    return data->indexValue;

  // Allocate a new compile-arg index. Find parent function to get the correct index counter.
  Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
  auto funcOp = dyn_cast<func::FuncOp>(parentOp);
  if (!funcOp)
    funcOp = parentOp->getParentOfType<func::FuncOp>();

  if (!funcOp) {
    // Should not happen in valid IR within a function.
    return nullptr;
  }

  Value compileArgOp = createGetArgValOp(loc, rewriter, funcOp,
                                        rewriter.getI32Type());

  // Cast i32 to index for compatibility.
  auto indexCast = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getIndexType(), compileArgOp);

  // Store the created values.
  indexArgToData[value] = IndexArgData{compileArgOp, indexCast.getResult()};

  return indexCast.getResult();
}

Value mlir::loom::CompileArgTracker::createTypedCompileArg(
    Location loc, OpBuilder &rewriter, Operation *funcOp, Type resultType) {
  if (!funcOp || !resultType)
    return nullptr;
  return createGetArgValOp(loc, rewriter, funcOp, resultType);
}

Value mlir::loom::CompileArgTracker::createTypedCompileArgAtIndex(
    Location loc, OpBuilder &rewriter, Operation *funcOp, int64_t argIndex,
    Type resultType) {
  if (!funcOp || !resultType || argIndex < 0)
    return nullptr;

  auto &explicitArgs = funcToExplicitCompileArgs[funcOp];
  auto it = explicitArgs.find(argIndex);
  if (it != explicitArgs.end()) {
    if (it->second.getType() != resultType)
      return nullptr;
    return it->second;
  }

  Value value = createGetArgValOpAtIndex(loc, rewriter, funcOp, argIndex, resultType);
  if (!value)
    return nullptr;

  explicitArgs.try_emplace(argIndex, value);
  return value;
}

std::optional<int64_t>
mlir::loom::CompileArgTracker::getRuntimeArgIndex(Operation *funcOp,
                                                  RuntimeArgKey key) const {
  if (!funcOp)
    return std::nullopt;

  auto it = funcToRuntimeArgLayout.find(funcOp);
  if (it == funcToRuntimeArgLayout.end())
    return std::nullopt;

  return it->second.indexOf(key);
}

Value mlir::loom::CompileArgTracker::createRuntimeArg(
    Location loc, OpBuilder &rewriter, Operation *funcOp, RuntimeArgKey key,
    Type resultType) {
  if (!funcOp || !resultType)
    return nullptr;

  auto layoutIt = funcToRuntimeArgLayout.find(funcOp);
  if (layoutIt == funcToRuntimeArgLayout.end())
    return nullptr;

  auto argIndex = layoutIt->second.indexOf(key);
  if (!argIndex)
    return nullptr;
  return createTypedCompileArgAtIndex(loc, rewriter, funcOp, *argIndex,
                                      resultType);
}

Value mlir::loom::CompileArgTracker::createCBConst(
    Location loc, OpBuilder &rewriter, Operation *funcOp, int64_t cbIndex,
    llvm::StringRef name, Type resultType,
    std::optional<int64_t> copyBindingSlot,
    std::optional<int64_t> internalSlot) {
  if (!funcOp || !resultType || cbIndex < 0 || name.empty())
    return nullptr;

  std::string nameStr = name.str();
  auto &cache = funcToCBConstValues[funcOp];
  auto cached = cache.find(nameStr);
  if (cached != cache.end()) {
    if (cached->second.getType() != resultType)
      return nullptr;
    return cached->second;
  }

  auto func = dyn_cast<func::FuncOp>(funcOp);
  if (!func)
    return nullptr;

  FailureOr<Value> cbConst =
      getOrCreateCBConst(loc, rewriter, func, cbIndex, name, resultType,
                         copyBindingSlot, internalSlot);
  if (failed(cbConst))
    return nullptr;

  cache[nameStr] = *cbConst;
  return *cbConst;
}

const mlir::loom::CompileArgTracker::ReduceRuntimeArgs *
mlir::loom::CompileArgTracker::getReduceRuntimeArgs(Operation *funcOp) const {
  auto it = funcToReduceRuntimeArgs.find(funcOp);
  if (it == funcToReduceRuntimeArgs.end())
    return nullptr;
  return &it->second;
}

mlir::loom::CompileArgTracker::ReduceRuntimeArgs *
mlir::loom::CompileArgTracker::getReduceRuntimeArgs(Operation *funcOp) {
  auto it = funcToReduceRuntimeArgs.find(funcOp);
  if (it == funcToReduceRuntimeArgs.end())
    return nullptr;
  return &it->second;
}

Value mlir::loom::CompileArgTracker::getScalarRuntimeArg(Operation *funcOp,
                                                         int64_t siteId) const {
  ArrayRef<Value> runtimeArgs = getScalarRuntimeArgs(funcOp, siteId);
  if (runtimeArgs.empty())
    return {};
  return runtimeArgs.front();
}

ArrayRef<Value>
mlir::loom::CompileArgTracker::getScalarRuntimeArgs(Operation *funcOp,
                                                    int64_t siteId) const {
  if (!funcOp || siteId < 0)
    return {};
  auto it = funcToScalarRuntimeArgs.find(funcOp);
  if (it == funcToScalarRuntimeArgs.end())
    return {};
  const auto &runtimeArgsBySite = it->second;
  if (siteId >= static_cast<int64_t>(runtimeArgsBySite.size()))
    return {};
  return runtimeArgsBySite[siteId];
}

void mlir::loom::CompileArgTracker::appendToCoreList(Operation *funcOp, Value value) {
  //add type transformation to i32
  funcToCoreList[funcOp].push_back(value);
}

ArrayRef<Value> mlir::loom::CompileArgTracker::getCoreList(Operation *funcOp) const {
  auto it = funcToCoreList.find(funcOp);
  if (it == funcToCoreList.end())
    return ArrayRef<Value>();
  return it->second;
}

void mlir::loom::CompileArgTracker::clearCoreList(Operation *funcOp) {
  funcToCoreList[funcOp].clear();
  funcToCoreCoordsByDim.erase(funcOp);
}

void mlir::loom::CompileArgTracker::setCoreCoordForDim(Operation *funcOp,
                                                        StringRef dimName,
                                                        Value value) {
  StringRef dim = dimName.trim();
  if (dim.starts_with("@"))
    dim = dim.drop_front();
  if (dim.equals_insensitive("x") || dim.equals_insensitive("dim_x"))
    funcToCoreCoordsByDim[funcOp].x = value;
  else if (dim.equals_insensitive("y") || dim.equals_insensitive("dim_y"))
    funcToCoreCoordsByDim[funcOp].y = value;
}

Value mlir::loom::CompileArgTracker::getCoreCoordForDim(Operation *funcOp,
                                                         StringRef dimName) const {
  auto it = funcToCoreCoordsByDim.find(funcOp);
  if (it == funcToCoreCoordsByDim.end())
    return {};

  StringRef dim = dimName.trim();
  if (dim.starts_with("@"))
    dim = dim.drop_front();
  if (dim.equals_insensitive("x") || dim.equals_insensitive("dim_x"))
    return it->second.x;
  if (dim.equals_insensitive("y") || dim.equals_insensitive("dim_y"))
    return it->second.y;
  return {};
}

int64_t mlir::loom::CompileArgTracker::getAndIncrementIndex(Operation *funcOp) {
  return funcToNextArgIndex[funcOp]++;
}

int64_t mlir::loom::CompileArgTracker::getNextTensorAccessorArgsIndex(Operation *funcOp) {
  return funcToNextTensorAccessorArgsIndex[funcOp]++;
}

Value mlir::loom::CompileArgTracker::createGetArgValOp(Location loc, OpBuilder &rewriter,
                                                  Operation *funcOp,
                                                  Type resultType) {
  int64_t index = getAndIncrementIndex(funcOp);
  Value idxValue = rewriter.create<arith::ConstantIntOp>(
      loc, rewriter.getI32Type(), static_cast<int64_t>(index));
  return rewriter.create<GetArgValOp>(loc, resultType, idxValue).getResult();
}

Value mlir::loom::CompileArgTracker::createGetArgValOpAtIndex(
    Location loc, OpBuilder &rewriter, Operation *funcOp, int64_t argIndex,
    Type resultType) {
  if (!funcOp || !resultType || argIndex < 0)
    return nullptr;

  int64_t &nextIndex = funcToNextArgIndex[funcOp];
  if (nextIndex <= argIndex)
    nextIndex = argIndex + 1;

  Value idxValue = rewriter.create<arith::ConstantIntOp>(
      loc, rewriter.getI32Type(), argIndex);
  return rewriter.create<GetArgValOp>(loc, resultType, idxValue).getResult();
}

//===----------------------------------------------------------------------===//
// replaceFuncArgsWithCompileArgs Implementation
//===----------------------------------------------------------------------===//

LogicalResult mlir::loom::replaceFuncArgsWithCompileArgs(
    func::FuncOp func, std::shared_ptr<CompileArgTracker> tracker,
    TypeConverter &typeConverter, OpBuilder &rewriter) {
  // Delegate to the tracker's processInputArgs method.
  return tracker->processInputArgs(func, typeConverter, rewriter);
}

namespace {

/**
 * @brief Check if a loom.copy is a store operation.
 *
 * @details A store is identified when the destination is a reinterpret_cast,
 *          indicating data flows from L1/CB to external DRAM.
 *
 * @param op The loom.copy operation to check.
 * @return true if this is a store operation, false otherwise.
 */
static bool isLoomStoreOp(::loom::CopyOp op) {
  return op.getDestination().getDefiningOp<memref::ReinterpretCastOp>() !=
         nullptr;
}

/**
 * @brief Check if a loom.copy is a load operation.
 *
 * @details A load is identified when the source is a reinterpret_cast,
 *          indicating data flows from external DRAM to L1/CB.
 *
 * @param op The loom.copy operation to check.
 * @return true if this is a load operation, false otherwise.
 */
static bool isLoomLoadOp(::loom::CopyOp op) {
  return op.getSource().getDefiningOp<memref::ReinterpretCastOp>() != nullptr;
}

static bool hasMatmulMergedBPartition(func::FuncOp func) {
  auto strategyAttr =
      func->getAttrOfType<StringAttr>(kPartitionStrategyAttrName);
  return strategyAttr &&
         strategyAttr.getValue() == kMergedBIntoWriterStrategy;
}

static StringRef getCopyDataMovementRole(::loom::CopyOp op) {
  if (auto roleAttr =
          op->getAttrOfType<StringAttr>(kCopyDataMovementRoleAttrName)) {
    return roleAttr.getValue();
  }
  return {};
}

static void clearMatmulBReaderMergeAttrs(func::FuncOp func) {
  func->removeAttr(kPartitionStrategyAttrName);
  func.walk([](::loom::CopyOp op) { op->removeAttr(kCopyDataMovementRoleAttrName); });
}

static void dropStaleSemaphoreCopyBindingAttrs(func::FuncOp func) {
  llvm::SmallSet<int64_t, 16> liveSlotIds;
  func.walk([&](::loom::CopyOp op) {
    if (auto slot = getCopyBindingSlot(op.getOperation()))
      liveSlotIds.insert(*slot);
  });

  func.walk([&](::loom::SemaphoreTakeOp sem) {
    auto slotAttr = sem->getAttrOfType<IntegerAttr>(kCopyBindingSlotAttrName);
    if (!slotAttr)
      return;
    if (!liveSlotIds.contains(slotAttr.getInt()))
      sem->removeAttr(kCopyBindingSlotAttrName);
  });
}

static void eraseUnboundSemaphoreTakes(func::FuncOp func) {
  SmallVector<::loom::SemaphoreTakeOp, 16> semaphores;
  func.walk([&](::loom::SemaphoreTakeOp sem) {
    if (sem->hasAttr(kCopyBindingSlotAttrName) ||
        sem->hasAttr(kSemaphoreSlotAttrName) ||
        sem->hasAttr(kReductionScaleCbAttrName) ||
        sem->hasAttr(kScalarSiteIdAttrName))
      return;
    semaphores.push_back(sem);
  });

  for (::loom::SemaphoreTakeOp sem : llvm::reverse(semaphores)) {
    sem.getResult().replaceAllUsesWith(sem.getSource());
    sem.erase();
  }
}

static FailureOr<::loom::CopyOp>
findUniqueCopyForBuffer(func::FuncOp func, Value buffer, bool wantLoad,
                        StringRef description) {
  Value canonicalBuffer = stripViewLikeWrappers(buffer);
  SmallVector<::loom::CopyOp, 2> matches;
  func.walk([&](::loom::CopyOp copyOp) {
    if (wantLoad != isLoomLoadOp(copyOp))
      return;
    if (!wantLoad && !isLoomStoreOp(copyOp))
      return;
    Value endpoint = wantLoad ? copyOp.getDestination() : copyOp.getSource();
    if (stripViewLikeWrappers(endpoint) == canonicalBuffer)
      matches.push_back(copyOp);
  });

  if (matches.empty()) {
    func.emitError() << "expected a " << (wantLoad ? "load" : "store")
                     << " loom.copy for " << description;
    return failure();
  }
  if (matches.size() != 1) {
    func.emitError() << "expected exactly one " << (wantLoad ? "load" : "store")
                     << " loom.copy for " << description << ", found "
                     << matches.size();
    return failure();
  }
  return matches.front();
}

static LogicalResult annotateMatmulBReaderMerge(func::FuncOp func) {
  SmallVector<linalg::MatmulOp, 2> matmuls;
  int loadCopyCount = 0;
  int storeCopyCount = 0;
  func.walk([&](linalg::MatmulOp matmulOp) { matmuls.push_back(matmulOp); });
  func.walk([&](::loom::CopyOp copyOp) {
    if (isLoomLoadOp(copyOp))
      ++loadCopyCount;
    else if (isLoomStoreOp(copyOp))
      ++storeCopyCount;
  });

  if (matmuls.empty())
    return success();

  if (matmuls.size() != 1) {
    return func.emitError()
           << "matmul B-reader merge expects exactly one linalg.matmul";
  }

  linalg::MatmulOp matmulOp = matmuls.front();
  if (matmulOp.getInputs().size() != 2 || matmulOp.getOutputs().size() != 1) {
    return matmulOp.emitOpError()
           << "matmul B-reader merge expects two inputs and one output";
  }
  if (loadCopyCount != 2 || storeCopyCount != 1) {
    return func.emitError()
           << "matmul B-reader merge expects exactly two load loom.copy ops "
              "and one store loom.copy op";
  }

  FailureOr<::loom::CopyOp> lhsLoad =
      findUniqueCopyForBuffer(func, matmulOp.getInputs()[0], /*wantLoad=*/true,
                              "matmul lhs");
  if (failed(lhsLoad))
    return failure();

  FailureOr<::loom::CopyOp> rhsLoad =
      findUniqueCopyForBuffer(func, matmulOp.getInputs()[1], /*wantLoad=*/true,
                              "matmul rhs");
  if (failed(rhsLoad))
    return failure();

  FailureOr<::loom::CopyOp> outputStore =
      findUniqueCopyForBuffer(func, matmulOp.getOutputs()[0],
                              /*wantLoad=*/false, "matmul output");
  if (failed(outputStore))
    return failure();

  OpBuilder builder(func.getContext());
  func->setAttr(kPartitionStrategyAttrName,
                builder.getStringAttr(kMergedBIntoWriterStrategy));
  (*lhsLoad)->setAttr(kCopyDataMovementRoleAttrName,
                      builder.getStringAttr(kReaderDataMovementRole));
  (*rhsLoad)->setAttr(kCopyDataMovementRoleAttrName,
                      builder.getStringAttr(kWriterDataMovementRole));
  (*outputStore)->setAttr(kCopyDataMovementRoleAttrName,
                          builder.getStringAttr(kWriterDataMovementRole));
  return success();
}

static bool isLoomDataMovementOrMetadataOp(Operation *op) {
  return isa<::loom::AllocOp, ::loom::SemaphoreTakeOp, ::loom::SemaphoreGiveOp,
             ::loom::CopyOp, ::loom::CopyToTensorOp,
             ::loom::CopyFromTensorOp, ::loom::BufferizeToTensorOp,
             ::loom::BufferizeToMemrefOp, ::loom::InitTensorOp,
             ::loom::PackToTensorOp, ::loom::SubviewOp, ::loom::BindOp,
             ::loom::BindShapeOp, ::loom::BindMemOp, ::loom::SymOp>(op);
}

/**
 * @brief Check if an operation belongs to compute, not data movement.
 *
 * @details This intentionally treats all `linalg` ops as compute and classifies
 *          Loom ops by role:
 *          - Data movement / metadata helpers are not compute.
 *          - Remaining Loom ops are treated as compute.
 *
 * This keeps reader/writer specialization generic without hardcoding only
 * matmul-like op classes.
 */
static bool isComputeOp(Operation *op) {
  if (isa<linalg::LinalgOp>(op))
    return true;

  Dialect *dialect = op->getDialect();
  if (!dialect || dialect->getNamespace() != StringRef("loom"))
    return false;

  return !isLoomDataMovementOrMetadataOp(op);
}

static bool containsGatherOp(func::FuncOp func) {
  bool hasReduce = false;
  func.walk([&](::loom::GatherOp) { hasReduce = true; });
  return hasReduce;
}

static bool isSpecializedKernelName(StringRef name) {
  return name.ends_with("__compute") || name.ends_with("__reader") ||
         name.ends_with("__writer") || name.ends_with("__host") ||
         name.ends_with("__host_cpp") || name.ends_with("__host_pybind");
}

static SmallVector<func::FuncOp, 8> collectEntryFuncs(ModuleOp module) {
  SmallVector<func::FuncOp, 8> funcs;
  module.walk([&](func::FuncOp func) {
    if (func.isExternal())
      return;
    if (isSpecializedKernelName(func.getName()))
      return;
    funcs.push_back(func);
  });
  return funcs;
}

/// Returns true when @p op is a gather transport op.
static bool isGatherTransportOp(Operation *op) {
  return isa<::loom::GatherOp>(op);
}

/**
 * @brief Decide whether a gather transport op should be kept or erased during
 *        function specialization for a given kernel role.
 *
 * @details Gather transport ops must survive into writer clones so the
 *          transport lowering pattern can match them:
 *          - __compute: erased (sum math stays in linalg.generic)
 *          - __writer:  kept for ConvertLoomGatherTransportOp
 *          - __reader:  erased (no gather transport in reader kernels)
 *          - __host_*:  kept so collectReduceRegion can analyze bounds;
 *                       the host emitter erases them later
 *
 * @return true if the gather op should be kept; false if it should be erased.
 */
static bool shouldKeepGatherOpInKernel(StringRef kernelSuffix) {
  return kernelSuffix == "__writer" ||
         kernelSuffix == "__host_cpp" || kernelSuffix == "__host_pybind";
}

/**
 * @brief Specialize a function for compute-only execution.
 *
 * @details Clones the function with `__compute` suffix. Both load and store
 *          operations are retained - they will be lowered to CB synchronization
 *          operations (cb_wait_front/cb_push_back) by the ConvertComputeLoadOp
 *          and ConvertComputeStoreOp patterns instead of NOC operations.
 *
 * @param func The original function to specialize.
 * @return The specialized compute function.
 */
static func::FuncOp makeComputeFunc(func::FuncOp func) {
  IRMapping mapping;
  auto computeFunc = cast<func::FuncOp>(func->clone(mapping));
  computeFunc.setName((func.getName() + "__compute").str());

  // Compute kernels keep both loads and stores - they will be lowered to
  // CB synchronization operations (cb_wait_front/cb_push_back) by the
  // ConvertComputeLoadOp and ConvertComputeStoreOp patterns.
  // Gather transport is writer-only and should not remain in compute clones.
  // Erase semaphore_give for gather inputs before erasing gather itself, since
  // later lowering cannot see gather users in compute clones.
  SmallVector<Operation *, 4> gatherOpsToErase;
  SmallVector<Value, 4> gatherInputValues;
  computeFunc.walk(
      [&](::loom::GatherOp op) {
        gatherOpsToErase.push_back(op.getOperation());
        Value gatherInput = stripMemrefCasts(op.getIns());
        if (!llvm::is_contained(gatherInputValues, gatherInput))
          gatherInputValues.push_back(gatherInput);
      });

  SmallVector<Operation *, 4> gatherInputGiveOpsToErase;
  if (!gatherInputValues.empty()) {
    computeFunc.walk([&](::loom::SemaphoreGiveOp op) {
      Value giveSource = stripMemrefCasts(op.getSource());
      if (llvm::is_contained(gatherInputValues, giveSource))
        gatherInputGiveOpsToErase.push_back(op.getOperation());
    });
  }

  for (Operation *op : llvm::reverse(gatherInputGiveOpsToErase))
    op->erase();
  for (Operation *op : llvm::reverse(gatherOpsToErase))
    op->erase();

  clearMatmulBReaderMergeAttrs(computeFunc);
  dropStaleSemaphoreCopyBindingAttrs(computeFunc);
  eraseUnboundSemaphoreTakes(computeFunc);
  return computeFunc;
}

/**
 * @brief Specialize a function for reader-only data movement.
 *
 * @details Clones the function with `__reader` suffix and erases all
 *          compute operations (linalg.matmul etc.) and store operations
 *          (memref.copy where target is reinterpret_cast). Loads are
 *          retained to model DRAM -> L1 traffic.
 *
 * @param func The original function to specialize.
 * @return The specialized reader function.
 */
static func::FuncOp makeReaderFunc(func::FuncOp func) {
  IRMapping mapping;
  auto readerFunc = cast<func::FuncOp>(func->clone(mapping));
  readerFunc.setName((func.getName() + "__reader").str());
  bool useMergedBPartition = hasMatmulMergedBPartition(readerFunc);

  // Collect compute ops (including gather transport ops) and stores to erase.
  // Gather ops are classified as compute by isComputeOp() and are erased
  // in reader kernels -- no gather transport runs on the reader.
  SmallVector<Operation *, 8> opsToErase;
  readerFunc.walk([&](Operation *op) {
    if (isComputeOp(op)) {
      opsToErase.push_back(op);
    }  else if (auto loomCopyOp = dyn_cast<::loom::CopyOp>(op)) {
      if (useMergedBPartition) {
        if (getCopyDataMovementRole(loomCopyOp) != kReaderDataMovementRole)
          opsToErase.push_back(op);
      } else if (isLoomStoreOp(loomCopyOp)) {
        opsToErase.push_back(op);
      }
    }
  });

  for (Operation *op : llvm::reverse(opsToErase))
    op->erase();

  clearMatmulBReaderMergeAttrs(readerFunc);
  dropStaleSemaphoreCopyBindingAttrs(readerFunc);
  eraseUnboundSemaphoreTakes(readerFunc);
  return readerFunc;
}

/**
 * @brief Specialize a function for writer-only data movement.
 *
 * @details Clones the function with `__writer` suffix and erases all
 *          compute operations and load operations (memref.copy where the
 *          source is reinterpret_cast). Stores are retained to model
 *          CB/L1 -> DRAM traffic.
 *
 * @param func The original function to specialize.
 * @return The specialized writer function.
 */
static func::FuncOp makeWriterFunc(func::FuncOp func) {
  IRMapping mapping;
  auto writerFunc = cast<func::FuncOp>(func->clone(mapping));
  writerFunc.setName((func.getName() + "__writer").str());
  bool useMergedBPartition = hasMatmulMergedBPartition(writerFunc);

  SmallVector<Operation *, 8> opsToErase;
  writerFunc.walk([&](Operation *op) {
    if (isGatherTransportOp(op) && shouldKeepGatherOpInKernel("__writer"))
      return;
    if (isComputeOp(op)) {
      opsToErase.push_back(op);
    }  else if (auto loomCopyOp = dyn_cast<::loom::CopyOp>(op)) {
      if (useMergedBPartition) {
        if (getCopyDataMovementRole(loomCopyOp) != kWriterDataMovementRole)
          opsToErase.push_back(op);
      } else if (isLoomLoadOp(loomCopyOp)) {
        opsToErase.push_back(op);
      }
    }
  });

  for (Operation *op : llvm::reverse(opsToErase))
    op->erase();

  clearMatmulBReaderMergeAttrs(writerFunc);
  dropStaleSemaphoreCopyBindingAttrs(writerFunc);
  eraseUnboundSemaphoreTakes(writerFunc);
  return writerFunc;
}

/**
 * @brief Emit host-side TT-Metal DRAM/CB setup and compile args.
 *
 * @details This helper consolidates host emission into one place:
 *          1. Either creates DRAM buffers or binds pre-existing buffers.
 *          2. Circular buffers from `loom.alloc`, linked back to DRAM memrefs.
 *          3. TensorAccessor compile args for each DRAM buffer.
 */
class TTMetalHostProgramEmitter {
public:
  TTMetalHostProgramEmitter(func::FuncOp originalFunc, func::FuncOp hostFunc,
                            HostProgramKind kind, func::FuncOp computeFunc,
                            func::FuncOp readerFunc, func::FuncOp writerFunc)
      : originalFunc(originalFunc), hostFunc(hostFunc), loc(hostFunc.getLoc()),
        builder(hostFunc.getContext()), kind(kind), computeFunc(computeFunc),
        readerFunc(readerFunc), writerFunc(writerFunc) {}

  LogicalResult run() {
    hasReduce = hostFunc->hasAttr(kHasReduceAttrName) ||
                   containsGatherOp(originalFunc);
    inferCoreCoordArgOrder();
    collectParallelIvExprs();
    collectDramInfos();
    collectScalarRuntimeSites();
    collectReduceRegion();
    collectCopyBindings();
    collectCbInfos();
    if (failed(status))
      return failure();
    if (!mlirCoreExtentX || !mlirCoreExtentY || *mlirCoreExtentX <= 0 ||
        *mlirCoreExtentY <= 0) {
      return hostFunc.emitError()
             << "host emission requires explicit x/y core-range metadata; "
                "device-grid defaults are not allowed";
    }
    FailureOr<RuntimeArgLayout> computeLayout =
        buildRuntimeArgLayout(computeFunc);
    FailureOr<RuntimeArgLayout> readerLayout =
        buildRuntimeArgLayout(readerFunc);
    FailureOr<RuntimeArgLayout> writerLayout =
        buildRuntimeArgLayout(writerFunc);
    if (failed(computeLayout) || failed(readerLayout) || failed(writerLayout))
      return failure();
    computeRuntimeLayout = *computeLayout;
    readerRuntimeLayout = *readerLayout;
    writerRuntimeLayout = *writerLayout;
    annotateHostSignatureMetadata();
    eraseNonHostOps();
    eraseDeadReinterpretCasts();
    eraseHostSpatialMesh();

    builder.setInsertionPointToStart(&hostFunc.getBody().front());
    emitPreamble();
    emitBufferSetup();
    emitCircularBuffers();
    emitInputSemaphores();
    emitCompileArgs();
    emitKernelRoles();

    // Runtime enqueue calls must be emitted at the end of the host function.
    builder.setInsertionPoint(hostFunc.front().getTerminator());
    emitCoreMulticastMappingAtEnd();
    emitRuntimeEnqueueEpilogue();
    return success();
  }

private:
  enum class MulticastKind { None, Horizontal, Vertical, All };

  struct DramBufferInfo {
    Value hostArg;
    MemRefType type;
    unsigned argIndex;
    unsigned hostParamIndex;
    unsigned memrefOrdinal;
    int inputTensorOrdinal;
    int outputTensorOrdinal;
    bool isInput;
    bool isOutput;
    std::string inputName;
    std::string configName;
    std::string bufferName;
    std::string sizeExpr;
  };

  struct CopyBindingInfo {
    unsigned slot = 0;
    unsigned dramInfoIndex = 0;
    int cbIndex = -1;
    bool isInput = false;
    bool isOutput = false;
    bool hasBroadcastRegion = false;
    Value l1Endpoint;
    std::string bindingName;
    std::string cbTilesExpr;
    std::string ulXExpr;
    std::string ulYExpr;
    std::string lrXExpr;
    std::string lrYExpr;
  };

  struct CircularBufferInfo {
    Value allocValue;
    std::string tilesVarName;
    std::string tilesExpr;
    SmallVector<unsigned, 4> cbIndices;
    SmallVector<std::string, 4> cbIndexNames;
  };

  struct KernelRoleInfo {
    std::string idVarName;
    std::string kernelSource;
    std::string configExpr;
  };

  struct ReduceRegionInfo {
    bool hasRegion = false;
    std::string ulXExpr;
    std::string ulYExpr;
    std::string lrXExpr;
    std::string lrYExpr;
  };

  struct ScalarRuntimeSiteInfo {
    int64_t siteId = -1;
    unsigned sourceArgIndex = 0;
    unsigned scalarHostArgOrdinal = 0;
    SmallVector<std::string, 4> offsetExprs;
    std::string inputName;
    std::string bufferName;
    std::string scalarValuesName;
  };

  struct ScalarRuntimeListDimHostInfo {
    Value iv;
    int64_t dimOrdinal = 0;
    int64_t lb = 0;
    int64_t step = 1;
    int64_t extent = 1;
  };

  struct ScalarRuntimeHostArgInfo {
    unsigned sourceArgIndex = 0;
    unsigned hostParamIndex = 0;
    std::string valuesName;
  };

  static Value stripCasts(Value value) {
    Value curr = value;
    while (auto cast = curr.getDefiningOp<memref::CastOp>())
      curr = cast.getSource();
    return curr;
  }

  static bool containsValue(ArrayRef<Value> values, Value value) {
    return llvm::find(values, value) != values.end();
  }

  static void appendUnique(SmallVectorImpl<Value> &values, Value value) {
    if (!containsValue(values, value))
      values.push_back(value);
  }

  static std::string getLetterName(size_t index) {
    if (index < 26)
      return std::string(1, static_cast<char>('A' + index));
    return "V" + std::to_string(index);
  }

  using IndexExprMap = llvm::DenseMap<Value, std::string>;

  static std::optional<int64_t> evaluateConstInt(Value value) {
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

    auto evalBinary = [&](Value lhs, Value rhs, auto fn) -> std::optional<int64_t> {
      auto lhsConst = evaluateConstInt(lhs);
      auto rhsConst = evaluateConstInt(rhs);
      if (!lhsConst || !rhsConst)
        return std::nullopt;
      return fn(*lhsConst, *rhsConst);
    };

    if (auto op = value.getDefiningOp<arith::AddIOp>())
      return evalBinary(op.getLhs(), op.getRhs(),
                        [](int64_t lhs, int64_t rhs) { return lhs + rhs; });
    if (auto op = value.getDefiningOp<arith::SubIOp>())
      return evalBinary(op.getLhs(), op.getRhs(),
                        [](int64_t lhs, int64_t rhs) { return lhs - rhs; });
    if (auto op = value.getDefiningOp<arith::MulIOp>())
      return evalBinary(op.getLhs(), op.getRhs(),
                        [](int64_t lhs, int64_t rhs) { return lhs * rhs; });
    if (auto op = value.getDefiningOp<arith::DivSIOp>()) {
      auto lhsConst = evaluateConstInt(op.getLhs());
      auto rhsConst = evaluateConstInt(op.getRhs());
      if (!lhsConst || !rhsConst || *rhsConst == 0)
        return std::nullopt;
      return *lhsConst / *rhsConst;
    }
    if (auto op = value.getDefiningOp<arith::DivUIOp>()) {
      auto lhsConst = evaluateConstInt(op.getLhs());
      auto rhsConst = evaluateConstInt(op.getRhs());
      if (!lhsConst || !rhsConst || *rhsConst == 0)
        return std::nullopt;
      return *lhsConst / *rhsConst;
    }
    if (auto op = value.getDefiningOp<arith::RemSIOp>()) {
      auto lhsConst = evaluateConstInt(op.getLhs());
      auto rhsConst = evaluateConstInt(op.getRhs());
      if (!lhsConst || !rhsConst || *rhsConst == 0)
        return std::nullopt;
      return *lhsConst % *rhsConst;
    }
    if (auto op = value.getDefiningOp<arith::RemUIOp>()) {
      auto lhsConst = evaluateConstInt(op.getLhs());
      auto rhsConst = evaluateConstInt(op.getRhs());
      if (!lhsConst || !rhsConst || *rhsConst == 0)
        return std::nullopt;
      return *lhsConst % *rhsConst;
    }
    if (auto op = value.getDefiningOp<arith::CeilDivSIOp>()) {
      auto lhsConst = evaluateConstInt(op.getLhs());
      auto rhsConst = evaluateConstInt(op.getRhs());
      if (!lhsConst || !rhsConst || *rhsConst <= 0)
        return std::nullopt;
      return (*lhsConst + *rhsConst - 1) / *rhsConst;
    }
    if (auto op = value.getDefiningOp<arith::CeilDivUIOp>()) {
      auto lhsConst = evaluateConstInt(op.getLhs());
      auto rhsConst = evaluateConstInt(op.getRhs());
      if (!lhsConst || !rhsConst || *rhsConst <= 0)
        return std::nullopt;
      return (*lhsConst + *rhsConst - 1) / *rhsConst;
    }

    return std::nullopt;
  }

  static std::string makeBinaryExpr(StringRef lhs, StringRef rhs, StringRef op) {
    return "((" + lhs.str() + ") " + op.str() + " (" + rhs.str() + "))";
  }

  std::optional<std::string> buildIndexExpr(Value value,
                                            const IndexExprMap &knownExprs) const {
    if (!value)
      return std::nullopt;

    if (auto it = knownExprs.find(value); it != knownExprs.end())
      return it->second;

    if (isa<BlockArgument>(value))
      return std::nullopt;

    if (auto constValue = evaluateConstInt(value))
      return std::to_string(*constValue);

    if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
      return buildIndexExpr(cast.getIn(), knownExprs);

    auto emitBinaryExpr = [&](Value lhs, Value rhs,
                              StringRef opSymbol) -> std::optional<std::string> {
      auto lhsExpr = buildIndexExpr(lhs, knownExprs);
      auto rhsExpr = buildIndexExpr(rhs, knownExprs);
      if (!lhsExpr || !rhsExpr)
        return std::nullopt;

      auto lhsConst = evaluateConstInt(lhs);
      auto rhsConst = evaluateConstInt(rhs);
      if (lhsConst && rhsConst) {
        if (opSymbol == "+")
          return std::to_string(*lhsConst + *rhsConst);
        if (opSymbol == "-")
          return std::to_string(*lhsConst - *rhsConst);
        if (opSymbol == "*")
          return std::to_string(*lhsConst * *rhsConst);
        if (opSymbol == "/" && *rhsConst != 0)
          return std::to_string(*lhsConst / *rhsConst);
        if (opSymbol == "%" && *rhsConst != 0)
          return std::to_string(*lhsConst % *rhsConst);
      }

      if (opSymbol == "+") {
        if (rhsConst && *rhsConst == 0)
          return lhsExpr;
        if (lhsConst && *lhsConst == 0)
          return rhsExpr;
      } else if (opSymbol == "-") {
        if (rhsConst && *rhsConst == 0)
          return lhsExpr;
      } else if (opSymbol == "*") {
        if ((lhsConst && *lhsConst == 0) || (rhsConst && *rhsConst == 0))
          return std::string("0");
        if (rhsConst && *rhsConst == 1)
          return lhsExpr;
        if (lhsConst && *lhsConst == 1)
          return rhsExpr;
      } else if (opSymbol == "/") {
        if (lhsConst && *lhsConst == 0)
          return std::string("0");
        if (rhsConst && *rhsConst == 1)
          return lhsExpr;
      } else if (opSymbol == "%") {
        if (lhsConst && *lhsConst == 0)
          return std::string("0");
        if (rhsConst && *rhsConst == 1)
          return std::string("0");
      }

      return makeBinaryExpr(*lhsExpr, *rhsExpr, opSymbol);
    };

    if (auto op = value.getDefiningOp<arith::AddIOp>())
      return emitBinaryExpr(op.getLhs(), op.getRhs(), "+");
    if (auto op = value.getDefiningOp<arith::SubIOp>())
      return emitBinaryExpr(op.getLhs(), op.getRhs(), "-");
    if (auto op = value.getDefiningOp<arith::MulIOp>())
      return emitBinaryExpr(op.getLhs(), op.getRhs(), "*");
    if (auto op = value.getDefiningOp<arith::DivSIOp>())
      return emitBinaryExpr(op.getLhs(), op.getRhs(), "/");
    if (auto op = value.getDefiningOp<arith::DivUIOp>())
      return emitBinaryExpr(op.getLhs(), op.getRhs(), "/");
    if (auto op = value.getDefiningOp<arith::RemSIOp>())
      return emitBinaryExpr(op.getLhs(), op.getRhs(), "%");
    if (auto op = value.getDefiningOp<arith::RemUIOp>())
      return emitBinaryExpr(op.getLhs(), op.getRhs(), "%");
    auto emitCeilDivExpr = [&](Value lhs,
                               Value rhs) -> std::optional<std::string> {
      auto lhsExpr = buildIndexExpr(lhs, knownExprs);
      auto rhsExpr = buildIndexExpr(rhs, knownExprs);
      if (!lhsExpr || !rhsExpr)
        return std::nullopt;

      auto lhsConst = evaluateConstInt(lhs);
      auto rhsConst = evaluateConstInt(rhs);
      if (lhsConst && rhsConst && *rhsConst > 0)
        return std::to_string((*lhsConst + *rhsConst - 1) / *rhsConst);
      if (lhsConst && *lhsConst == 0)
        return std::string("0");
      if (rhsConst && *rhsConst == 1)
        return lhsExpr;
      return makeBinaryExpr(
          makeBinaryExpr(*lhsExpr, makeBinaryExpr(*rhsExpr, "1", "-"), "+"),
          *rhsExpr, "/");
    };
    if (auto op = value.getDefiningOp<arith::CeilDivSIOp>())
      return emitCeilDivExpr(op.getLhs(), op.getRhs());
    if (auto op = value.getDefiningOp<arith::CeilDivUIOp>())
      return emitCeilDivExpr(op.getLhs(), op.getRhs());

    return std::nullopt;
  }

  void collectParallelIvExprs() {
    parallelIvExprByValue.clear();
    mlirCoreExtentX = std::nullopt;
    mlirCoreExtentY = std::nullopt;

    hostFunc.walk([&](scf::ParallelOp op) {
      auto mappedDims = op->getAttrOfType<ArrayAttr>("loom.physical_dims");
      if (!mappedDims)
        mappedDims = op->getAttrOfType<ArrayAttr>("loom.mapped_to_dims");
      if (!mappedDims)
        return;

      auto iterTypes = op->getAttrOfType<ArrayAttr>("loom.iter_types");
      auto logicalLevels = op->getAttrOfType<ArrayAttr>("loom.logical_levels");

      struct AxisComponent {
        size_t idx = 0;
        int64_t logicalLevel = 0;
      };

      SmallVector<AxisComponent, 4> xComponents;
      SmallVector<AxisComponent, 4> yComponents;

      auto getLogicalLevelForIndex = [&](size_t idx) -> int64_t {
        if (!logicalLevels || idx >= logicalLevels.size())
          return static_cast<int64_t>(idx);
        if (auto intAttr = dyn_cast<IntegerAttr>(logicalLevels[idx]))
          return intAttr.getInt();
        return static_cast<int64_t>(idx);
      };

      ValueRange lbs = op.getLowerBound();
      ValueRange ubs = op.getUpperBound();
      ValueRange steps = op.getStep();

      for (size_t idx = 0; idx < op.getInductionVars().size(); ++idx) {
        if (idx >= mappedDims.size() || idx >= lbs.size() || idx >= ubs.size() ||
            idx >= steps.size())
          break;
        if (iterTypes && idx < iterTypes.size() &&
            !isSpatialIterAttr(iterTypes[idx]))
          continue;

        std::string dimName = extractDimName(mappedDims[idx]);
        StringRef dim = StringRef(dimName).trim();
        if (dim.starts_with("@"))
          dim = dim.drop_front();

        if (dim.equals_insensitive("x") || dim.equals_insensitive("dim_x")) {
          xComponents.push_back(AxisComponent{idx, getLogicalLevelForIndex(idx)});
          continue;
        }
        if (dim.equals_insensitive("y") || dim.equals_insensitive("dim_y")) {
          yComponents.push_back(AxisComponent{idx, getLogicalLevelForIndex(idx)});
          continue;
        }
      }

      auto sortByLevelThenIndex = [](SmallVectorImpl<AxisComponent> &components) {
        std::stable_sort(
            components.begin(), components.end(),
            [](const AxisComponent &a, const AxisComponent &b) {
              if (a.logicalLevel != b.logicalLevel)
                return a.logicalLevel < b.logicalLevel;
              return a.idx < b.idx;
            });
      };

      sortByLevelThenIndex(xComponents);
      sortByLevelThenIndex(yComponents);

      auto collectAxisExtent = [&](ArrayRef<AxisComponent> components,
                                   std::optional<int64_t> &axisExtent) {
        if (components.empty())
          return;

        int64_t extentProduct = 1;
        for (const AxisComponent &component : components) {
          size_t idx = component.idx;
          if (idx >= lbs.size() || idx >= ubs.size() || idx >= steps.size())
            return;

          auto lbConst = evaluateConstInt(lbs[idx]);
          auto ubConst = evaluateConstInt(ubs[idx]);
          auto stepConst = evaluateConstInt(steps[idx]);
          if (!lbConst || !ubConst || !stepConst || *stepConst <= 0)
            return;

          int64_t spanConst = *ubConst - *lbConst;
          int64_t extentConst = (spanConst + *stepConst - 1) / *stepConst;
          if (extentConst <= 0)
            return;

          extentProduct *= extentConst;
        }

        if (!axisExtent || extentProduct > *axisExtent)
          axisExtent = extentProduct;
      };
      collectAxisExtent(xComponents, mlirCoreExtentX);
      collectAxisExtent(yComponents, mlirCoreExtentY);

      auto buildAxisIvExprs = [&](ArrayRef<AxisComponent> components,
                                  StringRef axisExpr) {
        std::string strideExpr = "1";
        std::optional<int64_t> strideConst = int64_t{1};
        for (const AxisComponent &component : components) {
          size_t idx = component.idx;
          auto lbExpr = buildIndexExpr(lbs[idx], parallelIvExprByValue);
          auto ubExpr = buildIndexExpr(ubs[idx], parallelIvExprByValue);
          auto stepExpr = buildIndexExpr(steps[idx], parallelIvExprByValue);
          if (!lbExpr || !ubExpr || !stepExpr)
            continue;

          auto lbConst = evaluateConstInt(lbs[idx]);
          auto ubConst = evaluateConstInt(ubs[idx]);
          auto stepConst = evaluateConstInt(steps[idx]);

          std::optional<int64_t> extentConst;
          std::string extentExpr;
          if (lbConst && ubConst && stepConst && *stepConst > 0) {
            int64_t spanConst = *ubConst - *lbConst;
            extentConst = (spanConst + *stepConst - 1) / *stepConst;
            extentExpr = std::to_string(*extentConst);
          } else {
            extentExpr = makeBinaryExpr(
                makeBinaryExpr(makeBinaryExpr(*ubExpr, *lbExpr, "-"),
                               makeBinaryExpr(*stepExpr, "1", "-"), "+"),
                *stepExpr, "/");
          }

          std::string quotientExpr = (strideConst && *strideConst == 1)
                                         ? axisExpr.str()
                                         : makeBinaryExpr(axisExpr, strideExpr, "/");

          std::string digitExpr = (extentConst && *extentConst == 1)
                                      ? "0"
                                      : makeBinaryExpr(quotientExpr, extentExpr, "%");
          std::string scaledExpr;
          if (stepConst && *stepConst == 1) {
            scaledExpr = digitExpr;
          } else if (stepConst && *stepConst == 0) {
            scaledExpr = "0";
          } else {
            scaledExpr = makeBinaryExpr(digitExpr, *stepExpr, "*");
          }

          std::string ivExpr;
          if (lbConst && *lbConst == 0)
            ivExpr = scaledExpr;
          else if (scaledExpr == "0")
            ivExpr = *lbExpr;
          else
            ivExpr = makeBinaryExpr(*lbExpr, scaledExpr, "+");

          parallelIvExprByValue[op.getInductionVars()[idx]] = ivExpr;
          if (strideConst && extentConst) {
            *strideConst *= *extentConst;
            strideExpr = std::to_string(*strideConst);
          } else {
            strideConst = std::nullopt;
            strideExpr = makeBinaryExpr(strideExpr, extentExpr, "*");
          }
        }
      };

      buildAxisIvExprs(xComponents, "core.x");
      buildAxisIvExprs(yComponents, "core.y");
    });
  }
  static FailureOr<std::string> buildTilesExpr(MemRefType memrefType,
                                               Location loc) {
    std::string expr;
    auto shape = memrefType.getShape();
    const size_t rank = shape.size();

    if (rank == 0)
      return emitError(loc)
             << "rank-0 memrefs do not carry tile geometry information";

    auto appendOne = [&](int64_t dim) -> bool {
      if (dim == ShapedType::kDynamic) return false;
      dim = std::max<int64_t>(32, dim);
      expr += std::to_string(dim);
      expr += " / TILE_HEIGHT";
      return true;
    };

    // Rank-1 memrefs are treated as [32, max(32, original_dim)].
    if (rank == 1) {
      if (!appendOne(32))
        return emitError(loc) << "dynamic memref dimension is unsupported in "
                                 "host tile-count expression";
      expr += " * ";
      if (!appendOne(shape[0]))
        return emitError(loc) << "dynamic memref dimension is unsupported in "
                                 "host tile-count expression";
      return expr;
    }

    // Dims before the last two: multiply them directly.
    for (size_t i = 0; i + 2 < rank; ++i) {
      int64_t dim = shape[i];
      if (dim == ShapedType::kDynamic)
        return emitError(loc) << "dynamic memref dimension is unsupported in "
                                 "host tile-count expression";
      if (!expr.empty())
        expr += " * ";
      expr += std::to_string(dim);
    }

    // Last two dims: keep the existing tiling logic (rank-2 and rank-1).
    if (!expr.empty())
      expr += " * ";
    if (!appendOne(shape[rank - 2]))
      return emitError(loc) << "dynamic memref dimension is unsupported in "
                               "host tile-count expression";
    expr += " * ";
    if (!appendOne(shape[rank - 1]))
      return emitError(loc) << "dynamic memref dimension is unsupported in "
                               "host tile-count expression";

    if (expr.empty())
      return emitError(loc) << "failed to build host tile-count expression";
    return expr;
  }

  static FailureOr<std::string> buildElementByteSizeExpr(Type elementType,
                                                         Location loc) {
    if (elementType.isBF16())
      return std::string("sizeof(bfloat16)");
    if (elementType.isF16())
      return std::string("sizeof(uint16_t)");
    if (elementType.isF32())
      return std::string("sizeof(float)");
    if (elementType.isF64())
      return std::string("sizeof(double)");
    if (elementType.isIndex())
      return std::string("sizeof(std::size_t)");

    if (auto intType = dyn_cast<IntegerType>(elementType)) {
      unsigned bytes = std::max<unsigned>(1, (intType.getWidth() + 7) / 8);
      return std::to_string(bytes);
    }

    return emitError(loc)
           << "unsupported memref element type in host byte-size expression: "
           << elementType;
  }

  static FailureOr<std::string> buildMemrefByteSizeExpr(MemRefType memrefType,
                                                        Location loc) {
    if (!memrefType.hasStaticShape())
      return emitError(loc) << "dynamic memref dimension is unsupported in "
                               "host byte-size expression";

    FailureOr<std::string> elementBytes =
        buildElementByteSizeExpr(memrefType.getElementType(), loc);
    if (failed(elementBytes))
      return failure();

    std::string expr = *elementBytes;
    for (int64_t dim : memrefType.getShape()) {
      if (dim == ShapedType::kDynamic)
        return emitError(loc) << "dynamic memref dimension is unsupported in "
                                 "host byte-size expression";
      if (dim < 0)
        return emitError(loc) << "negative memref dimension is unsupported in "
                                 "host byte-size expression";
      expr += " * " + std::to_string(dim);
    }
    return expr;
  }

  static std::string stringifyAttr(Attribute attr) {
    std::string text;
    llvm::raw_string_ostream os(text);
    attr.print(os);
    os.flush();
    return text;
  }

  static bool isSpatialIterAttr(Attribute iterTypeAttr) {
    std::string text = stringifyAttr(iterTypeAttr);
    return StringRef(text).contains("spatial");
  }

  static std::string extractDimName(Attribute dimAttr) {
    if (auto flat = dyn_cast<FlatSymbolRefAttr>(dimAttr))
      return flat.getValue().str();
    if (auto sym = dyn_cast<SymbolRefAttr>(dimAttr))
      return sym.getLeafReference().str();
    if (auto str = dyn_cast<StringAttr>(dimAttr))
      return str.getValue().str();

    std::string text = stringifyAttr(dimAttr);
    StringRef ref(text);
    if (ref.starts_with("@"))
      ref = ref.drop_front();
    return ref.trim().str();
  }

  static std::string normalizeDimName(StringRef dim) {
    StringRef t = dim.trim();
    if (t.starts_with("@"))
      t = t.drop_front();
    return t.lower();
  }

  static std::string coreCoordExprForDim(StringRef dim) {
    std::string d = normalizeDimName(dim);
    if (d == "x")
      return "core.x";
    if (d == "y")
      return "core.y";
    return {};
  }

  void inferCoreCoordArgOrder() {
    // SCF lowering now materializes exactly two physical core compile args in
    // fixed order: [x, y]. Keep host runtime-arg emission aligned.
    coreCoordArg0Expr = "core.x";
    coreCoordArg1Expr = "core.y";
  }

  static std::optional<std::pair<int64_t, int64_t>>
  parseBroadcastAttr(ArrayAttr broadcastAttr) {
    if (!broadcastAttr || broadcastAttr.size() < 2)
      return std::nullopt;

    auto xAttr = dyn_cast<IntegerAttr>(broadcastAttr[0]);
    auto yAttr = dyn_cast<IntegerAttr>(broadcastAttr[1]);
    if (!xAttr || !yAttr)
      return std::nullopt;

    return std::make_pair(xAttr.getInt(), yAttr.getInt());
  }

  static MulticastKind classifyBroadcast(ArrayAttr broadcastAttr) {
    auto parsed = parseBroadcastAttr(broadcastAttr);
    if (!parsed)
      return MulticastKind::None;

    const int64_t xBroadcast = parsed->first;
    const int64_t yBroadcast = parsed->second;

    // Keep compatibility with existing host runtime-arg conventions:
    //   [1, 8] => horizontal multicast
    //   [8, 1] => vertical multicast
    const bool hasHorizontal = yBroadcast > 1;
    const bool hasVertical = xBroadcast > 1;

    if (hasHorizontal && hasVertical)
      return MulticastKind::All;
    if (hasHorizontal)
      return MulticastKind::Horizontal;
    if (hasVertical)
      return MulticastKind::Vertical;
    return MulticastKind::None;
  }

  MulticastKind findInputMulticastKind(Value hostArg) {
    MulticastKind result = MulticastKind::None;
    hostFunc.walk([&](::loom::CopyOp op) {
      auto sourceRC = op.getSource().getDefiningOp<memref::ReinterpretCastOp>();
      if (!sourceRC)
        return;
      Value sourceArg = stripCasts(sourceRC.getSource());
      if (sourceArg != hostArg)
        return;

      MulticastKind opKind = classifyBroadcast(op.getBroadcastAttr());
      if (opKind != MulticastKind::None)
        result = opKind;
    });
    return result;
  }

  DramBufferInfo *findDramInfoMutable(Value hostArg) {
    for (DramBufferInfo &info : dramInfos) {
      if (info.hostArg == hostArg)
        return &info;
    }
    return nullptr;
  }

  int resolveCbIndexFromEndpoint(Value endpoint) const {
    Value current = stripCasts(endpoint);

    if (auto sem = current.getDefiningOp<::loom::SemaphoreTakeOp>()) {
      auto semIt = semaphoreToCbIndex.find(sem.getResult());
      if (semIt != semaphoreToCbIndex.end())
        return static_cast<int>(semIt->second);
      current = stripCasts(sem.getSource());
    }

    current = stripLoomSemaphores(stripCasts(current));
    auto alloc = current.getDefiningOp<::loom::AllocOp>();
    if (!alloc)
      return -1;

    auto allocIt = allocToCbInfo.find(alloc.getResult());
    if (allocIt == allocToCbInfo.end())
      return -1;

    const CircularBufferInfo &cbInfo = cbInfos[allocIt->second];
    if (cbInfo.cbIndices.empty())
      return -1;
    return static_cast<int>(cbInfo.cbIndices.front());
  }

  void collectDramInfos() {
    dramInfos.clear();
    scalarRuntimeHostArgs.clear();
    hostArgKinds.clear();
    SmallVector<Value, 8> loadMemrefs;
    SmallVector<Value, 8> storeMemrefs;

    hostFunc.walk([&](::loom::CopyOp op) {
      if (auto sourceRC = op.getSource().getDefiningOp<memref::ReinterpretCastOp>())
        appendUnique(loadMemrefs, stripCasts(sourceRC.getSource()));
      if (auto destRC = op.getDestination().getDefiningOp<memref::ReinterpretCastOp>())
        appendUnique(storeMemrefs, stripCasts(destRC.getSource()));
    });

    unsigned srcIdx = 0;
    unsigned dstIdx = 0;
    unsigned ioIdx = 0;
    unsigned argIdx = 0;
    unsigned inputIdx = 0;
    unsigned inputTensorIdx = 0;
    unsigned outputTensorIdx = 0;

    for (auto [index, arg] : llvm::enumerate(originalFunc.getArguments())) {
      if (isScalarRuntimeSiteArg(originalFunc, arg))
        continue;

      auto memrefType = dyn_cast<MemRefType>(arg.getType());
      if (!memrefType)
        continue;
      if (isScalarMemRefType(memrefType) && memrefType.getRank() == 0)
        continue;

      BlockArgument hostArgBlock =
          dyn_cast<BlockArgument>(stripCasts(hostFunc.getArgument(index)));
      if (hostArgBlock && isScalarOnlyRuntimeSourceArg(hostFunc, hostArgBlock)) {
        unsigned hostParamIndex = static_cast<unsigned>(hostArgKinds.size());
        std::string valuesName =
            "src" + std::to_string(static_cast<unsigned>(index)) +
            "_scalar_values";
        scalarRuntimeHostArgs.push_back(ScalarRuntimeHostArgInfo{
            static_cast<unsigned>(index), hostParamIndex, valuesName});
        hostArgKinds.push_back(HostArgKind::Scalar);
        continue;
      }

      FailureOr<std::string> sizeExpr =
          buildMemrefByteSizeExpr(memrefType, loc);
      if (failed(sizeExpr)) {
        status = failure();
        continue;
      }

      Value hostArg = stripCasts(hostFunc.getArgument(index));
      bool isLoad = containsValue(loadMemrefs, hostArg);
      bool isStore = containsValue(storeMemrefs, hostArg);

      std::string roleName;
      if (isLoad && !isStore)
        roleName = "src" + std::to_string(srcIdx++);
      else if (isStore && !isLoad)
        roleName = "dst" + std::to_string(dstIdx++);
      else if (isLoad && isStore)
        roleName = "io" + std::to_string(ioIdx++);
      else
        roleName = "arg" + std::to_string(argIdx++);

      std::string letter = getLetterName(dramInfos.size());
      bool isInput = isLoad;
      bool isOutput = isStore;
      int inputTensorOrdinal =
          isInput ? static_cast<int>(inputTensorIdx++) : -1;
      int outputTensorOrdinal =
          isOutput ? static_cast<int>(outputTensorIdx++) : -1;
      std::string inputName =
          isInput ? ("in" + std::to_string(inputIdx++)) : "";

      dramInfos.push_back(DramBufferInfo{
          hostArg,
          memrefType,
          static_cast<unsigned>(index),
          static_cast<unsigned>(hostArgKinds.size()),
          static_cast<unsigned>(dramInfos.size()),
          inputTensorOrdinal,
          outputTensorOrdinal,
          isInput,
          isOutput,
          inputName,
          "dram_config_" + letter,
          roleName + "_dram_buffer",
          *sizeExpr});
      hostArgKinds.push_back(HostArgKind::Memref);
    }
  }

  void collectCopyBindings() {
    copyBindings.clear();
    if (parallelIvExprByValue.empty())
      collectParallelIvExprs();

    SmallVector<::loom::CopyOp, 8> orderedBindings;
    if (failed(collectCopyBindingOps(hostFunc, orderedBindings))) {
      hostFunc.emitError("failed to collect annotated DRAM/L1 copy bindings");
      status = failure();
      return;
    }

    copyBindings.reserve(orderedBindings.size());
    for (::loom::CopyOp copyOp : orderedBindings) {
      auto slot = getCopyBindingSlot(copyOp.getOperation());
      if (!slot) {
        status = copyOp.emitOpError("missing required copy binding slot");
        continue;
      }

      Value dramMemref = getCopyBindingDramMemref(copyOp);
      DramBufferInfo *dramInfo = findDramInfoMutable(dramMemref);
      if (!dramInfo) {
        status = copyOp.emitOpError()
                 << "failed to resolve DRAM buffer info for copy binding";
        continue;
      }

      CopyBindingInfo info;
      info.slot = static_cast<unsigned>(*slot);
      info.dramInfoIndex = static_cast<unsigned>(dramInfo - dramInfos.data());
      info.isInput = isDramToL1Copy(copyOp);
      info.isOutput = isL1ToDramCopy(copyOp);
      info.l1Endpoint = getCopyBindingL1Endpoint(copyOp);
      info.bindingName = "binding" + std::to_string(*slot);

      if (auto cbMemrefType = getCopyBindingCBMemrefType(copyOp)) {
        FailureOr<std::string> cbTilesExpr =
            buildTilesExpr(cbMemrefType, copyOp.getLoc());
        if (failed(cbTilesExpr)) {
          status = failure();
          continue;
        }
        info.cbTilesExpr = *cbTilesExpr;
      } else {
        status = copyOp.emitOpError()
                 << "failed to infer L1 CB memref type for copy binding";
        continue;
      }

      if (info.isInput) {
        auto parsedBroadcast = parseBroadcastAttr(copyOp.getBroadcastAttr());
        if (parsedBroadcast &&
            (parsedBroadcast->first > 1 || parsedBroadcast->second > 1)) {
          Value ulX = copyOp.getUlX();
          Value ulY = copyOp.getUlY();
          Value lrX = copyOp.getLrX();
          Value lrY = copyOp.getLrY();
          if (ulX && ulY && lrX && lrY) {
            auto ulXExpr = buildIndexExpr(ulX, parallelIvExprByValue);
            auto ulYExpr = buildIndexExpr(ulY, parallelIvExprByValue);
            auto lrXExpr = buildIndexExpr(lrX, parallelIvExprByValue);
            auto lrYExpr = buildIndexExpr(lrY, parallelIvExprByValue);
            if (ulXExpr && ulYExpr && lrXExpr && lrYExpr) {
              info.hasBroadcastRegion = true;
              info.ulXExpr = *ulXExpr;
              info.ulYExpr = *ulYExpr;
              info.lrXExpr = *lrXExpr;
              info.lrYExpr = *lrYExpr;
            }
          }
        }
      }

      copyBindings.push_back(info);
    }
  }

  unsigned getOrCreateScalarRuntimeHostArg(unsigned sourceArgIndex) {
    for (auto [ordinal, info] : llvm::enumerate(scalarRuntimeHostArgs)) {
      if (info.sourceArgIndex == sourceArgIndex)
        return static_cast<unsigned>(ordinal);
    }

    unsigned hostParamIndex = static_cast<unsigned>(hostArgKinds.size());
    std::string valuesName =
        "src" + std::to_string(sourceArgIndex) + "_scalar_values";
    scalarRuntimeHostArgs.push_back(
        ScalarRuntimeHostArgInfo{sourceArgIndex, hostParamIndex, valuesName});
    hostArgKinds.push_back(HostArgKind::Scalar);
    return static_cast<unsigned>(scalarRuntimeHostArgs.size() - 1);
  }

  SmallVector<ScalarRuntimeListDimHostInfo, 4>
  collectScalarRuntimeListDims(Operation *anchor, int64_t siteId) const {
    SmallVector<ScalarRuntimeListDimHostInfo, 4> dims;
    for (Operation *parent = anchor ? anchor->getParentOp() : nullptr; parent;
         parent = parent->getParentOp()) {
      auto forOp = dyn_cast<scf::ForOp>(parent);
      if (!forOp)
        continue;

      auto attr =
          forOp->getAttrOfType<DenseI64ArrayAttr>(kScalarSiteListDimsAttrName);
      if (!attr)
        continue;

      ArrayRef<int64_t> values = attr.asArrayRef();
      for (size_t idx = 0; idx + 4 < values.size(); idx += 5) {
        if (values[idx] != siteId)
          continue;
        dims.push_back(ScalarRuntimeListDimHostInfo{
            forOp.getInductionVar(), values[idx + 1], values[idx + 2],
            values[idx + 3], values[idx + 4]});
      }
    }

    std::stable_sort(
        dims.begin(), dims.end(),
        [](const ScalarRuntimeListDimHostInfo &a,
           const ScalarRuntimeListDimHostInfo &b) {
          return a.dimOrdinal < b.dimOrdinal;
        });
    return dims;
  }

  FailureOr<int64_t>
  getScalarRuntimeSiteSize(int64_t siteId,
                           ArrayRef<ScalarRuntimeListDimHostInfo> dims,
                           Location loc) const {
    if (auto sizesAttr =
            hostFunc->getAttrOfType<DenseI64ArrayAttr>(kScalarSiteSizesAttrName)) {
      ArrayRef<int64_t> sizes = sizesAttr.asArrayRef();
      if (siteId < 0 || siteId >= static_cast<int64_t>(sizes.size()))
        return emitError(loc) << "missing scalar runtime list size for site "
                              << siteId;
      if (sizes[siteId] <= 0)
        return emitError(loc) << "invalid scalar runtime list size "
                              << sizes[siteId] << " for site " << siteId;
      return sizes[siteId];
    }

    int64_t siteSize = 1;
    for (const ScalarRuntimeListDimHostInfo &dim : dims) {
      if (dim.extent <= 0)
        return emitError(loc) << "invalid scalar runtime list extent "
                              << dim.extent << " for site " << siteId;
      if (siteSize > std::numeric_limits<int64_t>::max() / dim.extent)
        return emitError(loc)
               << "scalar runtime list size overflow for site " << siteId;
      siteSize *= dim.extent;
    }
    return siteSize;
  }

  FailureOr<SmallVector<std::string, 4>>
  buildScalarRuntimeOffsetExprs(
      memref::ReinterpretCastOp sourceRC,
      ArrayRef<ScalarRuntimeListDimHostInfo> dims, int64_t siteSize) const {
    SmallVector<std::string, 4> offsetExprs;

    Value offsetValue;
    auto offsets = sourceRC.getOffsets();
    if (!offsets.empty()) {
      offsetValue = offsets.front();
    } else {
      auto mixedOffsets = sourceRC.getMixedOffsets();
      if (!mixedOffsets.empty() && isa<Attribute>(mixedOffsets.front())) {
        auto intAttr =
            dyn_cast<IntegerAttr>(cast<Attribute>(mixedOffsets.front()));
        if (intAttr) {
          offsetExprs.push_back(std::to_string(intAttr.getInt()));
          return offsetExprs;
        }
      }
      return emitError(sourceRC.getLoc())
             << "scalar runtime site requires a dynamic or static offset";
    }

    for (int64_t elementIndex = 0; elementIndex < siteSize; ++elementIndex) {
      IndexExprMap knownExprs = parallelIvExprByValue;
      int64_t remaining = elementIndex;
      SmallVector<int64_t, 4> digits(dims.size(), 0);
      for (int64_t dimIdx = static_cast<int64_t>(dims.size()) - 1; dimIdx >= 0;
           --dimIdx) {
        int64_t extent = dims[dimIdx].extent;
        if (extent <= 0)
          return emitError(sourceRC.getLoc())
                 << "invalid scalar runtime list extent " << extent;
        digits[dimIdx] = remaining % extent;
        remaining /= extent;
      }

      for (auto [dim, digit] : llvm::zip(dims, digits)) {
        int64_t ivValue = dim.lb + digit * dim.step;
        knownExprs[dim.iv] = std::to_string(ivValue);
      }

      auto offsetExpr = buildIndexExpr(offsetValue, knownExprs);
      if (!offsetExpr)
        return emitError(sourceRC.getLoc())
               << "failed to build offset expression for scalar runtime site";
      offsetExprs.push_back(*offsetExpr);
    }

    return offsetExprs;
  }

  void collectScalarRuntimeSites() {
    scalarRuntimeSites.clear();
    if (parallelIvExprByValue.empty())
      collectParallelIvExprs();

    hostFunc.walk([&](::loom::CopyOp op) {
      auto siteAttr = op->getAttrOfType<IntegerAttr>(kScalarSiteIdAttrName);
      if (!siteAttr)
        return;

      auto sourceRC = op.getSource().getDefiningOp<memref::ReinterpretCastOp>();
      if (!sourceRC)
        return;

      Value memrefArg = stripCasts(sourceRC.getSource());
      auto blockArg = dyn_cast<BlockArgument>(memrefArg);
      if (!blockArg || blockArg.getOwner() != &hostFunc.front())
        return;
      unsigned sourceArgIndex = blockArg.getArgNumber();

      SmallVector<ScalarRuntimeListDimHostInfo, 4> dims =
          collectScalarRuntimeListDims(op.getOperation(), siteAttr.getInt());
      FailureOr<int64_t> siteSize =
          getScalarRuntimeSiteSize(siteAttr.getInt(), dims, op.getLoc());
      if (failed(siteSize)) {
        status = failure();
        return;
      }

      auto offsetExprs = buildScalarRuntimeOffsetExprs(sourceRC, dims, *siteSize);
      if (failed(offsetExprs)) {
        status = failure();
        return;
      }
      if (static_cast<int64_t>(offsetExprs->size()) != *siteSize) {
        op.emitError() << "scalar runtime site " << siteAttr.getInt()
                       << " expected " << *siteSize
                       << " offsets but built " << offsetExprs->size();
        status = failure();
        return;
      }

      unsigned scalarHostArgOrdinal =
          getOrCreateScalarRuntimeHostArg(sourceArgIndex);
      const ScalarRuntimeHostArgInfo &hostArgInfo =
          scalarRuntimeHostArgs[scalarHostArgOrdinal];
      scalarRuntimeSites.push_back(ScalarRuntimeSiteInfo{
          siteAttr.getInt(),
          sourceArgIndex,
          scalarHostArgOrdinal,
          *offsetExprs,
          "",
          "",
          hostArgInfo.valuesName});
    });

    std::stable_sort(scalarRuntimeSites.begin(), scalarRuntimeSites.end(),
                     [](const ScalarRuntimeSiteInfo &a,
                        const ScalarRuntimeSiteInfo &b) {
                       return a.siteId < b.siteId;
                     });
  }

  void collectReduceRegion() {
    reduceRegion = ReduceRegionInfo{};

    if (!hasReduce)
      return;
    if (parallelIvExprByValue.empty())
      collectParallelIvExprs();

    hostFunc.walk([&](::loom::GatherOp op) {
      if (reduceRegion.hasRegion)
        return;

      Value ulX = op.getUlX();
      Value ulY = op.getUlY();
      Value lrX = op.getLrX();
      Value lrY = op.getLrY();
      if (!ulX || !ulY || !lrX || !lrY)
        return;

      auto ulXExpr = buildIndexExpr(ulX, parallelIvExprByValue);
      auto ulYExpr = buildIndexExpr(ulY, parallelIvExprByValue);
      auto lrXExpr = buildIndexExpr(lrX, parallelIvExprByValue);
      auto lrYExpr = buildIndexExpr(lrY, parallelIvExprByValue);
      if (!ulXExpr || !ulYExpr || !lrXExpr || !lrYExpr)
        return;

      reduceRegion.hasRegion = true;
      reduceRegion.ulXExpr = *ulXExpr;
      reduceRegion.ulYExpr = *ulYExpr;
      reduceRegion.lrXExpr = *lrXExpr;
      reduceRegion.lrYExpr = *lrYExpr;
    });
  }

  void collectCbInfos() {
    cbInfos.clear();
    semaphoreToCbIndex.clear();
    allocToCbInfo.clear();
    internalSlotToCbIndex.clear();
    nextCbIndex = 0;

    llvm::DenseMap<Value, SmallVector<Value, 4>> semaphoresByAlloc;
    hostFunc.walk([&](::loom::SemaphoreTakeOp sem) {
      Value base = stripLoomSemaphores(stripCasts(sem.getSource()));
      if (!base.getDefiningOp<::loom::AllocOp>())
        return;
      semaphoresByAlloc[base].push_back(sem.getResult());
    });

    hostFunc.walk([&](::loom::AllocOp alloc) {
      auto cbMemrefType = dyn_cast<MemRefType>(alloc.getType());
      if (!cbMemrefType)
        return;

      CircularBufferInfo info;
      FailureOr<std::string> tilesExpr =
          buildTilesExpr(cbMemrefType, alloc.getLoc());
      if (failed(tilesExpr)) {
        status = failure();
        return;
      }
      info.tilesExpr = *tilesExpr;
      info.allocValue = alloc.getResult();
      info.tilesVarName =
          "cb_tiles_per_block_" + std::to_string(cbInfos.size());

      auto semIt = semaphoresByAlloc.find(alloc.getResult());
      if (semIt != semaphoresByAlloc.end()) {
        for (Value semValue : semIt->second) {
          auto sem = semValue.getDefiningOp<::loom::SemaphoreTakeOp>();
          if (!sem) {
            status = alloc.emitOpError()
                     << "failed to resolve semaphore for static CB index";
            return;
          }
          auto cbIndexAttr = sem->getAttrOfType<IntegerAttr>(kCBIndexAttrName);
          if (!cbIndexAttr) {
            status = sem.emitOpError("missing static CB index metadata");
            return;
          }
          if (cbIndexAttr.getInt() < 0) {
            status = sem.emitOpError("has negative static CB index");
            return;
          }
          unsigned cbIndex = static_cast<unsigned>(cbIndexAttr.getInt());
          nextCbIndex = std::max(nextCbIndex, cbIndex + 1);
          info.cbIndices.push_back(cbIndex);
          info.cbIndexNames.push_back("CBIndex::c_" + std::to_string(cbIndex));
          semaphoreToCbIndex[semValue] = cbIndex;
          if (auto slotAttr =
                  sem->getAttrOfType<IntegerAttr>(kSemaphoreSlotAttrName))
            internalSlotToCbIndex.try_emplace(slotAttr.getInt(), cbIndex);
        }
      }

      if (info.cbIndices.empty()) {
        status = alloc.emitOpError()
                 << "host CB emission requires at least one explicit "
                    "loom.semaphore_take for each loom.alloc";
        return;
      }

      allocToCbInfo[alloc.getResult()] = static_cast<unsigned>(cbInfos.size());
      cbInfos.push_back(info);
    });

    for (CopyBindingInfo &binding : copyBindings) {
      int cbIndex = resolveCbIndexFromEndpoint(binding.l1Endpoint);
      if (cbIndex < 0) {
        hostFunc.emitError()
            << "failed to resolve explicit CB index for copy binding "
            << binding.slot;
        status = failure();
        continue;
      }

      binding.cbIndex = cbIndex;
    }

  }

  void eraseNonHostOps() {
    SmallVector<Operation *, 16> opsToErase;
    hostFunc.walk([&](Operation *op) {
      if (isComputeOp(op)) {
        opsToErase.push_back(op);
      } else if (auto loomCopyOp = dyn_cast<::loom::CopyOp>(op)) {
        if (isLoomLoadOp(loomCopyOp) || isLoomStoreOp(loomCopyOp))
          opsToErase.push_back(op);
      }
    });

    for (Operation *op : llvm::reverse(opsToErase))
      op->erase();

    SmallVector<::loom::AllocOp, 8> allocOps;
    hostFunc.walk([&](::loom::AllocOp alloc) { allocOps.push_back(alloc); });
    for (::loom::AllocOp alloc : allocOps) {
      if (alloc.getResult().use_empty())
        alloc.erase();
    }
  }

  void eraseDeadReinterpretCasts() {
    SmallVector<Operation *, 8> unusedRcOps;
    hostFunc.walk([&](memref::ReinterpretCastOp rcOp) {
      if (rcOp.getResult().use_empty())
        unusedRcOps.push_back(rcOp);
    });
    for (Operation *op : unusedRcOps)
      op->erase();
  }

  /**
   * @brief Remove cloned mesh `scf.parallel` trees from the host clone.
   *
   * @details Host helpers are filled in with `emitc.verbatim` Metal bootstrap;
   *          the cloned spatial IR must not reach the main TileLoom→TTKernel
   *          conversion, or `ConvertSCFParallelOp` will emit
   *          `ttkernel.get_arg_val` (device runtime slots) into `__host_*`
   *          functions. Erasing the mesh shell keeps host IR host-only.
   */
  void eraseHostSpatialMesh() {
    SmallVector<scf::ParallelOp, 8> parallels;
    hostFunc.walk([&](scf::ParallelOp p) { parallels.push_back(p); });
    auto depthFromFuncRoot = [&](Operation *op) -> unsigned {
      unsigned depth = 0;
      for (Operation *parent = op->getParentOp();
           parent && !isa<func::FuncOp>(parent); parent = parent->getParentOp())
        ++depth;
      return depth;
    };
    llvm::sort(parallels, [&](scf::ParallelOp a, scf::ParallelOp b) {
      return depthFromFuncRoot(a) > depthFromFuncRoot(b);
    });
    for (scf::ParallelOp p : parallels) {
      if (!p->getBlock())
        continue;
      p.erase();
    }
  }

  void emitLine(const std::string &line) {
    builder.create<emitc::VerbatimOp>(loc, line);
  }

  void emitPreamble() {
    // Host signature is emitted by MLIR->C++ translation with synthesized
    // parameter names v1..vN.
    // Argument order is:
    //   host_cpp:    [all memrefs] [start_core_x start_core_y end_core_x end_core_y] [device]
    //   host_pybind: [all memrefs] [scalar host tables]
    const size_t hostArgPrefixCount = hostArgKinds.size();
    const size_t startCoreXArg = hostArgPrefixCount + 1;
    const size_t startCoreYArg = hostArgPrefixCount + 2;
    const size_t endCoreXArg = hostArgPrefixCount + 3;
    const size_t endCoreYArg = hostArgPrefixCount + 4;

    if (kind == HostProgramKind::Pybind) {
      emitLine("IDevice* device = v1.device();");
      emitLine("Program program = CreateProgram();");
      emitLine("uint32_t start_core_x = 0u;");
      emitLine("uint32_t start_core_y = 0u;");
      emitLine("uint32_t end_core_x = " +
               std::to_string(*mlirCoreExtentX - 1) + "u;");
      emitLine("uint32_t end_core_y = " +
               std::to_string(*mlirCoreExtentY - 1) + "u;");
    } else {
      const size_t deviceArg = hostArgPrefixCount + 5;
      emitLine("IDevice* device = v" + std::to_string(deviceArg) + ";");
      emitLine("CommandQueue& cq = device->command_queue();");
      emitLine("Program program{};");
      emitLine("uint32_t start_core_x = v" + std::to_string(startCoreXArg) +
               ";");
      emitLine("uint32_t start_core_y = v" + std::to_string(startCoreYArg) +
               ";");
      emitLine("uint32_t end_core_x = v" + std::to_string(endCoreXArg) +
               ";");
      emitLine("uint32_t end_core_y = v" + std::to_string(endCoreYArg) +
               ";");
    }
    emitLine("CoreRangeSet all_cores{ CoreRange{ {start_core_x, start_core_y}, {end_core_x, end_core_y} } };");
    emitLine("constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;");
    emitLine("const auto cb_data_format = tt::DataFormat::Float16_b;");
    emitLine("uint32_t cb_buffer_depth = 2;");
  }

  void emitDramBuffers() {
    if (kind != HostProgramKind::Cpp)
      return;

    for (const DramBufferInfo &info : dramInfos) {
      emitLine("tt_metal::InterleavedBufferConfig " + info.configName +
               "{.device = device, .size = " + info.sizeExpr +
               ", .page_size = single_tile_size, .buffer_type = "
               "tt_metal::BufferType::DRAM};");
      emitLine("auto " + info.bufferName +
               " = tt_metal::CreateBuffer(" + info.configName + ");");
    }
  }

  void emitBoundBuffers() {
    if (kind != HostProgramKind::Pybind)
      return;

    for (const DramBufferInfo &info : dramInfos) {
      emitLine("auto* " + info.bufferName + " = v" +
               std::to_string(info.hostParamIndex + 1) + ".buffer();");
    }
  }

  void emitScalarHostValues() {
    for (const ScalarRuntimeHostArgInfo &info : scalarRuntimeHostArgs) {
      emitLine("const auto& " + info.valuesName + " = v" +
               std::to_string(info.hostParamIndex + 1) + ";");
    }
  }

  void emitBufferSetup() {
    if (kind == HostProgramKind::Cpp) {
      emitDramBuffers();
      emitScalarHostValues();
      return;
    }
    emitBoundBuffers();
    emitScalarHostValues();
  }

  bool hasEmittedTilesVar(StringRef name) const {
    return llvm::is_contained(emittedTilesVars, name.str());
  }

  /**
   * @brief Check whether a circular buffer is fed by a function input memref.
   *
   * @param cbInfo Circular-buffer metadata collected for host emission.
   * @return True when any CB index in @p cbInfo maps to an input DRAM buffer.
   */
  bool isInputCircularBuffer(const CircularBufferInfo &cbInfo) const {
    for (unsigned cbIndex : cbInfo.cbIndices) {
      for (const CopyBindingInfo &binding : copyBindings) {
        if (binding.cbIndex < 0)
          continue;
        if (static_cast<unsigned>(binding.cbIndex) != cbIndex)
          continue;
        if (binding.isInput)
          return true;
      }
    }
    return false;
  }

  void annotateHostSignatureMetadata() {
    Builder b(hostFunc.getContext());
    hostFunc->setAttr(
        "loom.host_memref_count",
        b.getI32IntegerAttr(static_cast<int32_t>(dramInfos.size())));
    hostFunc->setAttr(
        "loom.host_scalar_count",
        b.getI32IntegerAttr(
            static_cast<int32_t>(scalarRuntimeHostArgs.size())));
    SmallVector<Attribute, 8> argKindAttrs;
    argKindAttrs.reserve(hostArgKinds.size());
    for (HostArgKind argKind : hostArgKinds) {
      argKindAttrs.push_back(b.getStringAttr(
          argKind == HostArgKind::Scalar ? "scalar" : "memref"));
    }
    hostFunc->setAttr("loom.host_arg_kinds", b.getArrayAttr(argKindAttrs));
  }

  void emitCircularBuffers() {
    auto emitCbInfo = [&](const CircularBufferInfo &info) {
      if (info.cbIndexNames.empty())
        return;

      if (!hasEmittedTilesVar(info.tilesVarName)) {
        emitLine("const uint32_t " + info.tilesVarName + " = " + info.tilesExpr +
                 ";");
        emittedTilesVars.push_back(info.tilesVarName);
      }

      std::string cbEntries;
      for (auto [idx, cbIndexName] : llvm::enumerate(info.cbIndexNames)) {
        if (idx > 0)
          cbEntries += "}, {";
        cbEntries += cbIndexName + ", cb_data_format";
      }

      std::string setPageSizeChain;
      for (const std::string &cbIndexName : info.cbIndexNames)
        setPageSizeChain +=
            ".set_page_size(" + cbIndexName + ", single_tile_size)";

      const std::string cbDepthExpr =
          isInputCircularBuffer(info) ? "cb_buffer_depth" : "1";
      emitLine("tt_metal::CreateCircularBuffer(program, all_cores, "
               "CircularBufferConfig(" +
               info.tilesVarName +
               " * " + cbDepthExpr + " * single_tile_size, {{" + cbEntries +
               "}})" + setPageSizeChain + ");");
    };

    for (const CircularBufferInfo &info : cbInfos)
      emitCbInfo(info);
  }

  void emitInputSemaphores() {
    for (const CopyBindingInfo &binding : copyBindings) {
      if (!binding.isInput || !binding.hasBroadcastRegion)
        continue;
      emitLine("auto " + binding.bindingName +
               "_mcast_sender_semaphore_addr = "
               "tt_metal::CreateSemaphore(program, all_cores, INVALID);");
      emitLine("auto " + binding.bindingName +
               "_mcast_receiver_semaphore_addr = "
               "tt_metal::CreateSemaphore(program, all_cores, INVALID);");
    }
    if (!hasReduce)
      return;
    emitLine("auto reduce_ready_semaphore_addr = "
             "tt_metal::CreateSemaphore(program, all_cores, INVALID);");
    emitLine("auto reduce_token_semaphore_addr = "
             "tt_metal::CreateSemaphore(program, all_cores, INVALID);");
  }

  std::string getPybindCallbackTensorExpr(const DramBufferInfo &info) const {
    if (info.isOutput && info.outputTensorOrdinal >= 0) {
      if (info.isInput && info.inputTensorOrdinal >= 0) {
        return "(output_tensors.empty() ? input_tensors.at(" +
               std::to_string(info.inputTensorOrdinal) +
               ") : output_tensors.at(" +
               std::to_string(info.outputTensorOrdinal) + "))";
      }
      return "output_tensors.at(" + std::to_string(info.outputTensorOrdinal) +
             ")";
    }
    if (info.inputTensorOrdinal >= 0)
      return "input_tensors.at(" + std::to_string(info.inputTensorOrdinal) +
             ")";
    return {};
  }

  static std::string getScalarRuntimePrefix(const ScalarRuntimeSiteInfo &site,
                                            int64_t elementIndex) {
    return "scalar_site_" + std::to_string(site.siteId) + "_" +
           std::to_string(elementIndex);
  }

  static std::string getScalarRuntimePackedName(
      const ScalarRuntimeSiteInfo &site, int64_t elementIndex) {
    return getScalarRuntimePrefix(site, elementIndex) + "_packed";
  }

  void emitScalarRuntimeIndex(const std::string &sitePrefix,
                              const std::string &offsetExpr) {
    emitLine("std::size_t " + sitePrefix +
             "_index = static_cast<std::size_t>(" + offsetExpr + ");");
  }

  void emitScalarRuntimePybindLoad(const ScalarRuntimeSiteInfo &site,
                                   const std::string &sitePrefix) {
    emitLine("bfloat16 " + sitePrefix + "_value = " +
             site.scalarValuesName + ".at(" + sitePrefix + "_index);");
    emitLine("uint32_t " + sitePrefix +
             "_packed = pack_two_bfloat16_into_uint32({" + sitePrefix +
             "_value, " + sitePrefix + "_value});");
  }

  bool scalarRuntimeOffsetsUseCore() const {
    return llvm::any_of(scalarRuntimeSites,
                        [](const ScalarRuntimeSiteInfo &site) {
                          return llvm::any_of(
                              site.offsetExprs,
                              [](StringRef offsetExpr) {
                                return offsetExpr.contains("core.");
                              });
                        });
  }

  const RuntimeArgLayout &runtimeLayoutForRole(KernelRuntimeRole role) const {
    switch (role) {
    case KernelRuntimeRole::Reader:
      return readerRuntimeLayout;
    case KernelRuntimeRole::Writer:
      return writerRuntimeLayout;
    case KernelRuntimeRole::Compute:
      return computeRuntimeLayout;
    }
    llvm_unreachable("unknown runtime role");
  }

  static StringRef runtimeVectorName(KernelRuntimeRole role) {
    switch (role) {
    case KernelRuntimeRole::Reader:
      return "reader_runtime_args_for_core";
    case KernelRuntimeRole::Writer:
      return "writer_runtime_args_for_core";
    case KernelRuntimeRole::Compute:
      return "compute_runtime_args_for_core";
    }
    llvm_unreachable("unknown runtime role");
  }

  static StringRef runtimeArgsLocalName(KernelRuntimeRole role) {
    switch (role) {
    case KernelRuntimeRole::Reader:
      return "reader_args";
    case KernelRuntimeRole::Writer:
      return "writer_args";
    case KernelRuntimeRole::Compute:
      return "compute_args";
    }
    llvm_unreachable("unknown runtime role");
  }

  const CopyBindingInfo *findCopyBindingBySlot(int64_t slot) const {
    for (const CopyBindingInfo &binding : copyBindings)
      if (static_cast<int64_t>(binding.slot) == slot)
        return &binding;
    return nullptr;
  }

  const ScalarRuntimeSiteInfo *findScalarRuntimeSite(int64_t siteId) const {
    for (const ScalarRuntimeSiteInfo &site : scalarRuntimeSites)
      if (site.siteId == siteId)
        return &site;
    return nullptr;
  }

  std::string runtimeArgExpr(const RuntimeArgKey &key) const {
    switch (key.kind) {
    case RuntimeArgKind::CopyBinding: {
      const CopyBindingInfo *binding = findCopyBindingBySlot(key.id);
      if (!binding)
        return "0";
      const DramBufferInfo &dramInfo = dramInfos[binding->dramInfoIndex];
      const std::string &prefix = binding->bindingName;
      switch (static_cast<CopyBindingRuntimeField>(key.field)) {
      case CopyBindingRuntimeField::Cb: {
        unsigned cbIndex = binding->cbIndex >= 0
                               ? static_cast<unsigned>(binding->cbIndex)
                               : binding->slot;
        return "static_cast<uint32_t>(CBIndex::c_" + std::to_string(cbIndex) +
               ")";
      }
      case CopyBindingRuntimeField::BaseAddr:
        return dramInfo.bufferName + "->address()";
      case CopyBindingRuntimeField::McastDestStartX:
        return binding->hasBroadcastRegion
                   ? prefix + "_multicast_dest_noc_start_x"
                   : "0";
      case CopyBindingRuntimeField::McastDestStartY:
        return binding->hasBroadcastRegion
                   ? prefix + "_multicast_dest_noc_start_y"
                   : "0";
      case CopyBindingRuntimeField::McastDestEndX:
        return binding->hasBroadcastRegion
                   ? prefix + "_multicast_dest_noc_end_x"
                   : "0";
      case CopyBindingRuntimeField::McastDestEndY:
        return binding->hasBroadcastRegion
                   ? prefix + "_multicast_dest_noc_end_y"
                   : "0";
      case CopyBindingRuntimeField::McastDestNum:
        return binding->hasBroadcastRegion ? prefix + "_multicast_dest_num"
                                           : "0";
      case CopyBindingRuntimeField::McastSenderNocX:
        return binding->hasBroadcastRegion
                   ? prefix + "_multicast_sender_noc_x"
                   : "0";
      case CopyBindingRuntimeField::McastSenderNocY:
        return binding->hasBroadcastRegion
                   ? prefix + "_multicast_sender_noc_y"
                   : "0";
      case CopyBindingRuntimeField::McastSenderSemaphore:
        return binding->hasBroadcastRegion
                   ? prefix + "_mcast_sender_semaphore_addr"
                   : "0";
      case CopyBindingRuntimeField::McastReceiverSemaphore:
        return binding->hasBroadcastRegion
                   ? prefix + "_mcast_receiver_semaphore_addr"
                   : "0";
      }
      llvm_unreachable("unknown copy-binding runtime field");
    }
    case RuntimeArgKind::Reduce:
      switch (static_cast<ReduceRuntimeField>(key.field)) {
      case ReduceRuntimeField::ReadySemaphore:
        return "reduce_ready_semaphore_addr";
      case ReduceRuntimeField::TokenSemaphore:
        return "reduce_token_semaphore_addr";
      case ReduceRuntimeField::TokenMcastDestStartX:
        return "reduce_token_mcast_dest_noc_start_x";
      case ReduceRuntimeField::TokenMcastDestStartY:
        return "reduce_token_mcast_dest_noc_start_y";
      case ReduceRuntimeField::TokenMcastDestEndX:
        return "reduce_token_mcast_dest_noc_end_x";
      case ReduceRuntimeField::TokenMcastDestEndY:
        return "reduce_token_mcast_dest_noc_end_y";
      }
      llvm_unreachable("unknown reduce runtime field");
    case RuntimeArgKind::ScalarSite:
      if (const auto *site = findScalarRuntimeSite(key.id)) {
        if (key.field >= 0 &&
            key.field < static_cast<int64_t>(site->offsetExprs.size()))
          return getScalarRuntimePackedName(*site, key.field);
      }
      return "0";
    case RuntimeArgKind::CoreCoord:
      switch (static_cast<CoreCoordRuntimeField>(key.field)) {
      case CoreCoordRuntimeField::X:
        return "core.x";
      case CoreCoordRuntimeField::Y:
        return "core.y";
      }
      llvm_unreachable("unknown core-coordinate runtime field");
    case RuntimeArgKind::InternalCb: {
      auto it = internalSlotToCbIndex.find(key.id);
      if (it == internalSlotToCbIndex.end())
        return "0";
      return "static_cast<uint32_t>(CBIndex::c_" + std::to_string(it->second) +
             ")";
    }
    }
    llvm_unreachable("unknown runtime arg kind");
  }

  void emitRuntimeArgVector(KernelRuntimeRole role) {
    const RuntimeArgLayout &layout = runtimeLayoutForRole(role);
    emitLine("std::vector<uint32_t> " + runtimeVectorName(role).str() +
             " = {");
    for (const RuntimeArgKey &key : layout.keys)
      emitLine(runtimeArgExpr(key) + ",");
    emitLine("};");
  }

  void emitRuntimePatchIfPresent(KernelRuntimeRole role, RuntimeArgKey key,
                                 const std::string &valueExpr) {
    const RuntimeArgLayout &layout = runtimeLayoutForRole(role);
    auto index = layout.indexOf(key);
    if (!index)
      return;
    emitLine(runtimeArgsLocalName(role).str() + "[" +
             std::to_string(*index) + "] = " + valueExpr + ";");
  }

  void emitPybindOverrideRuntimeCallback() {
    if (kind != HostProgramKind::Pybind)
      return;

    std::string captures =
        "reader_id, writer_id, compute_kernel_id, start_core_x, start_core_y, "
        "end_core_x, end_core_y";
    for (const ScalarRuntimeHostArgInfo &info : scalarRuntimeHostArgs)
      captures += ", " + info.valuesName;

    emitLine(
        "auto override_runtime_arguments_callback = [" + captures +
        "](const void* operation, Program& program, const "
        "std::vector<ttnn::Tensor>& input_tensors, const "
        "std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors, "
        "const std::vector<ttnn::Tensor>& output_tensors) {");
    emitLine("(void)operation;");
    emitLine("(void)optional_input_tensors;");

    for (const DramBufferInfo &info : dramInfos) {
      std::string tensorExpr = getPybindCallbackTensorExpr(info);
      if (tensorExpr.empty())
        continue;
      emitLine("auto* " + info.bufferName + " = " + tensorExpr + ".buffer();");
    }

    emitLine("auto& reader_args_by_core = GetRuntimeArgs(program, reader_id);");
    emitLine("auto& writer_args_by_core = GetRuntimeArgs(program, writer_id);");
    emitLine(
        "auto& compute_args_by_core = GetRuntimeArgs(program, compute_kernel_id);");
    emitLine("for (uint32_t core_x = start_core_x; core_x <= end_core_x; ++core_x) {");
    emitLine("for (uint32_t core_y = start_core_y; core_y <= end_core_y; ++core_y) {");
    emitLine("auto& reader_args = reader_args_by_core[core_x][core_y];");
    emitLine("auto& writer_args = writer_args_by_core[core_x][core_y];");
    emitLine("auto& compute_args = compute_args_by_core[core_x][core_y];");
    if (scalarRuntimeOffsetsUseCore())
      emitLine("CoreCoord core{core_x, core_y};");

    for (const CopyBindingInfo &binding : copyBindings) {
      const DramBufferInfo &dramInfo = dramInfos[binding.dramInfoIndex];
      RuntimeArgKey baseKey = RuntimeArgKey::copyBinding(
          binding.slot, CopyBindingRuntimeField::BaseAddr);
      std::string addressExpr = dramInfo.bufferName + "->address()";
      emitRuntimePatchIfPresent(KernelRuntimeRole::Reader, baseKey,
                                addressExpr);
      emitRuntimePatchIfPresent(KernelRuntimeRole::Writer, baseKey,
                                addressExpr);
      emitRuntimePatchIfPresent(KernelRuntimeRole::Compute, baseKey,
                                addressExpr);
    }

    for (const ScalarRuntimeSiteInfo &site : scalarRuntimeSites) {
      for (auto [elementIndex, offsetExpr] : llvm::enumerate(site.offsetExprs)) {
        std::string sitePrefix =
            getScalarRuntimePrefix(site, static_cast<int64_t>(elementIndex));
        emitScalarRuntimeIndex(sitePrefix, offsetExpr);
        emitScalarRuntimePybindLoad(site, sitePrefix);
        RuntimeArgKey scalarKey = RuntimeArgKey::scalarSite(
            site.siteId, static_cast<int64_t>(elementIndex));
        std::string packedExpr =
            getScalarRuntimePackedName(site, static_cast<int64_t>(elementIndex));
        emitRuntimePatchIfPresent(KernelRuntimeRole::Reader, scalarKey,
                                  packedExpr);
        emitRuntimePatchIfPresent(KernelRuntimeRole::Writer, scalarKey,
                                  packedExpr);
        emitRuntimePatchIfPresent(KernelRuntimeRole::Compute, scalarKey,
                                  packedExpr);
      }
    }

    emitLine("}");
    emitLine("}");
    emitLine("};");
  }

  void emitRuntimeEnqueueEpilogue() {
    if (kind == HostProgramKind::Pybind) {
      emitPybindOverrideRuntimeCallback();
      return;
    }

    if (kind == HostProgramKind::Cpp) {
      for (const DramBufferInfo &info : dramInfos) {
        if (!info.isInput)
          continue;
        emitLine("EnqueueWriteBuffer(cq, " + info.bufferName + ", v" +
                 std::to_string(info.hostParamIndex + 1) +
                 ".data(), false);");
      }
    }

    emitLine("EnqueueProgram(cq, program, false);");

    if (kind != HostProgramKind::Cpp)
      return;

    SmallVector<const DramBufferInfo *, 4> outputInfos;
    for (const DramBufferInfo &info : dramInfos) {
      if (info.isOutput)
        outputInfos.push_back(&info);
    }

    for (auto [idx, info] : llvm::enumerate(outputInfos)) {
      bool isLast = (idx + 1 == outputInfos.size());
      emitLine("EnqueueReadBuffer(cq, " + info->bufferName + ", v" +
               std::to_string(info->hostParamIndex + 1) + ".data(), " +
               (isLast ? "true" : "false") + ");");
    }
  }

  void emitCoreMulticastMappingAtEnd() {
    emitLine("constexpr bool row_major = true;");
    emitLine("auto cores = corerange_to_cores(all_cores, std::nullopt, row_major);");
    emitLine("for (const auto& core : cores) {");
    emitReaderRuntimeArgsForCore();
    emitLine("}");
  }

  void emitReaderRuntimeArgsForCore() {
    for (const CopyBindingInfo &binding : copyBindings) {
      if (!binding.isInput || !binding.hasBroadcastRegion)
        continue;

      const std::string &prefix = binding.bindingName;
      emitLine("std::size_t " + prefix + "_ul_x = static_cast<std::size_t>(" +
               binding.ulXExpr + ");");
      emitLine("std::size_t " + prefix + "_ul_y = static_cast<std::size_t>(" +
               binding.ulYExpr + ");");
      emitLine("std::size_t " + prefix + "_lr_x = static_cast<std::size_t>(" +
               binding.lrXExpr + ");");
      emitLine("std::size_t " + prefix + "_lr_y = static_cast<std::size_t>(" +
               binding.lrYExpr + ");");

      emitLine("CoreCoord " + prefix + "_sender_core = {" + prefix +
               "_ul_x, " + prefix + "_ul_y};");
      emitLine("CoreCoord " + prefix + "_dest_start_core = {" + prefix +
               "_ul_x, " + prefix + "_ul_y};");
      emitLine("CoreCoord " + prefix + "_dest_end_core = {" + prefix +
               "_lr_x, " + prefix + "_lr_y};");

      emitLine("auto " + prefix +
               "_sender_physical = device->worker_core_from_logical_core(" +
               prefix + "_sender_core);");
      emitLine("auto " + prefix +
               "_dest_start_physical = device->worker_core_from_logical_core(" +
               prefix + "_dest_start_core);");
      emitLine("auto " + prefix +
               "_dest_end_physical = device->worker_core_from_logical_core(" +
               prefix + "_dest_end_core);");

      emitLine("uint32_t " + prefix +
               "_multicast_dest_noc_start_x = (std::uint32_t)" + prefix +
               "_dest_start_physical.x;");
      emitLine("uint32_t " + prefix +
               "_multicast_dest_noc_start_y = (std::uint32_t)" + prefix +
               "_dest_start_physical.y;");
      emitLine("uint32_t " + prefix +
               "_multicast_dest_noc_end_x = (std::uint32_t)" + prefix +
               "_dest_end_physical.x;");
      emitLine("uint32_t " + prefix +
               "_multicast_dest_noc_end_y = (std::uint32_t)" + prefix +
               "_dest_end_physical.y;");
      emitLine("uint32_t " + prefix +
               "_multicast_sender_noc_x = (std::uint32_t)" + prefix +
               "_sender_physical.x;");
      emitLine("uint32_t " + prefix +
               "_multicast_sender_noc_y = (std::uint32_t)" + prefix +
               "_sender_physical.y;");
      emitLine("uint32_t " + prefix + "_multicast_dest_num = static_cast<uint32_t>(((" +
               prefix + "_lr_x - " + prefix + "_ul_x + 1) * (" + prefix +
               "_lr_y - " + prefix + "_ul_y + 1)) - 1);");
    }

    if (hasReduce && reduceRegion.hasRegion) {
      emitLine("std::size_t reduce_ul_x = static_cast<std::size_t>(" +
               reduceRegion.ulXExpr + ");");
      emitLine("std::size_t reduce_ul_y = static_cast<std::size_t>(" +
               reduceRegion.ulYExpr + ");");
      emitLine("std::size_t reduce_lr_x = static_cast<std::size_t>(" +
               reduceRegion.lrXExpr + ");");
      emitLine("std::size_t reduce_lr_y = static_cast<std::size_t>(" +
               reduceRegion.lrYExpr + ");");

      emitLine("CoreCoord reduce_dest_start_core = {reduce_ul_x, reduce_ul_y};");
      emitLine("CoreCoord reduce_dest_end_core = {reduce_lr_x, reduce_lr_y};");

      emitLine("auto reduce_dest_start_physical = device->worker_core_from_logical_core(reduce_dest_start_core);");
      emitLine("auto reduce_dest_end_physical = device->worker_core_from_logical_core(reduce_dest_end_core);");

      emitLine("uint32_t reduce_token_mcast_dest_noc_start_x = (std::uint32_t)reduce_dest_start_physical.x;");
      emitLine("uint32_t reduce_token_mcast_dest_noc_start_y = (std::uint32_t)reduce_dest_start_physical.y;");
      emitLine("uint32_t reduce_token_mcast_dest_noc_end_x = (std::uint32_t)reduce_dest_end_physical.x;");
      emitLine("uint32_t reduce_token_mcast_dest_noc_end_y = (std::uint32_t)reduce_dest_end_physical.y;");
    } else if (hasReduce) {
      emitLine("uint32_t reduce_token_mcast_dest_noc_start_x = 0;");
      emitLine("uint32_t reduce_token_mcast_dest_noc_start_y = 0;");
      emitLine("uint32_t reduce_token_mcast_dest_noc_end_x = 0;");
      emitLine("uint32_t reduce_token_mcast_dest_noc_end_y = 0;");
    }

    for (const ScalarRuntimeSiteInfo &site : scalarRuntimeSites) {
      for (auto [elementIndex, offsetExpr] : llvm::enumerate(site.offsetExprs)) {
        std::string sitePrefix =
            getScalarRuntimePrefix(site, static_cast<int64_t>(elementIndex));
        emitScalarRuntimeIndex(sitePrefix, offsetExpr);
        if (kind == HostProgramKind::Cpp) {
          emitLine("bfloat16 " + sitePrefix + "_value = " +
                   site.scalarValuesName + "[" + sitePrefix + "_index];");
          emitLine("uint32_t " + sitePrefix +
                   "_packed = pack_two_bfloat16_into_uint32({" + sitePrefix +
                   "_value, " + sitePrefix + "_value});");
        } else {
          emitScalarRuntimePybindLoad(site, sitePrefix);
        }
      }
    }

    emitRuntimeArgVector(KernelRuntimeRole::Reader);
    emitRuntimeArgVector(KernelRuntimeRole::Writer);
    emitRuntimeArgVector(KernelRuntimeRole::Compute);

    emitLine("tt_metal::SetRuntimeArgs(program, reader_id, core, "
             "reader_runtime_args_for_core);");
    emitLine("tt_metal::SetRuntimeArgs(program, writer_id, core, "
             "writer_runtime_args_for_core);");
    emitLine("tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, "
             "compute_runtime_args_for_core);");
  }

  void emitCompileArgs() {
    emitLine("std::vector<uint32_t> compile_args = {};");
    for (const DramBufferInfo &info : dramInfos) {
      if (kind == HostProgramKind::Pybind) {
        emitLine("tt::tt_metal::TensorAccessorArgs(" + info.bufferName +
                 ").append_to(compile_args);");
      } else {
        emitLine("tt::tt_metal::TensorAccessorArgs(*" + info.bufferName +
                 ").append_to(compile_args);");
      }
    }
  }

  void emitKernelRoles() {
    emitLine("MathFidelity math_fidelity = MathFidelity::HiFi4;");

    const DataMovementKernelSpec readerSpec =
        getDataMovementKernelSpec(DataMovementKernelRole::Reader);
    const DataMovementKernelSpec writerSpec =
        getDataMovementKernelSpec(DataMovementKernelRole::Writer);
    const SmallVector<KernelRoleInfo, 3> roles = {
        {"reader_id",
         "reader.cpp",
         buildHostDataMovementConfigExpr(readerSpec)},
        {"writer_id",
         "writer.cpp",
         buildHostDataMovementConfigExpr(writerSpec)},
        {"compute_kernel_id",
         "compute.cpp",
         "tt_metal::ComputeConfig{.math_fidelity = math_fidelity, "
         ".compile_args = compile_args}"}};

    for (const KernelRoleInfo &role : roles) {
      emitLine("auto " + role.idVarName + " = tt_metal::CreateKernel("
               "program, OVERRIDE_KERNEL_PREFIX "
               "\"tt_metal/programming_examples/mlir_matmul_simple/kernels/" +
               role.kernelSource + "\", all_cores, " + role.configExpr + ");");
    }
  }

  func::FuncOp originalFunc;
  func::FuncOp hostFunc;
  Location loc;
  OpBuilder builder;
  HostProgramKind kind;
  func::FuncOp computeFunc;
  func::FuncOp readerFunc;
  func::FuncOp writerFunc;
  RuntimeArgLayout computeRuntimeLayout;
  RuntimeArgLayout readerRuntimeLayout;
  RuntimeArgLayout writerRuntimeLayout;
  SmallVector<DramBufferInfo, 8> dramInfos;
  SmallVector<CopyBindingInfo, 8> copyBindings;
  SmallVector<CircularBufferInfo, 8> cbInfos;
  SmallVector<ScalarRuntimeSiteInfo, 4> scalarRuntimeSites;
  SmallVector<ScalarRuntimeHostArgInfo, 4> scalarRuntimeHostArgs;
  SmallVector<HostArgKind, 8> hostArgKinds;
  IndexExprMap parallelIvExprByValue;
  llvm::DenseMap<int64_t, unsigned> internalSlotToCbIndex;
  llvm::DenseMap<Value, unsigned> semaphoreToCbIndex;
  llvm::DenseMap<Value, unsigned> allocToCbInfo;
  unsigned nextCbIndex = 0;
  SmallVector<std::string, 8> emittedTilesVars;
  std::string coreCoordArg0Expr = "core.x";
  std::string coreCoordArg1Expr = "core.y";
  std::optional<int64_t> mlirCoreExtentX;
  std::optional<int64_t> mlirCoreExtentY;
  LogicalResult status = success();
  bool hasReduce = false;
  ReduceRegionInfo reduceRegion;
};

/**
 * @brief Specialize a function for host-only execution (no compute/load/store).
 *
 * @details Clones the function with either `__host_cpp` or
 *          `__host_pybind` suffix and erases all compute operations
 *          (linalg.matmul etc.), load operations (memref.copy where source is
 *          reinterpret_cast), and store operations (memref.copy where target
 *          is reinterpret_cast). This creates a host function with only
 *          control flow and other non-compute/memory operations.
 *
 * @param func The original function to specialize.
 * @return The specialized host function.
 */
static FailureOr<func::FuncOp> makeHostFunc(func::FuncOp func,
                                            HostProgramKind kind,
                                            func::FuncOp computeFunc,
                                            func::FuncOp readerFunc,
                                            func::FuncOp writerFunc) {
  IRMapping mapping;
  auto hostFunc = cast<func::FuncOp>(func->clone(mapping));
  hostFunc.setName((func.getName() +
                    (kind == HostProgramKind::Cpp ? "__host_cpp"
                                                  : "__host_pybind"))
                       .str());

  TTMetalHostProgramEmitter emitter(func, hostFunc, kind, computeFunc,
                                    readerFunc, writerFunc);
  if (failed(emitter.run()))
    return failure();
  clearMatmulBReaderMergeAttrs(hostFunc);
  return hostFunc;
}

/**
 * @brief Specializer class that manages function cloning and specialization.
 *
 * @details Follows the CoreSpecialize pattern from triton-tenstorrent.
 *          For each function, creates compute, reader, writer, host_cpp,
 *          and host_pybind specialized versions, inserts them into the
 *          module, and erases the original.
 */
class FunctionSpecializer {
public:
  FunctionSpecializer(ModuleOp module, func::FuncOp func)
      : module(module), originalFunc(func) {}

  LogicalResult run() {
    func::FuncOp func = originalFunc;
    const bool hasReduce = containsGatherOp(func);
    // Create specialized versions
    auto computeFunc = makeComputeFunc(func);
    auto readerFunc = makeReaderFunc(func);
    auto writerFunc = makeWriterFunc(func);

    // Attach TTKernel thread type attributes so downstream TTKernelToCpp
    // translation can recognize these as kernel entry points.
    auto *ctx = module.getContext();
    auto computeAttr = mlir::tt::ttkernel::ThreadTypeAttr::get(
        ctx, mlir::tt::ttkernel::ThreadType::Compute);
    auto nocAttr = mlir::tt::ttkernel::ThreadTypeAttr::get(
        ctx, mlir::tt::ttkernel::ThreadType::Noc);
    Builder attrBuilder(ctx);

    computeFunc->setAttr(mlir::tt::ttkernel::ThreadTypeAttr::name, computeAttr);
    readerFunc->setAttr(mlir::tt::ttkernel::ThreadTypeAttr::name, nocAttr);
    writerFunc->setAttr(mlir::tt::ttkernel::ThreadTypeAttr::name, nocAttr);
    readerFunc->setAttr(
        kDMProcessorAttrName,
        attrBuilder.getStringAttr(
            getDataMovementKernelSpec(DataMovementKernelRole::Reader)
                .processorAttrValue));
    writerFunc->setAttr(
        kDMProcessorAttrName,
        attrBuilder.getStringAttr(
            getDataMovementKernelSpec(DataMovementKernelRole::Writer)
                .processorAttrValue));
    if (hasReduce) {
      auto reduceAttr = attrBuilder.getUnitAttr();
      computeFunc->setAttr(kHasReduceAttrName, reduceAttr);
      readerFunc->setAttr(kHasReduceAttrName, reduceAttr);
      writerFunc->setAttr(kHasReduceAttrName, reduceAttr);
    }

    FailureOr<func::FuncOp> hostCppFunc =
        makeHostFunc(func, HostProgramKind::Cpp, computeFunc, readerFunc,
                     writerFunc);
    if (failed(hostCppFunc))
      return failure();
    FailureOr<func::FuncOp> hostPybindFunc =
        makeHostFunc(func, HostProgramKind::Pybind, computeFunc, readerFunc,
                     writerFunc);
    if (failed(hostPybindFunc))
      return failure();
    (*hostCppFunc)->setAttr(mlir::tt::ttkernel::ThreadTypeAttr::name, nocAttr);
    (*hostPybindFunc)
        ->setAttr(mlir::tt::ttkernel::ThreadTypeAttr::name, nocAttr);
    if (hasReduce) {
      auto reduceAttr = attrBuilder.getUnitAttr();
      (*hostCppFunc)->setAttr(kHasReduceAttrName, reduceAttr);
      (*hostPybindFunc)->setAttr(kHasReduceAttrName, reduceAttr);
    }

    // Insert specialized functions into the module (before the original)
    module.insert(func, computeFunc);
    module.insert(func, readerFunc);
    module.insert(func, writerFunc);
    module.insert(func, *hostCppFunc);
    module.insert(func, *hostPybindFunc);
    return success();
  }

private:
  ModuleOp module;
  func::FuncOp originalFunc;
};

} // namespace

LogicalResult mlir::loom::prepareMatmulBReaderMerge(ModuleOp module) {
  for (func::FuncOp func : collectEntryFuncs(module))
    if (failed(annotateMatmulBReaderMerge(func)))
      return failure();
  return success();
}

LogicalResult mlir::loom::annotateVecLoadUsage(ModuleOp module) {
  for (func::FuncOp func : collectEntryFuncs(module)) {
    auto annotated = annotateVecLoadUsageInFunc(func);
    if (failed(annotated))
      return failure();
  }
  return success();
}

LogicalResult mlir::loom::specializeFunctionsForTTKernel(ModuleOp module) {
  SmallVector<func::FuncOp, 8> funcs = collectEntryFuncs(module);
  if (funcs.empty())
    return success();

  for (func::FuncOp func : funcs) {
    if (!func || !func->getBlock())
      continue;

    // Mark scalar load-copy sites so host/device scalar runtime-arg
    // extraction remains in per-copy-site order.
    if (failed(annotateScalarRuntimeSites(func)))
      return failure();

    // Ensure every reduction generic has a dedicated scaler semaphore input.
    // This avoids reusing reduction outputs as scaler CBs during compute lowering.
    ensureReductionScaleInputs(func);

    // Temporary scalar path: replace scalar alloc/semaphore memory chains with
    // direct function inputs before function specialization.
    preprocessScalarMemoryOpsTmp(func);

    if (failed(annotateCopyBindingSlots(func)))
      return failure();

    // Assign stable per-semaphore internal CB slots before cloning so
    // specialized kernels share deterministic runtime-arg bindings.
    annotateSemaphoreSlots(func);

    if (failed(annotateStaticCbIndices(func)))
      return failure();

    ModuleOp parentModule = func->getParentOfType<ModuleOp>();
    if (!parentModule) {
      func.emitError("failed to locate parent module for specialization");
      return failure();
    }

    // Create specialized versions adjacent to the source function.
    FunctionSpecializer specializer(parentModule, func);
    if (failed(specializer.run()))
      return failure();

    // Erase the original function.
    func.erase();
  }

  return success();
}

LogicalResult mlir::loom::removeAllFunctionArguments(func::FuncOp func) {
  Block &entry = func.front();

  // Ensure no argument still has uses before erasing.
  for (BlockArgument arg : entry.getArguments()) {
    if (!arg.use_empty())
      return func.emitError()
             << "cannot erase arguments; argument " << arg.getArgNumber()
             << " still has uses";
  }

  // Erase arguments from last to first to avoid index shifting issues.
  for (int64_t idx = static_cast<int64_t>(entry.getNumArguments()) - 1;
       idx >= 0; --idx) {
    entry.eraseArgument(static_cast<unsigned>(idx));
  }

  auto funcType = func.getFunctionType();
  func.setType(FunctionType::get(func.getContext(), TypeRange(),
                                 funcType.getResults()));
  return success();
}
