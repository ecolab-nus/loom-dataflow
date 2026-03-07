/**
 * @file FuncOpToTTKernel.cpp
 * @brief Implementation for function specialization pass.
 *
 * @details This pass clones each func::FuncOp into four specialized versions:
 *          - `__compute`: retains compute ops, erases store operations
 *          - `__reader` : retains memory load ops, erases stores & compute ops
 *          - `__writer` : retains memory store ops, erases loads & compute ops
 *          - `__host`   : erases all compute, load, and store operations
 *          This loosely mimics the CoreSpecialize pattern from
 *          triton-tenstorrent while remaining TileLoom-specific.
 */

#include "FuncOpToTTKernel.h"

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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

// Loom dialect headers for ::::loom::CopyOp, ::loom::AllocOp
#include "mlir/Interfaces/ViewLikeInterface.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

// TTKernel thread type attribute and enum.
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

using namespace mlir;
using namespace tt::ttkernel;

//===----------------------------------------------------------------------===//
// CompileArgTracker Implementation
//===----------------------------------------------------------------------===//

namespace {

constexpr llvm::StringLiteral kReductionScaleCbAttrName =
    "loom.reduction_scale_cb";

static Value stripMemrefCasts(Value value) {
  Value current = value;
  while (auto cast = current.getDefiningOp<memref::CastOp>())
    current = cast.getSource();
  return current;
}

// Follow loom.semaphore wrappers to recover the underlying physical buffer
// producer (typically loom.alloc). Multiple semaphore ops may chain on the
// same base value in transformed IR.
static Value stripLoomSemaphores(Value value) {
  Value current = value;
  while (auto sem = current.getDefiningOp<::loom::SemaphoreTakeOp>())
    current = sem.getSource();
  return current;
}

// Infer per-argument CB memref shape from loom.copy links to loom.alloc.
// This decouples DRAM tensor shape (function args / host buffers) from
// on-core CB tile shape (loom.alloc).
static LogicalResult inferArgToCBMemrefType(
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
  return llvm::any_of(op.getIteratorTypesArray(), [](utils::IteratorType type) {
    return type == utils::IteratorType::reduction;
  });
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

} // namespace

LogicalResult mlir::loom::CompileArgTracker::processInputArgs(
    func::FuncOp func, TypeConverter &typeConverter, OpBuilder &rewriter) {
  if (func.getName().contains("__host"))
    return success();

  Block &entry = func.front();
  Location loc = func.getLoc();

  // Detect whether this function is a compute kernel. For compute kernels
  // we avoid creating TensorAccessor bookkeeping, since they do not perform
  // direct NOC reads/writes and only need CB handles.
  bool isComputeKernel = false;
  if (auto threadAttr =
          func->getAttrOfType<ThreadTypeAttr>(ThreadTypeAttr::name)) {
    isComputeKernel = threadAttr.getValue() == ThreadType::Compute;
  }

  // Save insertion point and set to start of function body.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&entry);

  llvm::DenseMap<Value, MemRefType> argToCBMemrefType;
  if (failed(inferArgToCBMemrefType(func, argToCBMemrefType)))
    return failure();

  // Process each function argument.
  for (BlockArgument arg : entry.getArguments()) {
    Type argType = arg.getType();

    if (isa<MemRefType, UnrankedMemRefType>(argType)) {
      // Memref type: create CB and base address.
      // CB uses nextCompileArgIndex, base address uses nextCompileArgIndex + 1.
      int64_t tensorAccessorArgsIndex = getNextTensorAccessorArgsIndex(func);

      // Create GetArgValOp for CB.
      Type cbType = nullptr;
      if (auto it = argToCBMemrefType.find(arg); it != argToCBMemrefType.end())
        cbType = CBType::get(it->second);
      else
        cbType = typeConverter.convertType(argType);
      if (!cbType)
        return func.emitError() << "failed to convert memref type to CB type";
      Value cbOp = createGetArgValOp(loc, rewriter, func, cbType);

      // Create GetArgValOp for base address.
      Value baseAddrOp = createGetArgValOp(loc, rewriter, func,
                                           rewriter.getI32Type());
      
      Value mcast_dest_noc_start_x_op = createGetArgValOp(
          loc, rewriter, func, rewriter.getI32Type());
      Value mcast_dest_noc_start_y_op = createGetArgValOp(
          loc, rewriter, func, rewriter.getI32Type());
      Value mcast_dest_noc_end_x_op = createGetArgValOp(
          loc, rewriter, func, rewriter.getI32Type());
      Value mcast_dest_noc_end_y_op = createGetArgValOp(
          loc, rewriter, func, rewriter.getI32Type());
      Value mcast_dest_num_op = createGetArgValOp(
          loc, rewriter, func, rewriter.getI32Type());
      Value mcast_sender_noc_x_op = createGetArgValOp(
          loc, rewriter, func, rewriter.getI32Type());
      Value mcast_sender_noc_y_op = createGetArgValOp(
          loc, rewriter, func, rewriter.getI32Type());
      Value mcast_sender_semaphore_addr_arg = createGetArgValOp(
          loc, rewriter, func, rewriter.getI32Type());
      Value mcast_sender_semaphore_addr_op;
      Value mcast_receiver_semaphore_addr_arg = createGetArgValOp(
        loc, rewriter, func, rewriter.getI32Type());
      Value mcast_receiver_semaphore_addr_op;


      // Create TensorAccessArgs and TensorAccess for base address only for
      // non-compute kernels (reader/writer/host). Compute kernels access data
      // via circular buffers and do not issue NOC reads/writes directly, so
      // they don't require TensorAccessor metadata.
      Value tensorAccessor;
      Value mcast_sender_semaphore_noc_addr_op;
      Value mcast_receiver_semaphore_noc_addr_op;
      Value mcast_receiver_semaphore_addr_ptr;
      Value mcast_sender_semaphore_addr_ptr;
      if (!isComputeKernel) {
        mcast_sender_semaphore_addr_op = GetSemaphoreOp::create(
          rewriter, loc, mcast_sender_semaphore_addr_arg);
        mcast_receiver_semaphore_addr_op = GetSemaphoreOp::create(
          rewriter, loc, mcast_receiver_semaphore_addr_arg);
        auto pagesize = GetTileSizeOp::create(rewriter, loc, cbOp);
        Value tensorAccessorArgsIdxVal = rewriter.create<arith::ConstantIntOp>(
            loc, rewriter.getI32Type(),
            static_cast<int64_t>(tensorAccessorArgsIndex));
        auto baseAddrArgs = rewriter.create<TensorAccessorArgsOp>(
            loc, tensorAccessorArgsIdxVal, tensorAccessorArgsIdxVal);
        auto baseAddrTensorAccess =
            rewriter.create<TensorAccessorOp>(loc, baseAddrArgs.getResult(),
                                              baseAddrOp, pagesize);
        tensorAccessor = baseAddrTensorAccess.getResult();
        //TODO, need to generate code like "ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>" instead of ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>, this is currently achieved by using python string processing
        mcast_receiver_semaphore_addr_ptr = CastToL1PtrOp::create(rewriter, loc, mcast_receiver_semaphore_addr_op);
        
        mcast_sender_semaphore_addr_ptr = CastToL1PtrOp::create(rewriter, loc, mcast_sender_semaphore_addr_op);
        mcast_sender_semaphore_noc_addr_op = GetNocAddrOp::create(rewriter, loc, 
                           mcast_sender_noc_x_op,
                           mcast_sender_noc_y_op,
                           mcast_sender_semaphore_addr_op);
        
        auto zeroVal = rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI8Type(), 0);
        mcast_receiver_semaphore_noc_addr_op =
          GetNocMulticastAddrOp::create(
              rewriter, loc, NocAddrType::get(rewriter.getContext()),
              mcast_dest_noc_end_x_op,
              mcast_dest_noc_end_y_op,
              mcast_dest_noc_start_x_op,
              mcast_dest_noc_start_y_op,
              mcast_receiver_semaphore_addr_op, zeroVal);
      }

      // pre-created base address when processing load/store ops.
      memrefArgToData[arg] = MemrefArgData{
        cbOp, // cb id
        baseAddrOp, // base address
        tensorAccessor, // tensor accessor
        mcast_dest_noc_start_x_op, // multicast destination NOC start x
        mcast_dest_noc_start_y_op, // multicast destination NOC start y
        mcast_dest_noc_end_x_op, // multicast destination NOC end x
        mcast_dest_noc_end_y_op, // multicast destination NOC end y
        mcast_dest_num_op, // multicast destination number
        mcast_sender_noc_x_op, // multicast sender NOC x
        mcast_sender_noc_y_op, // multicast sender NOC y
        mcast_sender_semaphore_addr_op, // multicast sender semaphore address
        mcast_receiver_semaphore_addr_op, // multicast receiver semaphore address
        mcast_sender_semaphore_addr_ptr, // L1 multicast sender semaphore address pointer
        mcast_receiver_semaphore_addr_ptr, // L1 multicast receiver semaphore address pointer
        mcast_sender_semaphore_noc_addr_op, // noc address of sender semaphore
        mcast_receiver_semaphore_noc_addr_op // noc address of receiver semaphore
      };
      memrefArgToData[arg].initLoc = loc;

    } else if (argType.isIndex()) {
      // Index type: create a single compile-arg.
      Value compileArgOp = createGetArgValOp(loc, rewriter, func,
                                              rewriter.getI32Type());

      // Cast i32 to index for compatibility.
      auto indexCast = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), compileArgOp);

      // Store the created values.
      indexArgToData[arg] = IndexArgData{compileArgOp, indexCast.getResult()};

      // Replace uses of the index argument with the casted value.
      arg.replaceAllUsesWith(indexCast.getResult());

    } else {
      // Other types: not supported yet.
      return func.emitError()
             << "unsupported argument type: " << argType;
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
  StringRef dim = dimName.trim().lower();
  if (dim == "x")
    funcToCoreCoordsByDim[funcOp].x = value;
  else if (dim == "y")
    funcToCoreCoordsByDim[funcOp].y = value;
}

Value mlir::loom::CompileArgTracker::getCoreCoordForDim(Operation *funcOp,
                                                         StringRef dimName) const {
  auto it = funcToCoreCoordsByDim.find(funcOp);
  if (it == funcToCoreCoordsByDim.end())
    return {};

  StringRef dim = dimName.trim().lower();
  if (dim == "x")
    return it->second.x;
  if (dim == "y")
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

/**
 * @brief Check if an operation is a compute operation.
 *
 * @details Currently identifies linalg.matmul as the primary compute op.
 *          Can be extended for other compute operations.
 *
 * @param op The operation to check.
 * @return true if this is a compute operation, false otherwise.
 */
static bool isComputeOp(Operation *op) {
  return isa<linalg::MatmulOp, linalg::BatchMatmulOp, linalg::GenericOp,
             linalg::FillOp, linalg::CopyOp>(op);
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

  // Collect compute ops and stores to erase.
  SmallVector<Operation *, 8> opsToErase;
  readerFunc.walk([&](Operation *op) {
    if (isComputeOp(op)) {
      opsToErase.push_back(op);
    }  else if (auto loomCopyOp = dyn_cast<::loom::CopyOp>(op)) {
      if (isLoomStoreOp(loomCopyOp))
        opsToErase.push_back(op);
    }
  });

  // Erase compute ops in reverse order to handle dependencies
  for (Operation *op : llvm::reverse(opsToErase))
    op->erase();

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

  SmallVector<Operation *, 8> opsToErase;
  writerFunc.walk([&](Operation *op) {
    if (isComputeOp(op)) {
      opsToErase.push_back(op);
    }  else if (auto loomCopyOp = dyn_cast<::loom::CopyOp>(op)) {
      if (isLoomLoadOp(loomCopyOp))
        opsToErase.push_back(op);
    }
  });

  for (Operation *op : llvm::reverse(opsToErase))
    op->erase();

  return writerFunc;
}

/**
 * @brief Emit host-side TT-Metal DRAM/CB setup and compile args.
 *
 * @details This helper consolidates host emission into one place:
 *          1. DRAM buffers from memref function arguments.
 *          2. Circular buffers from `loom.alloc`, linked back to DRAM memrefs.
 *          3. TensorAccessor compile args for each DRAM buffer.
 */
class TTMetalHostProgramEmitter {
public:
  TTMetalHostProgramEmitter(func::FuncOp originalFunc, func::FuncOp hostFunc)
      : originalFunc(originalFunc), hostFunc(hostFunc), loc(hostFunc.getLoc()),
        builder(hostFunc.getContext()) {}

  void run() {
    inferCoreCoordArgOrder();
    collectDramInfos();
    collectCbInfos();
    annotateHostSignatureMetadata();
    eraseNonHostOps();
    eraseDeadReinterpretCasts();

    builder.setInsertionPointToStart(&hostFunc.getBody().front());
    emitPreamble();
    emitDramBuffers();
    emitCircularBuffers();
    emitInputSemaphores();
    emitCompileArgs();
    emitKernelRoles();

    // Runtime enqueue calls must be emitted at the end of the host function.
    builder.setInsertionPoint(hostFunc.front().getTerminator());
    emitCoreMulticastMappingAtEnd();
    emitRuntimeEnqueueEpilogue();
  }

private:
  enum class MulticastKind { None, Horizontal, Vertical, All };

  struct DramBufferInfo {
    Value hostArg;
    MemRefType type;
    unsigned argIndex;
    unsigned memrefOrdinal;
    int cbIndex;
    bool isInput;
    bool isOutput;
    MulticastKind multicastKind;
    std::string inputName;
    std::string configName;
    std::string bufferName;
    std::string tilesVarName;
    std::string tilesExpr;
    std::string sizeExpr;
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
  //TODO: tmp fix, need to get the total size and then divide by the tile size
  static bool buildTilesExpr(MemRefType memrefType, std::string &expr) {
    expr.clear();
  
    auto shape = memrefType.getShape();
    const size_t rank = shape.size();
  
    // Need at least 2 dims to "tile the last two dimensions".
    // If you prefer returning "1" for rank < 2, change this behavior accordingly.
    if (rank < 2) {
      expr = "1";
      return true;
    }
  
    auto appendOne = [&](int64_t dim) -> bool {
      if (dim == ShapedType::kDynamic) return false;
      dim = std::max<int64_t>(32, dim);
      expr += std::to_string(dim);
      expr += " / TILE_HEIGHT";
      return true;
    };
  
    // Exactly last two dims: rank-2 and rank-1
    if (!appendOne(shape[rank - 2])) return false;
    expr += " * ";
    if (!appendOne(shape[rank - 1])) return false;
  
    // (Now expr can’t be empty if rank>=2, but keep the old fallback behavior.)
    if (expr.empty()) expr = "1";
    return true;
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
    coreCoordArg0Expr = "core.x";
    coreCoordArg1Expr = "core.y";

    bool foundOrder = false;
    hostFunc.walk([&](scf::ParallelOp op) {
      if (foundOrder)
        return;

      auto mappedAttr = op->getAttrOfType<ArrayAttr>("loom.mapped_to_dims");
      if (!mappedAttr || mappedAttr.size() < 2)
        return;

      auto iterTypesAttr = op->getAttrOfType<ArrayAttr>("loom.iter_types");
      SmallVector<std::string, 4> spatialMappedDims;
      for (auto [idx, mapped] : llvm::enumerate(mappedAttr)) {
        if (iterTypesAttr && idx < iterTypesAttr.size() &&
            !isSpatialIterAttr(iterTypesAttr[idx]))
          continue;
        spatialMappedDims.push_back(extractDimName(mapped));
      }

      if (spatialMappedDims.size() < 2)
        return;

      std::string firstExpr = coreCoordExprForDim(spatialMappedDims[0]);
      std::string secondExpr = coreCoordExprForDim(spatialMappedDims[1]);
      if (firstExpr.empty() || secondExpr.empty())
        return;

      coreCoordArg0Expr = firstExpr;
      coreCoordArg1Expr = secondExpr;
      foundOrder = true;
    });
  }

  static MulticastKind classifyInterconnect(ArrayAttr interconnectAttr) {
    if (!interconnectAttr || interconnectAttr.empty())
      return MulticastKind::None;

    bool hasHorizontal = false;
    bool hasVertical = false;
    for (Attribute attr : interconnectAttr) {
      if (auto symRef = dyn_cast<FlatSymbolRefAttr>(attr)) {
        StringRef name = symRef.getValue();
        if (name == "horizontal_links")
          hasHorizontal = true;
        if (name == "vertical_links")
          hasVertical = true;
      }
    }

    if (hasHorizontal && hasVertical)
      return MulticastKind::All;
    if (hasHorizontal && !hasVertical)
      return MulticastKind::Horizontal;
    if (!hasHorizontal && hasVertical)
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

      MulticastKind opKind = classifyInterconnect(op.getInterconnect());
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

    for (auto [index, arg] : llvm::enumerate(originalFunc.getArguments())) {
      auto memrefType = dyn_cast<MemRefType>(arg.getType());
      if (!memrefType)
        continue;

      std::string tilesExpr;
      if (!buildTilesExpr(memrefType, tilesExpr))
        continue;

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
      std::string sizeExpr = "single_tile_size";
      if (tilesExpr != "1")
        sizeExpr += " * " + tilesExpr;
      bool isInput = isLoad;
      bool isOutput = isStore;
      MulticastKind multicastKind = MulticastKind::None;
      if (isInput)
        multicastKind = findInputMulticastKind(hostArg);
      std::string inputName =
          isInput ? ("in" + std::to_string(inputIdx++)) : "";

      dramInfos.push_back(DramBufferInfo{
          hostArg,
          memrefType,
          static_cast<unsigned>(index),
          static_cast<unsigned>(dramInfos.size()),
          /*cbIndex=*/-1,
          isInput,
          isOutput,
          multicastKind,
          inputName,
          "dram_config_" + letter,
          roleName + "_dram_buffer",
          letter + "_tiles_per_block",
          tilesExpr,
          sizeExpr});
    }
  }

  void collectCbInfos() {
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
      if (!buildTilesExpr(cbMemrefType, info.tilesExpr))
        return;
      info.allocValue = alloc.getResult();
      info.tilesVarName =
          "cb_tiles_per_block_" + std::to_string(cbInfos.size());

      auto semIt = semaphoresByAlloc.find(alloc.getResult());
      if (semIt != semaphoresByAlloc.end()) {
        for (Value semValue : semIt->second) {
          unsigned cbIndex = nextCbIndex++;
          info.cbIndices.push_back(cbIndex);
          info.cbIndexNames.push_back("CBIndex::c_" + std::to_string(cbIndex));
          semaphoreToCbIndex[semValue] = cbIndex;
        }
      }

      // Keep one fallback entry if an alloc has no explicit semaphore.
      if (info.cbIndices.empty()) {
        unsigned cbIndex = nextCbIndex++;
        info.cbIndices.push_back(cbIndex);
        info.cbIndexNames.push_back("CBIndex::c_" + std::to_string(cbIndex));
      }

      allocToCbInfo[alloc.getResult()] = static_cast<unsigned>(cbInfos.size());
      cbInfos.push_back(info);
    });

    // Resolve memref argument CB mapping using copy edges:
    // reinterpret_cast(arg) <-> (semaphore|alloc) <-> CBIndex.
    hostFunc.walk([&](::loom::CopyOp copyOp) {
      if (auto sourceRC =
              copyOp.getSource().getDefiningOp<memref::ReinterpretCastOp>()) {
        Value memrefArg = stripCasts(sourceRC.getSource());
        int cbIndex = resolveCbIndexFromEndpoint(copyOp.getDestination());
        if (cbIndex >= 0) {
          if (DramBufferInfo *dramInfo = findDramInfoMutable(memrefArg)) {
            if (dramInfo->cbIndex < 0)
              dramInfo->cbIndex = cbIndex;
          }
        }
      }

      if (auto destRC =
              copyOp.getDestination().getDefiningOp<memref::ReinterpretCastOp>()) {
        Value memrefArg = stripCasts(destRC.getSource());
        int cbIndex = resolveCbIndexFromEndpoint(copyOp.getSource());
        if (cbIndex >= 0) {
          if (DramBufferInfo *dramInfo = findDramInfoMutable(memrefArg)) {
            if (dramInfo->cbIndex < 0)
              dramInfo->cbIndex = cbIndex;
          }
        }
      }
    });

    // Final fallback for memrefs with no copy->alloc/semaphore mapping.
    for (DramBufferInfo &dramInfo : dramInfos) {
      if (dramInfo.cbIndex >= 0)
        continue;

      CircularBufferInfo info;
      info.allocValue = {};
      info.tilesExpr = dramInfo.tilesExpr;
      info.tilesVarName =
          "cb_tiles_per_block_" + std::to_string(cbInfos.size());
      unsigned cbIndex = nextCbIndex++;
      info.cbIndices.push_back(cbIndex);
      info.cbIndexNames.push_back("CBIndex::c_" + std::to_string(cbIndex));
      cbInfos.push_back(info);

      dramInfo.cbIndex = static_cast<int>(cbIndex);
    }

    // Runtime-arg CB tail must only include internal CBs (those that do not
    // correspond to memref arguments). Memref-linked CB IDs are already emitted
    // at fixed per-memref positions at the beginning of runtime args.
    SmallVector<unsigned, 8> memrefCbIndices;
    for (const DramBufferInfo &dramInfo : dramInfos) {
      if (dramInfo.cbIndex < 0)
        continue;
      unsigned cbIndex = static_cast<unsigned>(dramInfo.cbIndex);
      if (!llvm::is_contained(memrefCbIndices, cbIndex))
        memrefCbIndices.push_back(cbIndex);
    }

    internalCbRuntimeArgOrder.clear();
    for (const CircularBufferInfo &info : cbInfos) {
      for (unsigned cbIndex : info.cbIndices) {
        if (llvm::is_contained(memrefCbIndices, cbIndex))
          continue;
        if (llvm::is_contained(internalCbRuntimeArgOrder, cbIndex))
          continue;
        internalCbRuntimeArgOrder.push_back(cbIndex);
      }
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

  void emitLine(const std::string &line) {
    builder.create<emitc::VerbatimOp>(loc, line);
  }

  void emitPreamble() {
    // Host signature is emitted by MLIR->C++ translation with synthesized
    // parameter names v1..vN.
    // Argument order is:
    //   [all memrefs] [start_core_x start_core_y end_core_x end_core_y] [device]
    const size_t startCoreXArg = dramInfos.size() + 1;
    const size_t startCoreYArg = dramInfos.size() + 2;
    const size_t endCoreXArg = dramInfos.size() + 3;
    const size_t endCoreYArg = dramInfos.size() + 4;
    const size_t deviceArg = dramInfos.size() + 5;

    emitLine("IDevice* device = v" + std::to_string(deviceArg) + ";");
    emitLine("CommandQueue& cq = device->command_queue();");
    emitLine("Program program{};");
    emitLine("auto core_grid = device->compute_with_storage_grid_size();");
    emitLine("uint32_t start_core_x = (v" + std::to_string(startCoreXArg) +
             " == UINT32_MAX) ? 0u : v" + std::to_string(startCoreXArg) + ";");
    emitLine("uint32_t start_core_y = (v" + std::to_string(startCoreYArg) +
             " == UINT32_MAX) ? 0u : v" + std::to_string(startCoreYArg) + ";");
    emitLine("uint32_t end_core_x = (v" + std::to_string(endCoreXArg) +
             " == UINT32_MAX) ? static_cast<uint32_t>(core_grid.x - 1) : v" +
             std::to_string(endCoreXArg) + ";");
    emitLine("uint32_t end_core_y = (v" + std::to_string(endCoreYArg) +
             " == UINT32_MAX) ? static_cast<uint32_t>(core_grid.y - 1) : v" +
             std::to_string(endCoreYArg) + ";");
    emitLine("CoreRangeSet all_cores{ CoreRange{ {start_core_x, start_core_y}, {end_core_x, end_core_y} } };");
    emitLine("constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;");
    emitLine("const auto cb_data_format = tt::DataFormat::Float16_b;");
    emitLine("uint32_t cb_buffer_depth = 2;");
  }

  void emitDramBuffers() {
    for (const DramBufferInfo &info : dramInfos) {
      emitLine("tt_metal::InterleavedBufferConfig " + info.configName +
               "{.device = device, .size = " + info.sizeExpr +
               ", .page_size = single_tile_size, .buffer_type = "
               "tt_metal::BufferType::DRAM};");
      emitLine("auto " + info.bufferName +
               " = tt_metal::CreateBuffer(" + info.configName + ");");
    }
  }

  bool hasEmittedTilesVar(StringRef name) const {
    return llvm::is_contained(emittedTilesVars, name.str());
  }

  void annotateHostSignatureMetadata() {
    Builder b(hostFunc.getContext());
    hostFunc->setAttr(
        "loom.host_memref_count",
        b.getI32IntegerAttr(static_cast<int32_t>(dramInfos.size())));
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

      emitLine("tt_metal::CreateCircularBuffer(program, all_cores, "
               "CircularBufferConfig(" +
               info.tilesVarName +
               " * cb_buffer_depth * single_tile_size, {{" + cbEntries +
               "}})" + setPageSizeChain + ");");
    };

    for (const CircularBufferInfo &info : cbInfos)
      emitCbInfo(info);
  }

  void emitInputSemaphores() {
    for (const DramBufferInfo &info : dramInfos) {
      if (!info.isInput)
        continue;
      emitLine("auto " + info.inputName +
               "_mcast_sender_semaphore_addr = "
               "tt_metal::CreateSemaphore(program, all_cores, INVALID);");
      emitLine("auto " + info.inputName +
               "_mcast_receiver_semaphore_addr = "
               "tt_metal::CreateSemaphore(program, all_cores, INVALID);");
    }
  }

  void emitRuntimeEnqueueEpilogue() {
    for (const DramBufferInfo &info : dramInfos) {
      if (!info.isInput)
        continue;
      emitLine("EnqueueWriteBuffer(cq, " + info.bufferName + ", v" +
               std::to_string(info.argIndex + 1) + ".data(), false);");
    }

    emitLine("EnqueueProgram(cq, program, false);");

    SmallVector<const DramBufferInfo *, 4> outputInfos;
    for (const DramBufferInfo &info : dramInfos) {
      if (info.isOutput)
        outputInfos.push_back(&info);
    }

    for (auto [idx, info] : llvm::enumerate(outputInfos)) {
      bool isLast = (idx + 1 == outputInfos.size());
      emitLine("EnqueueReadBuffer(cq, " + info->bufferName + ", v" +
               std::to_string(info->argIndex + 1) + ".data(), " +
               (isLast ? "true" : "false") + ");");
    }
  }

  void emitCoreMulticastMappingAtEnd() {
    emitLine("uint32_t num_cores_with_work_c = end_core_x - start_core_x + 1;");
    emitLine("uint32_t num_cores_with_work_r = end_core_y - start_core_y + 1;");
    emitLine("constexpr bool row_major = true;");
    emitLine("auto cores = corerange_to_cores(all_cores, std::nullopt, row_major);");

    auto emitMulticastMappingForKind = [&](StringRef prefix,
                                           StringRef senderCoreExpr,
                                           StringRef destStartCoreExpr,
                                           StringRef destEndCoreExpr) {
      emitLine("CoreCoord " + prefix.str() + "_sender_core = " +
               senderCoreExpr.str() + ";");
      emitLine("CoreCoord " + prefix.str() + "_dest_start_core = " +
               destStartCoreExpr.str() + ";");
      emitLine("CoreCoord " + prefix.str() + "_dest_end_core = " +
               destEndCoreExpr.str() + ";");

      emitLine("auto " + prefix.str() +
               "_sender_physical = device->worker_core_from_logical_core(" +
               prefix.str() + "_sender_core);");
      emitLine("auto " + prefix.str() +
               "_dest_start_physical = device->worker_core_from_logical_core(" +
               prefix.str() + "_dest_start_core);");
      emitLine("auto " + prefix.str() +
               "_dest_end_physical = device->worker_core_from_logical_core(" +
               prefix.str() + "_dest_end_core);");

      emitLine("uint32_t " + prefix.str() +
               "_multicast_dest_noc_start_x = (std::uint32_t)" + prefix.str() +
               "_dest_start_physical.x;");
      emitLine("uint32_t " + prefix.str() +
               "_multicast_dest_noc_start_y = (std::uint32_t)" + prefix.str() +
               "_dest_start_physical.y;");
      emitLine("uint32_t " + prefix.str() +
               "_multicast_dest_noc_end_x = (std::uint32_t)" + prefix.str() +
               "_dest_end_physical.x;");
      emitLine("uint32_t " + prefix.str() +
               "_multicast_dest_noc_end_y = (std::uint32_t)" + prefix.str() +
               "_dest_end_physical.y;");
      emitLine("uint32_t " + prefix.str() +
               "_multicast_sender_noc_x = (std::uint32_t)" + prefix.str() +
               "_sender_physical.x;");
      emitLine("uint32_t " + prefix.str() +
               "_multicast_sender_noc_y = (std::uint32_t)" + prefix.str() +
               "_sender_physical.y;");
    };

    emitLine("for (const auto& core : cores) {");

    emitMulticastMappingForKind(
        "horizontal",
        "{(std::size_t)0, (std::size_t)core.y}",
        "{(std::size_t)1, (std::size_t)core.y}",
        "{(std::size_t)(num_cores_with_work_c - 1), (std::size_t)core.y}");

    emitMulticastMappingForKind(
        "vertical",
        "{(std::size_t)core.x, (std::size_t)0}",
        "{(std::size_t)core.x, (std::size_t)1}",
        "{(std::size_t)core.x, (std::size_t)(num_cores_with_work_r - 1)}");

    emitMulticastMappingForKind(
        "all",
        "{(std::size_t)0, (std::size_t)0}",
        "{(std::size_t)0, (std::size_t)0}",
        "{(std::size_t)(num_cores_with_work_c - 1), "
        "(std::size_t)(num_cores_with_work_r - 1)}");

    emitReaderRuntimeArgsForCore();
    emitLine("}");
  }

  void emitReaderRuntimeArgsForCore() {
    auto emitMulticastTuple = [&](const DramBufferInfo &info) {
      if (!info.isInput || info.multicastKind == MulticastKind::None) {
        emitLine("0,");
        emitLine("0,");
        emitLine("0,");
        emitLine("0,");
        emitLine("0,");
        emitLine("0,");
        emitLine("0,");
        emitLine("0,");
        emitLine("0,");
        return;
      }
      if (info.multicastKind == MulticastKind::All) {
        emitLine("all_multicast_dest_noc_start_x,");
        emitLine("all_multicast_dest_noc_start_y,");
        emitLine("all_multicast_dest_noc_end_x,");
        emitLine("all_multicast_dest_noc_end_y,");
        emitLine("(num_cores_with_work_c * num_cores_with_work_r - 1),");
        emitLine("all_multicast_sender_noc_x,");
        emitLine("all_multicast_sender_noc_y,");
      } else if (info.multicastKind == MulticastKind::Horizontal) {
        emitLine("horizontal_multicast_dest_noc_start_x,");
        emitLine("horizontal_multicast_dest_noc_start_y,");
        emitLine("horizontal_multicast_dest_noc_end_x,");
        emitLine("horizontal_multicast_dest_noc_end_y,");
        emitLine("(num_cores_with_work_c - 1),");
        emitLine("horizontal_multicast_sender_noc_x,");
        emitLine("horizontal_multicast_sender_noc_y,");
      } else {
        emitLine("vertical_multicast_dest_noc_start_x,");
        emitLine("vertical_multicast_dest_noc_start_y,");
        emitLine("vertical_multicast_dest_noc_end_x,");
        emitLine("vertical_multicast_dest_noc_end_y,");
        emitLine("(num_cores_with_work_r - 1),");
        emitLine("vertical_multicast_sender_noc_x,");
        emitLine("vertical_multicast_sender_noc_y,");
      }

      emitLine(info.inputName + "_mcast_sender_semaphore_addr,");
      emitLine(info.inputName + "_mcast_receiver_semaphore_addr,");
    };

    emitLine("std::vector<uint32_t> runtime_args_for_core = {");

    for (const DramBufferInfo &info : dramInfos) {
      unsigned cbIndex = info.cbIndex >= 0
                             ? static_cast<unsigned>(info.cbIndex)
                             : info.memrefOrdinal;
      emitLine("static_cast<uint32_t>(CBIndex::c_" +
               std::to_string(cbIndex) + "),");
      emitLine(info.bufferName + "->address(),");
      emitMulticastTuple(info);
    }

    emitLine(coreCoordArg0Expr + ",");
    emitLine(coreCoordArg1Expr + ",");

    // Append CB indexes for internal-only buffers. CB IDs tied to memref
    // arguments are already emitted in the per-memref prefix above.
    for (unsigned cbIndex : internalCbRuntimeArgOrder) {
      emitLine("static_cast<uint32_t>(CBIndex::c_" + std::to_string(cbIndex) +
               "),");
    }

    // Keep ordered input CB indexes as the final tail of runtime args.
    for (const DramBufferInfo &info : dramInfos) {
      if (!info.isInput)
        continue;
      unsigned cbIndex = info.cbIndex >= 0
                             ? static_cast<unsigned>(info.cbIndex)
                             : info.memrefOrdinal;
      emitLine("static_cast<uint32_t>(CBIndex::c_" +
               std::to_string(cbIndex) + "),");
    }

    emitLine("};");

    emitLine("tt_metal::SetRuntimeArgs(program, reader_id, core, runtime_args_for_core);");
    emitLine("tt_metal::SetRuntimeArgs(program, writer_id, core, runtime_args_for_core);");
    emitLine("tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, runtime_args_for_core);");
  }

  void emitCompileArgs() {
    emitLine("std::vector<uint32_t> compile_args = {};");
    for (const DramBufferInfo &info : dramInfos) {
      emitLine("tt::tt_metal::TensorAccessorArgs(*" + info.bufferName +
               ").append_to(compile_args);");
    }
  }

  void emitKernelRoles() {
    emitLine("MathFidelity math_fidelity = MathFidelity::HiFi4;");

    const SmallVector<KernelRoleInfo, 3> roles = {
        {"reader_id",
         "reader.cpp",
         "tt_metal::DataMovementConfig{.processor = "
         "DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, "
         ".compile_args = compile_args}"},
        {"writer_id",
         "writer.cpp",
         "tt_metal::DataMovementConfig{.processor = "
         "DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, "
         ".compile_args = compile_args}"},
        {"compute_kernel_id",
         "compute.cpp",
         "tt_metal::ComputeConfig{.math_fidelity = math_fidelity, "
         ".compile_args = compile_args}"}};

    for (const KernelRoleInfo &role : roles) {
      emitLine("auto " + role.idVarName + " = tt_metal::CreateKernel("
               "program, OVERRIDE_KERNEL_PREFIX "
               "\"mlir_matmul_simple/kernels/" +
               role.kernelSource + "\", all_cores, " + role.configExpr + ");");
    }
  }

  func::FuncOp originalFunc;
  func::FuncOp hostFunc;
  Location loc;
  OpBuilder builder;
  SmallVector<DramBufferInfo, 8> dramInfos;
  SmallVector<CircularBufferInfo, 8> cbInfos;
  SmallVector<unsigned, 8> internalCbRuntimeArgOrder;
  llvm::DenseMap<Value, unsigned> semaphoreToCbIndex;
  llvm::DenseMap<Value, unsigned> allocToCbInfo;
  unsigned nextCbIndex = 0;
  SmallVector<std::string, 8> emittedTilesVars;
  std::string coreCoordArg0Expr = "core.x";
  std::string coreCoordArg1Expr = "core.y";
};

/**
 * @brief Specialize a function for host-only execution (no compute/load/store).
 *
 * @details Clones the function with `__host` suffix and erases all
 *          compute operations (linalg.matmul etc.), load operations
 *          (memref.copy where source is reinterpret_cast), and store
 *          operations (memref.copy where target is reinterpret_cast).
 *          This creates a host function with only control flow and
 *          other non-compute/memory operations.
 *
 * @param func The original function to specialize.
 * @return The specialized host function.
 */
static func::FuncOp makeHostFunc(func::FuncOp func) {
  IRMapping mapping;
  auto hostFunc = cast<func::FuncOp>(func->clone(mapping));
  hostFunc.setName((func.getName() + "__host").str());

  TTMetalHostProgramEmitter emitter(func, hostFunc);
  emitter.run();
  return hostFunc;
}

/**
 * @brief Specializer class that manages function cloning and specialization.
 *
 * @details Follows the CoreSpecialize pattern from triton-tenstorrent.
 *          For each function, creates compute, reader, writer, and host
 *          specialized versions, inserts them into the module, and
 *          erases the original.
 */
class FunctionSpecializer {
public:
  FunctionSpecializer(ModuleOp module, func::FuncOp func)
      : module(module), originalFunc(func) {
    // Create specialized versions
    auto computeFunc = makeComputeFunc(func);
    auto readerFunc = makeReaderFunc(func);
    auto writerFunc = makeWriterFunc(func);
    auto hostFunc = makeHostFunc(func);

    // Attach TTKernel thread type attributes so downstream TTKernelToCpp
    // translation can recognize these as kernel entry points.
    auto *ctx = module.getContext();
    auto computeAttr = mlir::tt::ttkernel::ThreadTypeAttr::get(
        ctx, mlir::tt::ttkernel::ThreadType::Compute);
    auto nocAttr = mlir::tt::ttkernel::ThreadTypeAttr::get(
        ctx, mlir::tt::ttkernel::ThreadType::Noc);

    computeFunc->setAttr(mlir::tt::ttkernel::ThreadTypeAttr::name, computeAttr);
    readerFunc->setAttr(mlir::tt::ttkernel::ThreadTypeAttr::name, nocAttr);
    writerFunc->setAttr(mlir::tt::ttkernel::ThreadTypeAttr::name, nocAttr);
    hostFunc->setAttr(mlir::tt::ttkernel::ThreadTypeAttr::name, nocAttr);

    // Insert specialized functions into the module (before the original)
    module.insert(func, computeFunc);
    module.insert(func, readerFunc);
    module.insert(func, writerFunc);
    module.insert(func, hostFunc);
  }

private:
  ModuleOp module;
  func::FuncOp originalFunc;
};

} // namespace

void mlir::loom::specializeFunctionsForTTKernel(ModuleOp module) {
  // Collect functions to specialize (use make_early_inc_range since we modify
  // the module during iteration)
  for (func::FuncOp func :
       llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
    // Skip functions that are already specialized
    StringRef name = func.getName();
    if (name.ends_with("__compute") || name.ends_with("__reader") ||
        name.ends_with("__writer") || name.ends_with("__host"))
      continue;

    // Skip external/declaration-only functions
    if (func.isExternal())
      continue;

    // Ensure every reduction generic has a dedicated scaler semaphore input.
    // This avoids reusing reduction outputs as scaler CBs during compute lowering.
    ensureReductionScaleInputs(func);

    // Create specialized versions
    FunctionSpecializer specializer(module, func);

    // Erase the original function
    func.erase();
  }
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
