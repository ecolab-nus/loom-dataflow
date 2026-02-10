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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

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

static Value stripMemrefCasts(Value value) {
  Value current = value;
  while (auto cast = current.getDefiningOp<memref::CastOp>())
    current = cast.getSource();
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
      Value destination = stripMemrefCasts(copyOp.getDestination());
      if (auto allocOp = destination.getDefiningOp<::loom::AllocOp>()) {
        if (auto allocMemrefType = dyn_cast<MemRefType>(allocOp.getType()))
          status = recordLinkedType(linkedArg, allocMemrefType);
      }
    }

    // L1 -> DRAM: source is loom.alloc, destination is reinterpret_cast(arg).
    if (auto destinationRC =
            copyOp.getDestination().getDefiningOp<memref::ReinterpretCastOp>()) {
      Value linkedArg = stripMemrefCasts(destinationRC.getSource());
      Value source = stripMemrefCasts(copyOp.getSource());
      if (auto allocOp = source.getDefiningOp<::loom::AllocOp>()) {
        if (auto allocMemrefType = dyn_cast<MemRefType>(allocOp.getType()))
          status = recordLinkedType(linkedArg, allocMemrefType);
      }
    }
  });

  return status;
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
  bool isReaderKernel = func.getName().ends_with("reader");


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
        //TODO, need to generate code like "ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>" instead of ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>
        mcast_receiver_semaphore_addr_ptr = CastToL1PtrOp::create(rewriter, loc, mcast_receiver_semaphore_addr_op);
        
        mcast_sender_semaphore_addr_ptr = CastToL1PtrOp::create(rewriter, loc, mcast_sender_semaphore_addr_op);
        // Store 1 to the semaphore pointer: *(mcast_receiver_semaphore_addr_ptr) = 1;
        Value oneValue = rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 1);
        Value zeroOffset = rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 0);
        if (isReaderKernel){
          StoreToL1Op::create(rewriter, loc, oneValue, mcast_receiver_semaphore_addr_ptr, zeroOffset);
        }
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

void mlir::loom::CompileArgTracker::clearCoreList(Operation *funcOp) { funcToCoreList[funcOp].clear(); }

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
  return isa<linalg::MatmulOp>(op);
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
    collectDramInfos();
    collectCbInfos();
    eraseNonHostOps();
    eraseDeadReinterpretCasts();

    builder.setInsertionPointToStart(&hostFunc.getBody().front());
    emitPreamble();
    emitDramBuffers();
    emitCircularBuffers();
    emitCompileArgs();
  }

private:
  struct DramBufferInfo {
    Value hostArg;
    MemRefType type;
    unsigned argIndex;
    unsigned memrefOrdinal;
    std::string configName;
    std::string bufferName;
    std::string tilesVarName;
    std::string tilesExpr;
    std::string sizeExpr;
  };

  struct CircularBufferInfo {
    int memrefOrdinal;
    std::string tilesVarName;
    std::string tilesExpr;
    std::string cbIndexName;
  };

  static Value stripCasts(Value value) {
    Value curr = value;
    while (auto cast = curr.getDefiningOp<memref::CastOp>())
      curr = cast.getSource();
    return curr;
  }

  static Value findInputMemref(::loom::AllocOp alloc) {
    for (Operation *user : alloc.getResult().getUsers()) {
      if (auto loomCopy = dyn_cast<::loom::CopyOp>(user)) {
        if (loomCopy.getDestination() != alloc.getResult())
          continue;
        if (auto rc = loomCopy.getSource().getDefiningOp<memref::ReinterpretCastOp>())
          return stripCasts(rc.getSource());
      }
    }
    return {};
  }

  static Value findOutputMemref(::loom::AllocOp alloc) {
    for (Operation *user : alloc.getResult().getUsers()) {
      if (auto loomCopy = dyn_cast<::loom::CopyOp>(user)) {
        if (loomCopy.getSource() != alloc.getResult())
          continue;
        if (auto rc = loomCopy.getDestination().getDefiningOp<memref::ReinterpretCastOp>())
          return stripCasts(rc.getSource());
      }
    }
    return {};
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

  static bool buildTilesExpr(MemRefType memrefType, std::string &expr) {
    expr.clear();
    bool first = true;
    for (int64_t dim : memrefType.getShape()) {
      if (dim == ShapedType::kDynamic)
        return false;
      if (!first)
        expr += " * ";
      expr += std::to_string(dim) + " / TILE_HEIGHT";
      first = false;
    }
    if (expr.empty())
      expr = "1";
    return true;
  }

  const DramBufferInfo *findDramInfo(Value hostArg) const {
    for (const DramBufferInfo &info : dramInfos) {
      if (info.hostArg == hostArg)
        return &info;
    }
    return nullptr;
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

      dramInfos.push_back(DramBufferInfo{
          hostArg,
          memrefType,
          static_cast<unsigned>(index),
          static_cast<unsigned>(dramInfos.size()),
          "dram_config_" + letter,
          roleName + "_dram_buffer",
          letter + "_tiles_per_block",
          tilesExpr,
          sizeExpr});
    }
  }

  void collectCbInfos() {
    unsigned fallbackIdx = 0;

    hostFunc.walk([&](::loom::AllocOp alloc) {
      auto cbMemrefType = dyn_cast<MemRefType>(alloc.getType());
      if (!cbMemrefType)
        return;

      Value linked = findInputMemref(alloc);
      if (!linked)
        linked = findOutputMemref(alloc);
      if (linked)
        linked = stripCasts(linked);

      CircularBufferInfo info;
      if (!buildTilesExpr(cbMemrefType, info.tilesExpr))
        return;
      info.tilesVarName =
          "cb_tiles_per_block_" + std::to_string(cbInfos.size());

      if (const DramBufferInfo *dramInfo = findDramInfo(linked)) {
        info.memrefOrdinal = static_cast<int>(dramInfo->memrefOrdinal);
        info.cbIndexName =
            "CBIndex::c_" + std::to_string(dramInfo->memrefOrdinal);
      } else {
        info.memrefOrdinal = -1;
        unsigned fallbackCbIndex =
            static_cast<unsigned>(dramInfos.size() + fallbackIdx);
        info.cbIndexName = "CBIndex::c_" + std::to_string(fallbackCbIndex);
        ++fallbackIdx;
      }

      cbInfos.push_back(info);
    });
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
    emitLine("CommandQueue& cq = device->command_queue();");
    emitLine("Program program{};");
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

  void emitCircularBuffers() {
    auto emitCbInfo = [&](const CircularBufferInfo &info) {
      if (!hasEmittedTilesVar(info.tilesVarName)) {
        emitLine("const uint32_t " + info.tilesVarName + " = " + info.tilesExpr +
                 ";");
        emittedTilesVars.push_back(info.tilesVarName);
      }

      emitLine("tt_metal::CreateCircularBuffer(program, all_cores, "
               "CircularBufferConfig(" +
               info.tilesVarName +
               " * cb_buffer_depth * single_tile_size, {{" + info.cbIndexName +
               ", cb_data_format}}).set_page_size(" + info.cbIndexName +
               ", single_tile_size));");
    };

    // Emit CBs linked to DRAM args in stable memref order (A, B, C, ...).
    for (const DramBufferInfo &dramInfo : dramInfos) {
      for (const CircularBufferInfo &info : cbInfos) {
        if (info.memrefOrdinal == static_cast<int>(dramInfo.memrefOrdinal))
          emitCbInfo(info);
      }
    }

    // Emit any fallback CBs not linked to a DRAM argument.
    for (const CircularBufferInfo &info : cbInfos) {
      if (info.memrefOrdinal < 0)
        emitCbInfo(info);
    }
  }

  void emitCompileArgs() {
    emitLine("std::vector<uint32_t> compile_args = {};");
    for (const DramBufferInfo &info : dramInfos) {
      emitLine("tt::tt_metal::TensorAccessorArgs(*" + info.bufferName +
               ").append_to(compile_args);");
    }
  }

  func::FuncOp originalFunc;
  func::FuncOp hostFunc;
  Location loc;
  OpBuilder builder;
  SmallVector<DramBufferInfo, 8> dramInfos;
  SmallVector<CircularBufferInfo, 8> cbInfos;
  SmallVector<std::string, 8> emittedTilesVars;
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
