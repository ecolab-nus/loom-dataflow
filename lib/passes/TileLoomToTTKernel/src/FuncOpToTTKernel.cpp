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

LogicalResult mlir::loom::CompileArgTracker::processInputArgs(
    func::FuncOp func, TypeConverter &typeConverter, OpBuilder &rewriter) {
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

  // Process each function argument.
  for (BlockArgument arg : entry.getArguments()) {
    Type argType = arg.getType();

    if (isa<MemRefType, UnrankedMemRefType>(argType)) {
      // Memref type: create CB and base address.
      // CB uses nextCompileArgIndex, base address uses nextCompileArgIndex + 1.
      int64_t tensorAccessorArgsIndex = getNextTensorAccessorArgsIndex(func);

      // Create GetArgValOp for CB.
      auto cbType = typeConverter.convertType(argType);
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

  // Collect all compute, load, and store operations to erase
  SmallVector<Operation *, 8> opsToErase;
  hostFunc.walk([&](Operation *op) {
    if (isComputeOp(op)) {
      opsToErase.push_back(op);
    }  else if (auto loomCopyOp = dyn_cast<::loom::CopyOp>(op)) {
      if (isLoomLoadOp(loomCopyOp) || isLoomStoreOp(loomCopyOp)) {
        opsToErase.push_back(op);
      }
    }
  });

  // Remove all collected operations first
  for (Operation *op : llvm::reverse(opsToErase)) {
    op->erase();
  }

  // Clean up unused reinterpret_cast ops that were only used by erased ops
  SmallVector<Operation *, 8> unusedRcOps;
  hostFunc.walk([&](memref::ReinterpretCastOp rcOp) {
    if (rcOp.getResult().use_empty()) {
      unusedRcOps.push_back(rcOp);
    }
  });

  for (Operation *op : unusedRcOps) {
    op->erase();
  }

  // Create DRAM buffers for input and output
  // Calculate single_tile_size: typically sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH = 2 * 32 * 32 = 2048
  Location loc = hostFunc.getLoc();
  OpBuilder builder(hostFunc.getContext());
  builder.setInsertionPointToStart(&hostFunc.getBody().front());

  // Create emitc statements for command queue and program
  builder.create<emitc::VerbatimOp>(
      loc, "CommandQueue& cq = device->command_queue();");
  builder.create<emitc::VerbatimOp>(loc, "Program program{};");

  // Create emitc variable for single_tile_size calculation
  builder.create<emitc::VerbatimOp>(loc, "constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;");
  
  // Create DeviceLocalBufferConfig struct with initializer list
  std::string dramConfigStr = std::string("distributed::DeviceLocalBufferConfig dram_config{") +
                               ".page_size = single_tile_size" + 
                               ", .buffer_type = tt_metal::BufferType::DRAM};";
  builder.create<emitc::VerbatimOp>(loc, dramConfigStr);
  
  // Create ReplicatedBufferConfig for each memref input argument
  // Iterate through original function arguments to get memref types
  // Map argument index to buffer config name
  SmallVector<std::pair<int, std::string>> argIndexToConfig; // Store (argIndex, configName) pairs
  
  for (auto [argIndex, arg] : llvm::enumerate(func.getArguments())) {
    Type argType = arg.getType();
    
    if (auto memrefType = dyn_cast<MemRefType>(argType)) {
      // Get memref shape/dimensions
      ArrayRef<int64_t> shape = memrefType.getShape();
      
      // Build size expression: single_tile_size * dim0 * dim1 * ...
      std::string sizeExpr = "single_tile_size";
      bool hasDynamicDim = false;
      
      for (int64_t dim : shape) {
        if (dim == ShapedType::kDynamic) {
          hasDynamicDim = true;
          break;
        }
        sizeExpr += " * " + std::to_string(dim);
      }
      
      // Skip if has dynamic dimensions (would need runtime values)
      if (hasDynamicDim) {
        continue;
      }
      
      // Generate variable name: buffer_config_A, buffer_config_B, etc.
      char varName = 'A' + argIndexToConfig.size();
      std::string configName = "buffer_config_" + std::string(1, varName);
      std::string bufferConfigStr = "distributed::ReplicatedBufferConfig " + configName + 
                                     "{.size = " + sizeExpr + "};";
      
      builder.create<emitc::VerbatimOp>(loc, bufferConfigStr);
      argIndexToConfig.push_back({argIndex, configName});
    }
  }
  
  // Create MeshBuffer objects for each buffer config
  // Name based on argument index: dram_buffer_0, dram_buffer_1, etc.
  for (const auto &[argIndex, configName] : argIndexToConfig) {
    std::string bufferName = "dram_buffer_" + std::to_string(argIndex);
    std::string meshBufferStr = "auto " + bufferName + " = distributed::MeshBuffer::create(" +
                                configName + ", dram_config, mesh_device.get());";
    builder.create<emitc::VerbatimOp>(loc, meshBufferStr);
  }
  
  // Create circular buffers for each memref argument
  // Calculate tiles_per_block from memref dimensions and create CircularBufferConfig
  builder.create<emitc::VerbatimOp>(loc, "const auto cb_data_format = tt::DataFormat::Float16_b;");
  builder.create<emitc::VerbatimOp>(loc, "uint32_t cb_buffer_depth = 2;");
  builder.create<emitc::VerbatimOp>(loc, "MathFidelity math_fidelity = MathFidelity::HiFi4;");
  
  // CBIndex mapping: arg 0 -> c_0, arg 1 -> c_1, arg 2 -> c_16, etc.
  SmallVector<int> cbIndexMap = {0, 1, 16}; // Default mapping, can be extended
  
  // Generate tiles_per_block variables and print statement
  std::string printArgs;
  for (const auto &[argIndex, configName] : argIndexToConfig) {
    // Get the memref type for this argument to extract dimensions
    BlockArgument arg = func.getArgument(argIndex);
    auto memrefType = cast<MemRefType>(arg.getType());
    ArrayRef<int64_t> shape = memrefType.getShape();
    
    // Calculate tiles_per_block: multiply all dimensions
    // For 2D: [Mt, Kt] -> Mt * Kt, [Kt, Nt] -> Kt * Nt, [Mt, Nt] -> Mt * Nt
    std::string tilesPerBlockExpr;
    bool firstDim = true;
    for (int64_t dim : shape) {
      if (dim == ShapedType::kDynamic) {
        // Skip if dynamic - would need runtime value
        tilesPerBlockExpr = "";
        break;
      }
      if (!firstDim) {
        tilesPerBlockExpr += " * ";
      }
      tilesPerBlockExpr += std::to_string(dim);
      firstDim = false;
    }
    
    if (tilesPerBlockExpr.empty()) {
      continue; // Skip if has dynamic dimensions
    }
    
    // Get CBIndex for this argument
    int cbIndex = (argIndex < cbIndexMap.size()) ? cbIndexMap[argIndex] : argIndex;
    std::string cbIndexStr = "CBIndex::c_" + std::to_string(cbIndex);
    
    // Generate variable name for tiles_per_block
    char varName = 'A' + argIndex;
    std::string tilesVarName = std::string(1, varName) + "_tiles_per_block";
    std::string tilesDecl = "const uint32_t " + tilesVarName + " = " + tilesPerBlockExpr + ";";
    builder.create<emitc::VerbatimOp>(loc, tilesDecl);
    
    // Collect for print statement
    if (!printArgs.empty()) {
      printArgs += ", ";
    }
    printArgs += tilesVarName;
  }

  // Create CircularBufferConfig and call CreateCircularBuffer for each memref
  for (const auto &[argIndex, configName] : argIndexToConfig) {
    // Get the memref type for this argument to extract dimensions
    BlockArgument arg = func.getArgument(argIndex);
    auto memrefType = cast<MemRefType>(arg.getType());
    ArrayRef<int64_t> shape = memrefType.getShape();
    
    // Calculate tiles_per_block: multiply all dimensions
    std::string tilesPerBlockExpr;
    bool firstDim = true;
    for (int64_t dim : shape) {
      if (dim == ShapedType::kDynamic) {
        tilesPerBlockExpr = "";
        break;
      }
      if (!firstDim) {
        tilesPerBlockExpr += " * ";
      }
      tilesPerBlockExpr += std::to_string(dim);
      firstDim = false;
    }
    
    if (tilesPerBlockExpr.empty()) {
      continue; // Skip if has dynamic dimensions
    }
    
    // Get CBIndex for this argument
    int cbIndex = (argIndex < cbIndexMap.size()) ? cbIndexMap[argIndex] : argIndex;
    std::string cbIndexStr = "CBIndex::c_" + std::to_string(cbIndex);
    
    // Generate variable name for tiles_per_block (matching the declaration above)
    char varName = 'A' + argIndex;
    std::string tilesVarName = std::string(1, varName) + "_tiles_per_block";
    
    // Create CircularBufferConfig and call CreateCircularBuffer
    std::string cbConfigStr = "tt_metal::CreateCircularBuffer("
                              "program, "
                              "all_cores, "
                              "CircularBufferConfig(" + tilesVarName + " * cb_buffer_depth * " + 
                              "single_tile_size" + ", {{" + cbIndexStr + ", cb_data_format}})"
                              ".set_page_size(" + cbIndexStr + ", single_tile_size));";
    builder.create<emitc::VerbatimOp>(loc, cbConfigStr);
  }
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
