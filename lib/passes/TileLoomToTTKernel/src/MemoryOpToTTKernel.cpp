/**
 * @file MemoryOpToTTKernel.cpp
 * @brief Implementation for memory operation to TT kernel conversion pass.
 * @details
 * This pass processes memory operations whose destination allocations carry
 * `{loom.alloc ...}` attributes and records their base address information
 * using the DataLoaderInfo structure.
 */

 #include "MemoryOpToTTKernel.h"

 #include "mlir/Dialect/Affine/IR/AffineOps.h"
 #include "mlir/Dialect/Func/IR/FuncOps.h"
 #include "mlir/Dialect/MemRef/IR/MemRef.h"
 #include "mlir/Dialect/Arith/IR/Arith.h"
 #include "mlir/Dialect/SCF/IR/SCF.h"
 #include "mlir/IR/Attributes.h"
 #include "mlir/IR/Operation.h"
 #include "mlir/IR/Value.h"
 #include "mlir/Transforms/DialectConversion.h"
 #include "llvm/ADT/DenseMap.h"
 #include "llvm/ADT/SmallVector.h"
 #include "llvm/Support/Casting.h"
 #include <memory>
 #include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
 #include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
 #include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
 #include "ttmlir/Dialect/TTCore/IR/TTCore.h"
 
 using namespace mlir;
 using namespace mlir::loom;
 using namespace tt::ttkernel;
 using namespace tt::ttcore;
 
 namespace {
 
 /**
  * @brief Per-function tracking data for compile-arg indices.
  *
  * @details Each function gets its own tracking state so that compile-arg
  *          indices start from 0 for each specialized function.
  */
 struct FunctionTrackingData {
   /// Map from input memref to its compile-arg indices.
   llvm::DenseMap<Value, std::pair<int64_t, int64_t>> inputToData;
   /// Map from output memref to its compile-arg indices.
   llvm::DenseMap<Value, std::pair<int64_t, int64_t>> outputToData;
   /// Map from alloc result to its assigned CB index.
   llvm::DenseMap<Value, int64_t> allocToCbIndex;
   /// Next compile-arg index (even indices are CBs; odd are base addrs).
   int64_t nextCompileArgIndex = 0;
 };

 /**
  * @brief Tracks input memrefs and assigns compile-arg indices for CB/base addr.
  *
  * @details Each input memref is assigned a pair of compile-arg indices:
  *          - cbIndex: index for the L1 circular buffer.
  *          - baseAddrIndex: index for the input base address (cbIndex + 1).
  *          Each function maintains its own independent set of indices.
  */
 class InputDataTracker {
 public:
   /**
    * @brief Encapsulates compile-arg indices for an input memref.
    */
   struct InputData {
     int64_t cbIndex;
     int64_t baseAddrIndex;
   };
 
   /**
    * @brief Get or create a CB index for the given L1 alloc.
    * @param alloc The `memref.alloc` op annotated with `{loom.alloc ...}`.
    * @return A stable, non-negative compile-arg index for this allocation.
    */
   int64_t getOrCreateAlloc(memref::AllocOp alloc) {
     auto *funcData = getOrCreateFuncData(alloc);
     Value v = alloc.getResult();
     auto it = funcData->allocToCbIndex.find(v);
     if (it != funcData->allocToCbIndex.end())
       return it->second;
     int64_t cbIndex = funcData->nextCompileArgIndex;
     funcData->nextCompileArgIndex += 2;
     funcData->allocToCbIndex.try_emplace(v, cbIndex);
     return cbIndex;
   }
 
   /**
    * @brief Get or create input data for the given input memref.
    * @param inputMemref Input memref value (typically a function argument).
    * @param alloc The L1 alloc associated with this input.
    * @return InputData containing CB index and base address index.
    */
   InputData getOrCreate(Value inputMemref, memref::AllocOp alloc) {
     auto *funcData = getOrCreateFuncData(alloc);
     auto it = funcData->inputToData.find(inputMemref);
     if (it != funcData->inputToData.end())
       return InputData{it->second.first, it->second.second};
     int64_t cbIndex = getOrCreateAlloc(alloc);
     int64_t baseAddrIndex = cbIndex + 1;
     funcData->inputToData.try_emplace(inputMemref, std::make_pair(cbIndex, baseAddrIndex));
     return InputData{cbIndex, baseAddrIndex};
   }
 
   /**
    * @brief Get or create output data for the given output memref.
    *
    * @details Stores use the output (DRAM) memref to recover a compile-arg index
    *          for the base address. Unlike inputs, the output memref may not be
    *          associated with a `{loom.alloc ...}` L1 allocation, so we assign a
    *          fresh (cbIndex, baseAddrIndex) pair keyed by the output memref
    *          value itself.
    *
    * @param outputMemref Output memref value (typically a function argument).
    * @param op The operation requesting the output data (used to find parent function).
    * @return InputData containing a stable CB index and base address index.
    */
   InputData getOrCreateOutput(Value outputMemref, Operation *op) {
     auto *funcData = getOrCreateFuncData(op);
     auto it = funcData->outputToData.find(outputMemref);
     if (it != funcData->outputToData.end())
       return InputData{it->second.first, it->second.second};
     int64_t cbIndex = funcData->nextCompileArgIndex;
     funcData->nextCompileArgIndex += 2;
     funcData->outputToData.try_emplace(outputMemref, std::make_pair(cbIndex, cbIndex + 1));
     return InputData{cbIndex, cbIndex + 1};
   }
 
 private:
   /// Get the parent function of an operation.
   func::FuncOp getParentFunc(Operation *op) const {
     return op->getParentOfType<func::FuncOp>();
   }

   /// Get or create tracking data for the function containing the given operation.
   FunctionTrackingData *getOrCreateFuncData(Operation *op) {
     func::FuncOp func = getParentFunc(op);
     assert(func && "Operation must be inside a function");
     auto it = funcToData.find(func);
     if (it != funcToData.end())
       return &it->second;
     return &funcToData.try_emplace(func, FunctionTrackingData{}).first->second;
   }

   /// Map from function to its tracking data.
   llvm::DenseMap<func::FuncOp, FunctionTrackingData> funcToData;
 };
 
 } // namespace
 /**
  * @brief Convert `memref.copy` with `loom.copy.choice` into TTKernel load ops.
  *
  * @details The conversion uses `DataLoaderInfo` to recover the base memref and
  *          offset from the source `memref.reinterpret_cast`. It then emits a
  *          TTKernel NOC read sequence to populate the destination circular
  *          buffer (CB). The destination is expected to be type-converted to a
  *          TTKernel CB type by the surrounding conversion pipeline.
  */
 struct ConvertLoadOp : public OpConversionPattern<memref::CopyOp> {
   using OpConversionPattern<memref::CopyOp>::OpConversionPattern;
 
   /**
    * @brief Construct the pattern with a type converter and context.
    * @param typeConverter Type converter for the conversion pipeline.
    * @param context MLIR context.
    */
   ConvertLoadOp(TypeConverter &typeConverter, MLIRContext *context,
                 std::shared_ptr<InputDataTracker> inputTracker)
       : OpConversionPattern<memref::CopyOp>(typeConverter, context),
         inputTracker(std::move(inputTracker)) {}
 
   /**
    * @brief Match and rewrite `memref.copy` into TTKernel NOC read operations.
    *
    * @details This pattern handles `memref.copy` operations where the source
    *          is a `memref.reinterpret_cast` (indicating data flows from external
    *          memory to L1/CB). It emits TTKernel NOC async read operations to
    *          load data from DRAM into the circular buffer.
    */
   LogicalResult
   matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
     // Load: source must be a reinterpret_cast (reading from external memory).
     Value source = op.getSource();
     if (!source.getDefiningOp<memref::ReinterpretCastOp>())
       return failure();
 
     Location loc = op.getLoc();
     Value cb;
     memref::AllocOp targetAlloc = op.getTarget().getDefiningOp<memref::AllocOp>();
     cb = rewriter.getRemappedValue(op.getTarget());

 
     auto opInsertionPt = rewriter.saveInsertionPoint();
     rewriter.setInsertionPointAfterValue(cb);
 
     auto dataFormat = GetDataFormatOp::create(rewriter, loc, cb);
     auto pageSize = GetTileSizeOp::create(rewriter, loc, cb);
 
     // Get the base address and offset from memref.reinterpret_cast
     // (source is already known to be a reinterpret_cast from the match check above)
     Value offset;
     Value inputMemref = source;
     auto reinterpretCastOp = source.getDefiningOp<memref::ReinterpretCastOp>();
     {
       inputMemref = reinterpretCastOp.getSource();
       // Get the offset from reinterpret_cast (use first offset if multiple)
       auto offsets = reinterpretCastOp.getOffsets();
       if (!offsets.empty()) {
         offset = offsets[0];
       } else {
         // If no dynamic offsets, check static offsets
         auto mixedOffsets = reinterpretCastOp.getMixedOffsets();
         if (!mixedOffsets.empty() && mixedOffsets[0].is<Attribute>()) {
           // Static offset - convert to value
           auto staticOffset = llvm::cast<IntegerAttr>(mixedOffsets[0].get<Attribute>());
           offset = rewriter.create<arith::ConstantIndexOp>(loc, staticOffset.getInt());
         } else {
           // Fallback: use constant 0 (index type)
           offset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
         }
       }
     }
 
     // If the source comes from a memref input and the target is an L1 alloc,
     // materialize the base address from a compile-time arg adjacent to the CB.
         const auto inputData =
             inputTracker->getOrCreate(inputMemref, targetAlloc);
         auto baseAddrIdxAttr = rewriter.getI32IntegerAttr(
             static_cast<int32_t>(inputData.baseAddrIndex));
         auto baseAddr = rewriter.create<GetCompileArgValOp>(
             loc, rewriter.getI32Type(), baseAddrIdxAttr);
 
     Value trueVal = rewriter.create<arith::ConstantIntOp>(
         loc, rewriter.getI1Type(), 1);
     Value addrGen = GetInterleavedAddrGenFastOp::create(
         rewriter, loc, /*dram=*/trueVal, baseAddr, pageSize, dataFormat);
 
     rewriter.restoreInsertionPoint(opInsertionPt);
 
     // convert bytes offset to tile index
     //
     // arith.divui requires operands/results to have the same type.
     // Our `offset` is typically `index` (from memref.reinterpret_cast), while
     // TTKernel tile size is `i32`. Convert offset to `i32` before dividing.
     Value offsetI32 = offset;
     if (offsetI32 && offsetI32.getType().isIndex()) {
       offsetI32 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(),
                                                       offsetI32);
     }
     Value baseTileIndex =
         arith::DivUIOp::create(rewriter, loc, offsetI32, pageSize);
 
     Value const0 = rewriter.create<arith::ConstantIntOp>(
         loc, rewriter.getI32Type(), 0);
 
     // determine how many tiles we need to load by converting the shape to tiles
     auto cbType = cast<CBType>(cb.getType());
     Value numPages;
     
     // Check if CB is tiled (element type is TileType) or scalar
     auto elementType = cbType.getElementType();
     if (auto tileType = llvm::dyn_cast<TileType>(elementType)) {
       // Tiled CB: use getNumTiles() directly
       const int32_t numTiles = cbType.getNumTiles();
       numPages = rewriter.create<arith::ConstantIntOp>(
           loc, rewriter.getI32Type(), numTiles);
     } else {
       // Scalar CB: calculate numTiles = (numElements * elementSizeInBytes) / pageSizeInBytes
       // Get number of elements and element size
       const int64_t numElements = cbType.getNumElements();
       
       // Get element size in bytes
       int32_t elementSizeBytes = 0;
       if (elementType.isF32()) {
         elementSizeBytes = 4;
       } else if (elementType.isF16() || elementType.isBF16()) {
         elementSizeBytes = 2;
       } else if (auto intType = llvm::dyn_cast<IntegerType>(elementType)) {
         elementSizeBytes = (intType.getWidth() + 7) / 8; // Round up to bytes
       } else {
         // Default: try to infer from type
         // For now, assume 4 bytes if unknown
         elementSizeBytes = 4;
       }
       
       // Calculate total size in bytes and divide by page size
       Value totalSizeBytes = rewriter.create<arith::ConstantIntOp>(
           loc, rewriter.getI32Type(), numElements * elementSizeBytes);
       numPages = arith::DivUIOp::create(rewriter, loc, totalSizeBytes, pageSize);
     }
     CBReserveBackOp::create(rewriter, loc, cb, numPages);
 
     Value l1Addr = GetWritePtrOp::create(rewriter, loc, cb);
 
     scf::ForOp loadTileLoop = scf::ForOp::create(
         rewriter, loc, rewriter.create<arith::ConstantIntOp>(
                           loc, rewriter.getI32Type(), 0),
         numPages,
         rewriter.create<arith::ConstantIntOp>(
             loc, rewriter.getI32Type(), 1),
         ValueRange{l1Addr, baseTileIndex});
     {
       rewriter.setInsertionPointToStart(loadTileLoop.getBody());
       Value crtL1Address = loadTileLoop.getRegionIterArgs()[0];
       Value crtTileIndex = loadTileLoop.getRegionIterArgs()[1];
       Value nocAddr = InterleavedAddrGenFastGetNocAddrOp::create(
           rewriter, loc, addrGen, crtTileIndex, const0, Value());
       NocAsyncReadOp::create(rewriter, loc, nocAddr, crtL1Address,
                                        pageSize);
       Value nextL1Address =
           arith::AddIOp::create(rewriter, loc, crtL1Address, pageSize);
       Value nextTileIndex =
           arith::AddIOp::create(rewriter, loc, crtTileIndex,
                                 rewriter.create<arith::ConstantIntOp>(
                                     loc, rewriter.getI32Type(), 1));
       scf::YieldOp::create(rewriter, loc,
                            ValueRange{nextL1Address, nextTileIndex});
     }
 
     rewriter.setInsertionPointAfter(loadTileLoop);
     NocAsyncReadBarrierOp::create(rewriter, loc);
 
 
     rewriter.eraseOp(op);
     return success();
   }
 
 private:
   /// Shared input tracker for base address indices.
   std::shared_ptr<InputDataTracker> inputTracker;
 };
 
 /**
  * @brief Convert `memref.copy` with `loom.copy.choice` into TTKernel store ops.
  *
  * @details The conversion handles store operations where data flows from L1/CB
  *          (source) to external memory (target). It uses the target memref to
  *          recover base address information and emits a TTKernel NOC write
  *          sequence to transfer data from the circular buffer to DRAM.
  */
 struct ConvertStoreOp : public OpConversionPattern<memref::CopyOp> {
   using OpConversionPattern<memref::CopyOp>::OpConversionPattern;
 
   /**
    * @brief Construct the pattern with a type converter and context.
    * @param typeConverter Type converter for the conversion pipeline.
    * @param context MLIR context.
    */
   ConvertStoreOp(TypeConverter &typeConverter, MLIRContext *context,
                  std::shared_ptr<InputDataTracker> inputTracker)
       : OpConversionPattern<memref::CopyOp>(typeConverter, context),
         inputTracker(std::move(inputTracker)) {}
 
   /**
    * @brief Match and rewrite `memref.copy` into TTKernel NOC write operations.
    *
    * @details This pattern handles `memref.copy` operations where the source
    *          is NOT a `memref.reinterpret_cast` (indicating data flows from
    *          L1/CB to external memory). It emits TTKernel NOC async write
    *          operations to transfer data from the circular buffer to DRAM.
    */
   LogicalResult
   matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
     // Store: DRAM side is the target reinterpret_cast (writing from L1/CB to external memory).
     Value target = op.getTarget();
     if (!target.getDefiningOp<memref::ReinterpretCastOp>())
       return failure();    
     Location loc = op.getLoc();
     
     // L1 side: materialize the CB from the compile-time args associated with
     // the DRAM memref (the reinterpret_cast source), matching the load path.
     Value outputMemref = target;
     if (auto reinterpretCastOp = target.getDefiningOp<memref::ReinterpretCastOp>())
       outputMemref = reinterpretCastOp.getSource();

      auto cbIdxAttr = rewriter.getI32IntegerAttr(
          static_cast<int32_t>(inputTracker->getOrCreateOutput(outputMemref, op).cbIndex));
      auto memrefType =
          cast<CBType>(typeConverter->convertType(op.getSource().getType()));
      auto cb = rewriter.create<GetCompileArgValOp>(loc, memrefType, cbIdxAttr);

 
     auto opInsertionPt = rewriter.saveInsertionPoint();
     rewriter.setInsertionPointAfterValue(cb);
 
     auto dataFormat = GetDataFormatOp::create(rewriter, loc, cb);
     auto pageSize = GetTileSizeOp::create(rewriter, loc, cb);
 
     // Get the base address and offset from the target (external memory)
     Value offset;
     if (auto reinterpretCastOp = target.getDefiningOp<memref::ReinterpretCastOp>()) {
       // Get the offset from reinterpret_cast (use first offset if multiple)
       auto offsets = reinterpretCastOp.getOffsets();
       if (!offsets.empty()) {
         offset = offsets[0];
       } else {
         // If no dynamic offsets, check static offsets
         auto mixedOffsets = reinterpretCastOp.getMixedOffsets();
         if (!mixedOffsets.empty() && mixedOffsets[0].is<Attribute>()) {
           // Static offset - convert to value
           auto staticOffset = llvm::cast<IntegerAttr>(mixedOffsets[0].get<Attribute>());
           offset = rewriter.create<arith::ConstantIndexOp>(loc, staticOffset.getInt());
         } else {
           // Fallback: use constant 0 (index type)
           offset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
         }
       }
     } else {
       // No reinterpret_cast - use offset 0
       offset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
     }
 
     // Materialize the base address from a compile-time arg associated with the
     // output (DRAM) memref (typically the "last input memref" function arg).
     const auto outputData = inputTracker->getOrCreateOutput(outputMemref, op);
     auto baseAddrIdxAttr = rewriter.getI32IntegerAttr(
         static_cast<int32_t>(outputData.baseAddrIndex));
     auto baseAddr = rewriter.create<GetCompileArgValOp>(
         loc, rewriter.getI32Type(), baseAddrIdxAttr);
 
     Value trueVal = rewriter.create<arith::ConstantIntOp>(
         loc, rewriter.getI1Type(), 1);
     Value addrGen = GetInterleavedAddrGenFastOp::create(
         rewriter, loc, /*dram=*/trueVal, baseAddr, pageSize, dataFormat);
 
     rewriter.restoreInsertionPoint(opInsertionPt);
 
     // Convert bytes offset to tile index.
     // arith.divui requires operands/results to have the same type.
     Value offsetI32 = offset;
     if (offsetI32 && offsetI32.getType().isIndex()) {
       offsetI32 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(),
                                                       offsetI32);
     }
     Value baseTileIndex =
         arith::DivUIOp::create(rewriter, loc, offsetI32, pageSize);
 
     Value const0 = rewriter.create<arith::ConstantIntOp>(
         loc, rewriter.getI32Type(), 0);
 
     // Determine how many tiles we need to store.
     auto cbType = cast<CBType>(cb.getType());
     Value numPages;
 
     // Check if CB is tiled (element type is TileType) or scalar
     auto elementType = cbType.getElementType();
     if (auto tileType = llvm::dyn_cast<TileType>(elementType)) {
       // Tiled CB: use getNumTiles() directly
       const int32_t numTiles = cbType.getNumTiles();
       numPages = rewriter.create<arith::ConstantIntOp>(
           loc, rewriter.getI32Type(), numTiles);
     } else {
       // Scalar CB: calculate numTiles = (numElements * elementSizeInBytes) / pageSizeInBytes
       const int64_t numElements = cbType.getNumElements();
 
       // Get element size in bytes
       int32_t elementSizeBytes = 0;
       if (elementType.isF32()) {
         elementSizeBytes = 4;
       } else if (elementType.isF16() || elementType.isBF16()) {
         elementSizeBytes = 2;
       } else if (auto intType = llvm::dyn_cast<IntegerType>(elementType)) {
         elementSizeBytes = (intType.getWidth() + 7) / 8; // Round up to bytes
       } else {
         // Default: assume 4 bytes if unknown
         elementSizeBytes = 4;
       }
 
       // Calculate total size in bytes and divide by page size
       Value totalSizeBytes = rewriter.create<arith::ConstantIntOp>(
           loc, rewriter.getI32Type(), numElements * elementSizeBytes);
       numPages = arith::DivUIOp::create(rewriter, loc, totalSizeBytes, pageSize);
     }
 
     // Get read pointer for store (reading from CB to write to DRAM)
     Value l1Addr = GetReadPtrOp::create(rewriter, loc, cb);
 
     scf::ForOp storeTileLoop = scf::ForOp::create(
         rewriter, loc,
         rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 0),
         numPages,
         rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 1),
         ValueRange{l1Addr, baseTileIndex});
     {
       rewriter.setInsertionPointToStart(storeTileLoop.getBody());
       Value crtL1Address = storeTileLoop.getRegionIterArgs()[0];
       Value crtTileIndex = storeTileLoop.getRegionIterArgs()[1];
 
       Value nocAddr = InterleavedAddrGenFastGetNocAddrOp::create(
           rewriter, loc, addrGen, crtTileIndex, const0, Value());
 
       // Use NocAsyncWriteOp instead of NocAsyncReadOp
       NocAsyncWriteOp::create(rewriter, loc, crtL1Address, nocAddr, pageSize);
 
       Value nextL1Address =
           arith::AddIOp::create(rewriter, loc, crtL1Address, pageSize);
       Value nextTileIndex =
           arith::AddIOp::create(rewriter, loc, crtTileIndex,
                                 rewriter.create<arith::ConstantIntOp>(
                                     loc, rewriter.getI32Type(), 1));
       scf::YieldOp::create(rewriter, loc,
                            ValueRange{nextL1Address, nextTileIndex});
     }
 
     rewriter.setInsertionPointAfter(storeTileLoop);
     NocAsyncWriteBarrierOp::create(rewriter, loc);
 
    // Pop the tiles from the CB after writing
    CBPopFrontOp::create(rewriter, loc, cb, numPages);

    // Remove the original memref.copy op.
    rewriter.eraseOp(op);
    return success();
   }
  private:
    /// Shared input tracker for base address indices.
    std::shared_ptr<InputDataTracker> inputTracker;
  };
 
 struct ConvertReuseReinterpretCastOp
     : public OpConversionPattern<memref::ReinterpretCastOp> {
   using OpConversionPattern<memref::ReinterpretCastOp>::OpConversionPattern;
 
   LogicalResult
   matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
     // Only handle casts carrying the loom.reuse attribute.
     auto reuseAttr = op->getAttrOfType<DictionaryAttr>("loom.reuse");
     if (!reuseAttr)
       return failure();
 
     rewriter.eraseOp(op);
     return success();
   }
 };
 
 
 struct ConvertAllocOp : public OpConversionPattern<memref::AllocOp> {
   using OpConversionPattern<memref::AllocOp>::OpConversionPattern;
 
   /**
    * @brief Construct the pattern with a type converter and context.
    * @param typeConverter Type converter for the conversion pipeline.
    * @param context MLIR context.
    */
   ConvertAllocOp(TypeConverter &typeConverter, MLIRContext *context,
                  std::shared_ptr<InputDataTracker> tracker)
       : OpConversionPattern<memref::AllocOp>(typeConverter, context),
         tracker(std::move(tracker)) {}
 
   /**
    * @brief Match and rewrite `memref.alloc` with `{loom.alloc ...}`.
    *
    * @details This pattern is intended to recognize allocations that are meant
    *          to become TTKernel circular buffers (CBs) later in the pipeline.
    *          For now, it is a no-op and only serves as a hook point for future
    *          lowering.
    */
   LogicalResult
   matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
     // Only handle memref.alloc annotated with {loom.alloc ...}.
     auto allocAttr = op->getAttrOfType<DictionaryAttr>("loom.alloc");
     if (!allocAttr)
       return failure();
 
     Location loc = op.getLoc();
     // Assign a stable index based on the number of tracked allocations so far.
     int64_t allocIdx = tracker->getOrCreateAlloc(op);
     auto idxAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(allocIdx));
     auto memrefType =
         cast<CBType>(typeConverter->convertType(op.getResult().getType()));
     auto cb =
         rewriter.create<GetCompileArgValOp>(loc, memrefType, idxAttr);
     
     //auto opInsertionPt = rewriter.saveInsertionPoint();
     //rewriter.setInsertionPointAfterValue(cb);
 
     //auto dataFormat = GetDataFormatOp::create(rewriter, loc, cb);
     //auto pageSize = GetTileSizeOp::create(rewriter, loc, cb);
 
     // Replace all uses of the alloc result with the CB value
     rewriter.replaceOp(op, cb);
     //rewriter.eraseOp(op);
     return success();
   }
 
 private:
   /// Shared input tracker used to assign deterministic indices.
   std::shared_ptr<InputDataTracker> tracker;
 };
 
 void loom::populateMemoryOpConversionPatterns(RewritePatternSet &patterns,
                                              TypeConverter &typeConverter,
                                              MLIRContext *context) {
   auto inputTracker = std::make_shared<InputDataTracker>();
   patterns.add<ConvertAllocOp>(typeConverter, context, inputTracker);
   patterns.add<ConvertLoadOp>(typeConverter, context, inputTracker);
   patterns.add<ConvertStoreOp>(typeConverter, context, inputTracker);
   patterns.add<ConvertReuseReinterpretCastOp>(typeConverter, context);
 }