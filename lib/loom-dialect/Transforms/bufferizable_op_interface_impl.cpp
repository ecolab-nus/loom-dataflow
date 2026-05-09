#include "Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"

#include "LoomDialect.h.inc"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;
using namespace mlir::bufferization;
using namespace loom;

namespace {

static DenseI64ArrayAttr toDenseI64ArrayAttr(OpBuilder &builder,
                                             ArrayAttr attr) {
  SmallVector<int64_t> values;
  for (Attribute value : attr)
    values.push_back(cast<IntegerAttr>(value).getInt());
  return builder.getDenseI64ArrayAttr(values);
}

struct InitTensorOpInterface
    : public BufferizableOpInterface::ExternalModel<InitTensorOpInterface,
                                                    loom::InitTensorOp> {
  bool bufferizesToMemoryRead(Operation * /*op*/, OpOperand & /*opOperand*/,
                              const AnalysisState & /*state*/) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation * /*op*/, OpOperand & /*opOperand*/,
                               const AnalysisState & /*state*/) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand & /*opOperand*/,
                                      const AnalysisState & /*state*/) const {
    return {AliasingValue(op->getOpResult(0), BufferRelation::Equivalent,
                          /*isDefinite=*/true)};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions & /*options*/,
                          BufferizationState & /*state*/) const {
    auto initOp = cast<loom::InitTensorOp>(op);
    replaceOpWithBufferizedValues(rewriter, op, initOp.getBuffer());
    return success();
  }
};

struct CopyToTensorOpInterface
    : public BufferizableOpInterface::ExternalModel<CopyToTensorOpInterface,
                                                    loom::CopyToTensorOp> {
  bool bufferizesToMemoryRead(Operation * /*op*/, OpOperand & /*opOperand*/,
                              const AnalysisState & /*state*/) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation * /*op*/, OpOperand & /*opOperand*/,
                               const AnalysisState & /*state*/) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand & /*opOperand*/,
                                      const AnalysisState & /*state*/) const {
    return {AliasingValue(op->getOpResult(0), BufferRelation::Equivalent,
                          /*isDefinite=*/true)};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions & /*options*/,
                          BufferizationState & /*state*/) const {
    auto copyOp = cast<loom::CopyToTensorOp>(op);
    auto loc = op->getLoc();

    auto dramSymbol = SymbolRefAttr::get(op->getContext(), "DRAM");
    auto l1Symbol = SymbolRefAttr::get(op->getContext(), "L1");

    loom::CopyOp::create(
        rewriter, loc, copyOp.getSourceView(), copyOp.getBuffer(),
        dramSymbol, l1Symbol, ValueRange{},
        toDenseI64ArrayAttr(rewriter, copyOp.getBroadcastAttr()),
        mlir::Value{}, mlir::Value{}, mlir::Value{}, mlir::Value{});

    replaceOpWithBufferizedValues(rewriter, op, copyOp.getBuffer());
    return success();
  }
};

struct CopyFromTensorOpInterface
    : public BufferizableOpInterface::ExternalModel<CopyFromTensorOpInterface,
                                                    loom::CopyFromTensorOp> {
  bool bufferizesToMemoryRead(Operation * /*op*/, OpOperand & /*opOperand*/,
                              const AnalysisState & /*state*/) const {
    return true; // reads from source_tensor
  }

  bool bufferizesToMemoryWrite(Operation * /*op*/, OpOperand & /*opOperand*/,
                               const AnalysisState & /*state*/) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation * /*op*/,
                                      OpOperand & /*opOperand*/,
                                      const AnalysisState & /*state*/) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto copyOp = cast<loom::CopyFromTensorOp>(op);
    auto loc = op->getLoc();

    FailureOr<Value> srcBuffer =
        getBuffer(rewriter, copyOp.getSourceTensor(), options, state);
    if (failed(srcBuffer))
      return failure();

    auto l1Symbol = SymbolRefAttr::get(op->getContext(), "L1");
    auto dramSymbol = SymbolRefAttr::get(op->getContext(), "DRAM");

    loom::CopyOp::create(rewriter, loc, *srcBuffer, copyOp.getTargetView(),
                         l1Symbol, dramSymbol, ValueRange{},
                         /*staticArea=*/rewriter.getDenseI64ArrayAttr({1, 1}),
                         mlir::Value{}, mlir::Value{},
                         mlir::Value{}, mlir::Value{});

    rewriter.eraseOp(op);
    return success();
  }
};

/// loom.bufferize_to_tensor is a pure alias from a memref to a tensor.
/// OSB can eliminate it by substituting the source buffer for the result.
struct BufferizeToTensorOpInterface
    : public BufferizableOpInterface::ExternalModel<
          BufferizeToTensorOpInterface, loom::BufferizeToTensorOp> {
  bool bufferizesToMemoryRead(Operation * /*op*/, OpOperand & /*opOperand*/,
                              const AnalysisState & /*state*/) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation * /*op*/, OpOperand & /*opOperand*/,
                               const AnalysisState & /*state*/) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand & /*opOperand*/,
                                      const AnalysisState & /*state*/) const {
    return {AliasingValue(op->getOpResult(0), BufferRelation::Equivalent,
                          /*isDefinite=*/true)};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions & /*options*/,
                          BufferizationState & /*state*/) const {
    auto bufferizeOp = cast<loom::BufferizeToTensorOp>(op);
    Value source = bufferizeOp.getSource();
    auto srcMemrefType = cast<MemRefType>(source.getType());
    auto resultTensorType =
        cast<RankedTensorType>(bufferizeOp.getResult().getType());

    // If the source memref rank exceeds the result tensor rank, insert a
    // collapse_shape to match ranks.  This occurs for the scalar (rank-0)
    // tensor case where the L1 alloc carries a unit dim (e.g. memref<1xf16>
    // backing tensor<f16>).  All static dims in the source must be 1 for this
    // to be valid, which is guaranteed by the unit-dim-only rank difference.
    if (srcMemrefType.getRank() > resultTensorType.getRank()) {
      SmallVector<ReassociationIndices> reassoc;
      auto targetType = MemRefType::get(resultTensorType.getShape(),
                                        srcMemrefType.getElementType());
      source = memref::CollapseShapeOp::create(rewriter, op->getLoc(),
                                               targetType, source, reassoc);
    }

    replaceOpWithBufferizedValues(rewriter, op, source);
    return success();
  }
};

/// loom.bufferize_to_memref is a pure alias from a tensor to a memref.
/// OSB can eliminate it by substituting the underlying buffer for the result.
struct BufferizeToMemrefOpInterface
    : public BufferizableOpInterface::ExternalModel<
          BufferizeToMemrefOpInterface, loom::BufferizeToMemrefOp> {
  bool bufferizesToMemoryRead(Operation * /*op*/, OpOperand & /*opOperand*/,
                              const AnalysisState & /*state*/) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation * /*op*/, OpOperand & /*opOperand*/,
                               const AnalysisState & /*state*/) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation * /*op*/,
                                      OpOperand & /*opOperand*/,
                                      const AnalysisState & /*state*/) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto bufferizeOp = cast<loom::BufferizeToMemrefOp>(op);
    FailureOr<Value> srcBuffer =
        getBuffer(rewriter, bufferizeOp.getSource(), options, state);
    if (failed(srcBuffer))
      return failure();
    replaceOpWithBufferizedValues(rewriter, op, *srcBuffer);
    return success();
  }
};

/// BroadcastOp bufferizes with in-place semantics on the init/outs buffer.
struct BroadcastOpInterface
    : public BufferizableOpInterface::ExternalModel<BroadcastOpInterface,
                                                    loom::BroadcastOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState & /*state*/) const {
    auto broadcastOp = cast<loom::BroadcastOp>(op);
    // DPS-style: broadcast reads source and writes destination.
    // `init` is destination-only and should not force preservation copies.
    return opOperand.get() == broadcastOp.getIns();
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState & /*state*/) const {
    auto broadcastOp = cast<loom::BroadcastOp>(op);
    return opOperand.get() == broadcastOp.getInit();
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState & /*state*/) const {
    auto broadcastOp = cast<loom::BroadcastOp>(op);
    if (opOperand.get() != broadcastOp.getInit())
      return {};
    if (op->getNumResults() == 0)
      return {};
    // Broadcast result is a view of destination buffer with possibly different
    // logical shape/strides. It is a definite alias, but not necessarily
    // equivalent type/layout to init.
    return {AliasingValue(op->getOpResult(0), BufferRelation::Unknown,
                          /*isDefinite=*/true)};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto broadcastOp = cast<loom::BroadcastOp>(op);

    FailureOr<Value> insBuffer =
        getBuffer(rewriter, broadcastOp.getIns(), options, state);
    if (failed(insBuffer))
      return failure();

    FailureOr<Value> initBuffer =
        getBuffer(rewriter, broadcastOp.getInit(), options, state);
    if (failed(initBuffer))
      return failure();

    SmallVector<Type> resultTypes;
    if (op->getNumResults() != 0) {
      auto resultTensorType =
          dyn_cast<TensorType>(op->getResult(0).getType());
      if (!resultTensorType)
        return failure();
      resultTypes.push_back(getMemRefType(resultTensorType, options));
    }

    auto newOp = loom::BroadcastOp::create(rewriter, op->getLoc(), resultTypes,
                                           *insBuffer, *initBuffer,
                                           broadcastOp.getDimAttr());

    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
      return success();
    }

    replaceOpWithBufferizedValues(rewriter, op, newOp->getResults());
    return success();
  }
};

/// SyncOp implements DestinationStyleOpInterface and bufferizes to a memref
/// form (no results) while preserving DPS in-place semantics.
struct SyncOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          SyncOpInterface, loom::SyncOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState & /*state*/) const {
    (void)op;
    return opOperand.getOperandNumber() == 0;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState & /*state*/) const {
    (void)op;
    return opOperand.getOperandNumber() == 1;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState & /*state*/) const {
    if (opOperand.getOperandNumber() != 1 || op->getNumResults() == 0)
      return {};

    return {AliasingValue(op->getOpResult(0), BufferRelation::Equivalent,
                          /*isDefinite=*/true)};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto syncOp = cast<loom::SyncOp>(op);

    auto resolveBuffer = [&](Value v) -> FailureOr<Value> {
      if (isa<BaseMemRefType>(v.getType()))
        return v;
      return getBuffer(rewriter, v, options, state);
    };

    FailureOr<Value> insBuffer = resolveBuffer(syncOp.getIns());
    if (failed(insBuffer))
      return failure();

    FailureOr<Value> initBuffer = resolveBuffer(syncOp.getInit());
    if (failed(initBuffer))
      return failure();

    loom::SyncOp::create(rewriter, op->getLoc(), /*resultTypes=*/TypeRange{},
                         *insBuffer, *initBuffer);

    replaceOpWithBufferizedValues(rewriter, op, *initBuffer);
    return success();
  }
};

} // namespace

void loom::registerBufferizableOpInterfaceExternalModels(MLIRContext *ctx) {
  loom::InitTensorOp::attachInterface<InitTensorOpInterface>(*ctx);
  loom::CopyToTensorOp::attachInterface<CopyToTensorOpInterface>(*ctx);
  loom::CopyFromTensorOp::attachInterface<CopyFromTensorOpInterface>(*ctx);
  loom::BufferizeToTensorOp::attachInterface<BufferizeToTensorOpInterface>(
      *ctx);
  loom::BufferizeToMemrefOp::attachInterface<BufferizeToMemrefOpInterface>(
      *ctx);
  loom::BroadcastOp::attachInterface<BroadcastOpInterface>(*ctx);
  loom::SyncOp::attachInterface<SyncOpInterface>(*ctx);
}
