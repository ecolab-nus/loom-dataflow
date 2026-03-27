#include "Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"

#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;
using namespace mlir::bufferization;
using namespace loom;

namespace {

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
        rewriter, loc, copyOp.getSourceView(), copyOp.getBuffer(), dramSymbol,
        l1Symbol, copyOp.getInterconnectAttr(), copyOp.getBroadcastAttr());

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
                         l1Symbol, dramSymbol,
                         /*interconnect=*/rewriter.getArrayAttr({}),
                         /*broadcast=*/rewriter.getI64ArrayAttr({1, 1}));

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
    replaceOpWithBufferizedValues(rewriter, op, bufferizeOp.getSource());
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

} // namespace

void loom::registerBufferizableOpInterfaceExternalModels(MLIRContext *ctx) {
  loom::InitTensorOp::attachInterface<InitTensorOpInterface>(*ctx);
  loom::CopyToTensorOp::attachInterface<CopyToTensorOpInterface>(*ctx);
  loom::CopyFromTensorOp::attachInterface<CopyFromTensorOpInterface>(*ctx);
  loom::BufferizeToTensorOp::attachInterface<BufferizeToTensorOpInterface>(
      *ctx);
  loom::BufferizeToMemrefOp::attachInterface<BufferizeToMemrefOpInterface>(
      *ctx);
}
