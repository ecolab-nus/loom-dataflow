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
        l1Symbol, copyOp.getBroadcastAttr(),
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
                         l1Symbol, dramSymbol,
                         /*broadcast=*/rewriter.getI64ArrayAttr({1, 1}),
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
    auto sourceType = cast<MemRefType>(source.getType());
    auto resultType = cast<RankedTensorType>(bufferizeOp.getType());

    // When the source memref has higher rank than the result tensor (rank-
    // reducing view that dropped unit dimensions), insert a
    // memref.collapse_shape to match ranks before replacing.
    if (sourceType.getRank() > resultType.getRank()) {
      // Build reassociation: group each unit dim with the next kept dim.
      // Kept dims are the non-unit static dims (and any dynamic dims).
      SmallVector<int64_t> srcShape(sourceType.getShape());
      int64_t srcRank = sourceType.getRank();
      int64_t dstRank = resultType.getRank();

      SmallVector<ReassociationIndices> reassoc;
      ReassociationIndices current;
      int64_t keptCount = 0;
      for (int64_t i = 0; i < srcRank; ++i) {
        current.push_back(i);
        bool isUnit = (srcShape[i] == 1);
        if (!isUnit || (keptCount == dstRank - 1)) {
          // This is a kept dimension — flush the group.
          if (keptCount < dstRank) {
            reassoc.push_back(current);
            current.clear();
            ++keptCount;
          }
        }
      }
      // Remaining trailing dims go into the last group.
      if (!current.empty() && !reassoc.empty()) {
        reassoc.back().insert(reassoc.back().end(), current.begin(),
                              current.end());
      }

      auto collapsedType = MemRefType::get(resultType.getShape(),
                                           resultType.getElementType());
      source = memref::CollapseShapeOp::create(rewriter, op->getLoc(),
                                               collapsedType, source, reassoc);
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

/// ReduceSumOp implements DestinationStyleOpInterface.  The base class
/// DstBufferizableOpInterfaceExternalModel provides bufferizesToMemoryRead,
/// bufferizesToMemoryWrite, and getAliasingValues.  We only need to supply
/// the bufferize() method.
struct ReduceSumOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          ReduceSumOpInterface, loom::ReduceSumOp> {

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto reduceOp = cast<loom::ReduceSumOp>(op);

    FailureOr<Value> inputBuffer =
        getBuffer(rewriter, reduceOp.getInput(), options, state);
    if (failed(inputBuffer))
      return failure();

    FailureOr<Value> initBuffer =
        getBuffer(rewriter, reduceOp.getInit(), options, state);
    if (failed(initBuffer))
      return failure();

    // Create memref-mode ReduceSumOp (no results — pure buffer semantics).
    loom::ReduceSumOp::create(rewriter, op->getLoc(), /*resultTypes=*/TypeRange{},
                              *inputBuffer, *initBuffer, reduceOp.getUlX(),
                              reduceOp.getUlY(), reduceOp.getLrX(),
                              reduceOp.getLrY());

    // The init buffer IS the result (DPS in-place semantics).
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
  loom::ReduceSumOp::attachInterface<ReduceSumOpInterface>(*ctx);
}
