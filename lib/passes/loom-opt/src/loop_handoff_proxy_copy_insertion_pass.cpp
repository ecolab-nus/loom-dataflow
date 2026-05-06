#include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "utils.h"
#include <optional>

#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace {

struct LoopHandoffProxy {
  Value loopResult;
  Operation *loopOp;
  unsigned resultIndex;
  SmallVector<OpOperand *, 4> uses;
};

static std::optional<std::pair<Operation *, unsigned>>
getLoopResultKey(Value value) {
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return std::nullopt;

  Operation *defOp = result.getOwner();
  if (!isa<scf::ForOp, affine::AffineForOp>(defOp))
    return std::nullopt;

  return std::make_pair(defOp, result.getResultNumber());
}

static void addProxyUse(Value value, OpOperand *operand,
                        SmallVectorImpl<LoopHandoffProxy> &proxies) {
  auto key = getLoopResultKey(value);
  if (!key)
    return;

  for (LoopHandoffProxy &proxy : proxies) {
    if (proxy.loopOp == key->first && proxy.resultIndex == key->second) {
      proxy.uses.push_back(operand);
      return;
    }
  }

  LoopHandoffProxy proxy;
  proxy.loopResult = value;
  proxy.loopOp = key->first;
  proxy.resultIndex = key->second;
  proxy.uses.push_back(operand);
  proxies.push_back(std::move(proxy));
}

static Value createProxyCopy(OpBuilder &builder, Location loc,
                             Value loopResult) {
  auto tensorType = cast<RankedTensorType>(loopResult.getType());
  SmallVector<loom::utils::SymbolicDim, 4> tracedShape =
      loom::utils::traceShape(loopResult);

  SmallVector<Value, 4> dynamicSizes;
  for (auto [idx, dim] : llvm::enumerate(tensorType.getShape())) {
    if (!ShapedType::isDynamic(dim))
      continue;

    Value dynamicSize;
    if (idx < tracedShape.size()) {
      if (auto tracedValue = dyn_cast<Value>(tracedShape[idx])) {
        dynamicSize = tracedValue;
      } else if (auto attr = dyn_cast<Attribute>(tracedShape[idx])) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr);
            intAttr && intAttr.getInt() >= 0) {
          dynamicSize = arith::ConstantIndexOp::create(
              builder, loc, intAttr.getInt());
        }
      }
    }

    if (!dynamicSize)
      dynamicSize = tensor::DimOp::create(builder, loc, loopResult, idx);
    dynamicSizes.push_back(dynamicSize);
  }

  auto empty = tensor::EmptyOp::create(builder, loc, tensorType.getShape(),
                                       tensorType.getElementType(),
                                       dynamicSizes);
  auto copy = linalg::CopyOp::create(builder, loc, loopResult,
                                     empty.getResult());
  return copy.getResult(0);
}

class LoopHandoffProxyCopyInsertionPass
    : public PassWrapper<LoopHandoffProxyCopyInsertionPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LoopHandoffProxyCopyInsertionPass)

  StringRef getArgument() const override {
    return "loom-loop-handoff-proxy-copy-insertion";
  }

  StringRef getDescription() const override {
    return "Insert post-loop linalg.copy proxy tensors for communication "
           "handoffs";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, bufferization::BufferizationDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect, loom::LoomDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<LoopHandoffProxy, 8> proxies;

    module.walk([&](Operation *op) {
      if (auto gatherOp = dyn_cast<loom::GatherOp>(op)) {
        addProxyUse(gatherOp.getIns(), &gatherOp->getOpOperand(0), proxies);
        return;
      }

      if (auto toMemrefOp = dyn_cast<loom::BufferizeToMemrefOp>(op)) {
        addProxyUse(toMemrefOp.getSource(), &toMemrefOp->getOpOperand(0),
                    proxies);
        return;
      }

      if (auto toBufferOp = dyn_cast<bufferization::ToBufferOp>(op)) {
        addProxyUse(toBufferOp.getTensor(), &toBufferOp->getOpOperand(0),
                    proxies);
        return;
      }
    });

    for (LoopHandoffProxy &proxy : proxies) {
      OpBuilder builder(proxy.loopOp);
      builder.setInsertionPointAfter(proxy.loopOp);

      Value proxyTensor =
          createProxyCopy(builder, proxy.loopOp->getLoc(), proxy.loopResult);
      for (OpOperand *use : proxy.uses)
        use->set(proxyTensor);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
loom::passes::createLoopHandoffProxyCopyInsertionPass() {
  return std::make_unique<LoopHandoffProxyCopyInsertionPass>();
}
