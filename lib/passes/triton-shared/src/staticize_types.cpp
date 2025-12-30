/**
 * @file staticize_types.cpp
 * @brief Implementation of staticize types pass.
 * @details
 * This pass converts operations with dynamic shapes to use static shapes
 * when the dynamic sizes can be resolved to constants. It handles:
 * - memref.alloc: Converts dynamic allocations to static when sizes are constants
 * - tensor.empty: Converts dynamic tensor creation to static when sizes are constants
 * - bufferization.to_tensor: Updates output type to match staticized memref input
 * - linalg.fill: Updates output type to match staticized tensor
 * - linalg.matmul: Updates output type to match staticized operand types
 * - linalg.generic: Updates output type to match staticized operand types
 */

#include "staticize_types.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

/// Try to extract a constant integer value from a Value.
static std::optional<int64_t> getConstantIntValue(Value v) {
  APInt value;
  if (matchPattern(v, m_ConstantInt(&value))) {
    return value.getSExtValue();
  }
  return std::nullopt;
}

/// Check if all values in the range are constants and extract them.
/// Returns true if all values are constants.
static bool extractConstantSizes(ValueRange dynamicSizes,
                                 SmallVectorImpl<int64_t> &staticSizes) {
  for (Value v : dynamicSizes) {
    auto constVal = getConstantIntValue(v);
    if (!constVal) {
      return false;
    }
    staticSizes.push_back(*constVal);
  }
  return true;
}

/// Pattern to convert dynamic memref.alloc to static memref.alloc.
struct StaticizeAllocPattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {
    auto memRefType = op.getType();

    // Skip if already fully static
    if (memRefType.hasStaticShape()) {
      return failure();
    }

    // Check if all dynamic sizes are constants
    ValueRange dynamicSizes = op.getDynamicSizes();
    if (dynamicSizes.empty()) {
      return failure();
    }

    SmallVector<int64_t> constantSizes;
    if (!extractConstantSizes(dynamicSizes, constantSizes)) {
      return failure();
    }

    // Build the new static shape by replacing dynamic dims with constants
    SmallVector<int64_t> newShape;
    auto shape = memRefType.getShape();
    unsigned dynamicIdx = 0;
    for (int64_t dim : shape) {
      if (ShapedType::isDynamic(dim)) {
        newShape.push_back(constantSizes[dynamicIdx++]);
      } else {
        newShape.push_back(dim);
      }
    }

    // Create the new static memref type
    auto newType = MemRefType::get(newShape, memRefType.getElementType(),
                                   memRefType.getLayout(),
                                   memRefType.getMemorySpace());

    // Create a new alloc operation with static type (no dynamic sizes)
    auto newAlloc = memref::AllocOp::create(
        rewriter, op.getLoc(), newType, ValueRange{}, op.getSymbolOperands(),
        op.getAlignmentAttr());

    // Copy attributes from the original operation
    for (auto attr : op->getAttrs()) {
      if (attr.getName() != "operandSegmentSizes" &&
          attr.getName() != "alignment") {
        newAlloc->setAttr(attr.getName(), attr.getValue());
      }
    }

    rewriter.replaceOp(op, newAlloc.getResult());
    return success();
  }
};

/// Pattern to convert dynamic tensor.empty to static tensor.empty.
struct StaticizeTensorEmptyPattern : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::EmptyOp op,
                                PatternRewriter &rewriter) const override {
    auto tensorType = op.getType();

    // Skip if already fully static
    if (tensorType.hasStaticShape()) {
      return failure();
    }

    // Check if all dynamic sizes are constants
    ValueRange dynamicSizes = op.getDynamicSizes();
    if (dynamicSizes.empty()) {
      return failure();
    }

    SmallVector<int64_t> constantSizes;
    if (!extractConstantSizes(dynamicSizes, constantSizes)) {
      return failure();
    }

    // Build the new static shape
    SmallVector<int64_t> newShape;
    auto shape = tensorType.getShape();
    unsigned dynamicIdx = 0;
    for (int64_t dim : shape) {
      if (ShapedType::isDynamic(dim)) {
        newShape.push_back(constantSizes[dynamicIdx++]);
      } else {
        newShape.push_back(dim);
      }
    }

    // Create the new static tensor type
    auto newType = RankedTensorType::get(newShape, tensorType.getElementType(),
                                         tensorType.getEncoding());

    // Create a new empty tensor with static type
    auto newEmpty = tensor::EmptyOp::create(rewriter, op.getLoc(), newType,
                                            ValueRange{});

    rewriter.replaceOp(op, newEmpty.getResult());
    return success();
  }
};

/// Pattern to update bufferization.to_tensor output type when input is static.
struct StaticizeToTensorPattern
    : public OpRewritePattern<bufferization::ToTensorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::ToTensorOp op,
                                PatternRewriter &rewriter) const override {
    // Get the memref operand (single operand for ToTensorOp)
    Value memrefOperand = op.getOperand();
    auto memrefType = llvm::dyn_cast<MemRefType>(memrefOperand.getType());
    if (!memrefType) {
      return failure();
    }

    auto tensorType =
        llvm::dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!tensorType) {
      return failure();
    }

    // Only update if memref is static but tensor is dynamic
    if (!memrefType.hasStaticShape() || tensorType.hasStaticShape()) {
      return failure();
    }

    // Create new static tensor type from memref shape
    auto newTensorType = RankedTensorType::get(memrefType.getShape(),
                                               memrefType.getElementType());

    auto newOp = bufferization::ToTensorOp::create(
        rewriter, op.getLoc(), newTensorType, memrefOperand, op.getRestrict(),
        op.getWritable());

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// Pattern to update linalg.fill output type when output operand is static.
struct StaticizeFillPattern : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getOutputs().empty()) {
      return failure();
    }

    auto outputType =
        llvm::dyn_cast<RankedTensorType>(op.getOutputs()[0].getType());
    if (!outputType) {
      return failure();
    }

    // Check if result type is dynamic but output operand type is static
    auto resultType =
        llvm::dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || resultType.hasStaticShape()) {
      return failure();
    }

    // If output operand is static, use it
    if (!outputType.hasStaticShape()) {
      return failure();
    }

    // Create new fill with correct result type
    auto newFill = linalg::FillOp::create(rewriter, op.getLoc(),
                                          TypeRange{outputType}, op.getInputs(),
                                          op.getOutputs());

    rewriter.replaceOp(op, newFill.getResults());
    return success();
  }
};

/// Pattern to update linalg.matmul output type when operands are static.
struct StaticizeMatmulPattern : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getResults().empty()) {
      return failure();
    }

    auto resultType =
        llvm::dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType) {
      return failure();
    }

    // Skip if result is already static
    if (resultType.hasStaticShape()) {
      return failure();
    }

    // Check if operand types are now static
    auto lhsType =
        llvm::dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    auto rhsType =
        llvm::dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
    auto outType =
        llvm::dyn_cast<RankedTensorType>(op.getOutputs()[0].getType());

    if (!lhsType || !rhsType || !outType) {
      return failure();
    }

    // Need at least one static operand to infer shape
    bool canInfer = (lhsType.hasStaticShape() && rhsType.hasStaticShape()) ||
                    outType.hasStaticShape();
    if (!canInfer) {
      return failure();
    }

    // Infer result shape: [M, N] from lhs[M, K] x rhs[K, N]
    SmallVector<int64_t> newShape;
    if (lhsType.hasStaticShape() && rhsType.hasStaticShape()) {
      newShape.push_back(lhsType.getShape()[0]); // M
      newShape.push_back(rhsType.getShape()[1]); // N
    } else if (outType.hasStaticShape()) {
      newShape = SmallVector<int64_t>(outType.getShape());
    } else {
      return failure();
    }

    auto newResultType =
        RankedTensorType::get(newShape, resultType.getElementType());

    auto newMatmul =
        linalg::MatmulOp::create(rewriter, op.getLoc(), TypeRange{newResultType},
                                 op.getInputs(), op.getOutputs());

    rewriter.replaceOp(op, newMatmul.getResults());
    return success();
  }
};

/// Pattern to update linalg.generic output type when operands are static.
struct StaticizeGenericPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Check if any result has dynamic shape
    bool hasAnyDynamic = false;
    for (Value result : op.getResults()) {
      if (auto tensorType =
              llvm::dyn_cast<RankedTensorType>(result.getType())) {
        if (!tensorType.hasStaticShape()) {
          hasAnyDynamic = true;
          break;
        }
      }
    }
    if (!hasAnyDynamic) {
      return failure();
    }

    // Check if output operands have static types now
    SmallVector<Type> newResultTypes;
    bool canUpdate = false;
    for (Value output : op.getOutputs()) {
      auto tensorType = llvm::dyn_cast<RankedTensorType>(output.getType());
      if (tensorType && tensorType.hasStaticShape()) {
        newResultTypes.push_back(tensorType);
        canUpdate = true;
      } else {
        newResultTypes.push_back(output.getType());
      }
    }

    if (!canUpdate) {
      return failure();
    }

    // Clone the operation: first clone, then update result types
    Operation *cloned = op->clone();
    rewriter.setInsertionPoint(op);
    rewriter.insert(cloned);

    // Update result types
    for (auto [idx, resultType] : llvm::enumerate(newResultTypes)) {
      cloned->getResult(idx).setType(resultType);
    }

    rewriter.replaceOp(op, cloned->getResults());
    return success();
  }
};

/// Pattern to update scf.for iter_args and results when operand types change.
struct StaticizeScfForPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    // Check if any init arg has static type but the corresponding block arg
    // and result have dynamic type
    bool needsUpdate = false;
    for (auto [initArg, result] :
         llvm::zip(op.getInitArgs(), op.getResults())) {
      auto initType = llvm::dyn_cast<RankedTensorType>(initArg.getType());
      auto resultType = llvm::dyn_cast<RankedTensorType>(result.getType());
      if (initType && resultType) {
        if (initType.hasStaticShape() && !resultType.hasStaticShape()) {
          needsUpdate = true;
          break;
        }
      }
    }

    if (!needsUpdate) {
      return failure();
    }

    // Collect new result types from init args
    SmallVector<Type> newResultTypes;
    for (Value initArg : op.getInitArgs()) {
      newResultTypes.push_back(initArg.getType());
    }

    // Create new for loop with updated types
    auto newForOp =
        scf::ForOp::create(rewriter, op.getLoc(), op.getLowerBound(),
                           op.getUpperBound(), op.getStep(), op.getInitArgs());

    // Move the body of the old loop to the new one
    newForOp.getBody()->erase();
    rewriter.inlineRegionBefore(op.getRegion(), newForOp.getRegion(),
                                newForOp.getRegion().end());

    // Update block argument types
    for (auto [idx, blockArg] :
         llvm::enumerate(newForOp.getRegionIterArgs())) {
      blockArg.setType(newResultTypes[idx]);
    }

    rewriter.replaceOp(op, newForOp.getResults());
    return success();
  }
};

class StaticizeTypesPass
    : public PassWrapper<StaticizeTypesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StaticizeTypesPass)

  StringRef getArgument() const override { return "loom-staticize-types"; }

  StringRef getDescription() const override {
    return "Convert dynamic memref/tensor types to static types when possible";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    // Apply patterns iteratively to handle cascading type updates
    RewritePatternSet patterns(&getContext());
    patterns.add<StaticizeAllocPattern, StaticizeTensorEmptyPattern,
                 StaticizeToTensorPattern, StaticizeFillPattern,
                 StaticizeMatmulPattern, StaticizeGenericPattern,
                 StaticizeScfForPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createStaticizeTypesPass() {
  return std::make_unique<StaticizeTypesPass>();
}

void loom::passes::registerStaticizeTypesPass() {
  PassRegistration<StaticizeTypesPass>();
}
