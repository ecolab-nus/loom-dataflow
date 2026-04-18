/**
 * @file SCFOpToTTKernel.cpp
 * @brief Conversion of SCF operations (notably scf.parallel) to TTKernel.
 *
 * @details
 * This file provides patterns that remove `scf.parallel` loops and replace
 * their induction variables with TTKernel compile-time arguments, similar to
 * the way `CompileArgTracker` handles function arguments.
 *
 * For each `scf.parallel` operation without reductions:
 * - A fresh compile-arg index is allocated for each induction variable using
 *   `CompileArgTracker::getOrCreateIndex`.
 * - A `ttkernel.get_compile_time_arg_val` is emitted (returning `i32`), and
 *   index materialization is delayed until replacing the original IV.
 * - All uses of the induction variable are replaced with this cast value.
 * - The loop body is inlined into the parent block and the `scf.parallel`
 *   op is erased.
 *
 * This treats the parallel iterators as per-kernel compile-time constants
 * rather than dynamic loop indices, which matches the intended mapping of
 * TileLoom spatial loops to hardware coordinates.
 */

#include "SCFOpToTTKernel.h"

#include "FuncOpToTTKernel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

using namespace mlir;
using namespace mlir::loom;
using namespace tt::ttkernel;

namespace {

/// Normalize unsigned index arithmetic to signed forms.
/// `ttmlir-opt --convert-ttkernel-to-emitc` does not legalize `arith.divui`
/// / `arith.remui` / `arith.ceildivui` on `index`.
class ConvertIndexDivUIOp : public OpConversionPattern<arith::DivUIOp> {
public:
  using OpConversionPattern<arith::DivUIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::DivUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getType().isIndex())
      return failure();
    rewriter.replaceOpWithNewOp<arith::DivSIOp>(op, adaptor.getLhs(),
                                                adaptor.getRhs());
    return success();
  }
};

class ConvertIndexRemUIOp : public OpConversionPattern<arith::RemUIOp> {
public:
  using OpConversionPattern<arith::RemUIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::RemUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getType().isIndex())
      return failure();
    rewriter.replaceOpWithNewOp<arith::RemSIOp>(op, adaptor.getLhs(),
                                                adaptor.getRhs());
    return success();
  }
};

class ConvertIndexCeilDivUIOp
    : public OpConversionPattern<arith::CeilDivUIOp> {
public:
  using OpConversionPattern<arith::CeilDivUIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CeilDivUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getType().isIndex())
      return failure();
    Value one = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    Value rhsMinusOne =
        rewriter.create<arith::SubIOp>(op.getLoc(), adaptor.getRhs(), one);
    Value numerator =
        rewriter.create<arith::AddIOp>(op.getLoc(), adaptor.getLhs(), rhsMinusOne);
    rewriter.replaceOpWithNewOp<arith::DivSIOp>(op, numerator, adaptor.getRhs());
    return success();
  }
};

class ConvertIndexCeilDivSIOp
    : public OpConversionPattern<arith::CeilDivSIOp> {
public:
  using OpConversionPattern<arith::CeilDivSIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CeilDivSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getType().isIndex())
      return failure();
    Value one = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    Value rhsMinusOne =
        rewriter.create<arith::SubIOp>(op.getLoc(), adaptor.getRhs(), one);
    Value numerator =
        rewriter.create<arith::AddIOp>(op.getLoc(), adaptor.getLhs(), rhsMinusOne);
    rewriter.replaceOpWithNewOp<arith::DivSIOp>(op, numerator, adaptor.getRhs());
    return success();
  }
};

static std::string stringifyAttr(Attribute attr) {
  std::string text;
  llvm::raw_string_ostream os(text);
  attr.print(os);
  os.flush();
  return text;
}

static bool isSpatialIterTypeAttr(Attribute attr) {
  return StringRef(stringifyAttr(attr)).contains("spatial");
}

static std::string normalizeDimNameFromAttr(Attribute attr) {
  if (auto flat = dyn_cast<FlatSymbolRefAttr>(attr))
    return flat.getValue().str();
  if (auto sym = dyn_cast<SymbolRefAttr>(attr))
    return sym.getLeafReference().str();
  if (auto str = dyn_cast<StringAttr>(attr))
    return str.getValue().str();

  StringRef text = stringifyAttr(attr);
  if (text.starts_with("@"))
    text = text.drop_front();
  return text.trim().str();
}

/**
 * @brief Convert `scf.parallel` to straight-line code with compile-time IVs.
 *
 * @details
 * This pattern targets `scf.parallel` operations with no reductions or
 * results. For each induction variable:
 *
 * - A fresh compile-time argument index is allocated via
 *   `CompileArgTracker::getOrCreateIndex`, using the IV value as the key.
 * - A `ttkernel.get_compile_time_arg_val` is emitted (type `i32`). Spatial
 *   decomposition math stays in `i32`; we cast to `index` only when replacing
 *   the original IV SSA value.
 * - All uses of the induction variable inside the loop body are rewritten to
 *   use this cast value instead.
 *
 * After induction variables are rewritten, the body of the `scf.parallel`
 * operation is inlined into the parent block and the original loop op is
 * erased. The original lower/upper bounds and steps are ignored, as the
 * parallel iterator is interpreted as a hardware coordinate provided as a
 * compile-time argument.
 */
class ConvertSCFParallelOp
    : public OpConversionPattern<scf::ParallelOp> {
public:
  using OpConversionPattern<scf::ParallelOp>::OpConversionPattern;

  ConvertSCFParallelOp(TypeConverter &typeConverter, MLIRContext *context,
                       std::shared_ptr<CompileArgTracker> tracker)
      : OpConversionPattern<scf::ParallelOp>(typeConverter, context),
        tracker(std::move(tracker)) {}

  LogicalResult
  matchAndRewrite(scf::ParallelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Do not attempt to convert parallel loops with reductions/results for now.
    if (!op.getResults().empty())
      return failure();

    Location loc = op.getLoc();

    // Rewritten IV values. We prefer deriving all mapped spatial IVs from
    // exactly two physical compile args (x/y). Any IV that cannot be derived
    // from mapping metadata falls back to a direct compile arg.
    SmallVector<Value, 4> newIvValues(op.getInductionVars().size(), Value{});
    auto parentFunc = op->getParentOfType<func::FuncOp>();
    if (!parentFunc)
      return failure();

    rewriter.setInsertionPoint(op);

    // Gather metadata used to map N logical spatial IVs <-> 2 physical axes.
    auto mappedDims = op->getAttrOfType<ArrayAttr>("loom.physical_dims");
    if (!mappedDims)
      mappedDims = op->getAttrOfType<ArrayAttr>("loom.mapped_to_dims");
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
    if (mappedDims) {
      for (size_t idx = 0; idx < newIvValues.size(); ++idx) {
        if (idx >= mappedDims.size() || idx >= lbs.size() || idx >= ubs.size() ||
            idx >= steps.size())
          break;

        if (iterTypes && idx < iterTypes.size() &&
            !isSpatialIterTypeAttr(iterTypes[idx]))
          continue;

        std::string dimName = normalizeDimNameFromAttr(mappedDims[idx]);
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

    // Materialize exactly two physical compile args (x, y) when we can map
    // logical IVs through physical dims metadata.
    if (!xComponents.empty() || !yComponents.empty()) {
      Value oneI32 = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
      Value xCompileArgI32 = tracker->createTypedCompileArg(
          loc, rewriter, parentFunc, rewriter.getI32Type());
      Value yCompileArgI32 = tracker->createTypedCompileArg(
          loc, rewriter, parentFunc, rewriter.getI32Type());
      if (!xCompileArgI32 || !yCompileArgI32)
        return failure();

      tracker->appendToCoreList(parentFunc, xCompileArgI32);
      tracker->appendToCoreList(parentFunc, yCompileArgI32);
      tracker->setCoreCoordForDim(parentFunc, "x", xCompileArgI32);
      tracker->setCoreCoordForDim(parentFunc, "y", yCompileArgI32);

      auto toI32 = [&](Value value) -> Value {
        if (!value)
          return {};
        Type ty = value.getType();
        if (ty.isInteger(32))
          return value;
        if (ty.isIndex())
          return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(),
                                                     value);
        if (auto intTy = dyn_cast<IntegerType>(ty)) {
          if (intTy.getWidth() < 32)
            return rewriter.create<arith::ExtSIOp>(loc, rewriter.getI32Type(),
                                                   value);
          if (intTy.getWidth() > 32)
            return rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(),
                                                    value);
        }
        return {};
      };

      // Decompose each axis coordinate into its logical components in
      // logical-level order:
      //   iv = lb + ((axis / stride) % extent) * step.
      auto materializeAxisIvs = [&](ArrayRef<AxisComponent> components,
                                    Value axisCoord) {
        Value stride = oneI32;
        for (const AxisComponent &component : components) {
          size_t idx = component.idx;
          Value lbI32 = toI32(lbs[idx]);
          Value ubI32 = toI32(ubs[idx]);
          Value stepI32 = toI32(steps[idx]);
          if (!lbI32 || !ubI32 || !stepI32) {
            newIvValues[idx] = {};
            continue;
          }
          Value span = rewriter.create<arith::SubIOp>(loc, ubI32, lbI32);
          // Emit ceildiv as (span + step - 1) / step to avoid downstream
          // legalization gaps for arith.ceildivsi.
          Value stepMinusOne =
              rewriter.create<arith::SubIOp>(loc, stepI32, oneI32);
          Value ceilNumer =
              rewriter.create<arith::AddIOp>(loc, span, stepMinusOne);
          Value extent = rewriter.create<arith::DivSIOp>(loc, ceilNumer, stepI32);
          Value quotient = rewriter.create<arith::DivSIOp>(loc, axisCoord, stride);
          Value digit = rewriter.create<arith::RemSIOp>(loc, quotient, extent);
          Value scaledDigit = rewriter.create<arith::MulIOp>(loc, digit, stepI32);
          Value ivValI32 = rewriter.create<arith::AddIOp>(loc, lbI32, scaledDigit);
          Value ivValIndex = rewriter.create<arith::IndexCastOp>(
              loc, rewriter.getIndexType(), ivValI32);
          newIvValues[idx] = ivValIndex;
          stride = rewriter.create<arith::MulIOp>(loc, stride, extent);
        }
      };

      materializeAxisIvs(xComponents, xCompileArgI32);
      materializeAxisIvs(yComponents, yCompileArgI32);
    }

    // Any IV not covered by x/y physical mapping falls back to a dedicated
    // compile-time argument.
    for (auto [idx, oldIv] : llvm::enumerate(op.getInductionVars())) {
      if (newIvValues[idx])
        continue;
      Value ivIndex = tracker->createIndexCompileArg(oldIv, loc, rewriter);
      if (!ivIndex)
        return failure();
      newIvValues[idx] = ivIndex;
    }

    // Replace all IV uses in the loop body with the new compile-arg-based
    // index values.
    for (auto [oldIv, newIv] :
         llvm::zip(op.getInductionVars(), newIvValues)) {
      oldIv.replaceAllUsesWith(newIv);
    }

    // Inline the body of the scf.parallel into the parent block, skipping
    // the terminator (scf.reduce).
    Block *bodyBlock = op.getBody();

    // Move all operations except the terminator before the parallel op.
    for (Operation &innerOp :
         llvm::make_early_inc_range(bodyBlock->getOperations())) {
      if (isa<scf::ReduceOp>(innerOp))
        continue;
      innerOp.moveBefore(op);
    }

    // Erase the now-empty scf.parallel operation.
    rewriter.eraseOp(op);

    return success();
  }

private:
  /// Shared tracker for compile-arg index assignment.
  std::shared_ptr<CompileArgTracker> tracker;
};

} // namespace

void loom::populateSCFOpConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *context, std::shared_ptr<CompileArgTracker> tracker) {
  patterns.add<ConvertIndexDivUIOp, ConvertIndexRemUIOp,
               ConvertIndexCeilDivUIOp, ConvertIndexCeilDivSIOp>(typeConverter,
                                                                  context);
  patterns.add<ConvertSCFParallelOp>(typeConverter, context,
                                     std::move(tracker));
}
