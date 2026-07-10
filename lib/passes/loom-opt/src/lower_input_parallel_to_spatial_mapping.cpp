#include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"

// ADL dialect headers for adl.arch.scale.
#define GET_TYPEDEF_CLASSES
#include "ADLTypes.h.inc"
#define GET_OP_CLASSES
#include "ADLOps.h.inc"

// Loom dialect headers for loom.sym, loom.mapping_matrix, and
// loom.spatial_mapping.
#include "LoomDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LoomTypes.h.inc"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace loom {
namespace passes {

#define GEN_PASS_DEF_LOWERINPUTPARALLELTOSPATIALMAPPING
#include "Passes.h.inc"

namespace {

struct LowerInputParallelToSpatialMappingPass
    : public impl::LowerInputParallelToSpatialMappingBase<
          LowerInputParallelToSpatialMappingPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    adl::ArchScaleOp archMesh = findArchMesh(module);
    if (!archMesh) {
      module.emitError("expected exactly one adl.arch.scale symbol @arch_mesh");
      return signalPassFailure();
    }

    SmallVector<affine::AffineParallelOp> parallelOps;
    module.walk([&](affine::AffineParallelOp op) { parallelOps.push_back(op); });
    if (parallelOps.size() != 1) {
      module.emitError("expected exactly one affine.parallel, found ")
          << parallelOps.size();
      return signalPassFailure();
    }

    affine::AffineParallelOp parallelOp = parallelOps.front();
    if (failed(validateParallelOp(parallelOp)))
      return signalPassFailure();

    if (failed(rewriteParallelOp(parallelOp, archMesh)))
      return signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    loom::LoomDialect>();
  }

private:
  adl::ArchScaleOp findArchMesh(ModuleOp module) {
    adl::ArchScaleOp result;
    unsigned count = 0;
    module.walk([&](adl::ArchScaleOp op) {
      if (op.getSymName() == "arch_mesh") {
        result = op;
        ++count;
      }
    });
    return count == 1 ? result : adl::ArchScaleOp();
  }

  LogicalResult validateParallelOp(affine::AffineParallelOp op) {
    if (op.getNumResults() != 0)
      return op.emitError("loom spatial mapping only supports affine.parallel "
                          "without reduction results");

    SmallVector<int64_t, 8> steps = op.getSteps();
    if (llvm::any_of(steps, [](int64_t step) { return step != 1; }))
      return op.emitError(
          "loom spatial mapping only supports affine.parallel step = 1");

    for (unsigned i = 0, e = op.getNumDims(); i < e; ++i) {
      AffineMap lbMap = op.getLowerBoundMap(i);
      if (lbMap.getNumResults() != 1)
        return op.emitError("expected single-result lower bound map");
      auto constExpr = dyn_cast<AffineConstantExpr>(lbMap.getResult(0));
      if (!constExpr || constExpr.getValue() != 0)
        return op.emitError(
            "loom spatial mapping only supports zero lower bounds");
    }

    return success();
  }

  LogicalResult rewriteParallelOp(affine::AffineParallelOp parallelOp,
                                  adl::ArchScaleOp archMesh) {
    OpBuilder builder(parallelOp);
    Location loc = parallelOp.getLoc();
    MLIRContext *ctx = parallelOp.getContext();

    unsigned numLogicalDims = parallelOp.getNumDims();
    unsigned numPhysicalDims = archMesh.getSpatialDims().size();
    if (numLogicalDims == 0 || numPhysicalDims == 0)
      return parallelOp.emitError("expected non-zero logical and physical "
                                  "dimension counts");

    SmallVector<Value> upperBounds;
    upperBounds.reserve(numLogicalDims);
    for (unsigned i = 0; i < numLogicalDims; ++i) {
      AffineMap ubMap = parallelOp.getUpperBoundMap(i);
      if (ubMap.getNumResults() != 1)
        return parallelOp.emitError("expected single-result upper bound map");
      upperBounds.push_back(affine::AffineApplyOp::create(
          builder, loc, ubMap, parallelOp.getUpperBoundsOperands()));
    }

    SmallVector<Value> matrixEntries;
    matrixEntries.reserve(numLogicalDims * numPhysicalDims);
    for (unsigned row = 0; row < numLogicalDims; ++row) {
      for (unsigned col = 0; col < numPhysicalDims; ++col) {
        std::string symName =
            llvm::formatv("logdim_{0}{1}", row, col).str();
        auto symRef = SymbolRefAttr::get(ctx, symName);
        auto symOp =
            loom::SymOp::create(builder, loc, builder.getIndexType(), symRef,
                                /*upper_bound=*/IntegerAttr{},
                                /*is_reduction=*/false,
                                /*asure_divisible=*/false);
        matrixEntries.push_back(symOp.getResult());
      }
    }

    auto mapType = loom::SpatialMapType::get(ctx, numLogicalDims,
                                             numPhysicalDims);
    SmallVector<Type> ldTypes(numLogicalDims, builder.getIndexType());
    SmallVector<Type> mappingResultTypes;
    mappingResultTypes.push_back(mapType);
    mappingResultTypes.append(ldTypes);

    auto mappingOp = loom::MappingMatrixOp::create(
        builder, loc, mappingResultTypes,
        FlatSymbolRefAttr::get(ctx, archMesh.getSymName()), matrixEntries);

    SmallVector<Value> lds(mappingOp.getLds().begin(), mappingOp.getLds().end());

    auto spatialOp = loom::SpatialMappingOp::create(
        builder, loc, upperBounds, mappingOp.getMap(), lds);

    Region &body = spatialOp.getBody();
    Block *bodyBlock = new Block();
    body.push_back(bodyBlock);
    for (unsigned i = 0; i < numLogicalDims * 2; ++i)
      bodyBlock->addArgument(builder.getIndexType(), loc);

    IRMapping mapping;
    for (unsigned i = 0; i < numLogicalDims; ++i)
      mapping.map(parallelOp.getBody()->getArgument(i),
                  bodyBlock->getArgument(i));

    builder.setInsertionPointToStart(bodyBlock);
    for (Operation &innerOp : parallelOp.getBody()->without_terminator())
      builder.clone(innerOp, mapping);

    builder.setInsertionPointToEnd(bodyBlock);
    loom::YieldOp::create(builder, loc);

    parallelOp.erase();
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerInputParallelToSpatialMappingPass() {
  return std::make_unique<LowerInputParallelToSpatialMappingPass>();
}

} // namespace passes
} // namespace loom
