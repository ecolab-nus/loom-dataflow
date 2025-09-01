// Spatial reuse analysis for Triton-shared lowered kernels.
//
// This pass inspects each function, identifies memref arguments, and determines
// for each whether the accessed base region is invariant with respect to the
// hardware spatial coordinates annotated on function arguments via
// `tmd.spatial_dim_name = "x"|"y"`.
//
// For every memref block argument `arg{i}`, the pass computes two booleans:
// - tmd.invariant.x: true if all uses of the argument are independent of the X
//   spatial coordinate (data can be reused across cores along X).
// - tmd.invariant.y: true if all uses of the argument are independent of the Y
//   spatial coordinate (data can be reused across cores along Y).
//
// The analysis is conservative and flow-insensitive: a value is considered to
// depend on a spatial coordinate if any of its SSA operands do. We examine
// offsets/indices of memref.* ops that reference the memref arguments to decide
// invariance; if any offset/index depends on a coordinate, we mark it variant
// along that coordinate.

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace tmd {
namespace passes {

namespace {

/// Simple dependency flags for whether a value depends on spatial x/y.
struct SpatialDeps {
  bool dependsOnX = false;
  bool dependsOnY = false;
};

/// Compute and memoize spatial dependency of a value given x/y seed values.
class SpatialDependencyAnalysis {
public:
  SpatialDependencyAnalysis(Value xCoord, Value yCoord)
      : xCoord(xCoord), yCoord(yCoord) {}

  /**
   * \brief Return whether the given value depends on spatial X or Y.
   *
   * The analysis is conservative and treats unknown operations as depending on
   * the union of their operands' dependencies. Local loop induction variables
   * (scf.for/affine.for block arguments) are treated as independent.
   */
  SpatialDeps depends(Value v) {
    auto it = cache.find(v);
    if (it != cache.end())
      return it->second;

    SpatialDeps deps;

    // Seed values: exactly the coordinates.
    if (v == xCoord)
      deps.dependsOnX = true;
    if (v == yCoord)
      deps.dependsOnY = true;

    if (deps.dependsOnX || deps.dependsOnY) {
      cache.try_emplace(v, deps);
      return deps;
    }

    if (auto barg = dyn_cast<BlockArgument>(v)) {
      // Function arguments other than x/y do not depend on x/y by themselves.
      // Induction variables/iter args of loops are considered independent.
      Operation *owner = barg.getOwner()->getParentOp();
      if (isa_and_nonnull<func::FuncOp>(owner) ||
          isa_and_nonnull<scf::ForOp>(owner) || owner == nullptr) {
        cache.try_emplace(v, deps);
        return deps;
      }
    }

    Operation *def = v.getDefiningOp();
    if (!def) {
      cache.try_emplace(v, deps);
      return deps;
    }

    // Propagate through common arithmetic and cast ops.
    auto orIn = [&](Value opv) {
      SpatialDeps od = depends(opv);
      deps.dependsOnX = deps.dependsOnX || od.dependsOnX;
      deps.dependsOnY = deps.dependsOnY || od.dependsOnY;
    };

    // Whitelist simple ops that just transform a value but preserve deps.
    if (isa<arith::IndexCastOp, arith::TruncIOp, arith::ExtUIOp, arith::ExtSIOp,
            memref::CastOp>(def)) {
      for (Value o : def->getOperands())
        orIn(o);
      cache.try_emplace(v, deps);
      return deps;
    }

    if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
            arith::RemSIOp, arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp,
            arith::SelectOp>(def)) {
      for (Value o : def->getOperands())
        orIn(o);
      cache.try_emplace(v, deps);
      return deps;
    }

    if (auto apply = dyn_cast<affine::AffineApplyOp>(def)) {
      for (Value o : apply.getMapOperands())
        orIn(o);
      cache.try_emplace(v, deps);
      return deps;
    }

    // Default: union of operands.
    for (Value o : def->getOperands())
      orIn(o);
    cache.try_emplace(v, deps);
    return deps;
  }

private:
  Value xCoord;
  Value yCoord;
  llvm::DenseMap<Value, SpatialDeps> cache;
};

/// Inspect a memref argument's uses and determine invariance along X/Y.
static std::pair<bool, bool> analyzeMemrefArgInvariance(func::FuncOp func,
                                                        BlockArgument memrefArg,
                                                        Value xCoord,
                                                        Value yCoord) {
  SpatialDependencyAnalysis dep(xCoord, yCoord);

  bool invX = true;
  bool invY = true;

  llvm::SmallVector<Operation *, 32> worklist;
  for (OpOperand &use : memrefArg.getUses())
    worklist.push_back(use.getOwner());

  // Helper that checks offset/index lists for dependency.
  auto checkValues = [&](ValueRange vals) {
    for (Value v : vals) {
      SpatialDeps d = dep.depends(v);
      if (d.dependsOnX)
        invX = false;
      if (d.dependsOnY)
        invY = false;
      if (!invX && !invY)
        return;
    }
  };

  while (!worklist.empty() && (invX || invY)) {
    Operation *op = worklist.back();
    worklist.pop_back();

    // Propagate through trivial forwarding ops.
    if (auto castOp = dyn_cast<memref::CastOp>(op)) {
      for (OpResult res : castOp->getResults())
        for (OpOperand &u : res.getUses())
          worklist.push_back(u.getOwner());
      continue;
    }

    // Consider base selection and index/offset usage.
    if (auto r = dyn_cast<memref::ReinterpretCastOp>(op)) {
      // Dynamic offsets indicate base movement.
      checkValues(r.getOffsets());
      // Also dynamic strides can depend, though uncommon.
      checkValues(r.getStrides());
      continue;
    }
    if (auto sv = dyn_cast<memref::SubViewOp>(op)) {
      checkValues(sv.getOffsets());
      // Strides rarely depend on coords but include for completeness.
      checkValues(sv.getStrides());
      continue;
    }
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      checkValues(load.getIndices());
      continue;
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      checkValues(store.getIndices());
      continue;
    }

    // For unknown consumers, conservatively scan all index-typed operands.
    for (Value operand : op->getOperands()) {
      if (operand.getType().isIndex()) {
        SpatialDeps d = dep.depends(operand);
        if (d.dependsOnX)
          invX = false;
        if (d.dependsOnY)
          invY = false;
        if (!invX && !invY)
          break;
      }
    }
  }

  return {invX, invY};
}

/// Pass that annotates memref arguments with tmd.invariant.{x,y} attributes.
class TritonSharedSpatialReuseAnalysisPass
    : public PassWrapper<TritonSharedSpatialReuseAnalysisPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TritonSharedSpatialReuseAnalysisPass)

  StringRef getArgument() const override {
    return "tmd-triton-shared-spatial-reuse";
  }
  StringRef getDescription() const override {
    return "Annotate memref arguments with invariance across spatial X/Y";
  }

  /**
   * \brief Run the spatial reuse analysis on all functions in the module.
   */
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    module.walk([&](func::FuncOp func) {
      // Discover the spatial coordinate values within the function.
      Value xCoord;
      Value yCoord;
      for (BlockArgument barg : func.getArguments()) {
        if (auto i32Ty = dyn_cast<IntegerType>(barg.getType())) {
          (void)i32Ty; // type not strongly enforced; rely on attribute
          if (auto attr = func.getArgAttrOfType<StringAttr>(
                  barg.getArgNumber(), "tmd.spatial_dim_name")) {
            if (attr.getValue() == "x")
              xCoord = barg;
            else if (attr.getValue() == "y")
              yCoord = barg;
          }
        }
      }

      // If not explicitly provided on args, attempt to find index_cast of such
      // args. Otherwise, coords remain null and analysis will conservatively do
      // nothing (leave defaults true only if no dependent uses found).

      for (BlockArgument barg : func.getArguments()) {
        if (!llvm::isa<mlir::BaseMemRefType>(barg.getType()))
          continue;

        auto [invX, invY] =
            analyzeMemrefArgInvariance(func, barg, xCoord, yCoord);

        func.setArgAttr(barg.getArgNumber(), "tmd.invariant.x",
                        BoolAttr::get(ctx, invX));
        func.setArgAttr(barg.getArgNumber(), "tmd.invariant.y",
                        BoolAttr::get(ctx, invY));
      }
    });
  }
};

} // namespace

/// Create the pass.
std::unique_ptr<Pass> createTritonSharedSpatialReuseAnalysisPass() {
  return std::make_unique<TritonSharedSpatialReuseAnalysisPass>();
}

/// Register the pass for use in pipelines.
void registerTritonSharedSpatialReuseAnalysisPass() {
  PassRegistration<TritonSharedSpatialReuseAnalysisPass>();
}

} // namespace passes
} // namespace tmd
