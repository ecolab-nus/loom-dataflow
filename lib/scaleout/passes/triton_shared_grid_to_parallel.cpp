//===- triton_shared_grid_to_parallel.cpp -----------------------*- C++ -*-===//
//
// Replace the last three grid index function arguments with a 3-D
// affine.parallel and erase those arguments from the signature.
//
// Expected: the last six args are (sizeX, sizeY, sizeZ, idxX, idxY, idxZ).
// We build `affine.parallel (%i, %j, %k) = (0,0,0) to (sizeX,sizeY,sizeZ)` and
// replace all uses of (idxX, idxY, idxZ) by (%i, %j, %k).
//
// Detailed semantics
// ------------------
// - Calling convention: The kernel function produced by Triton-shared
//   affinization carries at its tail six arguments encoding grid information:
//   three dynamic sizes followed by the three grid indices.
//     - sizeX, sizeY, sizeZ: upper bounds along x/y/z grid dimensions
//     - idxX,  idxY,  idxZ : current coordinate along x/y/z grid dimensions
// - Parallelization model: The kernel is intended to be executed independently
//   across all grid coordinates. This transformation makes that explicit by
//   introducing a single 3-D `affine.parallel` that enumerates all coordinates
//   in lexicographic order and uses its induction variables as the indices.
// - Signature change: The three index arguments are no longer needed and are
//   erased from the function type and entry block. The three size arguments are
//   preserved as function parameters and used as dynamic upper bounds of the
//   parallel loop. Steps are set to 1 for all dimensions.
// - In-body rewrites: All uses of the old index arguments are replaced by the
//   newly created parallel loop IVs. No other changes are performed.
//
// Preconditions and effects
// -------------------------
// - The function must have at least six arguments. If fewer, the pass is a
//   no-op for that function.
// - The last six arguments are interpreted as (sizeX, sizeY, sizeZ, idxX, idxY,
//   idxZ). The pass does not validate producer provenance beyond arity.
// - Upper bounds accept dynamic values. They are used as-is; if they are of
//   non-index types, downstream canonicalizations should normalize types.
// - The resulting `affine.parallel` has no reductions and yields no values.
// - This is a structural wrap; it does not alter memory effects or introduce
//   additional synchronization.
//
// Example (conceptual)
// --------------------
// Before (tail args: szX, szY, szZ, iX, iY, iZ):
//   func @kernel(..., %szX: index, %szY: index, %szZ: index,
//                %iX: index, %iY: index, %iZ: index) {
//     // body using %iX, %iY, %iZ
//     return
//   }
// After:
//   func @kernel(..., %szX: index, %szY: index, %szZ: index) {
//     affine.parallel (%iX, %iY, %iZ) = (0, 0, 0) to (%szX, %szY, %szZ) {
//       // same body with %iX/%iY/%iZ replacing the old args
//     }
//     return
//   }
//
//===----------------------------------------------------------------------===//

#include "triton_shared_grid_to_parallel.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace tmd {
namespace passes {

namespace {

class TritonSharedGridToParallelPass
    : public PassWrapper<TritonSharedGridToParallelPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonSharedGridToParallelPass)

  StringRef getArgument() const override {
    return "tmd-triton-shared-grid-to-parallel";
  }
  StringRef getDescription() const override {
    return "Wrap function body with 3-D affine.parallel and remove grid index"
           " arguments (last 3)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    module.walk([&](func::FuncOp func) {
      Block &entry = func.getBody().front();
      unsigned numArgs = func.getNumArguments();
      if (numArgs < 6)
        return; // Not our calling convention.

      // Identify the last six args as (sizeX, sizeY, sizeZ, idxX, idxY, idxZ).
      Value sizeX = entry.getArgument(numArgs - 6);
      Value sizeY = entry.getArgument(numArgs - 5);
      Value sizeZ = entry.getArgument(numArgs - 4);
      Value idxX = entry.getArgument(numArgs - 3);
      Value idxY = entry.getArgument(numArgs - 2);
      Value idxZ = entry.getArgument(numArgs - 1);

      // Builder set up.
      OpBuilder b(func);

      // Insert the affine.parallel at the very beginning of the entry block,
      // after any argument-related ops we may materialize.
      // We expect sizes to already be of index type after affinization. Use
      // them directly as dynamic upper bounds.
      Value sizeXi = sizeX;
      Value sizeYi = sizeY;
      Value sizeZi = sizeZ;

      // Build lb/ub maps for 3-D parallel: all lower bounds 0, dynamic uppers.
      SmallVector<AffineMap, 3> lbMaps(3, AffineMap::getConstantMap(0, ctx));
      // For upper bounds, the builder requires each map to have as many inputs
      // as the shared ubArgs vector. Provide maps (d0,d1,d2)->(d0),
      // (d0,d1,d2)->(d1), (d0,d1,d2)->(d2)
      SmallVector<AffineMap, 3> ubMaps{
          AffineMap::get(/*dimCount=*/3, /*symCount=*/0,
                         getAffineDimExpr(0, ctx)),
          AffineMap::get(/*dimCount=*/3, /*symCount=*/0,
                         getAffineDimExpr(1, ctx)),
          AffineMap::get(/*dimCount=*/3, /*symCount=*/0,
                         getAffineDimExpr(2, ctx))};

      SmallVector<Value, 3> lbArgs; // none for constant 0
      SmallVector<Value, 3> ubArgs{sizeXi, sizeYi, sizeZi};
      SmallVector<int64_t, 3> steps{1, 1, 1};

      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&entry);
      auto par = b.create<affine::AffineParallelOp>(
          func.getLoc(), /*resultTypes=*/TypeRange{},
          /*reductions=*/ArrayRef<arith::AtomicRMWKind>{}, lbMaps,
          /*lbArgs=*/ValueRange(lbArgs), ubMaps, /*ubArgs=*/ValueRange(ubArgs),
          steps);

      // Move the entire original body (except the terminator) into the
      // parallel region. We will then replace idx args with IVs.
      Block &parBody = par.getRegion().front();
      // Save all ops currently in entry except the terminator of func region
      // and the newly created parallel itself.
      SmallVector<Operation *, 32> opsToMove;
      for (Operation &op : entry.without_terminator())
        if (&op != par.getOperation())
          opsToMove.push_back(&op);

      // The parallel body initially contains only an affine.yield terminator;
      // insert before it.
      Operation *yieldOp = parBody.getTerminator();
      for (Operation *op : opsToMove)
        op->moveBefore(yieldOp);

      // Replace all uses of (idxX, idxY, idxZ) by the loop IVs.
      Value ivX = par.getIVs()[0];
      Value ivY = par.getIVs()[1];
      Value ivZ = par.getIVs()[2];
      idxX.replaceAllUsesWith(ivX);
      idxY.replaceAllUsesWith(ivY);
      idxZ.replaceAllUsesWith(ivZ);

      // Now remove the three index arguments from the function type and entry
      // block, keeping the first (numArgs - 3) arguments.
      SmallVector<Type, 8> newInputTypes;
      newInputTypes.reserve(numArgs - 3);
      for (unsigned i = 0; i < numArgs - 3; ++i)
        newInputTypes.push_back(entry.getArgument(i).getType());
      FunctionType oldTy = func.getFunctionType();
      FunctionType newTy =
          FunctionType::get(ctx, newInputTypes, oldTy.getResults());
      func.setType(newTy);

      // Erase the last three BlockArguments from the entry block (no uses
      // left).
      entry.eraseArgument(numArgs - 1);
      entry.eraseArgument(numArgs - 2);
      entry.eraseArgument(numArgs - 3);
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createTritonSharedGridToParallelPass() {
  return std::make_unique<TritonSharedGridToParallelPass>();
}

void registerTritonSharedGridToParallelPass() {
  PassRegistration<TritonSharedGridToParallelPass>();
}

} // namespace passes
} // namespace tmd
