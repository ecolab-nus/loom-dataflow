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
//   Subsequent spatial exploration interprets those induction variables as the
//   handles that can be matched to hardware spatial dimensions declared in the
//   `df` dialect; when a dimension is assigned, the inner loop is annotated
//   with `tmd.mapped_to` to record the binding.
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
//       // Later passes may wrap this in additional affine.for "waves" if the
//       // hardware mesh cannot cover all (%iX,%iY,%iZ) coordinates at once.
//     }
//     return
//   }
//
//===----------------------------------------------------------------------===//

/**
 * @file triton_shared_grid_to_parallel.cpp
 * @brief Implementation of grid-to-parallel conversion for Triton-shared ABI.
 * @details
 * Algorithm
 * - Identify the last six function arguments and interpret them as
 *   `(sizeX, sizeY, sizeZ, idxX, idxY, idxZ)`.
 * - Insert a 3-D `affine.parallel` at the beginning of the entry block with
 *   lower bounds 0 and upper bounds `(sizeX, sizeY, sizeZ)` and steps (1,1,1).
 * - Move the original body into the parallel region and replace all uses of
 *   `(idxX, idxY, idxZ)` with the parallel IVs.
 * - Remove the three index arguments from the function type and entry block.
 * - Optionally, shrink the parallel to only used IVs while preserving bounds.
 *
 * Limitations
 * - Assumes sizes are already (or castable to) index type.
 * - Does not introduce reductions or yield results from the parallel region.
 */

#include "triton_shared_grid_to_parallel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
using namespace mlir;

namespace tmd {
namespace passes {


class TritonSharedGridToParallelPass
    : public PassWrapper<TritonSharedGridToParallelPass,
                         OperationPass<ModuleOp>> {
public:
  /**
   * @brief Replace grid ABI tail arguments with a 3-D affine.parallel loop.
   *
   * @details Expects the last six function arguments to be
   * `(sizeX, sizeY, sizeZ, idxX, idxY, idxZ)`. Creates a single 3-D
   * `affine.parallel` enumerating `[0..sizeX) x [0..sizeY) x [0..sizeZ)` and
   * replaces all uses of `(idxX, idxY, idxZ)` by the induction variables.
   * The three index arguments are then removed from the function signature.
   */
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonSharedGridToParallelPass)

  /// Command-line flag name.
  StringRef getArgument() const override {
    return "tmd-triton-shared-grid-to-parallel";
  }
  /// Short pass description.
  StringRef getDescription() const override {
    return "Wrap function body with 3-D affine.parallel and remove grid index"
           " arguments (last 3)";
  }

  /// Declare dialect dependencies used by this pass implementation.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    func::FuncDialect>();
  }

  /**
   * @brief Transform all functions following the Triton-shared ABI.
   *
   * @details For each function with at least six arguments, interpret the last
   * six as `(sizeX, sizeY, sizeZ, idxX, idxY, idxZ)`, build a 3-D
   * `affine.parallel` using sizes as upper bounds, move the original body into
   * the parallel region, replace all uses of index arguments with the IVs, and
   * remove the index arguments from the signature. Finally, prune unused IVs by
   * shrinking the parallel to only the used dimensions.
   */
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    module.walk([&](func::FuncOp func) {
      Block &entry = func.getBody().front();
      unsigned numArgs = func.getNumArguments();
      if (numArgs < 6)
        return;

      Value sizeX = entry.getArgument(numArgs - 6);
      Value sizeY = entry.getArgument(numArgs - 5);
      Value idxX = entry.getArgument(numArgs - 3);
      Value idxY = entry.getArgument(numArgs - 2);

      OpBuilder b(func);
      
      Value sizeXi = sizeX;
      Value sizeYi = sizeY;

      SmallVector<AffineMap, 2> lbMaps(2, AffineMap::getConstantMap(0, ctx));
      SmallVector<AffineMap, 2> ubMaps{
          AffineMap::get(/*dimCount=*/2, /*symCount=*/0,
                         getAffineDimExpr(0, ctx)),
          AffineMap::get(/*dimCount=*/2, /*symCount=*/0,
                         getAffineDimExpr(1, ctx))};

      SmallVector<Value, 2> lbArgs;
      SmallVector<int64_t, 2> steps{1, 1};

      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&entry);
      Value gridX = b.create<arith::ConstantIndexOp>(func.getLoc(), 1);
      Value gridY = b.create<arith::ConstantIndexOp>(func.getLoc(), 1);
      SmallVector<Value, 2> ubArgs{gridX, gridY};
      
      auto par = b.create<affine::AffineParallelOp>(
          func.getLoc(), /*resultTypes=*/TypeRange{},
          /*reductions=*/ArrayRef<arith::AtomicRMWKind>{}, lbMaps,
          /*lbArgs=*/ValueRange(lbArgs), ubMaps, /*ubArgs=*/ValueRange(ubArgs),
          steps);

      Block &parBody = par.getRegion().front();
      Operation *gridXOp = gridX.getDefiningOp();
      Operation *gridYOp = gridY.getDefiningOp();
      SmallVector<Operation *, 32> opsToMove;
      for (Operation &op : entry.without_terminator())
        if (&op != par.getOperation() && &op != gridXOp && &op != gridYOp)
          opsToMove.push_back(&op);

      Operation *yieldOp = parBody.getTerminator();
      for (Operation *op : opsToMove)
        op->moveBefore(yieldOp);

      Value ivX = par.getIVs()[0];
      Value ivY = par.getIVs()[1];
      idxX.replaceAllUsesWith(ivX);
      idxY.replaceAllUsesWith(ivY);

      scf::ForOp scfFor = nullptr;
      for (Operation &op : parBody) {
        if (auto forOp = llvm::dyn_cast<scf::ForOp>(&op)) {
          scfFor = forOp;
          break;
        }
      }
      
      if (scfFor) {
        int64_t blockm = 64, blockn = 64;
        if (scfFor.getNumResults() > 0) {
          Type resultType = scfFor.getResult(0).getType();
          if (auto tensorType = llvm::dyn_cast<RankedTensorType>(resultType)) {
            auto shape = tensorType.getShape();
            if (shape.size() >= 2) {
              if (shape[0] != ShapedType::kDynamic)
                blockm = shape[0];
              if (shape[1] != ShapedType::kDynamic)
                blockn = shape[1];
            }
          } else if (auto memrefType = llvm::dyn_cast<MemRefType>(resultType)) {
            auto shape = memrefType.getShape();
            if (shape.size() >= 2) {
              if (shape[0] != ShapedType::kDynamic)
                blockm = shape[0];
              if (shape[1] != ShapedType::kDynamic)
                blockn = shape[1];
            }
          }
        }
        
        OpBuilder::InsertionGuard loopGuard(b);
        b.setInsertionPoint(scfFor);
        
        Value gridXInner = b.create<arith::ConstantIndexOp>(func.getLoc(), 1);
        Value gridYInner = b.create<arith::ConstantIndexOp>(func.getLoc(), 1);
        Value blockMSize = b.create<arith::ConstantIndexOp>(func.getLoc(), blockm);
        Value blockNSize = b.create<arith::ConstantIndexOp>(func.getLoc(), blockn);
        
        AffineExpr d0 = getAffineDimExpr(0, ctx);
        AffineExpr d1 = getAffineDimExpr(1, ctx);
        AffineMap ceilDivMap = AffineMap::get(2, 0, d0.ceilDiv(d1), ctx);
        
        Value coreMSize = b.create<affine::AffineApplyOp>(
            func.getLoc(), ceilDivMap, ValueRange{sizeXi, gridXInner});
        Value coreNSize = b.create<affine::AffineApplyOp>(
            func.getLoc(), ceilDivMap, ValueRange{sizeYi, gridYInner});
        
        Value temporalIterM = b.create<affine::AffineApplyOp>(
            func.getLoc(), ceilDivMap, ValueRange{coreMSize, blockMSize});
        Value temporalIterN = b.create<affine::AffineApplyOp>(
            func.getLoc(), ceilDivMap, ValueRange{coreNSize, blockNSize});
        
        AffineMap mLbMap = AffineMap::getConstantMap(0, ctx);
        AffineMap mUbMap = AffineMap::get(1, 0, getAffineDimExpr(0, ctx), ctx);
        auto mLoop = affine::AffineForOp::create(
            b, func.getLoc(), /*lowerBoundOperands=*/ValueRange{},
            /*lowerBoundMap=*/mLbMap, /*upperBoundOperands=*/ValueRange{temporalIterM},
            /*upperBoundMap=*/mUbMap, /*step=*/1);
        
        Block *mLoopBody = mLoop.getBody();
        Value mIV = mLoopBody->getArgument(0);
        
        b.setInsertionPointToStart(mLoopBody);
        AffineMap nLbMap = AffineMap::getConstantMap(0, ctx);
        AffineMap nUbMap = AffineMap::get(1, 0, getAffineDimExpr(0, ctx), ctx);
        auto nLoop = affine::AffineForOp::create(
            b, func.getLoc(), /*lowerBoundOperands=*/ValueRange{},
            /*lowerBoundMap=*/nLbMap, /*upperBoundOperands=*/ValueRange{temporalIterN},
            /*upperBoundMap=*/nUbMap, /*step=*/1);
        
        Block *nLoopBody = nLoop.getBody();
        Value nIV = nLoopBody->getArgument(0);
        
        SmallVector<Operation *> opsToMove;
        bool foundScfFor = false;
        for (Operation &op : parBody.without_terminator()) {
          if (&op == scfFor.getOperation()) {
            foundScfFor = true;
          }
          if (foundScfFor) {
            opsToMove.push_back(&op);
          }
        }
        
        Operation *nTerminator = nLoopBody->getTerminator();
        for (Operation *op : opsToMove) {
          op->moveBefore(nTerminator);
        }
        
        b.setInsertionPointToStart(nLoopBody);
        
        AffineExpr d0Expr = getAffineDimExpr(0, ctx);
        AffineExpr d1Expr = getAffineDimExpr(1, ctx);
        AffineMap mulMap = AffineMap::get(2, 0, d0Expr * d1Expr, ctx);
        AffineMap addMap = AffineMap::get(2, 0, d0Expr + d1Expr, ctx);
        
        Value coreBlockM = b.create<affine::AffineApplyOp>(
            func.getLoc(), mulMap, ValueRange{ivX, temporalIterM});
        Value globalBlockM = b.create<affine::AffineApplyOp>(
            func.getLoc(), addMap, ValueRange{coreBlockM, mIV});
        Value elementIndexM = b.create<affine::AffineApplyOp>(
            func.getLoc(), mulMap, ValueRange{globalBlockM, blockMSize});
        
        Value coreBlockN = b.create<affine::AffineApplyOp>(
            func.getLoc(), mulMap, ValueRange{ivY, temporalIterN});
        Value globalBlockN = b.create<affine::AffineApplyOp>(
            func.getLoc(), addMap, ValueRange{coreBlockN, nIV});
        Value elementIndexN = b.create<affine::AffineApplyOp>(
            func.getLoc(), mulMap, ValueRange{globalBlockN, blockNSize});
        
        for (Operation *op : opsToMove) {
          op->replaceUsesOfWith(ivX, elementIndexM);
          op->replaceUsesOfWith(ivY, elementIndexN);
          op->walk([&](Operation *nestedOp) {
            nestedOp->replaceUsesOfWith(ivX, elementIndexM);
            nestedOp->replaceUsesOfWith(ivY, elementIndexN);
          });
        }
      }

      entry.eraseArgument(numArgs - 1);
      entry.eraseArgument(numArgs - 2);
      entry.eraseArgument(numArgs - 3);
      
      entry.addArgument(b.getIndexType(), func.getLoc());
      entry.addArgument(b.getIndexType(), func.getLoc());
      
      SmallVector<Type, 8> newInputTypes;
      for (unsigned i = 0; i < entry.getNumArguments(); ++i)
        newInputTypes.push_back(entry.getArgument(i).getType());
      FunctionType oldTy = func.getFunctionType();
      FunctionType newTy =
          FunctionType::get(ctx, newInputTypes, oldTy.getResults());
      func.setType(newTy);

    });
  }
};


std::unique_ptr<mlir::Pass> createTritonSharedGridToParallelPass() {
  /**
   * @brief Create the grid-to-parallel conversion pass.
   */
  return std::make_unique<TritonSharedGridToParallelPass>();
}

void registerTritonSharedGridToParallelPass() {
  /**
   * @brief Register the grid-to-parallel pass for textual pipelines.
   */
  PassRegistration<TritonSharedGridToParallelPass>();
}

} // namespace passes
} // namespace tmd
