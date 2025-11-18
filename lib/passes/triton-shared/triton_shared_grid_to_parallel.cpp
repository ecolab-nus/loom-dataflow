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

      // Add two-level affine.for loops around scf.for to enable multi-block
      // processing per core.
      
      // Find scf.for loop in the parallel body
      scf::ForOp scfFor = nullptr;
      for (Operation &op : parBody) {
        if (auto forOp = llvm::dyn_cast<scf::ForOp>(&op)) {
          scfFor = forOp;
          break;
        }
      }
      
      if (scfFor) {
        // Extract blockm and blockn from scf.for return type
        int64_t blockm = 64, blockn = 64; // defaults
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
        
        // Create ceildiv maps for loop bounds
        // First loop: ceildiv(sizeX, blockm) where sizeX is ivX's bound
        // Second loop: ceildiv(sizeY, blockn) where sizeY is ivY's bound
        AffineExpr d0 = getAffineDimExpr(0, ctx);
        
        // Create ceildiv maps: (d0) -> (d0 ceildiv blockm) and (d0) -> (d0 ceildiv blockn)
        AffineMap mCeilDivMap = AffineMap::get(1, 0, d0.ceilDiv(blockm), ctx);
        AffineMap nCeilDivMap = AffineMap::get(1, 0, d0.ceilDiv(blockn), ctx);
        
        // Create first affine.for loop (m dimension)
        OpBuilder::InsertionGuard loopGuard(b);
        b.setInsertionPoint(scfFor);
        AffineMap mLbMap = AffineMap::getConstantMap(0, ctx);
        auto mLoop = affine::AffineForOp::create(
            b, func.getLoc(), /*lowerBoundOperands=*/ValueRange{},
            /*lowerBoundMap=*/mLbMap, /*upperBoundOperands=*/ValueRange{sizeXi},
            /*upperBoundMap=*/mCeilDivMap, /*step=*/1);
        
        Block *mLoopBody = mLoop.getBody();
        Value mIV = mLoopBody->getArgument(0);
        
        // Create second affine.for loop (n dimension) inside m loop
        b.setInsertionPointToStart(mLoopBody);
        AffineMap nLbMap = AffineMap::getConstantMap(0, ctx);
        auto nLoop = affine::AffineForOp::create(
            b, func.getLoc(), /*lowerBoundOperands=*/ValueRange{},
            /*lowerBoundMap=*/nLbMap, /*upperBoundOperands=*/ValueRange{sizeYi},
            /*upperBoundMap=*/nCeilDivMap, /*step=*/1);
        
        Block *nLoopBody = nLoop.getBody();
        Value nIV = nLoopBody->getArgument(0);
        
        // CRITICAL: Collect ALL operations starting from scf.for to the end of parBody
        // (excluding terminator). We need to move all these operations into nLoopBody.
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
        
        // Move all operations from scf.for onwards into n loop body (before terminator)
        Operation *nTerminator = nLoopBody->getTerminator();
        for (Operation *op : opsToMove) {
          op->moveBefore(nTerminator);
        }
        
        // Now create the new index calculations at the start of nLoopBody, before everything.
        // These will replace ivX with mIV*blockm + ivX, and ivY with nIV*blockn + ivY.
        b.setInsertionPointToStart(nLoopBody);
        AffineExpr mIVExpr = getAffineDimExpr(0, ctx); // mIV
        AffineExpr blockmExpr2 = getAffineConstantExpr(blockm, ctx);
        AffineExpr ivXExpr = getAffineDimExpr(1, ctx); // ivX (from parallel)
        AffineMap mIndexMap = AffineMap::get(2, 0, mIVExpr * blockmExpr2 + ivXExpr, ctx);
        Value newMIndex = b.create<affine::AffineApplyOp>(
            func.getLoc(), mIndexMap, ValueRange{mIV, ivX});
        
        AffineExpr nIVExpr = getAffineDimExpr(0, ctx); // nIV
        AffineExpr blocknExpr2 = getAffineConstantExpr(blockn, ctx);
        AffineExpr ivYExpr = getAffineDimExpr(1, ctx); // ivY (from parallel)
        AffineMap nIndexMap = AffineMap::get(2, 0, nIVExpr * blocknExpr2 + ivYExpr, ctx);
        Value newNIndex = b.create<affine::AffineApplyOp>(
            func.getLoc(), nIndexMap, ValueRange{nIV, ivY});
        
        // Replace all uses of ivX and ivY in the moved operations.
        // Since we collected operations BEFORE creating newMIndex and newNIndex,
        // those new operations are NOT in opsToMove, avoiding self-reference.
        // We need to recursively replace uses within nested regions (e.g., scf.for body).
        for (Operation *op : opsToMove) {
          op->replaceUsesOfWith(ivX, newMIndex);
          op->replaceUsesOfWith(ivY, newNIndex);
          // Also recursively replace uses in all nested regions
          op->walk([&](Operation *nestedOp) {
            nestedOp->replaceUsesOfWith(ivX, newMIndex);
            nestedOp->replaceUsesOfWith(ivY, newNIndex);
          });
        }
      }

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

      // After introducing the 3-D parallel, prune unused IVs.
      // Identify used IV indices among {0,1,2}.
      SmallVector<unsigned, 3> usedIdx;
      auto ivs = par.getIVs();
      for (unsigned i = 0; i < ivs.size(); ++i)
        if (!ivs[i].use_empty())
          usedIdx.push_back(i);
      if (usedIdx.size() < ivs.size()) {
        if (usedIdx.empty())
          usedIdx.push_back(0); // keep at least one dim

        MLIRContext *ctx2 = func.getContext();
        // Build new bounds: lb = 0 per used dim; ub selects corresponding size.
        SmallVector<AffineMap, 4> newLbMaps;
        SmallVector<AffineMap, 4> newUbMaps;
        SmallVector<int64_t, 4> newSteps;
        SmallVector<Value, 4> lbArgs2; // none for constant zero
        SmallVector<Value, 4> ubArgs2;

        // Original ub operands order: [sizeX, sizeY, sizeZ]
        SmallVector<Value, 3> oldUbOps{sizeXi, sizeYi, sizeZi};

        for (unsigned ignored : usedIdx)
          (void)ignored,
              newLbMaps.push_back(AffineMap::getConstantMap(0, ctx2));
        unsigned nUsed = usedIdx.size();
        for (unsigned i = 0; i < nUsed; ++i) {
          newUbMaps.push_back(AffineMap::get(/*dimCount=*/nUsed, /*symCount=*/0,
                                             getAffineDimExpr(i, ctx2)));
          ubArgs2.push_back(oldUbOps[usedIdx[i]]);
        }
        for (unsigned pos : usedIdx)
          newSteps.push_back(steps[pos]);

        OpBuilder::InsertionGuard g2(b);
        b.setInsertionPoint(par);
        auto newPar = b.create<affine::AffineParallelOp>(
            par.getLoc(), /*resultTypes=*/TypeRange{},
            /*reductions=*/ArrayRef<arith::AtomicRMWKind>{}, newLbMaps,
            /*lbArgs=*/ValueRange(lbArgs2), newUbMaps,
            /*ubArgs=*/ValueRange(ubArgs2), newSteps);

        // Remap IVs and clone old body.
        IRMapping mapper;
        Block &newBody = *newPar.getBody();
        for (auto it : llvm::enumerate(usedIdx))
          mapper.map(ivs[it.value()], newBody.getArgument(it.index()));

        Block &oldBody = *par.getBody();
        OpBuilder nb(&newBody, newBody.begin());
        for (Operation &op :
             llvm::make_early_inc_range(oldBody.without_terminator()))
          nb.clone(op, mapper);

        par.erase();
      }
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
