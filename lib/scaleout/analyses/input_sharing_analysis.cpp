#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace tmd_affine_analysis {

// Find the nearest enclosing affine.parallel and return its IV values in order.
static SmallVector<Value, 4> getEnclosingParallelIVs(Operation *op) {
  SmallVector<Value, 4> ivs;
  Operation *parent = op->getParentOp();
  while (parent) {
    if (auto par = dyn_cast<affine::AffineParallelOp>(parent)) {
      for (Value iv : par.getIVs())
        ivs.push_back(iv);
      break; // Use the innermost enclosing affine.parallel
    }
    parent = parent->getParentOp();
  }
  return ivs;
}

// Compute the set of enclosing parallel IVs that the load index map is
// independent of. We conservatively assume the load map's dims correspond to
// the concatenation of all surrounding IVs (parallel first, then others).
static void annotateLoadIndependence(affine::AffineLoadOp loadOp) {
  Operation *op = loadOp.getOperation();
  SmallVector<Value, 4> parIVs = getEnclosingParallelIVs(op);
  if (parIVs.empty())
    return;

  // Build a set of index operands used by this load.
  llvm::SmallPtrSet<Value, 8> indexOperandsSet;
  for (Value idx : loadOp.getIndices())
    indexOperandsSet.insert(idx);

  SmallVector<bool, 4> dependsOnPar(parIVs.size(), false);
  for (size_t i = 0; i < parIVs.size(); ++i) {
    if (indexOperandsSet.contains(parIVs[i]))
      dependsOnPar[i] = true;
  }

  // Build attribute: set of independent parallel IV names by SSA value name.
  // Since SSA values may not have stable names, we encode indices: 0..P-1.
  SmallVector<Attribute, 4> independent;
  MLIRContext *ctx = loadOp.getContext();
  for (size_t i = 0; i < dependsOnPar.size(); ++i) {
    if (!dependsOnPar[i])
      independent.push_back(
          IntegerAttr::get(IndexType::get(ctx), static_cast<int64_t>(i)));
  }

  ArrayAttr arr = ArrayAttr::get(ctx, independent);
  loadOp->setAttr("tmd.input_sharing.independent_parallel_iv_indices", arr);
}

void runInputSharingAnalysis(func::FuncOp funcOp) {
  funcOp.walk(
      [&](affine::AffineLoadOp loadOp) { annotateLoadIndependence(loadOp); });
}

} // namespace tmd_affine_analysis
