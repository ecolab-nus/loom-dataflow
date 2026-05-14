#include "gather_across_iv_tracer.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace loom::utils {
namespace {

bool isTemporalLoop(mlir::Operation *op) {
  if (!op)
    return false;
  mlir::Attribute iterAttr = op->getAttr("loom.iter_type");
  if (!iterAttr)
    return false;
  std::string printed;
  llvm::raw_string_ostream os(printed);
  iterAttr.print(os);
  os.flush();
  return llvm::StringRef(printed).contains("temporal");
}

} // namespace

llvm::SmallVector<mlir::Value, 8> collectDependentIVs(mlir::Value root) {
  llvm::SmallVector<mlir::Value, 8> ivs;
  if (!root)
    return ivs;

  llvm::SmallPtrSet<mlir::Value, 32> visited;
  llvm::SmallPtrSet<mlir::Value, 16> ivSeen;
  llvm::SmallVector<mlir::Value, 32> worklist = {root};

  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!current || !visited.insert(current).second)
      continue;

    if (auto barg = llvm::dyn_cast<mlir::BlockArgument>(current)) {
      mlir::Operation *parent = barg.getOwner()->getParentOp();
      if (auto par = llvm::dyn_cast_or_null<mlir::affine::AffineParallelOp>(parent)) {
        unsigned argNum = barg.getArgNumber();
        if (argNum < par.getNumDims() && ivSeen.insert(current).second)
          ivs.push_back(current);
      } else if (auto forOp = llvm::dyn_cast_or_null<mlir::scf::ForOp>(parent)) {
        if (current == forOp.getInductionVar() && ivSeen.insert(current).second)
          ivs.push_back(current);
      }
      continue;
    }

    if (mlir::Operation *defOp = current.getDefiningOp()) {
      for (mlir::Value operand : defOp->getOperands())
        if (!visited.contains(operand))
          worklist.push_back(operand);
    }
  }

  return ivs;
}

llvm::SmallVector<mlir::Operation *, 8>
collectTemporalDependentLoops(mlir::Value root) {
  llvm::SmallVector<mlir::Operation *, 8> loops;
  llvm::SmallPtrSet<mlir::Operation *, 8> seen;
  for (mlir::Value iv : collectDependentIVs(root)) {
    auto barg = llvm::cast<mlir::BlockArgument>(iv);
    mlir::Operation *loopOp = barg.getOwner()->getParentOp();
    if (!loopOp || seen.contains(loopOp) || !isTemporalLoop(loopOp))
      continue;
    seen.insert(loopOp);
    loops.push_back(loopOp);
  }
  return loops;
}

} // namespace loom::utils
