#include "loop_iv_dependency.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace loom::utils {
namespace {

IterTypeKind getIterTypeKind(mlir::Operation *op) {
  if (!op)
    return IterTypeKind::Unknown;
  mlir::Attribute iterAttr = op->getAttr("loom.iter_type");
  if (!iterAttr)
    return IterTypeKind::Unknown;

  std::string printed;
  llvm::raw_string_ostream os(printed);
  iterAttr.print(os);
  os.flush();
  llvm::StringRef text(printed);
  if (text.contains("spatial"))
    return IterTypeKind::Spatial;
  if (text.contains("temporal"))
    return IterTypeKind::Temporal;
  if (text.contains("sequential"))
    return IterTypeKind::Sequential;
  return IterTypeKind::Unknown;
}

bool matchesFilter(IterTypeKind kind, IterTypeFilter filter) {
  switch (filter) {
  case IterTypeFilter::All:
    return true;
  case IterTypeFilter::Spatial:
    return kind == IterTypeKind::Spatial;
  case IterTypeFilter::Temporal:
    return kind == IterTypeKind::Temporal;
  }
  return true;
}

bool tryAppendLoopIV(mlir::Value value,
                     llvm::SmallPtrSetImpl<mlir::Value> &ivSeen,
                     llvm::SmallVectorImpl<LoopIVDependency> &deps) {
  auto barg = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!barg)
    return false;

  mlir::Operation *parent = barg.getOwner()->getParentOp();
  if (auto par = llvm::dyn_cast_or_null<mlir::affine::AffineParallelOp>(parent)) {
    unsigned argNum = barg.getArgNumber();
    if (argNum >= par.getNumDims() || !ivSeen.insert(value).second)
      return true;
    deps.push_back({value, par.getOperation(), getIterTypeKind(parent), argNum});
    return true;
  }

  if (auto scfFor = llvm::dyn_cast_or_null<mlir::scf::ForOp>(parent)) {
    if (value != scfFor.getInductionVar() || !ivSeen.insert(value).second)
      return true;
    deps.push_back({value, scfFor.getOperation(), getIterTypeKind(parent), 0});
    return true;
  }

  return true;
}

} // namespace

llvm::SmallVector<LoopIVDependency, 8>
collectLoopIVDependencies(mlir::Value root, IterTypeFilter filter) {
  if (!root)
    return {};
  return collectLoopIVDependencies(llvm::ArrayRef<mlir::Value>(root), filter);
}

llvm::SmallVector<LoopIVDependency, 8>
collectLoopIVDependencies(llvm::ArrayRef<mlir::Value> roots,
                          IterTypeFilter filter) {
  llvm::SmallVector<LoopIVDependency, 8> allDeps;
  llvm::SmallPtrSet<mlir::Value, 32> visited;
  llvm::SmallPtrSet<mlir::Value, 16> ivSeen;
  llvm::SmallVector<mlir::Value, 32> worklist;

  for (mlir::Value root : roots)
    if (root)
      worklist.push_back(root);

  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!current || !visited.insert(current).second)
      continue;

    if (llvm::isa<mlir::BlockArgument>(current)) {
      tryAppendLoopIV(current, ivSeen, allDeps);
      continue;
    }

    if (mlir::Operation *defOp = current.getDefiningOp())
      for (mlir::Value operand : defOp->getOperands())
        if (!visited.contains(operand))
          worklist.push_back(operand);
  }

  if (filter == IterTypeFilter::All)
    return allDeps;

  llvm::SmallVector<LoopIVDependency, 8> filtered;
  for (const LoopIVDependency &dep : allDeps)
    if (matchesFilter(dep.iterType, filter))
      filtered.push_back(dep);
  return filtered;
}

llvm::SmallVector<LoopIVDependency, 8>
collectSpatialIVDependencies(mlir::Value root) {
  return collectLoopIVDependencies(root, IterTypeFilter::Spatial);
}

llvm::SmallVector<LoopIVDependency, 8>
collectSpatialIVDependencies(llvm::ArrayRef<mlir::Value> roots) {
  return collectLoopIVDependencies(roots, IterTypeFilter::Spatial);
}

llvm::SmallVector<LoopIVDependency, 8>
collectTemporalIVDependencies(mlir::Value root) {
  return collectLoopIVDependencies(root, IterTypeFilter::Temporal);
}

llvm::SmallVector<LoopIVDependency, 8>
collectTemporalIVDependencies(llvm::ArrayRef<mlir::Value> roots) {
  return collectLoopIVDependencies(roots, IterTypeFilter::Temporal);
}

} // namespace loom::utils

