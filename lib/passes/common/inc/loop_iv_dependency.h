#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace loom::utils {

enum class IterTypeKind { Spatial, Temporal, Sequential, Unknown };
enum class IterTypeFilter { All, Spatial, Temporal };

struct LoopIVDependency {
  mlir::Value iv;
  mlir::Operation *loop = nullptr;
  IterTypeKind iterType = IterTypeKind::Unknown;
  unsigned ivIndex = 0;
};

llvm::SmallVector<LoopIVDependency, 8>
collectLoopIVDependencies(mlir::Value root,
                          IterTypeFilter filter = IterTypeFilter::All);

llvm::SmallVector<LoopIVDependency, 8>
collectLoopIVDependencies(llvm::ArrayRef<mlir::Value> roots,
                          IterTypeFilter filter = IterTypeFilter::All);

llvm::SmallVector<LoopIVDependency, 8>
collectSpatialIVDependencies(mlir::Value root);

llvm::SmallVector<LoopIVDependency, 8>
collectSpatialIVDependencies(llvm::ArrayRef<mlir::Value> roots);

llvm::SmallVector<LoopIVDependency, 8>
collectTemporalIVDependencies(mlir::Value root);

llvm::SmallVector<LoopIVDependency, 8>
collectTemporalIVDependencies(llvm::ArrayRef<mlir::Value> roots);

} // namespace loom::utils

