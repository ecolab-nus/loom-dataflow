#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace loom::utils {

struct BinaryScalarChainMatch {
  mlir::linalg::GenericOp genericOp;
  mlir::Operation *intermediateOp = nullptr;
  mlir::Operation *splitOp = nullptr;
  llvm::SmallVector<mlir::Operation *, 4> firstOps;
  llvm::SmallVector<mlir::Operation *, 4> secondOps;
  llvm::SmallVector<unsigned, 4> firstInputIndices;
  llvm::SmallVector<unsigned, 4> secondInputIndices;
};

bool isPureBinaryScalarOp(mlir::Operation *op);
bool hasControlSemantics(mlir::Operation *op);

class BinaryScalarChainAnalyzer {
public:
  std::optional<BinaryScalarChainMatch>
  findFirstMatch(mlir::linalg::GenericOp genericOp) const;
};

} // namespace loom::utils
