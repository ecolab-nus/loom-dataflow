#ifndef LOOM_PASSES_COMMON_STATIC_MEMORY_ANALYSER_H
#define LOOM_PASSES_COMMON_STATIC_MEMORY_ANALYSER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace mlir {
namespace func {
class FuncOp;
}
namespace affine {
class AffineForOp;
}
} // namespace mlir

namespace loom {

struct VirtualBuffer {};

using SymbolicDim = mlir::OpFoldResult;

struct ShapeSignature {
  llvm::SmallVector<SymbolicDim, 4> dims;
  mlir::Type elementType;

  bool operator==(const ShapeSignature &other) const {
    if (elementType != other.elementType)
      return false;
    if (dims.size() != other.dims.size())
      return false;
    return dims == other.dims;
  }

  unsigned getHashValue() const;
};

} // namespace loom

namespace llvm {
template <> struct DenseMapInfo<loom::ShapeSignature> {
  static inline loom::ShapeSignature getEmptyKey() {
    return {llvm::SmallVector<loom::SymbolicDim, 4>{},
            mlir::Type::getFromOpaquePointer(
                reinterpret_cast<void *>(static_cast<uintptr_t>(-1)))};
  }
  static inline loom::ShapeSignature getTombstoneKey() {
    return {llvm::SmallVector<loom::SymbolicDim, 4>{},
            mlir::Type::getFromOpaquePointer(
                reinterpret_cast<void *>(static_cast<uintptr_t>(-2)))};
  }
  static unsigned getHashValue(const loom::ShapeSignature &val) {
    return val.getHashValue();
  }
  static bool isEqual(const loom::ShapeSignature &lhs,
                      const loom::ShapeSignature &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace loom {

struct TensorNode {
  mlir::Value value;
  mlir::Operation
      *definingOp;     // For BlockArgs, this is the parent affine.for op
  int linearIndex;     // Definition time
  int deathIndex = -1; // Last use time

  struct VirtualBuffer *mappedVB = nullptr;

  TensorNode(mlir::Value v, mlir::Operation *op, int idx)
      : value(v), definingOp(op), linearIndex(idx) {}
};

struct Bucket {
  ShapeSignature signature;
  std::vector<TensorNode> nodes;
  std::vector<struct VirtualBuffer> virtualBuffers;
  int maxColorsRequired = 0;
};

class MemoryAnalysisContext {
public:
  llvm::MapVector<ShapeSignature, Bucket> buckets;
  llvm::DenseMap<mlir::Operation *, int> opIndexMap;
  llvm::DenseMap<mlir::Value, TensorNode *> valueToNodeMap;

  void addTensor(mlir::Value v, ShapeSignature sig, mlir::Operation *defOp,
                 int idx);

  // --- Lifecycle Helpers ---
  int getOpIndex(mlir::Operation *op) const;
  int getLoopEndIndex(mlir::affine::AffineForOp forOp) const;
  int getValueDeathIndex(mlir::Value v) const;

  void dump(llvm::raw_ostream &os) const;
};

llvm::SmallVector<SymbolicDim, 4> traceShape(mlir::Value v);

MemoryAnalysisContext runMemoryAnalysis(mlir::func::FuncOp func);

} // namespace loom

#endif // LOOM_PASSES_COMMON_STATIC_MEMORY_ANALYSER_H
