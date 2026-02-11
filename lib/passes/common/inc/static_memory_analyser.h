#ifndef LOOM_PASSES_COMMON_STATIC_MEMORY_ANALYSER_H
#define LOOM_PASSES_COMMON_STATIC_MEMORY_ANALYSER_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <deque>
#include <memory>
#include <optional>
#include <set>
#include <vector>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

#include "utils.h"

namespace loom {

struct TensorNode; // Forward declaration

using loom::utils::SymbolicDim;

enum class VBType {
  Standard,   // Internal computation (Axiom 3)
  Fused,      // Loop Phi-node fusion (Axiom 1)
  Eternal,    // External read-only / template (Axiom 2)
  LoopCarried // Loop-carried but not fusing Init (Axiom 1 split)
};

llvm::StringRef toString(VBType type);

struct LivenessRange {
  int birth = -1;
  int death = -1;
  bool isValid() const { return birth >= 0 && death >= 0; }
};

struct VirtualBuffer {
  int id;
  VBType type;
  std::vector<TensorNode *> members;
  LivenessRange liveness;
  std::set<int> interferenceSet;
  int color = -1;
  mlir::Operation *definingOp = nullptr;

  VirtualBuffer(int id, VBType t) : id(id), type(t) {}

  void addMember(TensorNode *node);
  void updateDefiningOp();
};

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
  void print(llvm::raw_ostream &os) const;
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
  mlir::Operation *definingOp;
  int linearIndex;     // Definition time
  int deathIndex = -1; // Last use time

  std::optional<int> mappedVBId;

  TensorNode(mlir::Value v, mlir::Operation *op, int idx)
      : value(v), definingOp(op), linearIndex(idx) {}
};

struct Bucket {
  ShapeSignature signature;
  std::deque<TensorNode> nodes;
  std::vector<std::unique_ptr<VirtualBuffer>> virtualBuffers;
  int maxColorsRequired = 0;
  std::unique_ptr<class InterferenceGraph> interferenceGraph;

  VirtualBuffer *createVB(int id, VBType type);
  bool containsNode(const TensorNode *node) const;
  TensorNode *findNode(mlir::Value v);
};

class InterferenceGraph {
public:
  InterferenceGraph(const Bucket &bucket,
                    const class MemoryAnalysisContext &ctx);

  void build();
  void dump(llvm::raw_ostream &os) const;
  bool interferes(int vbIdA, int vbIdB) const;

private:
  const Bucket &bucket_;
  const class MemoryAnalysisContext &ctx_;
  std::set<std::pair<int, int>> edges_;

  int classifyOverlap(const VirtualBuffer &vbA, const VirtualBuffer &vbB) const;
  mlir::OpOperand *findOperandRelation(const VirtualBuffer &vbA,
                                       const VirtualBuffer &vbB) const;
  bool checkHandoffInterference(const VirtualBuffer &vbA,
                                const VirtualBuffer &vbB) const;
};

struct LoopContext {
  mlir::affine::AffineForOp forOp;
  int startIndex;
  int endIndex;
};

class MemoryAnalysisContext {
public:
  // --- Accessors ---
  const llvm::MapVector<ShapeSignature, Bucket> &getBuckets() const {
    return buckets_;
  }
  int getOpIndex(mlir::Operation *op) const;
  int getLoopEndIndex(mlir::affine::AffineForOp forOp) const;
  int getValueDeathIndex(mlir::Value v) const;

  // --- Analysis Entry Points ---
  void setOpIndex(mlir::Operation *op, int idx);
  void addTensor(mlir::Value v, ShapeSignature sig, mlir::Operation *defOp,
                 int idx);
  void computeDeathIndices();
  void buildVirtualBuffers();
  void buildInterferenceGraphs();

  void dump(llvm::raw_ostream &os) const;

private:
  llvm::MapVector<ShapeSignature, Bucket> buckets_;
  llvm::DenseMap<mlir::Operation *, int> opIndexMap_;
  llvm::DenseMap<mlir::Value, TensorNode *> valueToNodeMap_;
  int nextVBId_ = 0;

  // --- Internal Helpers ---
  std::optional<LoopContext> findLoopContext() const;
  void applyPhiFusionAxiom(Bucket &bucket, const LoopContext &loop);
  void applyExternalEternityAxiom(Bucket &bucket, const LoopContext &loop);
  void applyStandardAxiom(Bucket &bucket);
};

MemoryAnalysisContext runMemoryAnalysis(mlir::func::FuncOp func);

} // namespace loom

#endif // LOOM_PASSES_COMMON_STATIC_MEMORY_ANALYSER_H
