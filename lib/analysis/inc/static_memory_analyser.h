#ifndef LOOM_ANALYSIS_STATIC_MEMORY_ANALYSER_H
#define LOOM_ANALYSIS_STATIC_MEMORY_ANALYSER_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
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
  int color = -1;
  mlir::Operation *definingOp = nullptr;

  VirtualBuffer(int id, VBType t) : id(id), type(t) {}

  void addMember(TensorNode *node);
  void updateDefiningOp();
};

struct ShapeSignature {
  llvm::SmallVector<SymbolicDim, 4> dims;
  mlir::Type elementType;

  friend bool operator==(const ShapeSignature &lhs, const ShapeSignature &rhs) {
    return lhs.elementType == rhs.elementType && lhs.dims == rhs.dims;
  }

  friend bool operator!=(const ShapeSignature &lhs, const ShapeSignature &rhs) {
    return !(lhs == rhs);
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
  mlir::Operation *scopeOp =
      nullptr; // The affine.parallel enclosing this bucket

  VirtualBuffer *createVB(int id, VBType type);
  bool containsNode(const TensorNode *node) const;
  TensorNode *findNode(mlir::Value v);
};

struct ColorNode {
  int id = -1;
  llvm::SmallVector<int, 2> vbIds;
};

class InterferenceGraph {
public:
  InterferenceGraph(const Bucket &bucket,
                    const class MemoryAnalysisContext &ctx);

  void build();
  void dump(llvm::raw_ostream &os) const;
  bool interferes(int vbIdA, int vbIdB) const;
  bool colorNodesInterfere(int colorNodeA, int colorNodeB) const;
  int getColorNodeForVB(int vbId) const;
  llvm::ArrayRef<ColorNode> getColorNodes() const { return colorNodes_; }
  bool canRelay(const VirtualBuffer &vbA, const VirtualBuffer &vbB) const;

private:
  const Bucket &bucket_;
  const class MemoryAnalysisContext &ctx_;
  std::set<std::pair<int, int>> colorNodeEdges_;
  llvm::SmallVector<ColorNode, 8> colorNodes_;
  llvm::DenseMap<int, int> vbToColorNodeId_;

  int classifyOverlap(const VirtualBuffer &vbA, const VirtualBuffer &vbB) const;
  void buildColorNodes();
  mlir::OpOperand *findOperandRelation(const VirtualBuffer &vbA,
                                       const VirtualBuffer &vbB) const;
  bool checkHandoffInterference(const VirtualBuffer &vbA,
                                const VirtualBuffer &vbB) const;
};

/// First-fit greedy graph coloring solver.
/// Operates on a single Bucket, using its InterferenceGraph.
class ColoringSolver {
public:
  explicit ColoringSolver(Bucket &bucket);

  /// Run the greedy coloring algorithm.
  /// Returns the number of colors used (= max color + 1).
  int solve();

private:
  Bucket &bucket_;
};

/// Bridge data structure: carries coloring results to MLIR transform passes.
/// Self-contained — a Pass can read this and know exactly how to rewrite IR.
class LoomAllocationPlan {
public:
  /// Describes one physical buffer slot within a bucket.
  struct PhysicalBufferSlot {
    int colorId;            ///< Color ID (0, 1, 2...)
    ShapeSignature shape;   ///< Physical shape (for loom.alloc)
    mlir::Type elementType; ///< Element type
  };

  /// Maps an SSA tensor value to its assigned physical buffer.
  struct Assignment {
    ShapeSignature bucketKey; ///< Which bucket this belongs to
    int colorId;              ///< Which color within the bucket
  };

  /// Per-bucket slot allocations: BucketKey -> list of PhysicalBufferSlots
  llvm::DenseMap<ShapeSignature, std::vector<PhysicalBufferSlot>>
      bucketAllocations;

  /// Core mapping: SSA tensor Value -> Assignment (bucket + color)
  llvm::DenseMap<mlir::Value, Assignment> tensorToBufferMap;

  /// Per-bucket color count
  llvm::DenseMap<ShapeSignature, int> colorCountPerBucket;

  /// Dump the plan for debugging.
  void dump(llvm::raw_ostream &os) const;
};

struct LoopContext {
  mlir::Operation *loopOp; // Either affine::AffineForOp or scf::ForOp
  int startIndex;
  int endIndex;
};

struct SplitYieldInfo {
  unsigned iterArgIndex; // Index within affine.for iter_args
  mlir::Value iterArg;   // The iter_arg value
  mlir::Value yieldVal;  // The corresponding yield value
};

class MemoryAnalysisContext {
public:
  // --- Accessors ---
  const llvm::MapVector<ShapeSignature, Bucket> &getBuckets() const {
    return buckets_;
  }
  llvm::MapVector<ShapeSignature, Bucket> &getBucketsMutable() {
    return buckets_;
  }
  const LoomAllocationPlan &getAllocationPlan() const {
    return allocationPlan_;
  }
  const std::vector<SplitYieldInfo> &getSplitYields() const {
    return splitYields_;
  }

  int getOpIndex(mlir::Operation *op) const;
  mlir::Operation *getOpFromIndex(int index) const;
  int getLoopEndIndex(mlir::Operation *loopOp) const;
  int getValueDeathIndex(mlir::Value v) const;

  // --- Analysis Entry Points ---
  void setOpIndex(mlir::Operation *op, int idx);
  void addTensor(mlir::Value v, ShapeSignature sig, mlir::Operation *defOp,
                 int idx);
  void computeDeathIndices();
  void buildVirtualBuffers();
  void fuseRelayVirtualBuffers();
  void buildInterferenceGraphs();
  void solveColoring();
  void buildAllocationPlan();

  void dump(llvm::raw_ostream &os) const;

private:
  llvm::MapVector<ShapeSignature, Bucket> buckets_;
  llvm::DenseMap<mlir::Operation *, int> opIndexMap_;
  std::vector<mlir::Operation *> indexToOpMap_;
  llvm::DenseMap<mlir::Value, TensorNode *> valueToNodeMap_;
  std::vector<SplitYieldInfo> splitYields_;
  int nextVBId_ = 0;
  LoomAllocationPlan allocationPlan_;

  // --- Internal Helpers ---
  std::optional<LoopContext> findLoopContext() const;
  void applyPhiFusionAxiom(Bucket &bucket, const LoopContext &loop);
  void applyExternalEternityAxiom(Bucket &bucket, const LoopContext &loop);
  void applyStandardAxiom(Bucket &bucket);
};

MemoryAnalysisContext runMemoryAnalysis(mlir::func::FuncOp func);

} // namespace loom

#endif // LOOM_ANALYSIS_STATIC_MEMORY_ANALYSER_H
