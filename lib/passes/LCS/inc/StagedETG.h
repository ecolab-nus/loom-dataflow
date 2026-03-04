#ifndef LOOM_LCS_STAGED_ETG_H
#define LOOM_LCS_STAGED_ETG_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <string>
#include <vector>

namespace loom {
namespace lcs {

enum class HardwareUnitType { FPU, SFPU };

struct HardwareQueue {
  HardwareUnitType type;
  std::vector<mlir::Operation *> workloads;

  void dump(llvm::raw_ostream &os, int indent = 0) const;
  llvm::json::Value toJSON() const;
};

struct Stage {
  int stage_id;
  HardwareQueue fpu_queue;
  HardwareQueue sfpu_queue;

  Stage(int id);
  void assignWorkload(HardwareUnitType unit_type, mlir::Operation *op);
  void dump(llvm::raw_ostream &os, int indent = 0) const;
  llvm::json::Value toJSON() const;
};

struct Scope {
  std::string scope_name;
  std::map<int, Stage> stages;

  Scope(std::string name);
  Stage &getOrCreateStage(int id);
  void dump(llvm::raw_ostream &os, int indent = 0) const;
  llvm::json::Value toJSON() const;
};

class VariantETG {
public:
  std::string variant_name;
  Scope compute_scope;

  VariantETG(llvm::StringRef name);
  void buildFromAffineFor(mlir::affine::AffineForOp for_op);
  void dump(llvm::raw_ostream &os) const;
  llvm::json::Value toJSON() const;

private:
  void dispatchToHardwareQueues(mlir::Operation *op, Stage &target_stage);
};

} // namespace lcs
} // namespace loom

#endif // LOOM_LCS_STAGED_ETG_H
