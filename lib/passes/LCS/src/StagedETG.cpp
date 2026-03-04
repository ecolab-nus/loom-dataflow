#include "StagedETG.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"

namespace loom {
namespace lcs {

// ==========================================
// HardwareQueue
// ==========================================
void HardwareQueue::dump(llvm::raw_ostream &os, int indent) const {
  os.indent(indent) << (type == HardwareUnitType::FPU ? "FPU Queue:"
                                                      : "SFPU Queue:");
  if (workloads.empty()) {
    os << " <Empty>\n";
    return;
  }
  os << "\n";
  for (auto *op : workloads) {
    os.indent(indent + 2) << "└─ " << op->getName().getStringRef() << "\n";
  }
}

llvm::json::Value HardwareQueue::toJSON() const {
  llvm::json::Array workloads_json;
  for (auto *op : workloads) {
    workloads_json.push_back(op->getName().getStringRef());
  }
  return llvm::json::Object{
      {"type", type == HardwareUnitType::FPU ? "FPU" : "SFPU"},
      {"workloads", std::move(workloads_json)}};
}

// ==========================================
// Stage
// ==========================================
Stage::Stage(int id) : stage_id(id) {
  fpu_queue.type = HardwareUnitType::FPU;
  sfpu_queue.type = HardwareUnitType::SFPU;
}

void Stage::assignWorkload(HardwareUnitType unit_type, mlir::Operation *op) {
  if (unit_type == HardwareUnitType::FPU) {
    fpu_queue.workloads.push_back(op);
  } else {
    sfpu_queue.workloads.push_back(op);
  }
}

void Stage::dump(llvm::raw_ostream &os, int indent) const {
  os.indent(indent) << "├── Stage " << stage_id << ":\n";
  fpu_queue.dump(os, indent + 4);
  sfpu_queue.dump(os, indent + 4);
}

llvm::json::Value Stage::toJSON() const {
  return llvm::json::Object{{"stage_id", stage_id},
                            {"fpu_queue", fpu_queue.toJSON()},
                            {"sfpu_queue", sfpu_queue.toJSON()}};
}

// ==========================================
// Scope
// ==========================================
Scope::Scope(std::string name) : scope_name(name) {}

Stage &Scope::getOrCreateStage(int id) {
  if (stages.find(id) == stages.end()) {
    stages.emplace(id, Stage(id));
  }
  return stages.at(id);
}

void Scope::dump(llvm::raw_ostream &os, int indent) const {
  os.indent(indent) << "└── Scope [" << scope_name << "]:\n";
  for (auto &pair : stages) {
    pair.second.dump(os, indent + 4);
  }
}

llvm::json::Value Scope::toJSON() const {
  llvm::json::Array stages_json;
  for (auto &pair : stages) {
    stages_json.push_back(pair.second.toJSON());
  }
  return llvm::json::Object{{"scope_name", scope_name},
                            {"stages", std::move(stages_json)}};
}

// ==========================================
// VariantETG
// ==========================================
VariantETG::VariantETG(llvm::StringRef name)
    : variant_name(name.str()), compute_scope("ComputeScope") {}

void VariantETG::buildFromAffineFor(mlir::affine::AffineForOp for_op) {
  llvm::DenseMap<mlir::Value, int> value_ready_stage;

  // 0. Initialize Loop Args
  for (mlir::Value block_arg : for_op.getRegion().getArguments()) {
    value_ready_stage[block_arg] = 0;
  }

  // 1. Walk through the loop body
  for_op.getBody()->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    // Skip ops not in the loop body directly (except for internal
    // decomposition)
    if (op->getParentOp() != for_op)
      return;

    // Only process LinalgOps for now
    auto linalg_op = llvm::dyn_cast<mlir::linalg::LinalgOp>(op);
    if (!linalg_op)
      return;

    // A. Calculate ASAP stage based on dependencies
    int required_stage = 0;
    for (mlir::Value operand : op->getOperands()) {
      if (value_ready_stage.count(operand)) {
        required_stage = std::max(required_stage, value_ready_stage[operand]);
      }
    }

    // B. Classification and assignment
    bool is_infra = llvm::isa<mlir::linalg::FillOp, mlir::linalg::CopyOp>(op);
    if (!is_infra) {
      Stage &target_stage = compute_scope.getOrCreateStage(required_stage);
      dispatchToHardwareQueues(op, target_stage);
    }

    // C. Update output ready times
    // If it's infra (fill/copy), it's zero-latency -> ready at required_stage.
    // Otherwise, it takes time -> ready at required_stage + 1.
    int ready_time = is_infra ? required_stage : required_stage + 1;
    for (mlir::Value result : op->getResults()) {
      value_ready_stage[result] = ready_time;
    }
  });
}

void VariantETG::dispatchToHardwareQueues(mlir::Operation *op,
                                          Stage &target_stage) {
  // FPU: Matmul, BatchMatmul
  if (llvm::isa<mlir::linalg::BatchMatmulOp, mlir::linalg::MatmulOp>(op)) {
    target_stage.assignWorkload(HardwareUnitType::FPU, op);
    return;
  }

  // SFPU: generic body's arith/math ops
  if (auto generic_op = llvm::dyn_cast<mlir::linalg::GenericOp>(op)) {
    generic_op.getBody()->walk([&](mlir::Operation *inner_op) {
      // Filter for arith and math dialects, skip yield
      auto *dialect = inner_op->getDialect();
      if (dialect && (llvm::isa<mlir::arith::ArithDialect>(dialect) ||
                      llvm::isa<mlir::math::MathDialect>(dialect))) {
        target_stage.assignWorkload(HardwareUnitType::SFPU, inner_op);
      }
    });
    return;
  }
}

void VariantETG::dump(llvm::raw_ostream &os) const {
  os << "Variant ETG: [" << variant_name << "]\n";
  compute_scope.dump(os, 0);
}

llvm::json::Value VariantETG::toJSON() const {
  return llvm::json::Object{{"variant_name", variant_name},
                            {"compute_scope", compute_scope.toJSON()}};
}

} // namespace lcs
} // namespace loom
