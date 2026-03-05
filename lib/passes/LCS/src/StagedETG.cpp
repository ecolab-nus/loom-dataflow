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
  os.indent(indent) << unit_name << " Queue:";
  if (workloads.empty()) {
    os << " <Empty>\n";
    return;
  }
  os << "\n";
  for (const auto &label : workloads) {
    os.indent(indent + 2) << "└─ " << label << "\n";
  }
}

llvm::json::Value HardwareQueue::toJSON() const {
  llvm::json::Array workloads_json;
  for (const auto &label : workloads) {
    workloads_json.push_back(label);
  }
  return llvm::json::Object{{"unit_name", unit_name},
                            {"workloads", std::move(workloads_json)}};
}

// ==========================================
// Stage
// ==========================================
Stage::Stage(int id) : stage_id(id) {}

void Stage::pushWorkload(const std::string &unit_name,
                         const std::string &label) {
  if (queues.find(unit_name) == queues.end()) {
    queues[unit_name] = HardwareQueue{unit_name, {}};
  }
  queues[unit_name].workloads.push_back(label);
}

void Stage::dump(llvm::raw_ostream &os, int indent) const {
  os.indent(indent) << "├── Stage " << stage_id << ":\n";
  for (auto const &[name, queue] : queues) {
    queue.dump(os, indent + 4);
  }
}

llvm::json::Value Stage::toJSON() const {
  llvm::json::Object queues_json;
  for (auto const &[name, queue] : queues) {
    queues_json[name] = queue.toJSON();
  }
  return llvm::json::Object{{"stage_id", stage_id},
                            {"queues", std::move(queues_json)}};
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
    : variant_name(name.str()), compute_scope("ComputeScope"),
      memory_scope("MemoryScope") {}

void VariantETG::buildFromAffineFor(mlir::affine::AffineForOp for_op) {
  llvm::DenseMap<mlir::Value, int> value_ready_stage;

  // 0. Initialize Loop Args
  for (mlir::Value block_arg : for_op.getRegion().getArguments()) {
    value_ready_stage[block_arg] = 0;
  }

  // 1. Walk through the loop body
  for_op.getBody()->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (op->getParentOp() != for_op)
      return;

    // A. Calculate ASAP stage based on dependencies
    int required_stage = 0;
    for (mlir::Value operand : op->getOperands()) {
      if (value_ready_stage.count(operand)) {
        required_stage = std::max(required_stage, value_ready_stage[operand]);
      }
    }

    // B. Classification and assignment
    bool is_compute = llvm::isa<mlir::linalg::LinalgOp>(op);
    bool is_memory = op->getName().getStringRef() == "loom.copy_to_tensor";
    bool is_infra =
        is_compute && llvm::isa<mlir::linalg::FillOp, mlir::linalg::CopyOp>(op);

    if (is_compute && !is_infra) {
      Stage &target_stage = compute_scope.getOrCreateStage(required_stage);
      dispatchToComputeQueues(op, target_stage);
    }

    if (is_memory) {
      Stage &target_stage = memory_scope.getOrCreateStage(required_stage);
      dispatchToMemoryQueues(op, target_stage);
    }

    // C. Update output ready times
    // If it's infra or memory, assume it doesn't block compute timing for logic
    // dependencies (or is handled zero-latency in this simple model).
    int ready_time =
        (is_infra || is_memory) ? required_stage : required_stage + 1;
    for (mlir::Value result : op->getResults()) {
      value_ready_stage[result] = ready_time;
    }
  });
}

void VariantETG::dispatchToComputeQueues(mlir::Operation *op,
                                         Stage &target_stage) {
  // FPU: Matmul, BatchMatmul
  if (llvm::isa<mlir::linalg::BatchMatmulOp, mlir::linalg::MatmulOp>(op)) {
    target_stage.pushWorkload("FPU", op->getName().getStringRef().str());
    return;
  }

  // SFPU: generic body's arith/math ops
  if (auto generic_op = llvm::dyn_cast<mlir::linalg::GenericOp>(op)) {
    generic_op.getBody()->walk([&](mlir::Operation *inner_op) {
      auto *dialect = inner_op->getDialect();
      if (dialect && (llvm::isa<mlir::arith::ArithDialect>(dialect) ||
                      llvm::isa<mlir::math::MathDialect>(dialect))) {
        target_stage.pushWorkload("SFPU",
                                  inner_op->getName().getStringRef().str());
      }
    });
    return;
  }
}

std::string VariantETG::classifyCopyTransfer(mlir::Operation *op) {
  auto interconnect_attr = op->getAttrOfType<mlir::ArrayAttr>("interconnect");
  if (!interconnect_attr)
    return "d";

  bool has_h = false;
  bool has_v = false;
  for (mlir::Attribute attr : interconnect_attr) {
    if (auto symbol = llvm::dyn_cast<mlir::SymbolRefAttr>(attr)) {
      std::string name = symbol.getLeafReference().str();
      if (name.find("horizontal") != std::string::npos)
        has_h = true;
      if (name.find("vertical") != std::string::npos)
        has_v = true;
    }
  }

  if (has_h && has_v)
    return "a";
  if (has_h)
    return "h";
  if (has_v)
    return "v";
  return "d";
}

void VariantETG::dispatchToMemoryQueues(mlir::Operation *op,
                                        Stage &target_stage) {
  std::string label = classifyCopyTransfer(op);
  if (label == "d" || label == "a") {
    target_stage.pushWorkload("NoC_H", label);
    target_stage.pushWorkload("NoC_V", label);
  } else if (label == "h") {
    target_stage.pushWorkload("NoC_H", label);
  } else if (label == "v") {
    target_stage.pushWorkload("NoC_V", label);
  }
}

void VariantETG::dump(llvm::raw_ostream &os) const {
  os << "Variant ETG: [" << variant_name << "]\n";
  compute_scope.dump(os, 0);
  memory_scope.dump(os, 0);
}

llvm::json::Value VariantETG::toJSON() const {
  return llvm::json::Object{{"variant_name", variant_name},
                            {"compute_scope", compute_scope.toJSON()},
                            {"memory_scope", memory_scope.toJSON()}};
}

} // namespace lcs
} // namespace loom
