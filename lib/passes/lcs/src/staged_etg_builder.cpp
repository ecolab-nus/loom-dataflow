#include "staged_etg_builder.h"
#include "compute_op_registry.h"
#include "lcs_utils.h"
#include "ADLDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "ADLTypes.h.inc"
#define GET_OP_CLASSES
#include "ADLOps.h.inc"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"
#include <cassert>
#define GET_OP_CLASSES
#include "LoomEnums.h.inc"
#include "LoomOps.h.inc"
#define GET_ATTRDEF_CLASSES
#include "LoomAttributes.h.inc"

namespace loom {
namespace lcs {

// ==========================================
// Workload
// ==========================================
llvm::json::Value Workload::toJSON() const {
  llvm::json::Array symbols_json;
  llvm::json::Array entries_json;
  for (const auto &[key, expr] : dims) {
    symbols_json.push_back(key);
    llvm::json::Array entry;
    entry.push_back(key);
    entry.push_back(expr.toJSON());
    entries_json.push_back(std::move(entry));
  }

  llvm::json::Object func_inner;
  func_inner["name"] = op;
  func_inner["symbols"] = std::move(symbols_json);
  func_inner["sym_map"] =
      llvm::json::Object{{"entries", std::move(entries_json)}};

  llvm::json::Object func_envelope;
  func_envelope["func"] = std::move(func_inner);
  func_envelope["scenarios"] = llvm::json::Array{};

  return llvm::json::Object{{"Func", std::move(func_envelope)}};
}

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
  for (const auto &workload : workloads) {
    os.indent(indent + 2) << "└─ " << workload.op << " {";
    bool first = true;
    for (const auto &[key, expr] : workload.dims) {
      if (!first)
        os << ", ";
      os << key << ": " << expr;
      first = false;
    }
    os << "}\n";
  }
}

llvm::json::Value HardwareQueue::toJSON() const {
  llvm::json::Array schedules_json;
  for (const auto &workload : workloads) {
    schedules_json.push_back(workload.toJSON());
  }
  return llvm::json::Object{
      {"Sequential",
       llvm::json::Object{{"schedules", std::move(schedules_json)},
                           {"scenarios", llvm::json::Array{}}}}};
}

// ==========================================
// Stage
// ==========================================
Stage::Stage(int id) : stage_id(id) {}

void Stage::pushWorkload(const std::string &unit_name, const std::string &op,
                         std::map<std::string, Expr> dims) {
  if (queues.find(unit_name) == queues.end()) {
    queues[unit_name] = HardwareQueue{unit_name, {}, std::nullopt};
  }
  queues[unit_name].workloads.push_back(Workload{op, std::move(dims)});
}

void Stage::dump(llvm::raw_ostream &os, int indent) const {
  os.indent(indent) << "├── Stage " << stage_id << ":\n";
  for (auto const &[name, queue] : queues) {
    queue.dump(os, indent + 4);
  }
}

llvm::json::Value Stage::toJSON() const {
  // Merge all workloads from all queues into one Sequential
  llvm::json::Array schedules_json;
  for (auto const &[name, queue] : queues) {
    for (const auto &workload : queue.workloads) {
      schedules_json.push_back(workload.toJSON());
    }
  }

  llvm::json::Object sequential;
  sequential["schedules"] = std::move(schedules_json);
  sequential["scenarios"] = llvm::json::Array{};

  llvm::json::Object parallel;
  parallel["Sequential"] = std::move(sequential);

  return llvm::json::Object{{"stage_id", stage_id},
                            {"Parallel", std::move(parallel)}};
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
// ConstraintScope
// ==========================================
llvm::json::Value ConstraintScope::toJSON() const {
  // Build symbols JSON object
  llvm::json::Object symbols_json;
  for (const auto &[name, type] : symbols) {
    symbols_json[name] = type;
  }

  // Build temp_iter JSON array
  llvm::json::Array temp_iter_json;
  for (const auto &t : temp_iter) {
    temp_iter_json.push_back(t.toJSON());
  }

  // Build iter_num JSON object
  llvm::json::Object iter_num_json;
  iter_num_json["seq_iter"] = seq_iter.toJSON();
  iter_num_json["temp_iter"] = std::move(temp_iter_json);

  // Build L1_footprint JSON array
  llvm::json::Array footprint_json;
  for (const auto &term : l1_footprint) {
    footprint_json.push_back(term.toJSON());
  }

  // Build metadata JSON object
  llvm::json::Object metadata_json;
  metadata_json["symbols"] = std::move(symbols_json);
  metadata_json["L1_footprint"] = std::move(footprint_json);
  metadata_json["datatype"] = datatype;
  metadata_json["iter_num"] = std::move(iter_num_json);

  // Build hard_constraints JSON array
  llvm::json::Array hard_constraints_json;
  for (const auto &c : hard_constraints) {
    hard_constraints_json.push_back(c.toJSON());
  }

  // Build final ConstraintScope JSON object
  return llvm::json::Object{
      {"metadata", std::move(metadata_json)},
      {"hard_constraints", std::move(hard_constraints_json)}};
}

// ==========================================
// VariantETG Builder
// ==========================================
VariantETG::VariantETG(llvm::StringRef name, const HWOpRegistry *registry)
    : variant_name(name.str()), compute_scope("ComputeScope"),
      memory_scope("MemoryScope"), hw_registry_(registry) {}

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
    bool is_memory = op->getName().getStringRef() == "loom.copy";
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
    int ready_time =
        (is_infra || is_memory) ? required_stage : required_stage + 1;
    for (mlir::Value result : op->getResults()) {
      value_ready_stage[result] = ready_time;
    }
  });
}

void VariantETG::dispatchToComputeQueues(mlir::Operation *op,
                                         Stage &target_stage) {
  assert(hw_registry_ && "ComputeOpRegistry must be provided");

  if (llvm::isa<mlir::linalg::GenericOp>(op)) {
    dispatchGenericOp(op, target_stage);
  } else {
    dispatchNamedOp(op, target_stage);
  }
}

void VariantETG::dispatchNamedOp(mlir::Operation *op, Stage &target_stage) {
  std::string linalg_op_name = op->getName().getStringRef().str();
  const HWComputeFunc *hwFunc = hw_registry_->lookupMatrixOp(linalg_op_name);
  if (!hwFunc) {
    auto placeholder = HWOpRegistry::makePlaceholder(linalg_op_name);
    target_stage.pushWorkload(placeholder.hw_component,
                              placeholder.hw_func_name, {});
    return;
  }

  auto linalgOp = llvm::cast<mlir::linalg::LinalgOp>(op);
  std::map<std::string, Expr> dimMap;

  // Positional input dim matching
  auto inputs = linalgOp.getDpsInputs();
  for (size_t i = 0; i < inputs.size() && i < hwFunc->input_bindings.size();
       ++i) {
    std::vector<Expr> opDims = traceAllocDimsFromTensor(inputs[i]);
    const auto &hwBinding = hwFunc->input_bindings[i];
    for (size_t d = 0; d < hwBinding.dim_symbols.size() && d < opDims.size();
         ++d) {
      const std::string &hwSym = hwBinding.dim_symbols[d];
      if (dimMap.count(hwSym) == 0) {
        dimMap[hwSym] = opDims[d];
      }
    }
  }

  target_stage.pushWorkload(hwFunc->hw_component, hwFunc->hw_func_name,
                            std::move(dimMap));
}

void VariantETG::dispatchGenericOp(mlir::Operation *op, Stage &target_stage) {
  auto linalgOp = llvm::cast<mlir::linalg::LinalgOp>(op);

  // 1. Analyze generic dims
  GenericDimAnalysis analysis = analyzeGenericDims(linalgOp);

  // 2. Walk body for arith/math ops
  for (mlir::Operation &bodyOp : op->getRegion(0).front()) {
    if (llvm::isa<mlir::linalg::YieldOp>(&bodyOp))
      continue;
    mlir::Dialect *dialect = bodyOp.getDialect();
    if (!dialect)
      continue;
    llvm::StringRef ns = dialect->getNamespace();
    if (ns != "arith" && ns != "math")
      continue;

    std::string bodyOpName = bodyOp.getName().getStringRef().str();

    // 3. Lookup hw func
    const HWComputeFunc *hwFunc =
        hw_registry_->lookupVectorOp(bodyOpName, analysis.generic_class);
    if (!hwFunc) {
      auto placeholder = HWOpRegistry::makePlaceholder(bodyOpName);
      target_stage.pushWorkload(placeholder.hw_component,
                                placeholder.hw_func_name, {});
      continue;
    }

    // 4. Build dimMap from hw symbols
    std::map<std::string, Expr> dimMap;
    if (!hwFunc->parallel_symbol.empty() &&
        !analysis.parallel_product.isNone())
      dimMap[hwFunc->parallel_symbol] = analysis.parallel_product;
    if (!hwFunc->reduction_symbol.empty() &&
        !analysis.reduction_product.isNone())
      dimMap[hwFunc->reduction_symbol] = analysis.reduction_product;

    // 5. Dispatch by hw_component
    target_stage.pushWorkload(hwFunc->hw_component, hwFunc->hw_func_name,
                              std::move(dimMap));
  }
}

void VariantETG::dispatchToMemoryQueues(mlir::Operation *op,
                                        Stage &target_stage) {
  auto copyOp = llvm::dyn_cast<loom::CopyOp>(op);
  if (!copyOp)
    return;

  // Extract key attributes from the copy op.
  std::string srcMem, dstMem;
  if (auto attr = copyOp.getSrcMemSpaceAttr())
    srcMem = attr.getLeafReference().str();
  if (auto attr = copyOp.getDstMemSpaceAttr())
    dstMem = attr.getLeafReference().str();

  llvm::SmallVector<int64_t> broadcast;
  if (auto broadcastAttr = copyOp.getBroadcastAttr()) {
    for (auto val : broadcastAttr)
      broadcast.push_back(mlir::cast<mlir::IntegerAttr>(val).getInt());
  }

  // Registry lookup.
  const HWComputeFunc *hwFunc =
      hw_registry_->lookupDataMoverOp(srcMem, dstMem, broadcast);

  if (!hwFunc) {
    // Fallback placeholder.
    auto placeholder = HWOpRegistry::makePlaceholder(
        "loom.copy[" + srcMem + "->" + dstMem + "]", "data_movers");
    target_stage.pushWorkload(placeholder.hw_component,
                              placeholder.hw_func_name, {});
    return;
  }

  // Build dimMap from L1 memref bindings.
  // Try source first (handles L1→DRAM), then destination (handles DRAM→L1);
  // one side is always an L1 alloc traceable via loom.semaphore_take.
  std::map<std::string, Expr> dimMap;
  loom::AllocOp allocOp = traceToAlloc(copyOp.getSource());
  if (!allocOp)
    allocOp = traceToAlloc(copyOp.getDestination());
  if (allocOp && !hwFunc->input_bindings.empty()) {
    std::vector<Expr> opDims = formatAllocDims(allocOp);
    const auto &binding = hwFunc->input_bindings[0];
    for (size_t d = 0; d < binding.dim_symbols.size() && d < opDims.size();
         ++d) {
      if (dimMap.count(binding.dim_symbols[d]) == 0)
        dimMap[binding.dim_symbols[d]] = opDims[d];
    }
  }

  target_stage.pushWorkload(hwFunc->hw_component, hwFunc->hw_func_name,
                            std::move(dimMap));
}

void VariantETG::buildConstraintScope(mlir::func::FuncOp func_op) {
  // 1. Collect symbols from loom.sym ops
  func_op.walk([&](loom::SymOp op) {
    std::string name = op.getSymbolRef().getLeafReference().str();
    constraint_scope.symbols[name] = "int";
  });

  // 2. Walk affine.for loops in the func, check iter_type attribute
  func_op.walk([&](mlir::affine::AffineForOp forOp) {
    auto iterAttr = forOp->getAttrOfType<loom::IterTypeAttr>("loom.iter_type");
    if (!iterAttr)
      return;

    Expr tripCount = extractLoopTripCount(forOp);
    if (tripCount.isNone())
      return;

    if (iterAttr.getValue() == loom::IterType::Sequential) {
      constraint_scope.seq_iter = tripCount;
    } else if (iterAttr.getValue() == loom::IterType::Temporal) {
      constraint_scope.temp_iter.push_back(tripCount);
    }
  });

  // Helper: given an iter-count Expr containing exactly one Div(N, D), push:
  //   ge(iter, 1)        — the loop runs at least once
  //   divisible(N, D)    — the division is exact (no remainder)
  auto addIterConstraints = [&](const Expr &iter) {
    if (iter.isNone())
      return;

    // Walk the Expr tree to find the unique Div node.
    std::function<std::pair<Expr, Expr>(const Expr &)> findDiv =
        [&](const Expr &e) -> std::pair<Expr, Expr> {
      if (e.isNone())
        return {Expr::none(), Expr::none()};
      if (e.kind() == Expr::Kind::Div)
        return {e.lhs(), e.rhs()};
      auto fromLhs = findDiv(e.lhs());
      if (!fromLhs.first.isNone())
        return fromLhs;
      return findDiv(e.rhs());
    };

    constraint_scope.hard_constraints.push_back(
        ConstraintExpr::ge(iter, Expr::con(1)));

    auto [num, den] = findDiv(iter);
    if (!num.isNone())
      constraint_scope.hard_constraints.push_back(
          ConstraintExpr::divisible(num, den));
  };

  addIterConstraints(constraint_scope.seq_iter);
  for (const Expr &t : constraint_scope.temp_iter)
    addIterConstraints(t);

  // 3. Collect L1 allocations for footprint and datatype
  mlir::Type expectedElemType;
  func_op.walk([&](loom::AllocOp allocOp) {
    // Filter: only @L1 allocations
    if (allocOp.getMemory().getLeafReference() != "L1")
      return;

    // Extract element type from the memref result type
    auto memrefType =
        mlir::cast<mlir::MemRefType>(allocOp.getResult().getType());
    mlir::Type elemType = memrefType.getElementType();

    // Guard: assert all @L1 allocs share the same element type
    if (!expectedElemType) {
      expectedElemType = elemType;
      constraint_scope.datatype = formatElementType(elemType);
    } else {
      assert(elemType == expectedElemType &&
             "All L1 allocations must have the same element type");
    }

    // Collect symbolic footprint for this alloc
    Expr dims = productOfDims(formatAllocDims(allocOp));
    if (!dims.isNone()) {
      constraint_scope.l1_footprint.push_back(dims);
    }
  });
}

void VariantETG::dump(llvm::raw_ostream &os) const {
  os << "Variant ETG: [" << variant_name << "]\n";
  compute_scope.dump(os, 0);
  memory_scope.dump(os, 0);
}

llvm::json::Value VariantETG::toJSON() const {
  return llvm::json::Object{{"variant_name", variant_name},
                            {"compute_scope", compute_scope.toJSON()},
                            {"memory_scope", memory_scope.toJSON()},
                            {"constraint_scope", constraint_scope.toJSON()}};
}

void VariantETG::buildL1FootprintConstraint() {
  // TODO: should be removed after mlar is ready.
  //       This is a workaround method to extract L1 size from the ADL
  //       platform header (adl.memory.array / adl.memory.bank / adl.spatial_dim).
  if (!hw_registry_ || constraint_scope.l1_footprint.empty())
    return;

  mlir::ModuleOp platformModule = hw_registry_->getPlatformModule();
  if (!platformModule)
    return;

  int64_t l1_size = 0;
  platformModule.walk([&](adl::MemoryArrayOp arrayOp) {
    // Only examine direct children of the top-level module.
    if (arrayOp->getParentOp() != platformModule.getOperation())
      return mlir::WalkResult::skip();
    if (arrayOp.getSymName() != "L1")
      return mlir::WalkResult::advance();

    // Compute product of all spatial dimension sizes.
    int64_t spatial_product = 1;
    for (mlir::Value spatialVal : arrayOp.getSpatialDims()) {
      if (auto dimOp = spatialVal.getDefiningOp<adl::SpatialDimOp>())
        spatial_product *= static_cast<int64_t>(dimOp.getSize());
    }

    // Extract bsize and nblk from the bank operand.
    if (auto bankOp = arrayOp.getBank().getDefiningOp<adl::MemoryBankOp>()) {
      int64_t bsize = static_cast<int64_t>(bankOp.getBsize());
      int64_t nblk = static_cast<int64_t>(bankOp.getNblk());
      l1_size = spatial_product * bsize * nblk;
    }
    return mlir::WalkResult::advance();
  });

  if (l1_size == 0)
    return;

  // Sum all per-alloc footprint expressions.
  Expr footprint_sum = Expr::con(0);
  for (const Expr &term : constraint_scope.l1_footprint)
    footprint_sum = footprint_sum + term;

  // Push: sum(L1_footprint) <= L1_size.
  constraint_scope.hard_constraints.push_back(
      ConstraintExpr::le(footprint_sum, Expr::con(l1_size)));
}

} // namespace lcs
} // namespace loom
