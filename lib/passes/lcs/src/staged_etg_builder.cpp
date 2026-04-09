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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"
#include <cassert>
#include <set>
#define GET_OP_CLASSES
#include "LoomEnums.h.inc"
#include "LoomOps.h.inc"
#define GET_ATTRDEF_CLASSES
#include "LoomAttributes.h.inc"

namespace loom {
namespace lcs {

// ==========================================
// File-local helpers
// ==========================================
namespace {

/// Walk the Expr tree to find the first Div node.
/// Returns {numerator, denominator}; both Expr::none() if no Div is found.
std::pair<Expr, Expr> findDivNode(const Expr &e) {
  if (e.isNone())
    return {Expr::none(), Expr::none()};
  if (e.kind() == Expr::Kind::Div)
    return {e.lhs(), e.rhs()};
  auto fromLhs = findDivNode(e.lhs());
  if (!fromLhs.first.isNone())
    return fromLhs;
  return findDivNode(e.rhs());
}

/// A group of workloads that must execute sequentially (they share resources).
struct ResourceGroup {
  std::set<std::string> resources;
  std::vector<const Workload *> workloads;
};

/// Partition workloads by resource overlap using transitive closure.
/// Workloads with disjoint resource sets end up in separate groups
/// and can be scheduled in parallel.
std::vector<ResourceGroup>
groupWorkloadsByResources(const std::vector<const Workload *> &all_workloads) {
  std::vector<ResourceGroup> groups;

  for (const Workload *w : all_workloads) {
    std::set<std::string> rw(w->resources.begin(), w->resources.end());

    // Collect indices of existing groups that share any resource with w.
    std::vector<size_t> overlapping;
    for (size_t i = 0; i < groups.size(); ++i)
      for (const auto &r : rw)
        if (groups[i].resources.count(r)) {
          overlapping.push_back(i);
          break;
        }

    if (overlapping.empty()) {
      groups.push_back(ResourceGroup{rw, {w}});
    } else {
      // Merge w and all overlapping groups into the first one.
      auto &target = groups[overlapping[0]];
      target.resources.insert(rw.begin(), rw.end());
      target.workloads.push_back(w);
      // Merge remaining groups in reverse index order to keep indices valid.
      for (size_t j = overlapping.size(); j > 1; --j) {
        size_t idx = overlapping[j - 1];
        target.resources.insert(groups[idx].resources.begin(),
                                groups[idx].resources.end());
        target.workloads.insert(target.workloads.end(),
                                groups[idx].workloads.begin(),
                                groups[idx].workloads.end());
        groups.erase(groups.begin() + idx);
      }
    }
  }
  return groups;
}

} // namespace

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

// ==========================================
// Stage
// ==========================================
Stage::Stage(int id) : stage_id(id) {}

void Stage::pushWorkload(const std::string &unit_name, const std::string &op,
                         std::map<std::string, Expr> dims,
                         std::vector<std::string> resources) {
  if (queues.find(unit_name) == queues.end())
    queues[unit_name] = HardwareQueue{unit_name, {}};
  queues[unit_name].workloads.push_back(
      Workload{op, std::move(dims), std::move(resources)});
}

void Stage::dump(llvm::raw_ostream &os, int indent) const {
  os.indent(indent) << "├── Stage " << stage_id << ":\n";
  for (auto const &[name, queue] : queues)
    queue.dump(os, indent + 4);
}

llvm::json::Value Stage::toJSON() const {
  std::vector<const Workload *> all_workloads;
  for (auto const &[name, queue] : queues)
    for (const auto &w : queue.workloads)
      all_workloads.push_back(&w);

  llvm::json::Array parallel_arr;
  for (const auto &group : groupWorkloadsByResources(all_workloads)) {
    llvm::json::Array schedules;
    for (const Workload *w : group.workloads)
      schedules.push_back(w->toJSON());
    parallel_arr.push_back(llvm::json::Object{
        {"Sequential", llvm::json::Object{{"schedules", std::move(schedules)},
                                          {"scenarios", llvm::json::Array{}}}}});
  }

  return llvm::json::Object{{"stage_id", stage_id},
                            {"Parallel", std::move(parallel_arr)}};
}

// ==========================================
// Scope
// ==========================================
Scope::Scope(std::string name) : scope_name(name) {}

Stage &Scope::getOrCreateStage(int id) {
  if (stages.find(id) == stages.end())
    stages.emplace(id, Stage(id));
  return stages.at(id);
}

void Scope::dump(llvm::raw_ostream &os, int indent) const {
  os.indent(indent) << "└── Scope [" << scope_name << "]:\n";
  for (auto &pair : stages)
    pair.second.dump(os, indent + 4);
}

llvm::json::Value Scope::toJSON() const {
  llvm::json::Array stages_json;
  for (auto &pair : stages)
    stages_json.push_back(pair.second.toJSON());
  return llvm::json::Object{{"scope_name", scope_name},
                            {"stages", std::move(stages_json)}};
}

// ==========================================
// ConstraintScope
// ==========================================
llvm::json::Value ConstraintScope::toJSON() const {
  llvm::json::Object symbols_json;
  for (const auto &[name, info] : symbols) {
    llvm::json::Object sym_obj;
    sym_obj["type"] = info.type;
    if (info.natural_ub >= 0)
      sym_obj["natural_ub"] = info.natural_ub;
    symbols_json[name] = std::move(sym_obj);
  }

  llvm::json::Array temp_iter_json;
  for (const auto &t : temp_iter)
    temp_iter_json.push_back(t.toJSON());

  llvm::json::Array footprint_json;
  for (const auto &term : l1_footprint)
    footprint_json.push_back(term.toJSON());

  llvm::json::Array booleans_json;
  for (const auto &name : booleans)
    booleans_json.push_back(name);

  llvm::json::Object metadata_json;
  metadata_json["symbols"] = std::move(symbols_json);
  metadata_json["L1_footprint"] = std::move(footprint_json);
  metadata_json["datatype"] = datatype;
  metadata_json["iter_num"] = llvm::json::Object{
      {"seq_iter", seq_iter.toJSON()},
      {"temp_iter", std::move(temp_iter_json)}};
  metadata_json["booleans"] = std::move(booleans_json);

  llvm::json::Array hard_constraints_json;
  for (const auto &c : hard_constraints)
    hard_constraints_json.push_back(c.toJSON());

  return llvm::json::Object{
      {"metadata", std::move(metadata_json)},
      {"hard_constraints", std::move(hard_constraints_json)}};
}

// ==========================================
// VariantETG — construction
// ==========================================
VariantETG::VariantETG(llvm::StringRef name, const HWOpRegistry *registry)
    : variant_name_(name.str()), compute_scope_("ComputeScope"),
      memory_scope_("MemoryScope"), hw_registry_(registry) {}

// ==========================================
// VariantETG — ETG building
// ==========================================
void VariantETG::buildFromAffineFor(mlir::affine::AffineForOp for_op) {
  llvm::DenseMap<mlir::Value, int> value_ready_stage;

  for (mlir::Value block_arg : for_op.getRegion().getArguments())
    value_ready_stage[block_arg] = 0;

  for_op.getBody()->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (op->getParentOp() != for_op)
      return;

    // Calculate ASAP stage based on operand dependencies.
    int required_stage = 0;
    for (mlir::Value operand : op->getOperands())
      if (value_ready_stage.count(operand))
        required_stage = std::max(required_stage, value_ready_stage[operand]);

    bool is_compute = llvm::isa<mlir::linalg::LinalgOp>(op);
    bool is_memory  = op->getName().getStringRef() == "loom.copy";
    bool is_infra   =
        is_compute && llvm::isa<mlir::linalg::FillOp, mlir::linalg::CopyOp>(op);

    if (is_compute && !is_infra)
      dispatchToComputeQueues(op, compute_scope_.getOrCreateStage(required_stage));
    if (is_memory)
      dispatchToMemoryQueues(op, memory_scope_.getOrCreateStage(required_stage));

    int ready_time = (is_infra || is_memory) ? required_stage : required_stage + 1;
    for (mlir::Value result : op->getResults())
      value_ready_stage[result] = ready_time;
  });
}

void VariantETG::buildFromSCFFor(mlir::scf::ForOp for_op) {
  llvm::DenseMap<mlir::Value, int> value_ready_stage;

  for (mlir::Value block_arg : for_op.getBody()->getArguments())
    value_ready_stage[block_arg] = 0;

  for_op.getBody()->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (op->getParentOp() != for_op)
      return;

    int required_stage = 0;
    for (mlir::Value operand : op->getOperands())
      if (value_ready_stage.count(operand))
        required_stage = std::max(required_stage, value_ready_stage[operand]);

    bool is_compute = llvm::isa<mlir::linalg::LinalgOp>(op);
    bool is_memory  = op->getName().getStringRef() == "loom.copy";
    bool is_infra   =
        is_compute && llvm::isa<mlir::linalg::FillOp, mlir::linalg::CopyOp>(op);

    if (is_compute && !is_infra)
      dispatchToComputeQueues(op, compute_scope_.getOrCreateStage(required_stage));
    if (is_memory)
      dispatchToMemoryQueues(op, memory_scope_.getOrCreateStage(required_stage));

    int ready_time = (is_infra || is_memory) ? required_stage : required_stage + 1;
    for (mlir::Value result : op->getResults())
      value_ready_stage[result] = ready_time;
  });
}

void VariantETG::dispatchToComputeQueues(mlir::Operation *op,
                                         Stage &target_stage) {
  assert(hw_registry_ && "HWOpRegistry must be provided");
  if (llvm::isa<mlir::linalg::GenericOp>(op))
    dispatchGenericOp(op, target_stage);
  else
    dispatchNamedOp(op, target_stage);
}

void VariantETG::dispatchNamedOp(mlir::Operation *op, Stage &target_stage) {
  std::string linalg_op_name = op->getName().getStringRef().str();
  const HWComputeFunc *hwFunc =
      hw_registry_->lookup(HWOpKey::named(linalg_op_name));
  if (!hwFunc) {
    auto ph = HWOpRegistry::makePlaceholder(linalg_op_name);
    target_stage.pushWorkload(ph.hw_component, ph.hw_func_name, {}, {});
    return;
  }

  auto linalgOp = llvm::cast<mlir::linalg::LinalgOp>(op);
  std::map<std::string, Expr> dimMap;
  auto inputs = linalgOp.getDpsInputs();
  for (size_t i = 0; i < inputs.size() && i < hwFunc->input_bindings.size(); ++i) {
    std::vector<Expr> opDims = traceAllocDimsFromTensor(inputs[i]);
    const auto &hwBinding = hwFunc->input_bindings[i];
    for (size_t d = 0; d < hwBinding.dim_symbols.size() && d < opDims.size(); ++d)
      if (dimMap.count(hwBinding.dim_symbols[d]) == 0)
        dimMap[hwBinding.dim_symbols[d]] = opDims[d];
  }

  target_stage.pushWorkload(hwFunc->hw_component, hwFunc->hw_func_name,
                            std::move(dimMap), hwFunc->resources);
}

void VariantETG::dispatchGenericOp(mlir::Operation *op, Stage &target_stage) {
  auto linalgOp = llvm::cast<mlir::linalg::LinalgOp>(op);
  GenericDimAnalysis analysis = analyzeGenericDims(linalgOp);

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
    const HWComputeFunc *hwFunc =
        hw_registry_->lookup(HWOpKey::generic(bodyOpName, analysis.generic_class));
    if (!hwFunc) {
      auto ph = HWOpRegistry::makePlaceholder(bodyOpName);
      target_stage.pushWorkload(ph.hw_component, ph.hw_func_name, {}, {});
      continue;
    }

    std::map<std::string, Expr> dimMap;
    if (!hwFunc->parallel_symbol.empty() && !analysis.parallel_product.isNone())
      dimMap[hwFunc->parallel_symbol] = analysis.parallel_product;
    if (!hwFunc->reduction_symbol.empty() && !analysis.reduction_product.isNone())
      dimMap[hwFunc->reduction_symbol] = analysis.reduction_product;

    target_stage.pushWorkload(hwFunc->hw_component, hwFunc->hw_func_name,
                              std::move(dimMap), hwFunc->resources);
  }
}

void VariantETG::dispatchToMemoryQueues(mlir::Operation *op,
                                        Stage &target_stage) {
  auto copyOp = llvm::dyn_cast<loom::CopyOp>(op);
  if (!copyOp)
    return;

  std::string srcMem, dstMem;
  if (auto attr = copyOp.getSrcMemSpaceAttr())
    srcMem = attr.getLeafReference().str();
  if (auto attr = copyOp.getDstMemSpaceAttr())
    dstMem = attr.getLeafReference().str();

  std::vector<int64_t> bcastVec;
  if (auto broadcastAttr = copyOp.getBroadcastAttr())
    for (auto val : broadcastAttr)
      bcastVec.push_back(mlir::cast<mlir::IntegerAttr>(val).getInt());

  const HWComputeFunc *hwFunc =
      hw_registry_->lookup(HWOpKey::dataMover(srcMem, dstMem, bcastVec));
  if (!hwFunc) {
    auto ph = HWOpRegistry::makePlaceholder(
        "loom.copy[" + srcMem + "->" + dstMem + "]", "data_movers");
    target_stage.pushWorkload(ph.hw_component, ph.hw_func_name, {}, {});
    return;
  }

  // Trace the L1 alloc — try source first (L1→DRAM), then destination (DRAM→L1).
  std::map<std::string, Expr> dimMap;
  loom::AllocOp allocOp = traceToAlloc(copyOp.getSource());
  if (!allocOp)
    allocOp = traceToAlloc(copyOp.getDestination());
  if (allocOp && !hwFunc->input_bindings.empty()) {
    std::vector<Expr> opDims = formatAllocDims(allocOp);
    const auto &binding = hwFunc->input_bindings[0];
    for (size_t d = 0; d < binding.dim_symbols.size() && d < opDims.size(); ++d)
      if (dimMap.count(binding.dim_symbols[d]) == 0)
        dimMap[binding.dim_symbols[d]] = opDims[d];
  }

  target_stage.pushWorkload(hwFunc->hw_component, hwFunc->hw_func_name,
                            std::move(dimMap), hwFunc->resources);
}

// ==========================================
// VariantETG — constraint scope building
// ==========================================
void VariantETG::collectSymbols(mlir::func::FuncOp func_op) {
  func_op.walk([&](loom::SymOp op) {
    std::string name = op.getSymbolRef().getLeafReference().str();
    SymbolInfo info;
    info.type = "int";
    if (auto ubAttr = op.getUpperBound())
      info.natural_ub = ubAttr->getSExtValue();
    constraint_scope_.symbols[name] = std::move(info);
  });
}

void VariantETG::analyzeLoopIterations(mlir::func::FuncOp func_op) {
  func_op.walk([&](mlir::affine::AffineForOp forOp) {
    auto iterAttr =
        forOp->getAttrOfType<loom::IterTypeAttr>("loom.iter_type");
    if (!iterAttr)
      return;
    Expr tripCount = extractLoopTripCount(forOp);
    if (tripCount.isNone())
      return;
    if (iterAttr.getValue() == loom::IterType::Sequential)
      constraint_scope_.seq_iter = tripCount;
    else if (iterAttr.getValue() == loom::IterType::Temporal)
      constraint_scope_.temp_iter.push_back(tripCount);
  });
  // Also handle scf.for loops (e.g., the sequential K loop from helion frontend)
  func_op.walk([&](mlir::scf::ForOp forOp) {
    auto iterAttr =
        forOp->getAttrOfType<loom::IterTypeAttr>("loom.iter_type");
    if (!iterAttr)
      return;
    Expr tripCount = extractLoopTripCount(forOp);
    if (tripCount.isNone())
      return;
    if (iterAttr.getValue() == loom::IterType::Sequential)
      constraint_scope_.seq_iter = tripCount;
    else if (iterAttr.getValue() == loom::IterType::Temporal)
      constraint_scope_.temp_iter.push_back(tripCount);
  });
}

void VariantETG::addIterDivisibilityConstraints(const Expr &iter) {
  if (iter.isNone())
    return;
  constraint_scope_.hard_constraints.push_back(
      ConstraintExpr::ge(iter, Expr::con(1)));
  auto [num, den] = findDivNode(iter);
  if (!num.isNone())
    constraint_scope_.hard_constraints.push_back(
        ConstraintExpr::divisible(num, den));
}

void VariantETG::collectL1Footprint(mlir::func::FuncOp func_op) {
  mlir::Type expectedElemType;
  func_op.walk([&](loom::AllocOp allocOp) {
    if (allocOp.getMemory().getLeafReference() != "L1")
      return;
    auto memrefType =
        mlir::cast<mlir::MemRefType>(allocOp.getResult().getType());
    mlir::Type elemType = memrefType.getElementType();
    if (!expectedElemType) {
      expectedElemType = elemType;
      constraint_scope_.datatype = formatElementType(elemType);
    } else {
      assert(elemType == expectedElemType &&
             "All L1 allocations must have the same element type");
    }
    Expr dims = productOfDims(formatAllocDims(allocOp));
    if (!dims.isNone())
      constraint_scope_.l1_footprint.push_back(dims);
  });
}

void VariantETG::buildConstraintScope(mlir::func::FuncOp func_op) {
  collectSymbols(func_op);
  constraint_scope_.booleans.push_back("is_double_buffer");
  analyzeLoopIterations(func_op);
  // addIterDivisibilityConstraints(constraint_scope_.seq_iter);
  // for (const Expr &t : constraint_scope_.temp_iter)
    // addIterDivisibilityConstraints(t);
  collectL1Footprint(func_op);
}

// ==========================================
// VariantETG — output
// ==========================================
void VariantETG::dump(llvm::raw_ostream &os) const {
  os << "Variant ETG: [" << variant_name_ << "]\n";
  compute_scope_.dump(os, 0);
  memory_scope_.dump(os, 0);
}

llvm::json::Value VariantETG::toJSON() const {
  return llvm::json::Object{{"variant_name", variant_name_},
                            {"compute_scope", compute_scope_.toJSON()},
                            {"memory_scope", memory_scope_.toJSON()},
                            {"constraint_scope", constraint_scope_.toJSON()}};
}

void VariantETG::buildL1FootprintConstraint() {
  // TODO: should be removed after mlar is ready.
  //       This is a workaround method to extract L1 size from the ADL
  //       platform header (adl.memory.array / adl.memory.bank / adl.spatial_dim).
  if (!hw_registry_ || constraint_scope_.l1_footprint.empty())
    return;

  mlir::ModuleOp platformModule = hw_registry_->getPlatformModule();
  if (!platformModule)
    return;

  int64_t l1_size = 0;
  platformModule.walk([&](adl::MemoryArrayOp arrayOp) {
    if (arrayOp->getParentOp() != platformModule.getOperation())
      return mlir::WalkResult::skip();
    if (arrayOp.getSymName() != "mem_L1")
      return mlir::WalkResult::advance();

    int64_t spatial_product = 1;
    for (mlir::Value spatialVal : arrayOp.getSpatialDims())
      if (auto dimOp = spatialVal.getDefiningOp<adl::SpatialDimOp>())
        spatial_product *= static_cast<int64_t>(dimOp.getSize());

    if (auto bankOp = arrayOp.getBank().getDefiningOp<adl::MemoryBankOp>()) {
      int64_t bsize = static_cast<int64_t>(bankOp.getBsize());
      int64_t nblk  = static_cast<int64_t>(bankOp.getNblk());
      l1_size = spatial_product * bsize * nblk;
    }
    return mlir::WalkResult::advance();
  });

  if (l1_size == 0)
    return;

  Expr footprint_sum = Expr::con(0);
  for (const Expr &term : constraint_scope_.l1_footprint)
    footprint_sum = footprint_sum + term;

  // IfElse(is_double_buffer == 1, elem_bytes*2, elem_bytes):
  //   double-buffered → 2× buffer space needed in L1.
  int64_t elem_bytes = 2;
  auto db_cond = std::make_shared<ConstraintExpr>(
      ConstraintExpr::eq(Expr::sym("is_double_buffer"), Expr::con(1)));
  Expr multiplier =
      Expr::ifelse(db_cond, Expr::con(elem_bytes * 2), Expr::con(elem_bytes));

  constraint_scope_.hard_constraints.push_back(
      ConstraintExpr::le(footprint_sum * multiplier, Expr::con(l1_size)));
}

} // namespace lcs
} // namespace loom
