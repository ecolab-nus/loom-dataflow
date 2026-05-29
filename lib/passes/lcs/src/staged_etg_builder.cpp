#include "staged_etg_builder.h"
#include "hard_constraint_pipeline.h"
#include "hw_alignment.h"
#include "hw_op_registry.h"
#include "l1_footprint_estimator.h"
#include "lcs_utils.h"
#include "ssa_utils.h"
#include "ADLDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "ADLTypes.h.inc"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#define GET_OP_CLASSES
#include "ADLOps.h.inc"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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
#include <optional>
#include <set>
#define GET_OP_CLASSES
#include "LoomEnums.h.inc"
#include "LoomInterfaces.h.inc"
#include "LoomOps.h.inc"
#define GET_ATTRDEF_CLASSES
#include "LoomAttributes.h.inc"

#ifndef ComputeKernelPipeline
#define ComputeKernelPipeline 0
#endif

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

std::optional<int64_t> staticIndexFromOfr(mlir::OpFoldResult value) {
  if (auto attr = value.dyn_cast<mlir::Attribute>())
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr))
      return intAttr.getInt();
  mlir::Value ssaValue = value.dyn_cast<mlir::Value>();
  if (!ssaValue)
    return std::nullopt;
  if (auto constOp = ssaValue.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
      return intAttr.getInt();
  return std::nullopt;
}

void addBindingDimsFromValue(mlir::Value value, const HWTensorBinding &binding,
                             std::map<std::string, Expr> &dimMap) {
  loom::AllocOp allocOp = loom::utils::traceToRootAllocOp(value);
  if (!allocOp)
    return;
  std::vector<Expr> opDims = formatAllocDims(allocOp);
  for (size_t d = 0; d < binding.dim_symbols.size() && d < opDims.size(); ++d)
    if (dimMap.count(binding.dim_symbols[d]) == 0)
      dimMap[binding.dim_symbols[d]] = opDims[d];
}

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

const char *iterTypeTagToString(IterTypeTag t) {
  switch (t) {
  case IterTypeTag::Sequential:
    return "sequential";
  case IterTypeTag::Temporal:
    return "temporal";
  }
  return "temporal";
}

IterTypeTag iterTypeFromAttr(mlir::Operation *op) {
  if (auto attr = op->getAttrOfType<loom::IterTypeAttr>("loom.iter_type")) {
    if (attr.getValue() == loom::IterType::Sequential)
      return IterTypeTag::Sequential;
  }
  return IterTypeTag::Temporal;
}

std::string blockSymFromAttr(mlir::Operation *op) {
  if (auto attr = op->getAttrOfType<mlir::SymbolRefAttr>("loom.block_sym"))
    return attr.getLeafReference().str();
  return std::string();
}

std::vector<std::string>
resourcesForComputePipelineMode(const std::vector<std::string> &resources) {
  std::vector<std::string> result = resources;
#if !ComputeKernelPipeline
  result.push_back("__compute_kernel_pipeline_disabled__");
#endif
  return result;
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
// WorkloadStageBody
// ==========================================
void WorkloadStageBody::pushWorkload(const std::string &unit_name,
                                     const std::string &op,
                                     std::map<std::string, Expr> dims,
                                     std::vector<std::string> resources) {
  if (queues_.find(unit_name) == queues_.end())
    queues_[unit_name] = HardwareQueue{unit_name, {}};
  queues_[unit_name].workloads.push_back(
      Workload{op, std::move(dims), std::move(resources)});
}

llvm::json::Object WorkloadStageBody::toJSONFragment() const {
  std::vector<const Workload *> all_workloads;
  for (auto const &[name, queue] : queues_)
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

  return llvm::json::Object{{"Parallel", std::move(parallel_arr)}};
}

void WorkloadStageBody::dump(llvm::raw_ostream &os, int indent) const {
  for (auto const &[name, queue] : queues_)
    queue.dump(os, indent);
}

// ==========================================
// Stage
// ==========================================
llvm::json::Value Stage::toJSON() const {
  llvm::json::Object obj =
      body ? body->toJSONFragment() : llvm::json::Object{};
  obj["stage_id"] = stage_id;
  return obj;
}

void Stage::dump(llvm::raw_ostream &os, int indent) const {
  os.indent(indent) << "├── Stage " << stage_id << ":\n";
  if (body)
    body->dump(os, indent + 4);
}

// ==========================================
// Scope
// ==========================================
Scope::Scope(std::string name) : scope_name(std::move(name)) {}

WorkloadStageBody &Scope::getOrCreateWorkloadStage(int id) {
  auto it = stages.find(id);
  while (it != stages.end() && it->second.body &&
         it->second.body->getKind() != StageBody::Kind::Workload) {
    ++id;
    it = stages.find(id);
  }
  if (it == stages.end()) {
    auto inserted = stages.emplace(
        std::piecewise_construct, std::forward_as_tuple(id),
        std::forward_as_tuple(id, std::make_unique<WorkloadStageBody>()));
    it = inserted.first;
  }
  StageBody *raw = it->second.body.get();
  assert(raw && raw->getKind() == StageBody::Kind::Workload &&
         "Stage at this id is not a workload stage; cannot mix "
         "workloads with a for_loop_block at the same stage_id");
  return *static_cast<WorkloadStageBody *>(raw);
}

int Scope::placeStage(int min_id, std::unique_ptr<StageBody> body) {
  int id = min_id;
  while (stages.count(id))
    ++id;
  stages.emplace(std::piecewise_construct, std::forward_as_tuple(id),
                 std::forward_as_tuple(id, std::move(body)));
  return id;
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
// FusedOpBlock
// ==========================================
llvm::json::Object FusedOpBlock::emitScopesJSON() const {
  return llvm::json::Object{{"load_scope", load_scope.toJSON()},
                            {"compute_scope", compute_scope.toJSON()},
                            {"store_scope", store_scope.toJSON()}};
}

void FusedOpBlock::dumpScopes(llvm::raw_ostream &os, int indent) const {
  load_scope.dump(os, indent);
  compute_scope.dump(os, indent);
  store_scope.dump(os, indent);
}

// ==========================================
// KernelBlock
// ==========================================
llvm::json::Value KernelBlock::toJSON() const { return body.emitScopesJSON(); }

void KernelBlock::dump(llvm::raw_ostream &os, int indent) const {
  os.indent(indent) << "kernel_block:\n";
  body.dumpScopes(os, indent + 2);
}

// ==========================================
// ForLoopBlockStageBody
// ==========================================
ForLoopBlockStageBody::ForLoopBlockStageBody(std::string block_sym,
                                             IterTypeTag iter_type,
                                             Expr trip_count)
    : block_sym_(std::move(block_sym)), iter_type_(iter_type),
      trip_count_(std::move(trip_count)) {}

llvm::json::Object ForLoopBlockStageBody::toJSONFragment() const {
  llvm::json::Object inner = body.emitScopesJSON();
  inner["block_sym"] = block_sym_;
  inner["iter_type"] = iterTypeTagToString(iter_type_);
  inner["trip_count"] = trip_count_.toJSON();
  return llvm::json::Object{{"for_loop_block", std::move(inner)}};
}

void ForLoopBlockStageBody::dump(llvm::raw_ostream &os, int indent) const {
  os.indent(indent) << "for_loop_block [" << block_sym_ << ", "
                    << iterTypeTagToString(iter_type_) << "]:\n";
  body.dumpScopes(os, indent + 2);
}

// ==========================================
// ConstraintScope
// ==========================================
llvm::json::Value ConstraintScope::toJSON() const {
  llvm::json::Object symbols_json;
  for (const auto &[name, info] : symbols) {
    llvm::json::Object sym_obj;
    sym_obj["type"] = info.type;
    sym_obj["alignment"] = info.alignment;
    if (info.natural_ub >= 0)
      sym_obj["natural_ub"] = info.natural_ub;
    symbols_json[name] = std::move(sym_obj);
  }

  llvm::json::Array temp_iter_json;
  for (const auto &t : temp_iter)
    temp_iter_json.push_back(t.toJSON());

  auto footprintArray = [](const std::vector<Expr> &terms) {
    llvm::json::Array arr;
    for (const auto &term : terms)
      arr.push_back(term.toJSON());
    return arr;
  };
  llvm::json::Object footprint_json;
  footprint_json["load"] = footprintArray(l1_footprint.load);
  footprint_json["compute"] = footprintArray(l1_footprint.compute);
  footprint_json["store"] = footprintArray(l1_footprint.store);
  footprint_json["capacity"] = l1_footprint.capacity;

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

void ConstraintScope::pushHardConstraint(ConstraintExpr constraint) {
  hard_constraints.push_back(std::move(constraint));
}

// ==========================================
// VariantETG — construction
// ==========================================
VariantETG::VariantETG(llvm::StringRef name, const HWOpRegistry *registry)
    : variant_name_(name.str()), hw_registry_(registry) {}

// ==========================================
// VariantETG — ETG building
// ==========================================
void VariantETG::buildFromFunc(mlir::func::FuncOp func_op) {
  if (func_op.isExternal() || func_op.empty())
    return;
  populateScopesFromRegion(func_op.getRegion(),
                           kernel_block_.body.load_scope,
                           kernel_block_.body.compute_scope,
                           kernel_block_.body.store_scope);
}

void VariantETG::populateScopesFromRegion(mlir::Region &region,
                                          Scope &load_scope,
                                          Scope &compute_scope,
                                          Scope &store_scope) {
  if (region.empty())
    return;

  // Per-region readiness map; seeded with this region's block arguments at
  // stage 0 (matching the original walk semantics).
  llvm::DenseMap<mlir::Value, int> value_ready_stage;
  for (mlir::Block &block : region)
    for (mlir::Value arg : block.getArguments())
      value_ready_stage[arg] = 0;

  // Recursively walk a block. For nested scf.for, build a child for_loop_block
  // and recurse with fresh scopes. For affine.parallel, walk through
  // transparently into the same target scopes (spatial loops are not modelled
  // — confirmed in the design plan).
  std::function<void(mlir::Block &)> walkBlock = [&](mlir::Block &block) {
    for (mlir::Operation &op_ref : block) {
      mlir::Operation *op = &op_ref;

      int required_stage = 0;
      for (mlir::Value operand : op->getOperands()) {
        auto it = value_ready_stage.find(operand);
        if (it != value_ready_stage.end())
          required_stage = std::max(required_stage, it->second);
      }

      bool advances_stage = true; // default: generic ops advance stage by 1
      bool dispatched = false;

      if (auto for_op = llvm::dyn_cast<mlir::scf::ForOp>(op)) {
        auto child = std::make_unique<ForLoopBlockStageBody>(
            blockSymFromAttr(op), iterTypeFromAttr(op),
            extractLoopTripCount(for_op));
        // Recurse into the loop body with fresh scopes (a fresh region walk
        // re-seeds its own value_ready_stage from the for_op block args).
        populateScopesFromRegion(for_op.getRegion(), child->loadScope(),
                                 child->computeScope(), child->storeScope());
        compute_scope.placeStage(required_stage, std::move(child));
        dispatched = true;
        // for_op result tensors are visible to siblings; treat them as
        // becoming ready one stage after the loop.
        // (advances_stage = true)
      } else if (llvm::isa<mlir::scf::IfOp>(op)) {
        for (mlir::Region &inner_region : op->getRegions())
          for (mlir::Block &inner_block : inner_region) {
            for (mlir::Value barg : inner_block.getArguments())
              value_ready_stage[barg] = required_stage;
            walkBlock(inner_block);
          }
        dispatched = true;
        advances_stage = false;
      } else if (llvm::isa<mlir::affine::AffineParallelOp>(op)) {
        // Transparent: descend with the same target scopes and the same
        // readiness map. Block args of the parallel op inherit the current
        // required_stage so that ops inside can reference them correctly.
        for (mlir::Region &inner_region : op->getRegions())
          for (mlir::Block &inner_block : inner_region) {
            for (mlir::Value barg : inner_block.getArguments())
              value_ready_stage[barg] = required_stage;
            walkBlock(inner_block);
          }
        dispatched = true;
        advances_stage = false; // parallel op itself produces no modelled work
      } else {
        bool is_compute = llvm::isa<mlir::linalg::LinalgOp>(op);
        bool is_data_mover = op->getName().getStringRef() == "loom.copy";
        bool is_linalg_infra =
            is_compute &&
            llvm::isa<mlir::linalg::FillOp, mlir::linalg::CopyOp>(op);

        if (is_compute && !is_linalg_infra) {
          dispatchToComputeQueues(
              op, compute_scope.getOrCreateWorkloadStage(required_stage));
          dispatched = true;
#if !ComputeKernelPipeline
          advances_stage = false;
#endif
        } else if (is_data_mover) {
          loom::utils::CopyMemoryDirection direction =
              loom::utils::classifyCopyMemoryDirection(op);
          if (direction == loom::utils::CopyMemoryDirection::Load) {
            dispatchToDataMoverQueues(
                op, load_scope.getOrCreateWorkloadStage(required_stage));
          } else if (direction == loom::utils::CopyMemoryDirection::Store) {
            dispatchToDataMoverQueues(
                op, store_scope.getOrCreateWorkloadStage(required_stage));
          }
          dispatched = true;
          advances_stage = false; // data-mover ops are non-blocking
        } else if (is_linalg_infra) {
          advances_stage = false; // matches legacy: linalg.fill/copy don't bump
        }
        (void)dispatched;
      }

      int ready_time = advances_stage ? required_stage + 1 : required_stage;
      for (mlir::Value result : op->getResults())
        value_ready_stage[result] = ready_time;
    }
  };

  for (mlir::Block &block : region)
    walkBlock(block);
}

void VariantETG::dispatchToComputeQueues(mlir::Operation *op,
                                         WorkloadStageBody &target) {
  assert(hw_registry_ && "HWOpRegistry must be provided");
  if (llvm::isa<mlir::linalg::GenericOp>(op))
    dispatchGenericOp(op, target);
  else
    dispatchNamedOp(op, target);
}

void VariantETG::dispatchNamedOp(mlir::Operation *op,
                                 WorkloadStageBody &target) {
  std::string linalg_op_name = op->getName().getStringRef().str();
  const HWComputeFunc *hwFunc =
      hw_registry_->lookup(HWOpKey::named(linalg_op_name));
  if (!hwFunc) {
    auto ph = HWOpRegistry::makePlaceholder(linalg_op_name);
    target.pushWorkload(ph.hw_component, ph.hw_func_name, {},
                        resourcesForComputePipelineMode(ph.resources));
    return;
  }

  auto linalgOp = llvm::cast<mlir::linalg::LinalgOp>(op);
  std::map<std::string, Expr> dimMap;
  auto inputs = linalgOp.getDpsInputs();
  for (size_t i = 0; i < inputs.size() && i < hwFunc->input_bindings.size();
       ++i) {
    std::vector<Expr> opDims = traceAllocDimsFromTensor(inputs[i]);
    const auto &hwBinding = hwFunc->input_bindings[i];
    for (size_t d = 0;
         d < hwBinding.dim_symbols.size() && d < opDims.size(); ++d)
      if (dimMap.count(hwBinding.dim_symbols[d]) == 0)
        dimMap[hwBinding.dim_symbols[d]] = opDims[d];
  }

  target.pushWorkload(hwFunc->hw_component, hwFunc->hw_func_name,
                      std::move(dimMap),
                      resourcesForComputePipelineMode(hwFunc->resources));
}

void VariantETG::dispatchGenericOp(mlir::Operation *op,
                                   WorkloadStageBody &target) {
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
      target.pushWorkload(ph.hw_component, ph.hw_func_name, {},
                          resourcesForComputePipelineMode(ph.resources));
      continue;
    }

    std::map<std::string, Expr> dimMap;
    if (!hwFunc->parallel_symbol.empty() && !analysis.parallel_product.isNone())
      dimMap[hwFunc->parallel_symbol] = analysis.parallel_product;
    if (!hwFunc->reduction_symbol.empty() &&
        !analysis.reduction_product.isNone())
      dimMap[hwFunc->reduction_symbol] = analysis.reduction_product;

    target.pushWorkload(hwFunc->hw_component, hwFunc->hw_func_name,
                        std::move(dimMap),
                        resourcesForComputePipelineMode(hwFunc->resources));
  }
}

void VariantETG::dispatchToDataMoverQueues(mlir::Operation *op,
                                           WorkloadStageBody &target) {
  auto copyOp = llvm::dyn_cast<loom::CopyOp>(op);
  if (!copyOp)
    return;

  std::string srcMem, dstMem;
  mlir::Value source;
  mlir::Value destination;
  mlir::SmallVector<mlir::OpFoldResult, 4> mixedArea;
  DataMoverKind kind = DataMoverKind::Copy;
  std::string opName = "loom.copy";

  source = copyOp.getSource();
  destination = copyOp.getDestination();
  mixedArea = copyOp.getMixedArea();
  if (auto attr = copyOp.getSrcMemSpaceAttr())
    srcMem = attr.getLeafReference().str();
  if (auto attr = copyOp.getDstMemSpaceAttr())
    dstMem = attr.getLeafReference().str();

  std::vector<int64_t> bcastVec;
  for (mlir::OpFoldResult area : mixedArea) {
    if (std::optional<int64_t> value = staticIndexFromOfr(area)) {
      bcastVec.push_back(*value);
    } else {
      bcastVec.push_back(mlir::ShapedType::kDynamic);
    }
  }

  const HWComputeFunc *hwFunc =
      hw_registry_->lookupDataMover(kind, srcMem, dstMem, bcastVec);
  if (!hwFunc) {
    auto ph = HWOpRegistry::makePlaceholder(
        opName + "[" + srcMem + "->" + dstMem + "]", "data_movers");
    target.pushWorkload(ph.hw_component, ph.hw_func_name, {}, {});
    return;
  }

  std::map<std::string, Expr> dimMap;

  for (size_t i = 0; i < bcastVec.size() && i < hwFunc->area_symbols.size();
       ++i) {
    const std::string &symbol = hwFunc->area_symbols[i];
    if (!symbol.empty() && symbol != "?" &&
        !mlir::ShapedType::isDynamic(bcastVec[i]))
      dimMap[symbol] = Expr::con(bcastVec[i]);
  }

  if (!hwFunc->input_bindings.empty())
    addBindingDimsFromValue(source, hwFunc->input_bindings[0], dimMap);
  if (!hwFunc->output_bindings.empty())
    addBindingDimsFromValue(destination, hwFunc->output_bindings[0], dimMap);

  target.pushWorkload(hwFunc->hw_component, hwFunc->hw_func_name,
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
  constraint_scope_.pushHardConstraint(ConstraintExpr::ge(iter, Expr::con(1)));
  auto [num, den] = findDivNode(iter);
  if (!num.isNone())
    constraint_scope_.pushHardConstraint(ConstraintExpr::divisible(num, den));
}

void VariantETG::buildConstraintScope(mlir::func::FuncOp func_op) {
  collectSymbols(func_op);
  applyHardwareAlignments(func_op, constraint_scope_.symbols);
  constraint_scope_.booleans.push_back("is_double_buffer");
  analyzeLoopIterations(func_op);
  // addIterDivisibilityConstraints(constraint_scope_.seq_iter);
  // for (const Expr &t : constraint_scope_.temp_iter)
    // addIterDivisibilityConstraints(t);
  L1FootprintResult l1Result =
      L1FootprintEstimator::estimateFromFunc(func_op, hw_registry_);
  constraint_scope_.datatype = std::move(l1Result.datatype);
  constraint_scope_.l1_footprint = std::move(l1Result.l1_footprint);
}

// ==========================================
// VariantETG — output
// ==========================================
void VariantETG::dump(llvm::raw_ostream &os) const {
  os << "Variant ETG: [" << variant_name_ << "]\n";
  kernel_block_.dump(os, 0);
}

llvm::json::Value VariantETG::toJSON() const {
  return llvm::json::Object{{"variant_name", variant_name_},
                            {"constraint_scope", constraint_scope_.toJSON()},
                            {"kernel_block", kernel_block_.toJSON()}};
}

void VariantETG::buildHardConstraints(mlir::func::FuncOp func_op) {
  HardConstraintPipeline::pushAll(func_op, hw_registry_, constraint_scope_);
}

} // namespace lcs
} // namespace loom
