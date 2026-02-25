#include "cost_model_types.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

namespace loom {
namespace cost_model {
namespace {

std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::optional<double> getNumericAttr(mlir::Operation *op, llvm::StringRef name) {
  if (!op) {
    return std::nullopt;
  }
  if (auto intAttr = op->getAttrOfType<mlir::IntegerAttr>(name)) {
    return static_cast<double>(intAttr.getInt());
  }
  if (auto floatAttr = op->getAttrOfType<mlir::FloatAttr>(name)) {
    return floatAttr.getValueAsDouble();
  }
  return std::nullopt;
}

mlir::MemRefType tryGetStaticMemRefType(mlir::Value value) {
  auto checkType = [](mlir::Type type) -> mlir::MemRefType {
    if (auto memref = llvm::dyn_cast<mlir::MemRefType>(type)) {
      if (memref.hasStaticShape()) {
        return memref;
      }
    }
    return {};
  };

  if (auto direct = checkType(value.getType())) {
    return direct;
  }

  if (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
    if (auto referred = tryGetStaticMemRefType(cast.getSource())) {
      return referred;
    }
  }

  if (auto reinterpret = value.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
    if (auto concrete = checkType(reinterpret.getResult().getType())) {
      return concrete;
    }
  }

  return {};
}

mlir::Value stripCastsAndViews(mlir::Value value) {
  while (mlir::Operation *def = value.getDefiningOp()) {
    if (auto cast = llvm::dyn_cast<mlir::memref::CastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    if (auto reinterpret = llvm::dyn_cast<mlir::memref::ReinterpretCastOp>(def)) {
      value = reinterpret.getSource();
      continue;
    }
    if (auto viewLike = llvm::dyn_cast<mlir::ViewLikeOpInterface>(def)) {
      value = viewLike.getViewSource();
      continue;
    }
    break;
  }
  return value;
}

struct TensorLayout {
  std::vector<std::int64_t> dims;
  std::vector<std::int64_t> strides;
};

class LayoutAnalyzer {
public:
  std::optional<TensorLayout> fromMemRef(mlir::MemRefType type) const {
    if (!type || !type.hasStaticShape()) {
      return std::nullopt;
    }

    TensorLayout layout;
    layout.dims.reserve(type.getRank());
    for (int64_t dim : type.getShape()) {
      layout.dims.push_back(static_cast<std::int64_t>(dim));
    }

    llvm::SmallVector<int64_t, 4> strideTmp;
    int64_t offset = 0;
    if (mlir::failed(type.getStridesAndOffset(strideTmp, offset))) {
      return std::nullopt;
    }

    layout.strides.reserve(strideTmp.size());
    for (int64_t stride : strideTmp) {
      layout.strides.push_back(static_cast<std::int64_t>(stride));
    }
    return layout;
  }

  std::optional<TensorLayout> fromTensor(mlir::RankedTensorType type) const {
    if (!type || !type.hasStaticShape()) {
      return std::nullopt;
    }

    TensorLayout layout;
    layout.dims.reserve(type.getRank());
    for (int64_t dim : type.getShape()) {
      layout.dims.push_back(static_cast<std::int64_t>(dim));
    }

    const unsigned rank = type.getRank();
    if (rank == 0) {
      return layout;
    }

    layout.strides.resize(rank);
    layout.strides.back() = 1;
    for (int i = rank - 2; i >= 0; --i) {
      const std::int64_t size = type.getDimSize(static_cast<unsigned>(i + 1));
      layout.strides[static_cast<std::size_t>(i)] =
          size <= 0 ? 1 : layout.strides[static_cast<std::size_t>(i + 1)] * size;
    }

    return layout;
  }

  std::optional<TensorLayout> fromValue(mlir::Value value) const {
    if (!value) {
      return std::nullopt;
    }
    if (auto memrefType = tryGetStaticMemRefType(value)) {
      return fromMemRef(memrefType);
    }
    if (auto ranked = llvm::dyn_cast<mlir::RankedTensorType>(value.getType())) {
      return fromTensor(ranked);
    }
    return std::nullopt;
  }

  std::optional<TensorLayout> fromType(mlir::Type type) const {
    if (auto memref = llvm::dyn_cast<mlir::MemRefType>(type)) {
      return fromMemRef(memref);
    }
    if (auto tensor = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
      return fromTensor(tensor);
    }
    return std::nullopt;
  }
};

bool tryGetConstantIndex(mlir::Value value, std::int64_t &result) {
  if (!value) {
    return false;
  }
  if (auto constIndex = value.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
    if (auto integerAttr = llvm::dyn_cast<mlir::IntegerAttr>(constIndex.getValue())) {
      result = integerAttr.getInt();
      return true;
    }
  }
  if (auto constOp = value.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto integerAttr = llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
      result = integerAttr.getInt();
      return true;
    }
  }
  return false;
}

std::optional<std::int64_t> tryEvaluateAffineBound(mlir::AffineMap map,
                                                   mlir::ValueRange operands) {
  if (!map || map.getNumResults() != 1) {
    return std::nullopt;
  }

  llvm::SmallVector<mlir::Value, 4> operandVec(operands.begin(), operands.end());
  map = mlir::simplifyAffineMap(map);

  const unsigned numDims = map.getNumDims();
  const unsigned numSymbols = map.getNumSymbols();
  if (operandVec.size() != numDims + numSymbols) {
    return std::nullopt;
  }

  auto makeConstantExpr = [](mlir::Value value,
                             mlir::MLIRContext *context)
      -> std::optional<mlir::AffineExpr> {
    std::int64_t constant = 0;
    if (!tryGetConstantIndex(value, constant)) {
      return std::nullopt;
    }
    return mlir::getAffineConstantExpr(constant, context);
  };

  mlir::MLIRContext *context = map.getContext();
  llvm::SmallVector<mlir::AffineExpr, 4> dimReplacements;
  llvm::SmallVector<mlir::AffineExpr, 4> symbolReplacements;
  dimReplacements.reserve(numDims);
  symbolReplacements.reserve(numSymbols);

  auto operandIt = operandVec.begin();
  for (unsigned i = 0; i < numDims; ++i, ++operandIt) {
    auto expr = makeConstantExpr(*operandIt, context);
    if (!expr) {
      return std::nullopt;
    }
    dimReplacements.push_back(*expr);
  }
  for (unsigned i = 0; i < numSymbols; ++i, ++operandIt) {
    auto expr = makeConstantExpr(*operandIt, context);
    if (!expr) {
      return std::nullopt;
    }
    symbolReplacements.push_back(*expr);
  }

  mlir::AffineExpr resultExpr =
      map.getResult(0).replaceDimsAndSymbols(dimReplacements, symbolReplacements);
  if (auto constExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(resultExpr)) {
    return constExpr.getValue();
  }
  return std::nullopt;
}

double computeMatmulCalculations(mlir::linalg::MatmulOp matmul) {
  auto inputs = matmul.getInputs();
  if (inputs.size() < 2) {
    return 0.0;
  }

  LayoutAnalyzer analyzer;
  auto lhs = analyzer.fromValue(inputs[0]);
  auto rhs = analyzer.fromValue(inputs[1]);
  if (!lhs || !rhs || lhs->dims.size() < 2 || rhs->dims.size() < 2) {
    return 0.0;
  }

  const std::int64_t m = lhs->dims[0];
  const std::int64_t k = lhs->dims[1];
  const std::int64_t n = rhs->dims[1];
  if (m <= 0 || n <= 0 || k <= 0) {
    return 0.0;
  }

  return static_cast<double>(m) * static_cast<double>(k) *
         static_cast<double>(n) * 2.0;
}

double computeTotalCalculations(mlir::linalg::LinalgOp op) {
  if (auto matmul = llvm::dyn_cast<mlir::linalg::MatmulOp>(op.getOperation())) {
    return computeMatmulCalculations(matmul);
  }
  return 0.0;
}

std::int64_t computeBytesFromMemRef(mlir::MemRefType type) {
  if (!type || !type.hasStaticShape()) {
    return 0;
  }

  std::int64_t elements = 1;
  for (int64_t dim : type.getShape()) {
    if (dim <= 0) {
      return 0;
    }
    if (elements > std::numeric_limits<std::int64_t>::max() / dim) {
      return 0;
    }
    elements *= dim;
  }

  const unsigned elementBits = type.getElementTypeBitWidth();
  if (elementBits == 0) {
    return 0;
  }

  const std::int64_t bytesPerElt = static_cast<std::int64_t>((elementBits + 7) / 8);
  if (bytesPerElt <= 0) {
    return 0;
  }
  if (elements > std::numeric_limits<std::int64_t>::max() / bytesPerElt) {
    return 0;
  }

  return elements * bytesPerElt;
}

std::string describeGenericOp(mlir::linalg::GenericOp generic) {
  llvm::SmallVector<std::string, 4> arithOps;
  generic->walk([&](mlir::Operation *op) -> mlir::WalkResult {
    if (!op || op == generic.getOperation()) {
      return mlir::WalkResult::advance();
    }

    if (llvm::isa<mlir::linalg::LinalgOp>(op)) {
      return mlir::WalkResult::skip();
    }

    if (op->getDialect() &&
        op->getDialect()->getNamespace() ==
            mlir::arith::ArithDialect::getDialectNamespace()) {
      std::string name = op->getName().getStringRef().str();
      if (!llvm::is_contained(arithOps, name)) {
        arithOps.push_back(std::move(name));
      }
    }
    return mlir::WalkResult::advance();
  });

  if (arithOps.empty()) {
    return generic->getName().getStringRef().str();
  }

  llvm::sort(arithOps);
  std::string name = arithOps.front();
  for (unsigned i = 1; i < arithOps.size(); ++i) {
    name += ",";
    name += arithOps[i];
  }
  return name;
}

TensorLayout computeLayoutForOp(mlir::linalg::LinalgOp op,
                                const LayoutAnalyzer &analyzer) {
  for (mlir::Value init : op.getDpsInits()) {
    if (auto layout = analyzer.fromValue(init)) {
      return *layout;
    }
  }

  for (mlir::OpResult result : op->getResults()) {
    if (auto layout = analyzer.fromType(result.getType())) {
      return *layout;
    }
  }

  return {};
}

llvm::StringMap<std::int64_t> collectSpatialDims(mlir::ModuleOp module) {
  llvm::StringMap<std::int64_t> dims;
  module.walk([&](mlir::Operation *operation) {
    if (!operation) {
      return;
    }

    llvm::StringRef name = operation->getName().getStringRef();
    if (!name.ends_with("spatial_dim")) {
      return;
    }

    auto dimNameAttr = operation->getAttrOfType<mlir::StringAttr>("name");
    if (!dimNameAttr) {
      dimNameAttr = operation->getAttrOfType<mlir::StringAttr>("label");
    }
    auto sizeAttr = operation->getAttrOfType<mlir::IntegerAttr>("size");
    if (!dimNameAttr || !sizeAttr) {
      return;
    }

    dims.try_emplace(dimNameAttr.getValue(), sizeAttr.getInt());
  });
  return dims;
}

bool isDMAConnectedCoordinate(std::int64_t x, std::int64_t y) {
  return x == 0 || y == 0;
}

CoreIndex selectNearestDMACore(std::int64_t x, std::int64_t y,
                               std::int64_t extentX, std::int64_t extentY) {
  if (isDMAConnectedCoordinate(x, y)) {
    return CoreIndex{x, y};
  }

  CoreIndex rowCandidate{x, 0};
  CoreIndex colCandidate{0, y};

  auto inBounds = [&](const CoreIndex &core) -> bool {
    return core.x >= 0 && core.y >= 0 && core.x < extentX && core.y < extentY;
  };

  const bool rowValid = inBounds(rowCandidate);
  const bool colValid = inBounds(colCandidate);

  if (rowValid && colValid) {
    return (y <= x) ? rowCandidate : colCandidate;
  }
  if (rowValid) {
    return rowCandidate;
  }
  if (colValid) {
    return colCandidate;
  }
  return CoreIndex{0, 0};
}

std::size_t linearIndex(std::int64_t x, std::int64_t y, std::int64_t extentY) {
  return static_cast<std::size_t>(x * extentY + y);
}

std::string getSymbolRefName(mlir::Attribute attr) {
  if (auto flat = llvm::dyn_cast_or_null<mlir::FlatSymbolRefAttr>(attr)) {
    return flat.getValue().str();
  }
  if (auto sym = llvm::dyn_cast_or_null<mlir::SymbolRefAttr>(attr)) {
    return sym.getLeafReference().str();
  }
  return {};
}

struct CopyPlan {
  std::string kind = "mem";
  std::string dim;
  std::string network;
  bool vertical = false;
  bool horizontal = false;
  bool allLinks = false;
  std::string broadcastNetwork = "broadcast";
  bool isStore = false;
};

void applyNetworkFlags(CopyPlan &plan, llvm::StringRef network) {
  std::string lowered = toLower(network.str());
  if (lowered.find("vertical") != std::string::npos ||
      lowered.find("ver") != std::string::npos) {
    plan.vertical = true;
  }
  if (lowered.find("horizontal") != std::string::npos ||
      lowered.find("hor") != std::string::npos) {
    plan.horizontal = true;
  }
  if (lowered.find("all_links") != std::string::npos) {
    plan.allLinks = true;
  }
}

class WorkloadBuilder {
public:
  WorkloadBuilder(mlir::func::FuncOp entryFunc, std::int64_t extentX,
                  std::int64_t extentY)
      : entryFunc_(entryFunc), extentX_(std::max<std::int64_t>(extentX, 1)),
        extentY_(std::max<std::int64_t>(extentY, 1)) {
    perCore_.reserve(static_cast<std::size_t>(extentX_ * extentY_));
    for (std::int64_t x = 0; x < extentX_; ++x) {
      for (std::int64_t y = 0; y < extentY_; ++y) {
        CoreSequentialWorkload workload;
        workload.core = CoreIndex{x, y};
        perCore_.push_back(std::move(workload));
      }
    }
  }

  FunctionWorkload build() {
    FunctionWorkload result;
    result.functionName = entryFunc_.getSymName().str();
    result.funcOp = entryFunc_;

    processRegion(entryFunc_.getFunctionBody(), /*emit=*/false);

    result.workloads = std::move(perCore_);
    result.hasComputeWorkload = hasCompute_;
    return result;
  }

private:
  CoreSequentialWorkload &coreWork(std::int64_t x, std::int64_t y) {
    return perCore_[linearIndex(x, y, extentY_)];
  }

  void appendLoopBegin(const LoopBegin &loop) {
    for (auto &core : perCore_) {
      core.events.emplace_back(loop);
    }
  }

  void appendLoopEnd(const LoopEnd &loop) {
    for (auto &core : perCore_) {
      core.events.emplace_back(loop);
    }
  }

  void appendCompute(const std::string &opName,
                     const std::vector<std::int64_t> &layout,
                     const std::vector<std::int64_t> &strides,
                     std::int64_t totalCalculations) {
    hasCompute_ = true;
    for (auto &core : perCore_) {
      ComputeWorkload workload;
      workload.core = core.core;
      workload.opName = opName;
      workload.outputLayout = layout;
      workload.outputStrides = strides;
      workload.totalCalculations = totalCalculations;
      core.events.emplace_back(std::move(workload));
    }
  }

  void processRegion(mlir::Region &region, bool emit) {
    for (mlir::Block &block : region) {
      processBlock(block, emit);
    }
  }

  void processBlock(mlir::Block &block, bool emit) {
    for (mlir::Operation &operation : block) {
      if (llvm::isa<mlir::func::ReturnOp>(operation)) {
        continue;
      }
      processOperation(operation, emit);
    }
  }

  void processOperation(mlir::Operation &operation, bool emit) {
    if (auto affineFor = llvm::dyn_cast<mlir::affine::AffineForOp>(operation)) {
      handleAffineFor(affineFor, emit);
      return;
    }
    if (auto affineParallel =
            llvm::dyn_cast<mlir::affine::AffineParallelOp>(operation)) {
      handleAffineParallel(affineParallel);
      return;
    }
    if (auto scfParallel = llvm::dyn_cast<mlir::scf::ParallelOp>(operation)) {
      handleScfParallel(scfParallel);
      return;
    }
    if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(operation)) {
      handleScfFor(forOp, emit);
      return;
    }

    if (!emit) {
      if (operation.getNumRegions() > 0) {
        for (mlir::Region &region : operation.getRegions()) {
          processRegion(region, false);
        }
      }
      return;
    }

    if (auto copyOp = llvm::dyn_cast<mlir::memref::CopyOp>(operation)) {
      handleMemrefCopy(copyOp);
      return;
    }

    if (auto copyOp = llvm::dyn_cast<::loom::CopyOp>(operation)) {
      handleLoomCopy(copyOp);
      return;
    }

    if (auto materialize =
            llvm::dyn_cast<mlir::bufferization::MaterializeInDestinationOp>(
                operation)) {
      handleMaterialize(materialize);
      return;
    }

    if (auto linalgOp = llvm::dyn_cast<mlir::linalg::LinalgOp>(operation)) {
      handleLinalg(linalgOp);
      return;
    }

    if (operation.getNumRegions() > 0) {
      for (mlir::Region &region : operation.getRegions()) {
        processRegion(region, emit);
      }
    }
  }

  void handleAffineFor(mlir::affine::AffineForOp affineFor, bool emit) {
    LoopBegin begin;
    begin.loopType = "affine.for";

    LoopRange range;
    range.inductionVar = "iv" + std::to_string(loopId_++);
    if (affineFor.hasConstantLowerBound()) {
      range.lowerBound = affineFor.getConstantLowerBound();
      range.hasLowerBound = true;
    } else if (auto evaluated = tryEvaluateAffineBound(affineFor.getLowerBoundMap(),
                                                        affineFor.getLowerBoundOperands())) {
      range.lowerBound = *evaluated;
      range.hasLowerBound = true;
    }
    if (affineFor.hasConstantUpperBound()) {
      range.upperBound = affineFor.getConstantUpperBound();
      range.hasUpperBound = true;
    } else if (auto evaluated = tryEvaluateAffineBound(affineFor.getUpperBoundMap(),
                                                        affineFor.getUpperBoundOperands())) {
      range.upperBound = *evaluated;
      range.hasUpperBound = true;
    }
    range.step = affineFor.getStep().getSExtValue();
    range.hasStep = true;
    begin.ranges.push_back(range);

    appendLoopBegin(begin);
    for (mlir::Region &region : affineFor->getRegions()) {
      processRegion(region, emit);
    }
    appendLoopEnd(LoopEnd{begin.loopType});
  }

  void handleAffineParallel(mlir::affine::AffineParallelOp parallelOp) {
    if (mlir::Block *body = parallelOp.getBody()) {
      for (mlir::Operation &inner : *body) {
        if (llvm::isa<mlir::affine::AffineYieldOp>(inner)) {
          continue;
        }
        processOperation(inner, /*emit=*/true);
      }
    }
  }

  void handleScfParallel(mlir::scf::ParallelOp parallelOp) {
    if (!parallelOp.getBody()) {
      return;
    }

    for (mlir::Operation &inner : *parallelOp.getBody()) {
      if (llvm::isa<mlir::scf::ReduceOp>(inner)) {
        continue;
      }
      processOperation(inner, /*emit=*/true);
    }
  }

  void handleScfFor(mlir::scf::ForOp forOp, bool emit) {
    if (!emit) {
      for (mlir::Region &region : forOp->getRegions()) {
        processRegion(region, false);
      }
      return;
    }

    LoopBegin begin;
    begin.loopType = "scf.for";

    LoopRange range;
    range.inductionVar = "iv" + std::to_string(loopId_++);
    std::int64_t constant = 0;
    if (tryGetConstantIndex(forOp.getLowerBound(), constant)) {
      range.lowerBound = constant;
      range.hasLowerBound = true;
    }
    if (tryGetConstantIndex(forOp.getUpperBound(), constant)) {
      range.upperBound = constant;
      range.hasUpperBound = true;
    }
    if (tryGetConstantIndex(forOp.getStep(), constant)) {
      range.step = constant == 0 ? 1 : constant;
      range.hasStep = true;
    } else {
      range.step = 1;
      range.hasStep = false;
    }

    begin.ranges.push_back(range);
    appendLoopBegin(begin);
    for (mlir::Region &region : forOp->getRegions()) {
      processRegion(region, true);
    }
    appendLoopEnd(LoopEnd{begin.loopType});
  }

  void handleLinalg(mlir::linalg::LinalgOp op) {
    TensorLayout layout = computeLayoutForOp(op, layoutAnalyzer_);
    std::string opName = op->getName().getStringRef().str();
    if (auto generic = llvm::dyn_cast<mlir::linalg::GenericOp>(op.getOperation())) {
      opName = describeGenericOp(generic);
    }

    const auto totalCalcs =
        static_cast<std::int64_t>(computeTotalCalculations(op));
    appendCompute(opName, layout.dims, layout.strides, totalCalcs);
  }

  static bool isBlockArgValue(mlir::Value value) {
    return llvm::isa<mlir::BlockArgument>(stripCastsAndViews(value));
  }

  mlir::MemRefType inferCopyMemrefType(mlir::Value source, mlir::Value target) const {
    mlir::MemRefType memrefType = tryGetStaticMemRefType(source);
    if (!memrefType) {
      memrefType = tryGetStaticMemRefType(target);
    }
    return memrefType;
  }

  void finalizePlan(CopyPlan &plan) const {
    if (plan.vertical && plan.horizontal) {
      plan.allLinks = true;
    }
    if (plan.allLinks) {
      plan.network = "all_links";
      plan.broadcastNetwork = "all_links";
      return;
    }
    if (!plan.network.empty()) {
      plan.broadcastNetwork = plan.network;
    } else if (!plan.dim.empty()) {
      plan.broadcastNetwork = plan.dim;
    } else {
      plan.broadcastNetwork = "broadcast";
    }
  }

  void handleMemrefCopy(mlir::memref::CopyOp copyOp) {
    CopyPlan plan;

    if (auto choice = copyOp->getAttrOfType<mlir::DictionaryAttr>("loom.copy.choice")) {
      if (auto kindAttr = choice.getAs<mlir::StringAttr>("kind")) {
        plan.kind = kindAttr.getValue().str();
      }
      if (auto dimAttr = choice.getAs<mlir::StringAttr>("dim")) {
        plan.dim = dimAttr.getValue().str();
      }
      if (auto networkAttr = choice.getAs<mlir::StringAttr>("interconnect_name")) {
        plan.network = networkAttr.getValue().str();
        applyNetworkFlags(plan, networkAttr.getValue());
      }
    }

    const bool sourceIsArg = isBlockArgValue(copyOp.getSource());
    const bool targetIsArg = isBlockArgValue(copyOp.getTarget());
    if (sourceIsArg && !targetIsArg) {
      plan.isStore = false;
    } else if (!sourceIsArg && targetIsArg) {
      plan.isStore = true;
    }

    finalizePlan(plan);

    mlir::MemRefType memrefType = inferCopyMemrefType(copyOp.getSource(), copyOp.getTarget());
    if (!memrefType) {
      return;
    }

    auto layout = layoutAnalyzer_.fromMemRef(memrefType);
    if (!layout) {
      return;
    }

    const std::int64_t bytes = computeBytesFromMemRef(memrefType);

    if (plan.isStore) {
      emitStoreTransfer(plan, *layout, bytes);
      return;
    }

    if (plan.allLinks) {
      emitAllLinksBroadcast(plan, *layout, bytes);
      return;
    }
    if (plan.vertical) {
      emitVerticalBroadcast(plan, *layout, bytes);
      return;
    }
    if (plan.horizontal) {
      emitHorizontalBroadcast(plan, *layout, bytes);
      return;
    }

    emitDmaLoad(plan, *layout, bytes);
  }

  void handleLoomCopy(::loom::CopyOp copyOp) {
    CopyPlan plan;

    std::string srcSpace = getSymbolRefName(copyOp->getAttr("src_mem_space"));
    std::string dstSpace = getSymbolRefName(copyOp->getAttr("dst_mem_space"));
    std::string srcLower = toLower(srcSpace);
    std::string dstLower = toLower(dstSpace);

    if (srcLower == "l1" && dstLower == "dram") {
      plan.isStore = true;
      plan.kind = "store";
    } else if (srcLower == "dram" && dstLower == "l1") {
      plan.isStore = false;
      plan.kind = "load";
    }

    mlir::ArrayAttr interconnectAttr = copyOp.getInterconnect();
    if (interconnectAttr && !interconnectAttr.empty()) {
      for (mlir::Attribute attr : interconnectAttr) {
        std::string name = getSymbolRefName(attr);
        if (name.empty()) {
          continue;
        }
        if (plan.network.empty()) {
          plan.network = name;
        }
        applyNetworkFlags(plan, name);
      }
    }

    const bool sourceIsArg = isBlockArgValue(copyOp.getSource());
    const bool targetIsArg = isBlockArgValue(copyOp.getDestination());
    if (sourceIsArg && !targetIsArg) {
      plan.isStore = false;
    } else if (!sourceIsArg && targetIsArg) {
      plan.isStore = true;
    }

    finalizePlan(plan);

    mlir::MemRefType memrefType =
        inferCopyMemrefType(copyOp.getSource(), copyOp.getDestination());
    if (!memrefType) {
      return;
    }

    auto layout = layoutAnalyzer_.fromMemRef(memrefType);
    if (!layout) {
      return;
    }

    const std::int64_t bytes = computeBytesFromMemRef(memrefType);

    if (plan.isStore) {
      emitStoreTransfer(plan, *layout, bytes);
      return;
    }

    if (plan.allLinks) {
      emitAllLinksBroadcast(plan, *layout, bytes);
      return;
    }
    if (plan.vertical) {
      emitVerticalBroadcast(plan, *layout, bytes);
      return;
    }
    if (plan.horizontal) {
      emitHorizontalBroadcast(plan, *layout, bytes);
      return;
    }

    emitDmaLoad(plan, *layout, bytes);
  }

  void handleMaterialize(mlir::bufferization::MaterializeInDestinationOp materialize) {
    CopyPlan plan;
    plan.isStore = true;
    plan.network = "core_to_dma";

    if (auto choice =
            materialize->getAttrOfType<mlir::DictionaryAttr>("loom.copy.choice")) {
      if (auto kindAttr = choice.getAs<mlir::StringAttr>("kind")) {
        plan.kind = kindAttr.getValue().str();
      }
      if (auto networkAttr = choice.getAs<mlir::StringAttr>("interconnect_name")) {
        plan.network = networkAttr.getValue().str();
      }
    }

    mlir::MemRefType memrefType = tryGetStaticMemRefType(materialize.getDest());
    if (!memrefType) {
      return;
    }
    auto layout = layoutAnalyzer_.fromMemRef(memrefType);
    if (!layout) {
      return;
    }

    const std::int64_t bytes = computeBytesFromMemRef(memrefType);
    emitStoreTransfer(plan, *layout, bytes);
  }

  void emitVerticalBroadcast(const CopyPlan &plan, const TensorLayout &layout,
                             std::int64_t bytes) {
    if (extentY_ <= 0 || extentX_ <= 0) {
      return;
    }

    for (std::int64_t x = 0; x < extentX_; ++x) {
      for (std::int64_t y = 0; y < extentY_; ++y) {
        auto &current = coreWork(x, y);
        if (y == 0) {
          MemoryLoad load;
          load.core = CoreIndex{x, y};
          load.startCore = load.core;
          load.requestedCore = load.core;
          load.sourceLayout = layout.dims;
          load.sourceStrides = layout.strides;
          load.guardReason = plan.kind;
          current.events.emplace_back(std::move(load));
        }

        MulticastWorkload multicast;
        multicast.core = CoreIndex{x, 0};
        multicast.start = CoreIndex{x, 0};
        multicast.end = CoreIndex{x, extentY_ - 1};
        multicast.bytes = bytes;
        multicast.reason = plan.kind;
        multicast.network = plan.broadcastNetwork;
        current.events.emplace_back(std::move(multicast));
      }
    }
  }

  void emitHorizontalBroadcast(const CopyPlan &plan, const TensorLayout &layout,
                               std::int64_t bytes) {
    if (extentY_ <= 0 || extentX_ <= 0) {
      return;
    }

    for (std::int64_t x = 0; x < extentX_; ++x) {
      for (std::int64_t y = 0; y < extentY_; ++y) {
        auto &current = coreWork(x, y);
        if (x == 0) {
          MemoryLoad load;
          load.core = CoreIndex{x, y};
          load.startCore = load.core;
          load.requestedCore = load.core;
          load.sourceLayout = layout.dims;
          load.sourceStrides = layout.strides;
          load.guardReason = plan.kind;
          current.events.emplace_back(std::move(load));
        }

        MulticastWorkload multicast;
        multicast.core = CoreIndex{0, y};
        multicast.start = CoreIndex{0, y};
        multicast.end = CoreIndex{extentX_ - 1, y};
        multicast.bytes = bytes;
        multicast.reason = plan.kind;
        multicast.network = plan.broadcastNetwork;
        current.events.emplace_back(std::move(multicast));
      }
    }
  }

  void emitAllLinksBroadcast(const CopyPlan &plan, const TensorLayout &layout,
                             std::int64_t bytes) {
    if (extentY_ <= 0 || extentX_ <= 0) {
      return;
    }

    bool memoryLoaded = false;
    for (std::int64_t x = 0; x < extentX_; ++x) {
      for (std::int64_t y = 0; y < extentY_; ++y) {
        auto &current = coreWork(x, y);
        if (!memoryLoaded) {
          MemoryLoad load;
          load.core = CoreIndex{x, y};
          load.startCore = load.core;
          load.requestedCore = load.core;
          load.sourceLayout = layout.dims;
          load.sourceStrides = layout.strides;
          load.guardReason = plan.kind;
          current.events.emplace_back(std::move(load));
          memoryLoaded = true;
        }

        MulticastWorkload multicast;
        multicast.core = CoreIndex{0, 0};
        multicast.start = CoreIndex{0, 0};
        multicast.end = CoreIndex{extentX_ - 1, extentY_ - 1};
        multicast.bytes = bytes;
        multicast.reason = plan.kind;
        multicast.network = plan.broadcastNetwork;
        current.events.emplace_back(std::move(multicast));
      }
    }
  }

  void emitDmaLoad(const CopyPlan &plan, const TensorLayout &layout,
                   std::int64_t bytes) {
    for (std::int64_t x = 0; x < extentX_; ++x) {
      for (std::int64_t y = 0; y < extentY_; ++y) {
        CoreIndex core{x, y};
        CoreIndex sourceCore = selectNearestDMACore(x, y, extentX_, extentY_);

        MemoryLoad load;
        load.core = sourceCore;
        load.startCore = sourceCore;
        load.requestedCore = core;
        load.sourceLayout = layout.dims;
        load.sourceStrides = layout.strides;
        load.guardReason = plan.kind;

        auto &current = coreWork(x, y);
        auto &source = coreWork(sourceCore.x, sourceCore.y);
        current.events.emplace_back(load);
        if (sourceCore != core) {
          source.events.emplace_back(load);
        }

        if (!isDMAConnectedCoordinate(x, y)) {
          NoCWorkload noc;
          noc.core = sourceCore;
          noc.start = sourceCore;
          noc.end = core;
          noc.bytes = bytes;
          noc.reason = plan.kind;
          noc.network = plan.network.empty() ? "dma_to_core" : plan.network;
          current.events.emplace_back(noc);
          source.events.emplace_back(std::move(noc));
        }
      }
    }
  }

  void emitStoreTransfer(const CopyPlan &plan, const TensorLayout &layout,
                         std::int64_t bytes) {
    for (std::int64_t x = 0; x < extentX_; ++x) {
      for (std::int64_t y = 0; y < extentY_; ++y) {
        CoreIndex core{x, y};
        auto &current = coreWork(x, y);

        MemoryStore localStore;
        localStore.core = core;
        localStore.endCore = core;
        localStore.targetLayout = layout.dims;
        localStore.targetStrides = layout.strides;
        localStore.guardReason = plan.kind;

        if (!isDMAConnectedCoordinate(x, y)) {
          CoreIndex endCore = selectNearestDMACore(x, y, extentX_, extentY_);
          auto &endWork = coreWork(endCore.x, endCore.y);

          NoCWorkload noc;
          noc.core = core;
          noc.start = core;
          noc.end = endCore;
          noc.bytes = bytes;
          noc.reason = plan.kind;
          noc.network = plan.network.empty() ? "core_to_dma" : plan.network;

          current.events.emplace_back(noc);
          endWork.events.emplace_back(std::move(noc));

          MemoryStore dmaStore = localStore;
          dmaStore.core = endCore;
          dmaStore.endCore = endCore;
          endWork.events.emplace_back(std::move(dmaStore));
        }

        current.events.emplace_back(std::move(localStore));
      }
    }
  }

  mlir::func::FuncOp entryFunc_;
  std::int64_t extentX_ = 1;
  std::int64_t extentY_ = 1;
  std::vector<CoreSequentialWorkload> perCore_;
  std::int64_t loopId_ = 0;
  LayoutAnalyzer layoutAnalyzer_;
  bool hasCompute_ = false;
};

FunctionWorkload buildFunctionWorkload(mlir::func::FuncOp func, std::int64_t extentX,
                                       std::int64_t extentY) {
  WorkloadBuilder builder(func, extentX, extentY);
  return builder.build();
}

} // namespace

llvm::Expected<std::vector<FunctionWorkload>>
buildCoreSequentialWorkloads(mlir::ModuleOp module) {
  if (!module) {
    return llvm::make_error<llvm::StringError>(
        "invalid module provided", llvm::inconvertibleErrorCode());
  }

  llvm::StringMap<std::int64_t> spatialDims = collectSpatialDims(module);
  auto clampExtent = [](std::int64_t extent) { return extent > 0 ? extent : 1; };
  const std::int64_t extentX = clampExtent(spatialDims.lookup("x"));
  const std::int64_t extentY = clampExtent(spatialDims.lookup("y"));

  std::vector<FunctionWorkload> workloads;
  module.walk([&](mlir::func::FuncOp func) {
    workloads.push_back(buildFunctionWorkload(func, extentX, extentY));
  });

  return workloads;
}

void mergeHardwareSpecFromDF(mlir::ModuleOp module, HardwareSpec &spec) {
  if (!module) {
    return;
  }

  bool sawExplicitNoc = false;

  module.walk([&](mlir::Operation *op) {
    if (!op) {
      return;
    }

    const llvm::StringRef opName = op->getName().getStringRef();

    if (opName == "df.memory") {
      auto labelAttr = op->getAttrOfType<mlir::StringAttr>("label");
      if (!labelAttr) {
        return;
      }
      std::string label = toLower(labelAttr.getValue().str());
      if (label != "dram") {
        return;
      }
      if (auto bandwidth = getNumericAttr(op, "bandwidth")) {
        spec.dramBandwidthGBps = *bandwidth;
      }
      return;
    }

    if (opName == "df.interconnects") {
      auto bandwidth = getNumericAttr(op, "bandwidth");
      if (!bandwidth) {
        return;
      }

      auto labelAttr = op->getAttrOfType<mlir::StringAttr>("label");
      std::string label = labelAttr ? toLower(labelAttr.getValue().str()) : "";
      const bool allLinks = label.find("all_links") != std::string::npos;
      const bool explicitNoc = label.find("noc") != std::string::npos;

      if (allLinks) {
        spec.allLinksBandwidthGBps = *bandwidth;
      } else if (explicitNoc) {
        spec.nocBandwidthGBps = *bandwidth;
        sawExplicitNoc = true;
      } else if (!sawExplicitNoc) {
        spec.nocBandwidthGBps = *bandwidth;
      }
      return;
    }

    if (opName == "df.mat") {
      if (auto throughput = getNumericAttr(op, "throughput")) {
        spec.matrixGflops = *throughput;
      }
      return;
    }

    if (opName == "df.vec") {
      if (auto throughput = getNumericAttr(op, "throughput")) {
        spec.vectorGflops = *throughput;
      }
      return;
    }
  });
}

} // namespace cost_model
} // namespace loom
