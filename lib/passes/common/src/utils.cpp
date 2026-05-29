/**
 * @file utils.cpp
 * @brief Implementation of common utilities for function cloning.
 */

#include "utils.h"
#include "hardware_info.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <cassert>
#include <set>
#include <string>

// Include the generated Loom dialect headers
#include "LoomDialect.h.inc"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace loom {
namespace utils {
namespace {

using SizeChoice = std::optional<int64_t>;

SmallVector<SizeChoice> getOccupancyChoices(const SpatialDimInfo &dim) {
  SmallVector<SizeChoice> choices;
  if (!dim.size || *dim.size < 2) {
    choices.push_back(dim.size);
    return choices;
  }

  for (int64_t size = 2; size <= *dim.size; size += 2)
    choices.push_back(size);
  return choices;
}

bool containsSize(ArrayRef<SizeChoice> choices, SizeChoice size) {
  return llvm::is_contained(choices, size);
}

std::string buildOccupancyKey(ArrayRef<SizeChoice> sizes) {
  std::string key;
  llvm::raw_string_ostream os(key);
  bool first = true;
  for (SizeChoice size : sizes) {
    if (!first)
      os << ",";
    first = false;
    if (size)
      os << *size;
    else
      os << "?";
  }
  return key;
}

HardwareInfo withOccupancySizes(const HardwareInfo &base,
                                ArrayRef<SizeChoice> sizes) {
  HardwareInfo variant = base;
  for (auto [idx, size] : llvm::enumerate(sizes))
    variant.spatialDimInfoVec[idx].size = size;
  return variant;
}

} // namespace

ModuleOp getParentModule(func::FuncOp func) {
  Operation *parent = func->getParentOp();
  if (auto module = dyn_cast_or_null<ModuleOp>(parent)) {
    return module;
  }
  return nullptr;
}

func::FuncOp cloneFunc(OpBuilder &builder, func::FuncOp originalFunc,
                       llvm::StringRef newName, DictionaryAttr moduleAttrs,
                       std::function<LogicalResult(func::FuncOp)> modifier,
                       Operation *insertAfter) {
  if (insertAfter)
    builder.setInsertionPointAfter(insertAfter);

  ModuleOp wrapperModule = nullptr;
  OpBuilder effectiveBuilder = builder;
  if (moduleAttrs) {
    wrapperModule = ModuleOp::create(originalFunc.getLoc());
    wrapperModule->setAttrs(moduleAttrs);
    builder.insert(wrapperModule);
    effectiveBuilder = OpBuilder(wrapperModule.getBodyRegion());
  }

  IRMapping mapping;
  auto clonedFunc =
      cast<func::FuncOp>(effectiveBuilder.clone(*originalFunc, mapping));
  clonedFunc.setName(newName);

  if (modifier && failed(modifier(clonedFunc))) {
    if (wrapperModule)
      wrapperModule.erase();
    else
      clonedFunc.erase();
    return nullptr;
  }

  builder.setInsertionPointAfter(wrapperModule ? (Operation *)wrapperModule
                                               : (Operation *)clonedFunc);
  return clonedFunc;
}

llvm::SmallVector<func::FuncOp> collectFunctions(ModuleOp module) {
  llvm::SmallVector<func::FuncOp> funcs;
  module.walk([&](func::FuncOp func) { funcs.push_back(func); });
  return funcs;
}

StringRef traceToSymbolicVar(Value val) {
  if (!val)
    return "";

  // Handle direct loom.sym
  if (auto getSym = val.getDefiningOp<loom::SymOp>()) {
    return getSym.getSymbolRef().getLeafReference().getValue();
  }

  // Handle arith.muli/addi/etc. if needed, but for now we follow the user's
  // sketch where block sizes are directly used from
  // loom.get_symbolic_block_size.

  return "";
}

SmallVector<HardwareInfo>
generateHardwareOccupancyVariants(const HardwareInfo &hardwareInfo) {
  SmallVector<HardwareInfo> variants;
  const unsigned dimCount =
      static_cast<unsigned>(hardwareInfo.spatialDimInfoVec.size());
  if (dimCount == 0) {
    variants.push_back(hardwareInfo);
    return variants;
  }

  SmallVector<SmallVector<SizeChoice>> choicesByDim;
  choicesByDim.reserve(dimCount);
  for (const SpatialDimInfo &dim : hardwareInfo.spatialDimInfoVec)
    choicesByDim.push_back(getOccupancyChoices(dim));

  std::set<std::string> seen;
  auto addVariant = [&](ArrayRef<SizeChoice> sizes) {
    std::string key = buildOccupancyKey(sizes);
    if (!seen.insert(key).second)
      return;
    variants.push_back(withOccupancySizes(hardwareInfo, sizes));
  };

  if (dimCount == 2) {
    const auto &xChoices = choicesByDim[0];
    const auto &yChoices = choicesByDim[1];
    for (SizeChoice xCandidate : xChoices) {
      for (SizeChoice yCandidate : yChoices) {
        SmallVector<SizeChoice, 2> canonical;
        if (xCandidate && yCandidate) {
          SizeChoice larger = std::max(*xCandidate, *yCandidate);
          SizeChoice smaller = std::min(*xCandidate, *yCandidate);
          if (containsSize(xChoices, larger) &&
              containsSize(yChoices, smaller)) {
            canonical = {larger, smaller};
          } else if (containsSize(xChoices, smaller) &&
                     containsSize(yChoices, larger)) {
            canonical = {smaller, larger};
          } else {
            canonical = {xCandidate, yCandidate};
          }
        } else {
          canonical = {xCandidate, yCandidate};
        }
        addVariant(canonical);
      }
    }
    return variants;
  }

  SmallVector<SizeChoice> current;
  current.reserve(dimCount);
  std::function<void(unsigned)> enumerate = [&](unsigned dimIdx) {
    if (dimIdx == dimCount) {
      addVariant(current);
      return;
    }
    for (SizeChoice choice : choicesByDim[dimIdx]) {
      current.push_back(choice);
      enumerate(dimIdx + 1);
      current.pop_back();
    }
  };
  enumerate(0);
  return variants;
}

namespace {

bool hasRankedTensorInitArg(scf::ForOp forOp) {
  return llvm::any_of(forOp.getInitArgs(), [](Value initArg) {
    return isa<RankedTensorType>(initArg.getType());
  });
}

scf::ForOp findUniqueLoopCarriedForOrFail(affine::AffineParallelOp parallelOp) {
  SmallVector<scf::ForOp, 2> loopCarriedForOps;
  parallelOp.walk([&](scf::ForOp forOp) {
    if (!forOp.getInitArgs().empty())
      loopCarriedForOps.push_back(forOp);
  });

  if (loopCarriedForOps.size() != 1) {
    parallelOp.emitError()
        << "memory binding expects exactly one loop-carried scf.for under "
           "affine.parallel, found "
        << loopCarriedForOps.size();
    assert(false && "invalid loop-carried scf.for count");
    llvm::report_fatal_error("invalid loop-carried scf.for count");
  }

  scf::ForOp loopCarriedFor = loopCarriedForOps.front();
  if (!hasRankedTensorInitArg(loopCarriedFor)) {
    loopCarriedFor.emitError()
        << "memory binding expects the unique loop-carried scf.for to carry "
           "at least one ranked tensor iter_arg";
    assert(false && "loop-carried scf.for has no ranked tensor iter_arg");
    llvm::report_fatal_error(
        "loop-carried scf.for has no ranked tensor iter_arg");
  }

  return loopCarriedFor;
}

bool isAllowedLoopNestPrefixOp(Operation *op) {
  if (op->getNumRegions() != 0)
    return false;
  if (!isMemoryEffectFree(op))
    return false;
  if (op->getNumResults() == 0)
    return false;
  return llvm::all_of(op->getResults(), [](Value result) {
    Type type = result.getType();
    return type.isIndex() || isa<IntegerType>(type);
  });
}

void validateLoopCarriedForIsInnermost(scf::ForOp loopCarriedFor) {
  bool hasNestedLoop = false;
  loopCarriedFor.getBody()->walk([&](Operation *op) {
    if (isa<scf::ForOp, affine::AffineForOp>(op)) {
      hasNestedLoop = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (hasNestedLoop) {
    loopCarriedFor.emitError()
        << "loop-carried scf.for must be the innermost loop for memory binding";
    assert(false && "loop-carried scf.for must be innermost");
    llvm::report_fatal_error("loop-carried scf.for must be innermost");
  }

  unsigned yieldCount = 0;
  loopCarriedFor.getBody()->walk([&](scf::YieldOp yieldOp) {
    ++yieldCount;
    return WalkResult::advance();
  });
  auto yieldOp =
      dyn_cast<scf::YieldOp>(loopCarriedFor.getBody()->getTerminator());
  if (yieldCount != 1 || !yieldOp) {
    loopCarriedFor.emitError()
        << "loop-carried scf.for must have exactly one scf.yield terminator";
    assert(false && "loop-carried scf.for must have exactly one yield");
    llvm::report_fatal_error(
        "loop-carried scf.for must have exactly one yield");
  }
}

void validateSingleChildLoopContainer(Operation *container,
                                      scf::ForOp expectedChild) {
  if (container->getNumRegions() != 1 ||
      !llvm::hasSingleElement(container->getRegion(0))) {
    container->emitError()
        << "memory binding loop-nest validation expects a single-block region";
    assert(false && "unsupported loop container shape");
    llvm::report_fatal_error("unsupported loop container shape");
  }

  bool sawChild = false;
  unsigned childLoopCount = 0;
  Block &body = container->getRegion(0).front();
  for (Operation &bodyOp : body) {
    Operation *op = &bodyOp;
    if (op->hasTrait<OpTrait::IsTerminator>())
      continue;

    if (op == expectedChild.getOperation()) {
      sawChild = true;
      ++childLoopCount;
      continue;
    }

    if (isa<scf::ForOp, affine::AffineForOp>(op)) {
      container->emitError()
          << "memory binding requires exactly one child loop at each "
             "supported serial nest level";
      assert(false && "unsupported non-perfect loop nest");
      llvm::report_fatal_error("unsupported non-perfect loop nest");
    }

    if (!sawChild) {
      if (isAllowedLoopNestPrefixOp(op))
        continue;
      op->emitError()
          << "only side-effect-free index/integer bound computations may "
             "precede the child loop in a memory-binding serial envelope";
      assert(false && "unsupported pre-loop operation in serial envelope");
      llvm::report_fatal_error(
          "unsupported pre-loop operation in serial envelope");
    }

    op->emitError() << "no non-terminator operations may follow the child loop "
                       "in a memory-binding serial envelope";
    assert(false && "unsupported post-loop operation in serial envelope");
    llvm::report_fatal_error(
        "unsupported post-loop operation in serial envelope");
  }

  if (!sawChild || childLoopCount != 1) {
    container->emitError()
        << "memory binding serial envelope must contain the expected child "
           "loop exactly once";
    assert(false && "expected child loop missing from serial envelope");
    llvm::report_fatal_error(
        "expected child loop missing from serial envelope");
  }
}

void validateLoopNestPathOrFail(affine::AffineParallelOp parallelOp,
                                scf::ForOp loopCarriedFor) {
  validateLoopCarriedForIsInnermost(loopCarriedFor);

  // The loop-carried loop's immediate parent is the normalized compute scope.
  // It may contain regular compute before and after the carried loop. Only the
  // serial envelope outside that scope must be perfectly nested.
  Operation *normalizedScope = loopCarriedFor->getParentOp();
  if (normalizedScope == parallelOp.getOperation())
    return;

  auto scopeFor = dyn_cast_or_null<scf::ForOp>(normalizedScope);
  if (!scopeFor) {
    loopCarriedFor.emitError()
        << "loop-carried scf.for must be directly enclosed by either an "
           "outer scf.for normalized scope or the affine.parallel";
    assert(false && "unsupported loop-carried parent scope");
    llvm::report_fatal_error("unsupported loop-carried parent scope");
  }

  scf::ForOp child = scopeFor;
  Operation *parent = child->getParentOp();
  while (parent != parallelOp.getOperation()) {
    auto parentFor = dyn_cast_or_null<scf::ForOp>(parent);
    if (!parentFor) {
      loopCarriedFor.emitError()
          << "normalized memory scope must be enclosed only by scf.for ops "
             "between itself and the affine.parallel";
      assert(false && "unsupported non-scf parent in serial envelope");
      llvm::report_fatal_error(
          "unsupported non-scf parent in serial envelope");
    }
    validateSingleChildLoopContainer(parent, child);
    child = parentFor;
    parent = child->getParentOp();
  }

  validateSingleChildLoopContainer(parallelOp.getOperation(), child);
}

/// Returns the L1 alloc's element Type by inspecting its tensor-side users.
/// Returns null Type if no compatible user exists.
Type findL1AllocElementType(loom::AllocOp alloc) {
  for (auto user : alloc.getResult().getUsers()) {
    if (auto init = dyn_cast<loom::InitTensorOp>(user))
      return cast<RankedTensorType>(init.getResult().getType()).getElementType();
    if (auto copy = dyn_cast<loom::CopyToTensorOp>(user))
      return cast<RankedTensorType>(copy.getResult().getType()).getElementType();
  }
  return Type{};
}

/// Discriminates a dynamic alloc operand: either a plain symbolic variable
/// or a `(numerator ceildiv symbolic)` form (from `arith.ceildivsi` or
/// `affine.apply` with a CeilDiv map).
struct DynDim {
  StringRef sym;                                       // plain symbolic
  std::optional<std::pair<int64_t, StringRef>> ceildiv; // (numerator, denomSym)
};

/// Try to recognize `(constant numerator ceildiv symbolic-denominator)` from
/// either `arith.ceildivsi` or an `affine.apply` whose map is a CeilDiv with a
/// constant LHS and a single symbol RHS. Returns the (numerator, denomSym)
/// pair or nullopt.
std::optional<std::pair<int64_t, StringRef>>
matchConstCeildivSym(Value val) {
  if (auto ceildiv = val.getDefiningOp<arith::CeilDivSIOp>()) {
    int64_t numerator = -1;
    if (auto c = ceildiv.getLhs().getDefiningOp<arith::ConstantIndexOp>())
      numerator = c.value();
    else if (auto c = ceildiv.getLhs().getDefiningOp<arith::ConstantIntOp>())
      numerator = c.value();
    if (numerator <= 0)
      return std::nullopt;
    StringRef denom = traceToSymbolicVar(ceildiv.getRhs());
    if (denom.empty())
      return std::nullopt;
    return std::make_pair(numerator, denom);
  }

  if (auto apply = val.getDefiningOp<affine::AffineApplyOp>()) {
    auto map = apply.getAffineMap();
    if (map.getNumResults() != 1)
      return std::nullopt;
    auto expr = map.getResult(0);
    if (expr.getKind() != AffineExprKind::CeilDiv)
      return std::nullopt;
    auto binary = cast<AffineBinaryOpExpr>(expr);
    auto lhs = binary.getLHS();
    auto rhs = binary.getRHS();
    if (lhs.getKind() != AffineExprKind::Constant ||
        rhs.getKind() != AffineExprKind::SymbolId)
      return std::nullopt;
    int64_t numerator = cast<AffineConstantExpr>(lhs).getValue();
    unsigned symIdx = cast<AffineSymbolExpr>(rhs).getPosition();
    if (symIdx >= apply.getNumOperands())
      return std::nullopt;
    StringRef denom = traceToSymbolicVar(apply.getOperand(symIdx));
    if (denom.empty())
      return std::nullopt;
    return std::make_pair(numerator, denom);
  }

  return std::nullopt;
}

DynDim classifyDynamicAllocOperand(Value val) {
  if (auto cd = matchConstCeildivSym(val))
    return DynDim{StringRef{}, cd};
  return DynDim{traceToSymbolicVar(val), std::nullopt};
}

/// Cancel `block_K * (K_total ceildiv block_K) -> K_total`: when a ceildiv's
/// denominator symbol appears among `syms`, drop it from `syms` and fold the
/// numerator into `elemSize` (a constant total-size multiplier).
void cancelHoistedDivPairs(
    SmallVectorImpl<StringRef> &syms,
    ArrayRef<std::pair<int64_t, StringRef>> ceildivs,
    int64_t &elemSize) {
  for (const auto &[numerator, denom] : ceildivs) {
    auto it = llvm::find(syms, denom);
    if (it != syms.end()) {
      syms.erase(it);
      elemSize *= numerator;
    }
  }
}

} // namespace

Operation *
getNormalizedMemoryBindingScope(affine::AffineParallelOp parallelOp) {
  scf::ForOp loopCarriedFor = findUniqueLoopCarriedForOrFail(parallelOp);
  validateLoopNestPathOrFail(parallelOp, loopCarriedFor);

  Operation *parent = loopCarriedFor->getParentOp();
  if (isa<scf::ForOp>(parent))
    return parent;
  return parallelOp.getOperation();
}

llvm::SmallVector<AllocInfo> collectL1AllocInfos(func::FuncOp func) {
  llvm::SmallVector<AllocInfo> allocInfos;

  func.walk([&](loom::AllocOp alloc) {
    if (alloc.getMemory().getLeafReference() != "L1")
      return;

    Type elementType = findL1AllocElementType(alloc);
    if (!elementType)
      return;

    int64_t baseElemSize = elementType.getIntOrFloatBitWidth() / 8;
    AllocInfo info;
    info.elemSize = baseElemSize * alloc.getBufferCount();

    SmallVector<StringRef> syms;
    SmallVector<std::pair<int64_t, StringRef>> ceildivs;
    for (Value val : alloc.getSizes()) {
      DynDim d = classifyDynamicAllocOperand(val);
      if (d.ceildiv)
        ceildivs.push_back(*d.ceildiv);
      else if (!d.sym.empty())
        syms.push_back(d.sym);
    }

    cancelHoistedDivPairs(syms, ceildivs, info.elemSize);
    info.dims = syms;

    // Record only if it has symbolic dims or total size exceeds a single elem
    // (fixed-size multi-buffer or hoisted buffer).
    if (!info.dims.empty() || info.elemSize > baseElemSize)
      allocInfos.push_back(std::move(info));
  });

  return allocInfos;
}

} // namespace utils
} // namespace loom
