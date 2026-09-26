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

} // namespace utils
} // namespace loom
