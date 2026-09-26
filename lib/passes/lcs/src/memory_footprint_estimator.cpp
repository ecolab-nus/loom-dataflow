#include "utils.h"
#include "memory_footprint_estimator.h"
#include "ADL/IR/ADLDialect.h"
#include "ADL/IR/ADLOps.h"
#include "ADL/IR/ADLTypes.h"
#include "LoomInterfaces.h.inc"
#include "hw_op_registry.h"
#include "lcs_utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ssa_utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ErrorHandling.h"
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

namespace loom {
namespace lcs {
namespace {

constexpr int64_t kStorageAlignTile = 32;
constexpr int64_t kTenstorrentReductionConfigElements = 1024;

struct AllocInfo {
  loom::AllocOp alloc_op;
  std::string memory;
  std::vector<int64_t> static_sizes;
  std::vector<Expr> expr_dims;
  mlir::Type elem_type;
  int64_t elem_bytes = 0;
  Expr footprint_bytes;
};

enum class FootprintClass { Compute, Load, Store };

mlir::FailureOr<int64_t> checkedProduct(int64_t lhs, int64_t rhs,
                                        mlir::Operation *diagnosticOp,
                                        const llvm::Twine &description) {
  if (lhs < 0 || rhs < 0 ||
      (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs)) {
    diagnosticOp->emitError()
        << "staged-etg: " << description << " exceeds signed 64-bit range";
    return mlir::failure();
  }
  return lhs * rhs;
}

mlir::FailureOr<std::string> resolveMemoryIdentity(loom::AllocOp allocOp) {
  mlir::SymbolRefAttr memory = allocOp.getMemory();
  if (!memory.getNestedReferences().empty()) {
    allocOp.emitError()
        << "staged-etg: memory capacity accounting supports only flat memory "
           "symbols, got "
        << memory;
    return mlir::failure();
  }
  return memory.getRootReference().str();
}

mlir::FailureOr<int64_t> getElementBytes(loom::AllocOp allocOp,
                                         mlir::Type elemType) {
  if (!mlir::isa<mlir::IntegerType, mlir::FloatType>(elemType)) {
    allocOp.emitError() << "staged-etg: unsupported allocation element type "
                        << elemType << " for byte capacity accounting";
    return mlir::failure();
  }
  unsigned bits = elemType.getIntOrFloatBitWidth();
  if (bits == 0 || bits % 8 != 0) {
    allocOp.emitError() << "staged-etg: allocation element type " << elemType
                        << " does not have a whole-byte storage width";
    return mlir::failure();
  }
  return static_cast<int64_t>(bits / 8);
}

mlir::FailureOr<std::vector<AllocInfo>>
readAllLocalAllocs(mlir::func::FuncOp funcOp) {
  std::vector<AllocInfo> allocs;
  bool valid = true;
  funcOp.walk([&](loom::AllocOp allocOp) {
    if (allocOp.getBufferCount() != 1) {
      allocOp.emitError()
          << "staged-etg: capacity accounting requires buffer_count = 1; "
             "buffering is selected by the metadata is_double_buffer decision";
      valid = false;
      return;
    }
    auto memrefType =
        mlir::dyn_cast<mlir::MemRefType>(allocOp.getResult().getType());
    if (!memrefType) {
      allocOp.emitError()
          << "staged-etg: capacity accounting requires a memref allocation";
      valid = false;
      return;
    }
    auto memory = resolveMemoryIdentity(allocOp);
    auto elemBytes = getElementBytes(allocOp, memrefType.getElementType());
    if (mlir::failed(memory) || mlir::failed(elemBytes)) {
      valid = false;
      return;
    }
    allocs.push_back(AllocInfo{
        allocOp,
        *memory,
        std::vector<int64_t>(allocOp.getStaticSizes().begin(),
                             allocOp.getStaticSizes().end()),
        formatAllocDims(allocOp),
        memrefType.getElementType(),
        *elemBytes,
        Expr::none(),
    });
  });
  if (!valid)
    return mlir::failure();
  return allocs;
}

mlir::LogicalResult validateBottom2Dims(const AllocInfo &info) {
  const size_t rank = info.static_sizes.size();
  if (rank < 2) {
    info.alloc_op->emitError()
        << "staged-etg: storage footprint requires allocation rank >= 2";
    return mlir::failure();
  }

  int oneCount = 0;
  for (size_t idx : {rank - 2, rank - 1}) {
    int64_t dim = info.static_sizes[idx];
    if (mlir::ShapedType::isDynamic(dim))
      continue;
    if (dim == 1) {
      ++oneCount;
      continue;
    }
    if (dim <= 0 || dim % kStorageAlignTile != 0) {
      info.alloc_op->emitError()
          << "staged-etg: bottom-2 static allocation dimensions must be 1 or "
             "a positive multiple of 32";
      return mlir::failure();
    }
  }
  if (oneCount > 1) {
    info.alloc_op->emitError()
        << "staged-etg: bottom-2 allocation dimensions may contain at most "
           "one static dimension of size 1";
    return mlir::failure();
  }
  return mlir::success();
}

std::vector<Expr> applyBottom2Padding(const AllocInfo &info) {
  std::vector<Expr> aligned = info.expr_dims;
  const size_t rank = info.static_sizes.size();
  for (size_t idx : {rank - 2, rank - 1})
    if (info.static_sizes[idx] == 1)
      aligned[idx] = Expr::con(kStorageAlignTile);
  return aligned;
}

mlir::LogicalResult
markAllocClass(llvm::DenseMap<mlir::Operation *, FootprintClass> &classes,
               loom::AllocOp allocOp, FootprintClass nextClass) {
  mlir::Operation *key = allocOp.getOperation();
  auto [it, inserted] = classes.try_emplace(key, nextClass);
  if (inserted || it->second == nextClass)
    return mlir::success();
  allocOp.emitError()
      << "staged-etg: the same allocation cannot be classified as both load "
         "and store storage";
  return mlir::failure();
}

mlir::LogicalResult classifyCopyEndpoints(
    mlir::func::FuncOp funcOp,
    const llvm::DenseMap<mlir::Operation *, size_t> &allocIndex,
    llvm::DenseMap<mlir::Operation *, FootprintClass> &classes) {
  bool valid = true;
  funcOp.walk([&](loom::CopyOp copyOp) {
    loom::utils::CopyMemoryDirection direction =
        loom::utils::classifyCopyMemoryDirection(copyOp.getOperation());
    if (direction == loom::utils::CopyMemoryDirection::Other)
      return;

    loom::AllocOp endpoint =
        loom::utils::traceCopyAllocationEndpoint(copyOp.getOperation());
    if (!endpoint || !allocIndex.count(endpoint.getOperation()))
      return;

    FootprintClass cls = direction == loom::utils::CopyMemoryDirection::Load
                             ? FootprintClass::Load
                             : FootprintClass::Store;
    if (mlir::failed(markAllocClass(classes, endpoint, cls)))
      valid = false;
  });
  return mlir::success(valid);
}

void pushFootprint(MemoryFootprint &footprint, FootprintClass cls, Expr term) {
  switch (cls) {
  case FootprintClass::Load:
    footprint.load_bytes.push_back(std::move(term));
    return;
  case FootprintClass::Store:
    footprint.store_bytes.push_back(std::move(term));
    return;
  case FootprintClass::Compute:
    footprint.compute_bytes.push_back(std::move(term));
    return;
  }
  llvm_unreachable("unknown memory footprint class");
}

mlir::FailureOr<int64_t> extractInstanceSize(mlir::Value handle,
                                             mlir::Operation *diagnosticOp,
                                             llvm::StringRef memory) {
  if (auto bankOp = handle.getDefiningOp<mlir::adl::MemoryBankOp>()) {
    return checkedProduct(static_cast<int64_t>(bankOp.getBsize()),
                          static_cast<int64_t>(bankOp.getNblk()), diagnosticOp,
                          "capacity of memory '" + memory + "'");
  }
  auto arrayOp = handle.getDefiningOp<mlir::adl::MemoryArrayOp>();
  if (!arrayOp) {
    diagnosticOp->emitError() << "staged-etg: memory '" << memory
                              << "' has an unsupported capacity hierarchy";
    return mlir::failure();
  }

  int64_t count = 1;
  for (mlir::Value spatialVal : arrayOp.getSpatialDims()) {
    auto dimOp = spatialVal.getDefiningOp<mlir::adl::SpatialDimOp>();
    if (!dimOp) {
      diagnosticOp->emitError() << "staged-etg: memory '" << memory
                                << "' has a non-constant spatial dimension";
      return mlir::failure();
    }
    auto next =
        checkedProduct(count, static_cast<int64_t>(dimOp.getSize()),
                       diagnosticOp, "capacity of memory '" + memory + "'");
    if (mlir::failed(next))
      return mlir::failure();
    count = *next;
  }
  auto nested = extractInstanceSize(arrayOp.getBank(), diagnosticOp, memory);
  if (mlir::failed(nested))
    return mlir::failure();
  return checkedProduct(count, *nested, diagnosticOp,
                        "capacity of memory '" + memory + "'");
}

mlir::FailureOr<int64_t>
extractLocalSizeFromPlatform(const HWOpRegistry *registry,
                             mlir::func::FuncOp funcOp,
                             llvm::StringRef memory) {
  if (!registry || !registry->getPlatformModule()) {
    funcOp.emitError()
        << "staged-etg: hardware platform is unavailable for memory capacity "
           "accounting";
    return mlir::failure();
  }

  std::string target = loom::utils::physicalMemorySymbol(memory);
  mlir::adl::MemoryArrayOp matched;
  bool duplicate = false;
  registry->getPlatformModule().walk([&](mlir::adl::MemoryArrayOp arrayOp) {
    if (arrayOp->getParentOp() != registry->getPlatformModule().getOperation())
      return mlir::WalkResult::skip();
    if (arrayOp.getSymName() != target)
      return mlir::WalkResult::advance();
    if (matched)
      duplicate = true;
    else
      matched = arrayOp;
    return mlir::WalkResult::advance();
  });
  if (!matched || duplicate) {
    funcOp.emitError() << "staged-etg: expected exactly one top-level hardware "
                          "memory named @"
                       << target << " for allocation memory @" << memory;
    return mlir::failure();
  }

  // The outer array enumerates local instances. Its bank hierarchy is the
  // capacity visible to one workload instance.
  auto capacity = extractInstanceSize(matched.getBank(), funcOp, memory);
  if (mlir::failed(capacity) || *capacity <= 0) {
    if (mlir::succeeded(capacity))
      funcOp.emitError() << "staged-etg: memory @" << memory
                         << " must have positive per-instance capacity";
    return mlir::failure();
  }
  return capacity;
}

} // namespace

mlir::FailureOr<MemoryFootprintResult>
MemoryFootprintEstimator::estimateFromFunc(mlir::func::FuncOp funcOp,
                                           const HWOpRegistry *registry,
                                           Target target) {
  auto allocResult = readAllLocalAllocs(funcOp);
  if (mlir::failed(allocResult))
    return mlir::failure();
  std::vector<AllocInfo> allocs = std::move(*allocResult);

  llvm::DenseMap<mlir::Operation *, size_t> allocIndex;
  std::map<std::string, MemoryFootprint> footprints;
  for (size_t i = 0; i < allocs.size(); ++i) {
    AllocInfo &info = allocs[i];
    allocIndex[info.alloc_op.getOperation()] = i;
    std::vector<Expr> aligned = info.expr_dims;
    if (target == Target::TT) {
      if (mlir::failed(validateBottom2Dims(info)))
        return mlir::failure();
      aligned = applyBottom2Padding(info);
    }
    for (const Expr &dim : aligned) {
      if (dim.isNone()) {
        info.alloc_op.emitError()
            << "staged-etg: allocation dimension could not be represented as "
               "a capacity expression";
        return mlir::failure();
      }
    }
    Expr elements = productOfDims(aligned);
    if (elements.isNone()) {
      info.alloc_op.emitError()
          << "staged-etg: could not construct allocation footprint";
      return mlir::failure();
    }
    info.footprint_bytes = elements * Expr::con(info.elem_bytes);
    footprints.try_emplace(info.memory, MemoryFootprint{info.memory});
  }

  for (auto &[memory, footprint] : footprints) {
    auto capacity = extractLocalSizeFromPlatform(registry, funcOp, memory);
    if (mlir::failed(capacity))
      return mlir::failure();
    footprint.capacity_bytes = *capacity;
  }

  llvm::DenseMap<mlir::Operation *, FootprintClass> classes;
  if (mlir::failed(classifyCopyEndpoints(funcOp, allocIndex, classes)))
    return mlir::failure();

  for (AllocInfo &info : allocs) {
    FootprintClass cls = FootprintClass::Compute;
    auto it = classes.find(info.alloc_op.getOperation());
    if (it != classes.end())
      cls = it->second;
    pushFootprint(footprints.at(info.memory), cls, info.footprint_bytes);
  }

  if (target == Target::Generic) {
    MemoryFootprintResult result;
    for (auto &[memory, footprint] : footprints)
      result.memory_footprints.push_back(std::move(footprint));
    return result;
  }

  std::map<std::string, std::set<int64_t>> reductionWidths;
  bool reductionOwnerMissing = false;
  funcOp.walk([&](mlir::linalg::GenericOp genericOp) {
    auto iteratorTypes = genericOp.getIteratorTypesArray();
    if (iteratorTypes.empty() ||
        iteratorTypes.back() != mlir::utils::IteratorType::reduction)
      return;
    auto linalgOp = mlir::cast<mlir::linalg::LinalgOp>(genericOp.getOperation());
    if (linalgOp.getDpsInits().empty()) {
      reductionOwnerMissing = true;
      return;
    }
    loom::AllocOp owner =
        loom::utils::traceToRootAllocOp(linalgOp.getDpsInits().front());
    if (!owner) {
      reductionOwnerMissing = true;
      return;
    }
    auto memory = resolveMemoryIdentity(owner);
    auto type = mlir::cast<mlir::ShapedType>(owner.getResult().getType());
    auto width = getElementBytes(owner, type.getElementType());
    if (mlir::failed(memory) || mlir::failed(width)) {
      reductionOwnerMissing = true;
      return;
    }
    reductionWidths[*memory].insert(*width);
  });
  if (reductionOwnerMissing) {
    funcOp.emitError()
        << "staged-etg: reduction configuration storage has no explicit "
           "output allocation";
    return mlir::failure();
  }
  for (const auto &[memory, widths] : reductionWidths) {
    if (widths.size() != 1) {
      funcOp.emitError()
          << "staged-etg: reduction configuration byte size is ambiguous for "
             "memory '" << memory << "'";
      return mlir::failure();
    }
    auto scratchBytes = checkedProduct(kTenstorrentReductionConfigElements,
                                       *widths.begin(), funcOp,
                                       "reduction configuration footprint");
    if (mlir::failed(scratchBytes))
      return mlir::failure();
    footprints.at(memory).compute_bytes.push_back(Expr::con(*scratchBytes));
  }

  MemoryFootprintResult result;
  for (auto &[memory, footprint] : footprints)
    result.memory_footprints.push_back(std::move(footprint));
  return result;
}

} // namespace lcs
} // namespace loom
