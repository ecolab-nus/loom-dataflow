#include "l1_footprint_estimator.h"
#include "lcs_utils.h"
#include "ssa_utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

namespace loom {
namespace lcs {
namespace {

constexpr int64_t kL1AlignTile = 32;

struct AllocInfo {
  loom::AllocOp alloc_op;
  std::vector<int64_t> static_sizes;
  std::vector<Expr> expr_dims;
  mlir::Type elem_type;
  Expr footprint;
};

enum class FootprintClass { Compute, Load, Store };

std::string locToString(mlir::Location loc) {
  std::string s;
  llvm::raw_string_ostream os(s);
  loc.print(os);
  os.flush();
  return s;
}

[[noreturn]] void failAlignAssert(const AllocInfo &info,
                                  llvm::StringRef reason) {
  llvm::errs() << "L1 memory cannot align to hardware granularity (32x32): "
               << reason << "\n";
  llvm::errs() << "  alloc location: " << locToString(info.alloc_op->getLoc())
               << "\n";
  assert(false && "L1 memory cannot align to hardware granularity (32x32)");
  llvm_unreachable("assert should have terminated");
}

[[noreturn]] void failClassifyAssert(loom::AllocOp allocOp,
                                     llvm::StringRef reason) {
  llvm::errs() << "L1 footprint classification failed: " << reason << "\n";
  llvm::errs() << "  alloc location: " << locToString(allocOp->getLoc())
               << "\n";
  assert(false && "L1 footprint classification failed");
  llvm_unreachable("assert should have terminated");
}

std::vector<AllocInfo> readAllL1Allocs(mlir::func::FuncOp funcOp) {
  std::vector<AllocInfo> allocs;
  funcOp.walk([&](loom::AllocOp allocOp) {
    if (allocOp.getMemory().getLeafReference() != "L1")
      return;
    auto memrefType = mlir::cast<mlir::MemRefType>(allocOp.getResult().getType());
    allocs.push_back(AllocInfo{
        allocOp,
        std::vector<int64_t>(allocOp.getStaticSizes().begin(),
                             allocOp.getStaticSizes().end()),
        formatAllocDims(allocOp),
        memrefType.getElementType(),
        Expr::none(),
    });
  });
  return allocs;
}

void validateBottom2Dims(const AllocInfo &info) {
  const size_t rank = info.static_sizes.size();
  if (rank < 2)
    failAlignAssert(info, "L1 alloc rank is smaller than 2");

  const int64_t d0 = info.static_sizes[rank - 2];
  const int64_t d1 = info.static_sizes[rank - 1];
  int one_count = 0;

  auto validateStatic = [&](int64_t dim) {
    if (mlir::ShapedType::isDynamic(dim))
      return;
    if (dim == 1) {
      ++one_count;
      return;
    }
    if (dim % kL1AlignTile != 0)
      failAlignAssert(info, "bottom-2 static dims must be 1 or multiple of 32");
  };

  validateStatic(d0);
  validateStatic(d1);
  if (one_count > 1)
    failAlignAssert(info, "bottom-2 dims can contain at most one static 1");
}

std::vector<Expr> applyBottom2Padding(const AllocInfo &info) {
  std::vector<Expr> aligned = info.expr_dims;
  const size_t rank = info.static_sizes.size();
  assert(rank >= 2 && "rank already validated");
  assert(aligned.size() == rank && "expr dim size should match memref rank");

  auto padIfStaticOne = [&](size_t idx) {
    if (info.static_sizes[idx] == 1)
      aligned[idx] = Expr::con(kL1AlignTile);
  };

  padIfStaticOne(rank - 2);
  padIfStaticOne(rank - 1);
  return aligned;
}

Expr buildFootprintExpr(const std::vector<Expr> &alignedDims) {
  return productOfDims(alignedDims);
}

void markAllocClass(llvm::DenseMap<mlir::Operation *, FootprintClass> &classes,
                    loom::AllocOp allocOp, FootprintClass nextClass) {
  mlir::Operation *key = allocOp.getOperation();
  auto [it, inserted] = classes.try_emplace(key, nextClass);
  if (inserted || it->second == nextClass)
    return;
  failClassifyAssert(allocOp, "same alloc is classified as both load and store");
}

void classifyCopyEndpoints(
    mlir::func::FuncOp funcOp,
    const llvm::DenseMap<mlir::Operation *, size_t> &allocIndex,
    llvm::DenseMap<mlir::Operation *, FootprintClass> &classes) {
  funcOp.walk([&](loom::CopyOp copyOp) {
    loom::utils::CopyMemoryDirection direction =
        loom::utils::classifyCopyMemoryDirection(copyOp.getOperation());
    if (direction == loom::utils::CopyMemoryDirection::Other)
      return;

    loom::AllocOp endpoint =
        loom::utils::traceCopyL1EndpointRootAlloc(copyOp.getOperation());
    if (!endpoint || !allocIndex.count(endpoint.getOperation()))
      return;

    FootprintClass cls =
        direction == loom::utils::CopyMemoryDirection::Load
            ? FootprintClass::Load
            : FootprintClass::Store;
    markAllocClass(classes, endpoint, cls);
  });
}

void pushFootprint(L1FootprintByScope &footprint, FootprintClass cls,
                   Expr term) {
  if (term.isNone())
    return;
  switch (cls) {
  case FootprintClass::Load:
    footprint.load.push_back(term);
    return;
  case FootprintClass::Store:
    footprint.store.push_back(term);
    return;
  case FootprintClass::Compute:
    footprint.compute.push_back(term);
    return;
  }
  llvm_unreachable("unknown L1 footprint class");
}

} // namespace

L1FootprintResult L1FootprintEstimator::estimateFromFunc(mlir::func::FuncOp funcOp) {
  L1FootprintResult result;
  std::vector<AllocInfo> allocs = readAllL1Allocs(funcOp);
  llvm::DenseMap<mlir::Operation *, size_t> allocIndex;
  for (size_t i = 0; i < allocs.size(); ++i)
    allocIndex[allocs[i].alloc_op.getOperation()] = i;

  mlir::Type expectedElemType;
  for (AllocInfo &info : allocs) {
    if (!expectedElemType) {
      expectedElemType = info.elem_type;
      result.datatype = formatElementType(info.elem_type);
    } else {
      assert(info.elem_type == expectedElemType &&
             "All L1 allocations must have the same element type");
    }

    validateBottom2Dims(info);
    std::vector<Expr> aligned = applyBottom2Padding(info);
    info.footprint = buildFootprintExpr(aligned);
  }

  llvm::DenseMap<mlir::Operation *, FootprintClass> classes;
  classifyCopyEndpoints(funcOp, allocIndex, classes);

  for (AllocInfo &info : allocs) {
    FootprintClass cls = FootprintClass::Compute;
    auto it = classes.find(info.alloc_op.getOperation());
    if (it != classes.end())
      cls = it->second;
    pushFootprint(result.l1_footprint, cls, info.footprint);
  }

  return result;
}

} // namespace lcs
} // namespace loom
