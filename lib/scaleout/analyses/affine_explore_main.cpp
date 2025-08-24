#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/FileSystem.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include <algorithm>
// df dialect
#include "DataflowDialect.h.inc"
// Dialect registry for robust pretty-printing

using namespace mlir;

namespace {

struct SpatialDimInfo {
  Operation *op; // df.spatial_dim
  int64_t size;  // static size
  Value value;   // SSA result (index)
};

// A plan assigns an ordered list of spatial dimension indices to each loop.
// - Outer vector indexes loops in order: 0..P-1
// - Inner vectors contain indices into the collected `dims` array, preserving
//   left-to-right order per loop.
using LoopPlan = llvm::SmallVector<int, 4>;
using Plan = llvm::SmallVector<LoopPlan, 4>;

// Enumerate all weak compositions of S into P non-negative parts.
//
// Definition: A weak composition of an integer S into P parts is a length-P
// vector of non-negative integers that sums to S. Order matters, zeros are
// allowed. The number of such compositions is binomial(S + P - 1, P - 1).
//
// Parameters:
// - S: the remaining sum to distribute across the remaining P parts.
// - P: the number of parts left to fill.
// - out: accumulator receiving all completed length-P vectors.
// - current: working prefix (length increases until it reaches P).
//
// Usage pattern: call with `current` empty; on return, `out` contains all
// length-P vectors whose elements sum to the original S.
static void
enumerateWeakCompositions(int S, int P,
                          llvm::SmallVector<llvm::SmallVector<int, 4>, 8> &out,
                          llvm::SmallVector<int, 4> &current) {
  // Base case: only one part left; the last part must take the entire
  // remaining sum S (which can be zero), completing a length-P vector.
  if (P == 1) {
    current.push_back(S);
    out.push_back(current);
    current.pop_back();
    return;
  }
  // Recursive step: choose the first part (0..S), then distribute the
  // remaining S - first across the remaining P - 1 parts.
  for (int first = 0; first <= S; ++first) {
    current.push_back(first);
    enumerateWeakCompositions(S - first, P - 1, out, current);
    current.pop_back();
  }
}

// Generate all permutations of indices 0..N-1.
//
// The permutations are produced in lexicographic order by iterating with
// std::next_permutation starting from the sorted seed [0, 1, ..., N-1].
// Complexity is O(N! * N) overall for generating and copying each vector.
static void
enumeratePermutations(int N,
                      llvm::SmallVector<llvm::SmallVector<int, 8>, 16> &perms) {
  llvm::SmallVector<int, 8> v;
  v.reserve(N);
  for (int i = 0; i < N; ++i)
    v.push_back(i);
  // Enumerate in lexicographic order using std::next_permutation.
  std::sort(v.begin(), v.end());
  do {
    perms.push_back(v);
  } while (std::next_permutation(v.begin(), v.end()));
}

// Collect df.spatial_dim ops and sizes in textual order within the function.
static llvm::SmallVector<SpatialDimInfo, 4>
collectSpatialDims(func::FuncOp funcOp) {
  llvm::SmallVector<SpatialDimInfo, 4> dims;
  funcOp.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "df.spatial_dim") {
      auto sz = op->getAttrOfType<IntegerAttr>("size");
      if (sz)
        dims.push_back({op, sz.getInt(), op->getResult(0)});
    }
  });
  return dims;
}

// Find the first (outermost) affine.parallel in the entry block.
static affine::AffineParallelOp getOutermostParallel(func::FuncOp funcOp) {
  for (Operation &op : funcOp.getBody().front()) {
    if (auto par = dyn_cast<affine::AffineParallelOp>(&op))
      return par;
  }
  return nullptr;
}

//

// Apply a plan assignment: for each parallel IV i, use the ordered list of
// spatial dimensions in plan[i]. This function modifies the cloned function
// in-place. For now, we materialize a single spatial affine.parallel whose IVs
// are laid out by concatenating each loop's list in order (loop 0 list, then
// loop 1 list, ...). The per-loop ordering is preserved in this flattening.
static LogicalResult applyPlanVariant(func::FuncOp clonedFunc, const Plan &plan,
                                      ArrayRef<SpatialDimInfo> dims) {
  affine::AffineParallelOp parOp = getOutermostParallel(clonedFunc);
  if (!parOp)
    return clonedFunc.emitError("expected outermost affine.parallel");

  Location loc = parOp.getLoc();
  OpBuilder b(parOp);
  (void)b.getContext();

  // Build flat list of spatial dims in the per-loop order.
  SmallVector<int64_t, 8> flatSizes;
  for (unsigned loopIndex = 0; loopIndex < plan.size(); ++loopIndex) {
    for (int dimIdx : plan[loopIndex]) {
      flatSizes.push_back(dims[static_cast<size_t>(dimIdx)].size);
    }
  }
  if (flatSizes.empty())
    return success();

  // Create the spatial affine.parallel and move original parallel inside it
  SmallVector<arith::AtomicRMWKind, 0> reductions;
  auto spatialPar = b.create<affine::AffineParallelOp>(loc, TypeRange{},
                                                       reductions, flatSizes);
  Block &spBody = spatialPar.getRegion().front();
  Operation *spTerm = spBody.getTerminator();
  parOp->moveBefore(spTerm);

  return success();
}

// Clone a function and give it a unique name.
static func::FuncOp cloneFunctionWithSuffix(ModuleOp module, func::FuncOp func,
                                            llvm::StringRef suffix) {
  OpBuilder b(module.getContext());
  b.setInsertionPointAfter(func);
  auto newFunc = cast<func::FuncOp>(b.clone(*func.getOperation()));
  std::string newName = func.getName().str();
  newName += suffix.str();
  newFunc.setName(newName);
  return newFunc;
}

// Create human-readable suffix for a plan variant, e.g., __plan_p0[8,16]_p1[]
static std::string makePlanSuffix(const Plan &plan,
                                  ArrayRef<SpatialDimInfo> dims) {
  std::string s = std::string("__plan");
  for (size_t loopIdx = 0; loopIdx < plan.size(); ++loopIdx) {
    s += std::string("_p") + std::to_string(loopIdx) + '[';
    for (size_t j = 0; j < plan[loopIdx].size(); ++j) {
      int dimIdx = plan[loopIdx][j];
      s += std::to_string(dims[static_cast<size_t>(dimIdx)].size);
      if (j + 1 < plan[loopIdx].size())
        s += ',';
    }
    s += ']';
  }
  return s;
}

} // end anonymous namespace

int main(int argc, char **argv) {
  // Initialize context and load required dialects explicitly.
  MLIRContext context;
  (void)context.getOrLoadDialect<mlir::BuiltinDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::memref::MemRefDialect>();
  context.loadDialect<mlir::affine::AffineDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<tmd::df::DataflowDialect>();

  llvm::SourceMgr sourceMgr;
  // CLI: [--restrict-distribution] [--outdir <dir>] <file>
  bool restrictDistribution = false;
  const char *filename = "-";
  std::string outFilePath;
  std::string outDirPath;
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg == "--restrict-distribution") {
      restrictDistribution = true;
    } else if (arg.consume_front("--outdir=")) {
      outDirPath = arg.str();
    } else if (arg == "--outdir" && i + 1 < argc) {
      outDirPath = argv[++i];
    } else if (arg.consume_front("--out=")) {
      outFilePath = arg.str();
    } else if (arg == "--out" && i + 1 < argc) {
      outFilePath = argv[++i];
    } else {
      filename = argv[i];
    }
  }
  auto file = mlir::openInputFile(filename);
  if (!file) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to open input file: " << filename << "\n";
    return 1;
  }
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::WithColor::error(llvm::errs()) << "Failed to parse MLIR module\n";
    return 1;
  }

  // If writing per-variant files, ensure directory exists.
  if (!outDirPath.empty())
  {
    auto created = llvm::sys::fs::create_directories(outDirPath);
    (void)created;
  }

  // For each function: collect spatial dims and outermost parallel, enumerate
  // variants, clone, transform, and optionally emit per-variant modules.
  module->walk([&](func::FuncOp funcOp) {
    auto dims = collectSpatialDims(funcOp);
    affine::AffineParallelOp parOp = getOutermostParallel(funcOp);
    if (!parOp)
      return;

    const int numParallelLoops = static_cast<int>(parOp.getIVs().size());
    const int numSpatialDims = static_cast<int>(dims.size());
    if (numParallelLoops <= 0 || numSpatialDims <= 0)
      return;
    if (numSpatialDims < numParallelLoops)
      return; // Not enough spatial dims to assign at least one per loop

    // Generate plans according to the requested policy.
    // All permutations of spatial dimension indices.
    llvm::SmallVector<llvm::SmallVector<int, 8>, 16> perms;
    enumeratePermutations(numSpatialDims, perms);

    if (restrictDistribution) {
      // Each plan assigns all spatial dims (in some order) to exactly one loop.
      for (const auto &perm : perms) {
        for (int chosen = 0; chosen < numParallelLoops; ++chosen) {
          Plan plan(static_cast<size_t>(numParallelLoops));
          plan[static_cast<size_t>(chosen)].assign(perm.begin(), perm.end());
          std::string suffix = makePlanSuffix(plan, dims);
          func::FuncOp clone = cloneFunctionWithSuffix(*module, funcOp, suffix);
          (void)applyPlanVariant(clone, plan, dims);
          if (!outDirPath.empty()) {
            ModuleOp outMod = ModuleOp::create(UnknownLoc::get(&context));
            OpBuilder mb(&context);
            mb.setInsertionPointToStart(outMod.getBody());
            mb.clone(*clone.getOperation());
            std::string filePath = outDirPath + "/" + clone.getName().str() + ".mlir";
            if (auto out = mlir::openOutputFile(filePath)) {
              mlir::OpPrintingFlags flags;
              flags.assumeVerified();
              outMod.print(out->os(), flags);
              out->os() << "\n";
              out->keep();
            } else {
              llvm::WithColor::error(llvm::errs())
                  << "Failed to open output file: " << filePath << "\n";
            }
            outMod.erase();
          }
        }
      }
    } else {
      // Full design space: split permutations into numParallelLoops ordered
      // buckets via a weak composition of numSpatialDims into numParallelLoops
      // parts (allowing zero-length buckets).
      llvm::SmallVector<llvm::SmallVector<int, 4>, 8> compositions;
      llvm::SmallVector<int, 4> curr;
      enumerateWeakCompositions(numSpatialDims, numParallelLoops, compositions, curr);
      for (const auto &perm : perms) {
        for (const auto &comp : compositions) {
          Plan plan(static_cast<size_t>(numParallelLoops));
          int offset = 0;
          for (int i = 0; i < numParallelLoops; ++i) {
            int len = comp[static_cast<size_t>(i)];
            plan[static_cast<size_t>(i)].reserve(len);
            for (int j = 0; j < len; ++j) {
              plan[static_cast<size_t>(i)].push_back(perm[static_cast<size_t>(offset + j)]);
            }
            offset += len;
          }
          std::string suffix = makePlanSuffix(plan, dims);
          func::FuncOp clone = cloneFunctionWithSuffix(*module, funcOp, suffix);
          (void)applyPlanVariant(clone, plan, dims);
          if (!outDirPath.empty()) {
            ModuleOp outMod = ModuleOp::create(UnknownLoc::get(&context));
            OpBuilder mb(&context);
            mb.setInsertionPointToStart(outMod.getBody());
            mb.clone(*clone.getOperation());
            std::string filePath = outDirPath + "/" + clone.getName().str() + ".mlir";
            if (auto out = mlir::openOutputFile(filePath)) {
              mlir::OpPrintingFlags flags;
              flags.assumeVerified();
              outMod.print(out->os(), flags);
              out->os() << "\n";
              out->keep();
            } else {
              llvm::WithColor::error(llvm::errs())
                  << "Failed to open output file: " << filePath << "\n";
            }
            outMod.erase();
          }
        }
      }
    }
  });

  // If no outdir given, honor --out single file or print to stdout.
  if (outDirPath.empty()) {
    if (!outFilePath.empty()) {
      auto out = mlir::openOutputFile(outFilePath);
      if (!out) {
        llvm::WithColor::error(llvm::errs())
            << "Failed to open output file: " << outFilePath << "\n";
        return 1;
      }
      llvm::raw_pwrite_stream &os = out->os();
      mlir::OpPrintingFlags flags;
      flags.assumeVerified();
      module->print(os, flags);
      os << "\n";
      out->keep();
    } else {
      mlir::OpPrintingFlags flags;
      flags.assumeVerified();
      module->print(llvm::outs(), flags);
      llvm::outs() << "\n";
    }
  }
  return 0;
}
