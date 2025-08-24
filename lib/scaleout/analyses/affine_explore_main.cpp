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

// Enumerate all compositions of S into P positive integers.
static void
enumerateWeakCompositions(int S, int P,
                          llvm::SmallVector<llvm::SmallVector<int, 4>, 8> &out,
                          llvm::SmallVector<int, 4> &current) {
  if (P == 1) {
    current.push_back(S);
    out.push_back(current);
    current.pop_back();
    return;
  }
  for (int first = 0; first <= S; ++first) {
    current.push_back(first);
    enumerateWeakCompositions(S - first, P - 1, out, current);
    current.pop_back();
  }
}

// Generate all permutations of indices 0..N-1.
static void
enumeratePermutations(int N,
                      llvm::SmallVector<llvm::SmallVector<int, 8>, 16> &perms) {
  llvm::SmallVector<int, 8> v;
  v.reserve(N);
  for (int i = 0; i < N; ++i)
    v.push_back(i);
  // Simple Heap's algorithm iterative implementation via std::next_permutation
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

// Apply one tiling assignment: for each parallel IV i, use sizes in
// perIvSizes[i] and order to build tile loops and rewrite accesses. This
// function modifies the cloned function in-place.
static LogicalResult
applyOneTilingVariant(func::FuncOp clonedFunc,
                      ArrayRef<llvm::SmallVector<int64_t, 4>> perIvSizes,
                      ArrayRef<llvm::SmallVector<Value, 4>> perIvDimValues) {
  affine::AffineParallelOp parOp = getOutermostParallel(clonedFunc);
  if (!parOp)
    return clonedFunc.emitError("expected outermost affine.parallel");

  Location loc = parOp.getLoc();
  OpBuilder b(parOp);
  MLIRContext *ctx = b.getContext();

  // Build flat lists of spatial dims: sizes (constants), owners (par iv index),
  // df ordinals
  (void)parOp; // silence unused warnings if invariants are not needed later
  SmallVector<int64_t, 8> flatSizes;
  SmallVector<int64_t, 8> flatOwners;
  SmallVector<int64_t, 8> flatOrdinals;
  DenseMap<Value, int64_t> dimToOrd;
  int64_t nextOrd = 0;
  clonedFunc.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "df.spatial_dim")
      dimToOrd[op->getResult(0)] = nextOrd++;
  });
  for (unsigned p = 0; p < perIvSizes.size(); ++p) {
    for (unsigned j = 0; j < perIvSizes[p].size(); ++j) {
      flatSizes.push_back(perIvSizes[p][j]);
      flatOwners.push_back(static_cast<int64_t>(p));
      auto it = dimToOrd.find(perIvDimValues[p][j]);
      flatOrdinals.push_back(it == dimToOrd.end() ? -1 : it->second);
    }
  }
  if (flatSizes.empty())
    return success();

  // Prepare mapping for wrapper
  SmallVector<Attribute, 8> elems;
  for (unsigned i = 0; i < flatSizes.size(); ++i) {
    NamedAttrList d;
    d.append("df_ordinal",
             IntegerAttr::get(IndexType::get(ctx), flatOrdinals[i]));
    d.append("par_iv", IntegerAttr::get(IndexType::get(ctx), flatOwners[i]));
    d.append("size", IntegerAttr::get(IndexType::get(ctx), flatSizes[i]));
    elems.push_back(DictionaryAttr::get(ctx, d));
  }
  OperationState wrapState(loc, "df.spatial_wrap");
  wrapState.addAttribute("mapping", ArrayAttr::get(ctx, elems));
  wrapState.addRegion();
  Operation *wrap = b.create(wrapState);
  Region &body = wrap->getRegion(0);
  body.push_back(new Block());
  OpBuilder wb(&body);
  wb.setInsertionPointToStart(&body.front());
  // Create the actual spatial affine.parallel inside wrapper
  SmallVector<arith::AtomicRMWKind, 0> reductions;
  auto spatialPar = wb.create<affine::AffineParallelOp>(loc, TypeRange{},
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

// Create human-readable suffix for a variant, e.g., __tile_p0[8,8]_p1[8]
static std::string
makeVariantSuffix(ArrayRef<llvm::SmallVector<int64_t, 4>> perIvSizes) {
  std::string s = std::string("__tile");
  for (size_t i = 0; i < perIvSizes.size(); ++i) {
    s += std::string("_p") + std::to_string(i) + '[';
    for (size_t j = 0; j < perIvSizes[i].size(); ++j) {
      s += std::to_string(perIvSizes[i][j]);
      if (j + 1 < perIvSizes[i].size())
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

    const int P = static_cast<int>(parOp.getIVs().size());
    const int S = static_cast<int>(dims.size());
    if (P <= 0 || S <= 0)
      return;
    if (S < P)
      return; // Not enough spatial dims to assign at least one per IV

    // Prepare size array and SSA values of spatial dims
    llvm::SmallVector<int64_t, 8> sizes;
    sizes.reserve(S);
    for (auto &d : dims)
      sizes.push_back(d.size);

    // All permutations of dims
    llvm::SmallVector<llvm::SmallVector<int, 8>, 16> perms;
    enumeratePermutations(S, perms);

    if (restrictDistribution) {
      // Assign all spatial dims to exactly one parallel IV (others get none)
      for (const auto &perm : perms) {
        for (int chosen = 0; chosen < P; ++chosen) {
          llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> perIvSizes(P);
          llvm::SmallVector<llvm::SmallVector<Value, 4>, 4> perIvDimValues(P);
          for (int j = 0; j < S; ++j) {
            int dimIdx = perm[j];
            perIvSizes[chosen].push_back(sizes[dimIdx]);
            perIvDimValues[chosen].push_back(dims[dimIdx].value);
          }
          std::string suffix = makeVariantSuffix(perIvSizes);
          func::FuncOp clone = cloneFunctionWithSuffix(*module, funcOp, suffix);
          (void)applyOneTilingVariant(clone, perIvSizes, perIvDimValues);
          if (!outDirPath.empty()) {
            ModuleOp outMod = ModuleOp::create(UnknownLoc::get(&context));
            OpBuilder mb(&context);
            mb.setInsertionPointToStart(outMod.getBody());
            mb.clone(*clone.getOperation());
            std::string filePath = outDirPath + "/" + clone.getName().str() + ".mlir";
            if (auto out = mlir::openOutputFile(filePath)) {
              outMod.print(out->os());
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
      // Full design space: distribute spatial dims across parallel IVs
      // using weak compositions to allow multiple dims per IV and allow zeros.
      llvm::SmallVector<llvm::SmallVector<int, 4>, 8> compositions;
      llvm::SmallVector<int, 4> curr;
      enumerateWeakCompositions(S, P, compositions, curr);
      for (const auto &perm : perms) {
        for (const auto &comp : compositions) {
          llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> perIvSizes(P);
          llvm::SmallVector<llvm::SmallVector<Value, 4>, 4> perIvDimValues(P);
          int offset = 0;
          for (int i = 0; i < P; ++i) {
            int len = comp[i];
            for (int j = 0; j < len; ++j) {
              int dimIdx = perm[offset + j];
              perIvSizes[i].push_back(sizes[dimIdx]);
              perIvDimValues[i].push_back(dims[dimIdx].value);
            }
            offset += len;
          }
          std::string suffix = makeVariantSuffix(perIvSizes);
          func::FuncOp clone = cloneFunctionWithSuffix(*module, funcOp, suffix);
          (void)applyOneTilingVariant(clone, perIvSizes, perIvDimValues);
          if (!outDirPath.empty()) {
            ModuleOp outMod = ModuleOp::create(UnknownLoc::get(&context));
            OpBuilder mb(&context);
            mb.setInsertionPointToStart(outMod.getBody());
            mb.clone(*clone.getOperation());
            std::string filePath = outDirPath + "/" + clone.getName().str() + ".mlir";
            if (auto out = mlir::openOutputFile(filePath)) {
              outMod.print(out->os());
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
      module->print(os);
      os << "\n";
      out->keep();
    } else {
      module->print(llvm::outs());
      llvm::outs() << "\n";
    }
  }
  return 0;
}
