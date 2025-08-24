#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
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
#include "mlir/IR/DialectRegistry.h"

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

// Compute ceilDiv(ub, C) using affine.apply with a constant divisor C.
static Value buildCeilDivByConst(OpBuilder &b, Location loc, Value ub,
                                 int64_t C) {
  assert(C > 0 && "tiling size must be positive");
  // Map: (s0) -> ((s0 + (C-1)) floordiv C)
  auto ctx = b.getContext();
  AffineExpr s0 = getAffineSymbolExpr(0, ctx);
  AffineExpr expr = (s0 + getAffineConstantExpr(C - 1, ctx)).floorDiv(C);
  AffineMap m = AffineMap::get(/*dims=*/0, /*syms=*/1, {expr}, ctx);
  SmallVector<OpFoldResult, 1> ops;
  ops.push_back(ub);
  return affine::makeComposedAffineApply(b, loc, m, ops).getResult();
}

// Build nested outer/inner affine.parallel loops that tile the original
// parallel op. The outer has one IV per original parallel IV. The inner has one
// IV per spatial dimension assigned across all original IVs (in order). The
// original loop body is moved into the inner loop.
// Returns the created outer and inner parallel ops and fills outerIVs and
// innerIVs vectors.
// Build composed index value: outer * prod + sum(inner_j * stride_j)
static Value buildComposedIndex(OpBuilder &b, Location loc, Value outerIv,
                                ArrayRef<Value> innerIvs,
                                ArrayRef<int64_t> sizes) {
  MLIRContext *ctx = b.getContext();
  const unsigned dims = 1 + innerIvs.size();
  SmallVector<AffineExpr, 8> dimExprs;
  for (unsigned i = 0; i < dims; ++i)
    dimExprs.push_back(getAffineDimExpr(i, ctx));
  int64_t prod = 1;
  for (int64_t s : sizes)
    prod *= s;
  AffineExpr expr = (prod == 1)
                        ? dimExprs[0]
                        : dimExprs[0] * getAffineConstantExpr(prod, ctx);
  int64_t stride = 1;
  for (unsigned j = 0; j < innerIvs.size(); ++j) {
    AffineExpr term =
        (stride == 1) ? dimExprs[1 + j]
                      : dimExprs[1 + j] * getAffineConstantExpr(stride, ctx);
    expr = expr + term;
    stride *= sizes[j];
  }
  AffineMap map = AffineMap::get(dims, 0, expr, ctx);
  SmallVector<OpFoldResult, 8> ops;
  ops.push_back(outerIv);
  for (Value v : innerIvs)
    ops.push_back(v);
  return affine::makeComposedAffineApply(b, loc, map, ops).getResult();
}

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
  const unsigned numPar = parOp.getIVs().size();
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
  SmallVector<AffineExpr, 8> lbExprs(flatSizes.size(),
                                     getAffineConstantExpr(0, ctx));
  SmallVector<AffineExpr, 8> ubExprs;
  for (int64_t s : flatSizes)
    ubExprs.push_back(getAffineConstantExpr(s, ctx));
  AffineMap lbMap = AffineMap::get(0, 0, lbExprs, ctx);
  AffineMap ubMap = AffineMap::get(0, 0, ubExprs, ctx);
  SmallVector<int64_t, 8> steps(flatSizes.size(), 1);
  SmallVector<arith::AtomicRMWKind, 0> reductions;
  auto spatialPar = wb.create<affine::AffineParallelOp>(
      loc, TypeRange{}, reductions, lbMap, ValueRange{}, ubMap, ValueRange{},
      steps);
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
  // Initialize context with a registry to ensure all dialects are properly
  // registered for custom printers/parsers.
  DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                  mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect, tmd::df::DataflowDialect>();
  MLIRContext context(registry);
  // Explicitly load dialects to enable custom printers instead of generic form.
  context.loadDialect<mlir::BuiltinDialect, mlir::func::FuncDialect,
                      mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                      mlir::memref::MemRefDialect, tmd::df::DataflowDialect>();
  // Avoid falling back to generic op form by disallowing unregistered dialects.
  context.allowUnregisteredDialects(false);
  // Ensure they are fully materialized in this context.
  (void)context.getOrLoadDialect<mlir::BuiltinDialect>();
  (void)context.getOrLoadDialect<mlir::func::FuncDialect>();
  (void)context.getOrLoadDialect<mlir::affine::AffineDialect>();
  (void)context.getOrLoadDialect<mlir::arith::ArithDialect>();
  (void)context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  (void)context.getOrLoadDialect<tmd::df::DataflowDialect>();

  llvm::SourceMgr sourceMgr;
  // CLI: [--restrict-distribution] <file>
  bool restrictDistribution = false;
  const char *filename = "-";
  std::string outFilePath;
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg == "--restrict-distribution") {
      restrictDistribution = true;
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

  // For each function: collect spatial dims and outermost parallel, enumerate
  // variants, clone and transform.
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
        }
      }
    }
  });

  // Print using simplified/custom assembly form for readability.
  OpPrintingFlags printFlags;
  // Prefer local scope names and avoid verbose generic form.
  printFlags.useLocalScope();
  // Elide huge elements attributes if present to keep output concise.
  printFlags.elideLargeElementsAttrs();
  if (!outFilePath.empty()) {
    auto out = mlir::openOutputFile(outFilePath);
    if (!out) {
      llvm::WithColor::error(llvm::errs())
          << "Failed to open output file: " << outFilePath << "\n";
      return 1;
    }
    llvm::raw_pwrite_stream &os = out->os();
    module->print(os, printFlags);
    os << "\n";
    out->keep();
  } else {
    module->print(llvm::outs(), printFlags);
    llvm::outs() << "\n";
  }
  return 0;
}
