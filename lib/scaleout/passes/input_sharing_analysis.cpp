#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace tmd_affine_analysis {

/**
 * Run a basic syntax check on the function. Currently acts as a stub that
 * returns success. Extend here if you want validation before analyses.
 */
LogicalResult runSyntaxCheck(func::FuncOp /*funcOp*/) { return success(); }

// Return the nearest enclosing affine.parallel op for the given op, or null.
static affine::AffineParallelOp getNearestEnclosingParallelOp(Operation *op) {
  Operation *parent = op->getParentOp();
  while (parent) {
    if (auto par = dyn_cast<affine::AffineParallelOp>(parent))
      return par;
    parent = parent->getParentOp();
  }
  return nullptr;
}

// Removed the old attribute-based independence annotation; we now only print
// a textual reuse report.

// Compute and emit reuse volume (0/1) for each affine.load under each enclosing
// affine.parallel for all movement directions formed by +1 along any subset of
// parallel IVs (excluding the all-zero move). Results are written to the given
// stream in a simple, stable text format.
// Collect enclosing affine.for IVs in lexical order from outermost to innermost
// starting above 'start' and stopping before reaching 'stopAt' (exclusive).
static SmallVector<Value, 4> collectEnclosingForIVs(Operation *start,
                                                    Operation *stopAt) {
  SmallVector<Value, 4> ivsReversed;
  Operation *parent = start->getParentOp();
  while (parent && parent != stopAt) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(parent))
      ivsReversed.push_back(forOp.getInductionVar());
    parent = parent->getParentOp();
  }
  SmallVector<Value, 4> ivs;
  ivs.assign(ivsReversed.rbegin(), ivsReversed.rend());
  return ivs;
}

// Collect all enclosing induction variables (affine.parallel IVs in order and
// affine.for IVs) from outermost to innermost.
static SmallVector<Value, 8> collectAllEnclosingIVs(Operation *start) {
  SmallVector<Operation *, 8> parents;
  for (Operation *p = start->getParentOp(); p; p = p->getParentOp())
    parents.push_back(p); // inner .. outer

  SmallVector<Value, 8> ivs;
  // iterate outer .. inner, preserving per-op IV order
  for (auto it = parents.rbegin(); it != parents.rend(); ++it) {
    Operation *parent = *it;
    if (auto par = dyn_cast<affine::AffineParallelOp>(parent)) {
      for (Value iv : par.getIVs())
        ivs.push_back(iv);
    } else if (auto forOp = dyn_cast<affine::AffineForOp>(parent)) {
      ivs.push_back(forOp.getInductionVar());
    }
  }
  return ivs;
}

// Simple integer rational number to perform exact RREF.
struct IntRational {
  int64_t num;
  int64_t den;
};

static int64_t gcdll(int64_t a, int64_t b) {
  if (a < 0)
    a = -a;
  if (b < 0)
    b = -b;
  while (b != 0) {
    int64_t t = a % b;
    a = b;
    b = t;
  }
  return a == 0 ? 1 : a;
}

static IntRational makeRat(int64_t n, int64_t d) {
  if (d < 0) {
    n = -n;
    d = -d;
  }
  if (n == 0)
    return {0, 1};
  int64_t g = gcdll(n, d);
  return {n / g, d / g};
}

static IntRational subRat(const IntRational &a, const IntRational &b) {
  return makeRat(a.num * b.den - b.num * a.den, a.den * b.den);
}
static IntRational mulRat(const IntRational &a, const IntRational &b) {
  return makeRat(a.num * b.num, a.den * b.den);
}
static IntRational divRat(const IntRational &a, const IntRational &b) {
  return makeRat(a.num * b.den, a.den * b.num);
}

// Compute RREF over rationals; returns pivot column per pivot row.
static SmallVector<int, 8>
rref(SmallVector<SmallVector<IntRational, 8>, 8> &M) {
  const int m = static_cast<int>(M.size());
  const int n = m ? static_cast<int>(M[0].size()) : 0;
  int r = 0;
  SmallVector<int, 8> pivotCol;
  for (int c = 0; c < n && r < m; ++c) {
    int pivot = -1;
    for (int i = r; i < m; ++i) {
      if (M[i][c].num != 0) {
        pivot = i;
        break;
      }
    }
    if (pivot < 0)
      continue;
    if (pivot != r)
      std::swap(M[pivot], M[r]);
    // Normalize row r
    IntRational lead = M[r][c];
    for (int j = c; j < n; ++j)
      M[r][j] = divRat(M[r][j], lead);
    // Eliminate other rows
    for (int i = 0; i < m; ++i) {
      if (i == r)
        continue;
      IntRational factor = M[i][c];
      if (factor.num == 0)
        continue;
      for (int j = c; j < n; ++j)
        M[i][j] = subRat(M[i][j], mulRat(factor, M[r][j]));
    }
    pivotCol.push_back(c);
    ++r;
  }
  return pivotCol;
}

// Compute primitive integer nullspace basis vectors of integer matrix A
// (m x n). Returns a list of integer vectors (size n).
static SmallVector<SmallVector<int64_t, 8>, 4> computePrimitiveIntegerNullspace(
    const SmallVector<SmallVector<int64_t, 8>, 8> &A) {
  const int m = static_cast<int>(A.size());
  const int n = m ? static_cast<int>(A[0].size()) : 0;
  SmallVector<SmallVector<IntRational, 8>, 8> M(m,
                                                SmallVector<IntRational, 8>(n));
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      M[i][j] = makeRat(A[i][j], 1);
  SmallVector<int, 8> pivots = rref(M);
  llvm::SmallBitVector isPivot(n, false);
  for (int c : pivots)
    isPivot.set(c);
  SmallVector<int, 8> freeCols;
  for (int c = 0; c < n; ++c)
    if (!isPivot.test(c))
      freeCols.push_back(c);
  SmallVector<SmallVector<int64_t, 8>, 4> basis;
  if (freeCols.empty())
    return basis; // trivial nullspace

  // Map from pivot row to pivot column index
  SmallVector<int, 8> pivotRowToCol;
  pivotRowToCol.assign(pivots.begin(), pivots.end());

  for (int fCol : freeCols) {
    SmallVector<IntRational, 8> x(n, makeRat(0, 1));
    x[fCol] = makeRat(1, 1);
    // For each pivot row r, pivot column pc
    for (int r = 0; r < static_cast<int>(pivotRowToCol.size()); ++r) {
      int pc = pivotRowToCol[r];
      // x_pc = - M[r][fCol]
      x[pc] = subRat(makeRat(0, 1), M[r][fCol]);
    }
    // Convert rational vector to primitive integer vector
    int64_t lcmDen = 1;
    for (int j = 0; j < n; ++j) {
      int64_t den = x[j].den;
      int64_t g = gcdll(lcmDen, den);
      lcmDen = (lcmDen / g) * den;
    }
    SmallVector<int64_t, 8> vec(n, 0);
    for (int j = 0; j < n; ++j)
      vec[j] = x[j].num * (lcmDen / x[j].den);
    // Normalize by gcd
    int64_t g = 0;
    for (int j = 0; j < n; ++j)
      g = gcdll(g, vec[j]);
    if (g == 0)
      g = 1;
    for (int j = 0; j < n; ++j)
      vec[j] /= g;
    // Normalize sign: make first nonzero positive
    for (int j = 0; j < n; ++j) {
      if (vec[j] != 0) {
        if (vec[j] < 0) {
          for (int k = 0; k < n; ++k)
            vec[k] = -vec[k];
        }
        break;
      }
    }
    basis.push_back(std::move(vec));
  }
  return basis;
}

// Attach primitive reuse vectors attribute to each affine.load.
void attachPrimitiveReuseVectors(func::FuncOp funcOp) {
  MLIRContext *ctx = funcOp.getContext();
  funcOp.walk([&](affine::AffineLoadOp loadOp) {
    // Collect iterators
    SmallVector<Value, 8> iterIVs =
        collectAllEnclosingIVs(loadOp.getOperation());
    if (iterIVs.empty())
      return;

    // Compose and canonicalize map and operands
    AffineMap map = loadOp.getAffineMap();
    SmallVector<Value, 8> operands(loadOp.getMapOperands());
    mlir::affine::fullyComposeAffineMapAndOperands(&map, &operands);
    mlir::affine::canonicalizeMapAndOperands(&map, &operands);

    const unsigned numDims = map.getNumDims();
    const unsigned numSyms = map.getNumSymbols();

    // Map each map dim operand to iterator index or -1
    llvm::DenseMap<Value, unsigned> iterIndexOf;
    for (unsigned i = 0; i < iterIVs.size(); ++i)
      iterIndexOf[iterIVs[i]] = i;
    SmallVector<int, 8> dimToIterIdx(numDims, -1);
    for (unsigned d = 0; d < numDims; ++d) {
      auto it = iterIndexOf.find(operands[d]);
      if (it != iterIndexOf.end())
        dimToIterIdx[d] = static_cast<int>(it->second);
    }

    // Build coefficient matrix A (rows = results, cols = numIters).
    SmallVector<SmallVector<int64_t, 8>, 8> A;
    A.resize(map.getNumResults());
    for (unsigned r = 0; r < map.getNumResults(); ++r) {
      A[r].assign(iterIVs.size(), 0);
      AffineExpr expr = map.getResult(r);
      AffineExpr orig = simplifyAffineExpr(expr, numDims, numSyms);
      // Prebuild identity sym replacements
      SmallVector<AffineExpr, 8> symRepls;
      symRepls.reserve(numSyms);
      for (unsigned s = 0; s < numSyms; ++s)
        symRepls.push_back(getAffineSymbolExpr(s, ctx));
      for (unsigned j = 0; j < iterIVs.size(); ++j) {
        // Build dim replacements: increment dims that correspond to iter j by 1
        SmallVector<AffineExpr, 8> dimRepls;
        dimRepls.reserve(numDims);
        for (unsigned d = 0; d < numDims; ++d) {
          AffineExpr dExpr = getAffineDimExpr(d, ctx);
          int iterIdx = dimToIterIdx[d];
          if (iterIdx == static_cast<int>(j))
            dimRepls.push_back(dExpr + getAffineConstantExpr(1, ctx));
          else
            dimRepls.push_back(dExpr);
        }
        AffineExpr shifted = simplifyAffineExpr(
            expr.replaceDimsAndSymbols(dimRepls, symRepls), numDims, numSyms);
        AffineExpr delta = simplifyAffineExpr(shifted - orig, numDims, numSyms);
        if (auto c = dyn_cast<AffineConstantExpr>(delta))
          A[r][j] = static_cast<int64_t>(c.getValue());
        else
          A[r][j] = 0; // Non-constant delta: treat as zero (conservative)
      }
    }

    SmallVector<SmallVector<int64_t, 8>, 4> basis =
        computePrimitiveIntegerNullspace(A);

    // Expand to include both directions (v and -v) for each primitive vector.
    SmallVector<SmallVector<int64_t, 8>, 4> expanded;
    expanded.reserve(basis.size() * 2);
    for (auto &v : basis) {
      bool allZero = true;
      for (int64_t x : v)
        if (x != 0) {
          allZero = false;
          break;
        }
      if (allZero)
        continue;
      expanded.push_back(v);
      SmallVector<int64_t, 8> neg(v.begin(), v.end());
      for (int64_t &x : neg)
        x = -x;
      expanded.push_back(std::move(neg));
    }

    // Convert to attribute: array<array<index>> named
    // tmd.reuse.primitive_vectors
    SmallVector<Attribute, 4> vecAttrs;
    vecAttrs.reserve(expanded.size());
    for (auto &vec : expanded) {
      SmallVector<Attribute, 8> ints;
      ints.reserve(vec.size());
      for (int64_t v : vec)
        ints.push_back(IntegerAttr::get(IndexType::get(ctx), v));
      vecAttrs.push_back(ArrayAttr::get(ctx, ints));
    }
    loadOp->setAttr("tmd.reuse.primitive_vectors",
                    ArrayAttr::get(ctx, vecAttrs));
  });
}

void runInputSharingReuseAnalysis(func::FuncOp funcOp, llvm::raw_ostream &os) {
  MLIRContext *ctx = funcOp.getContext();

  funcOp.walk([&](affine::AffineParallelOp parOp) {
    SmallVector<Value, 4> parIVs;
    for (Value iv : parOp.getIVs())
      parIVs.push_back(iv);

    const unsigned numPar = parIVs.size();
    if (numPar == 0)
      return;

    os << "parallel ";
    parOp.getLoc().print(os);
    os << "\n";

    // Map parallel IV Value to its index 0..P-1
    llvm::DenseMap<Value, unsigned> parIndexOf;
    for (unsigned i = 0; i < numPar; ++i)
      parIndexOf[parIVs[i]] = i;

    // Collect all loads nested within this parallel op.
    SmallVector<affine::AffineLoadOp, 8> loads;
    parOp.getOperation()->walk([&](affine::AffineLoadOp loadOp) {
      // Attribute this load only to its nearest enclosing parallel.
      auto nearest = getNearestEnclosingParallelOp(loadOp.getOperation());
      if (nearest && nearest.getOperation() == parOp.getOperation())
        loads.push_back(loadOp);
    });

    if (loads.empty()) {
      os << "  (no loads)\n";
      return;
    }

    // Build union iterator set: [parallel IVs] + [all enclosing for IVs across
    // loads], preserving lexical order.
    SmallVector<Value, 8> iterIVs;
    iterIVs.append(parIVs.begin(), parIVs.end());
    llvm::SmallPtrSet<Value, 16> seen(iterIVs.begin(), iterIVs.end());
    for (affine::AffineLoadOp loadOp : loads) {
      SmallVector<Value, 4> forIVs =
          collectEnclosingForIVs(loadOp.getOperation(), parOp.getOperation());
      for (Value v : forIVs) {
        if (seen.insert(v).second)
          iterIVs.push_back(v);
      }
    }

    // Labels for pretty printing: p<i> for parallel IVs, f<j> for for-IVs.
    llvm::DenseMap<Value, std::string> ivToLabel;
    SmallVector<std::string, 8> iterLabels;
    iterLabels.reserve(iterIVs.size());
    for (unsigned i = 0; i < numPar; ++i) {
      std::string lab = std::string("p") + std::to_string(i);
      ivToLabel[parIVs[i]] = lab;
    }
    for (unsigned i = numPar; i < iterIVs.size(); ++i) {
      std::string lab = std::string("f") + std::to_string(i - numPar);
      ivToLabel[iterIVs[i]] = lab;
    }
    for (Value v : iterIVs)
      iterLabels.push_back(ivToLabel.lookup(v));

    // Print iterator label summary for this parallel region.
    os << "  iters=(";
    for (size_t i = 0; i < iterLabels.size(); ++i) {
      os << iterLabels[i];
      if (i + 1 < iterLabels.size())
        os << ", ";
    }
    os << ") [p:parallel, f:for]\n";

    // Precompute per-load canonicalized map and mapping from map dims to
    // iterator indices.
    struct LoadInfo {
      affine::AffineLoadOp load;
      AffineMap map;
      SmallVector<int, 8> dimToIterIdx;
      unsigned numDims;
      unsigned numSyms;
    };
    SmallVector<LoadInfo, 8> loadInfos;
    loadInfos.reserve(loads.size());

    llvm::DenseMap<Value, unsigned> iterIndexOf;
    for (unsigned i = 0; i < iterIVs.size(); ++i)
      iterIndexOf[iterIVs[i]] = i;

    for (affine::AffineLoadOp loadOp : loads) {
      AffineMap map = loadOp.getAffineMap();
      SmallVector<Value, 8> operands(loadOp.getMapOperands());
      mlir::affine::fullyComposeAffineMapAndOperands(&map, &operands);
      mlir::affine::canonicalizeMapAndOperands(&map, &operands);

      const unsigned numDims = map.getNumDims();
      const unsigned numSyms = map.getNumSymbols();

      SmallVector<int, 8> dimToIterIdx(numDims, -1);
      for (unsigned d = 0; d < numDims; ++d) {
        auto it = iterIndexOf.find(operands[d]);
        if (it != iterIndexOf.end())
          dimToIterIdx[d] = static_cast<int>(it->second);
      }

      loadInfos.push_back(
          LoadInfo{loadOp, map, std::move(dimToIterIdx), numDims, numSyms});
    }

    const unsigned numIters = iterIVs.size();
    const unsigned maxMask = (numIters >= 32) ? 0u : ((1u << numIters) - 1u);
    if (numIters >= 32) {
      os << "    [skip: too many iterators=" << numIters << "]\n";
      return;
    }

    for (unsigned mask = 1; mask <= maxMask; ++mask) {
      // Direction header like [+m +n +k]
      os << "  dir [";
      bool first = true;
      for (unsigned i = 0; i < numIters; ++i) {
        if ((mask >> i) & 1u) {
          if (!first)
            os << ' ';
          os << '+' << iterLabels[i];
          first = false;
        }
      }
      os << "]\n";

      for (LoadInfo &li : loadInfos) {
        // Build dim replacements for this mask: d_i -> d_i + (mask has iterIdx)
        SmallVector<AffineExpr, 8> dimRepls;
        dimRepls.reserve(li.numDims);
        for (unsigned d = 0; d < li.numDims; ++d) {
          AffineExpr dExpr = getAffineDimExpr(d, ctx);
          int iterIdx = li.dimToIterIdx[d];
          if (iterIdx >= 0 && ((mask >> static_cast<unsigned>(iterIdx)) & 1u))
            dimRepls.push_back(dExpr + getAffineConstantExpr(1, ctx));
          else
            dimRepls.push_back(dExpr);
        }

        // Identity symbol replacements sized for this map.
        SmallVector<AffineExpr, 8> symRepls;
        symRepls.reserve(li.numSyms);
        for (unsigned s = 0; s < li.numSyms; ++s)
          symRepls.push_back(getAffineSymbolExpr(s, ctx));

        bool reused = true;
        for (AffineExpr expr : li.map.getResults()) {
          AffineExpr orig = simplifyAffineExpr(expr, li.numDims, li.numSyms);
          AffineExpr shifted =
              simplifyAffineExpr(expr.replaceDimsAndSymbols(dimRepls, symRepls),
                                 li.numDims, li.numSyms);
          if (shifted != orig) {
            reused = false;
            break;
          }
        }

        // Make a compact, readable label for the loaded memref value.
        auto memrefVal = li.load.getMemRef();
        std::string memrefLabel;
        if (auto barg = llvm::dyn_cast<BlockArgument>(memrefVal)) {
          memrefLabel =
              std::string("arg") + std::to_string(barg.getArgNumber());
        } else if (Operation *def = memrefVal.getDefiningOp()) {
          memrefLabel = def->getName().getStringRef().str();
        } else {
          memrefLabel = "val";
        }

        os << "    load " << memrefLabel;
        os << " at ";
        li.load.getLoc().print(os);
        os << " reuse=" << (reused ? 1 : 0) << "\n";
      }
    }
  });
}

} // namespace tmd_affine_analysis
