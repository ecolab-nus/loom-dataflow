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

// (deprecated helper removed)

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
