/**
 * @file explore_alloc_copy_mapping.cpp
 * @brief Implementation for alloc/copy mapping exploration.
 * @details
 * Dataflow assumptions
 * - Exactly one `df.memory` declares the local memory pool (`memory_name`).
 * - Optional `df.interconnects` define broadcast-capable links between tiles.
 *   We classify simple maps as horizontal (x) or vertical (y) heuristically.
 * - Prior reuse analysis must have attached `tmd.reuse` on relevant
 *   `memref.reinterpret_cast` ops to expose spatial total-reuse.
 *
 * Behavior
 * - Always annotate `memref.alloc` with `tmd.alloc = { local=true,
 *   memory_name=... }`.
 * - For each `memref.copy`, build a candidate set:
 *   - `kind=mem`: local memory copy into the single `df.memory`.
 *   - `kind=broadcast`, `dim=x|y`, `interconnect_name=...` for each eligible
 *     interconnect along a dimension with total-reuse.
 * - Analysis-only mode attaches `tmd.copy.candidates` per site.
 * - Enumeration mode clones per cross-product of site candidates and attaches
 *   `tmd.copy.choice` in each clone; function names are suffixed accordingly.
 */

#include "explore_alloc_copy_mapping.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

#include "DataflowDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "DataflowTypes.h.inc"
#define GET_OP_CLASSES
#include "DataflowOps.h.inc"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/WithColor.h"

using namespace mlir;

namespace {

struct DFContext {
  std::string singleMemoryName;
  // Map dimension name (e.g., "x") to list of interconnect value + name.
  llvm::SmallVector<std::pair<Value, std::string>, 4> xICs;
  llvm::SmallVector<std::pair<Value, std::string>, 4> yICs;
  llvm::SmallVector<std::string, 4> spatialDimNames;
};

static std::string getStringOr(mlir::StringAttr a, StringRef fallback) {
  return a ? a.getValue().str() : fallback.str();
}

static bool collectDFContext(ModuleOp module, DFContext &out) {
  // Find single df.memory and record its name.
  tmd::df::MemoryOp singleMem;
  module.walk([&](tmd::df::MemoryOp m) {
    if (!singleMem)
      singleMem = m;
  });
  if (!singleMem) {
    llvm::WithColor::error(llvm::errs()) << "No df.memory found in DF module\n";
    return false;
  }
  out.singleMemoryName = getStringOr(singleMem.getLabelAttr(), "mem0");

  // Collect spatial dim names.
  module.walk([&](tmd::df::SpatialDimOp sd) {
    out.spatialDimNames.push_back(getStringOr(sd.getNameAttr(), "dim"));
  });

  // Collect memory-to-memory interconnects and classify by simple map shape.
  module.walk([&](tmd::df::InterconnectsOp ic) {
    // Only consider memory-to-memory ICs.
    if (!llvm::isa<tmd::df::MemoryHandleType>(ic.getSource().getType()) ||
        !llvm::isa<tmd::df::MemoryHandleType>(ic.getTarget().getType()))
      return;
    // Heuristic: if map is (d0+1,d1) treat as x; if (d0,d1+1) treat as y.
    AffineMap map = ic.getMap();
    if (!map || map.getNumResults() < 1)
      return;
    auto results = map.getResults();
    auto classifyDim = [&](AffineExpr e0, AffineExpr e1) -> char {
      // d0 + c, d1
      if (auto sum = llvm::dyn_cast<AffineBinaryOpExpr>(e0)) {
        if (sum.getKind() == AffineExprKind::Add) {
          if (llvm::isa<AffineDimExpr>(sum.getLHS()) &&
              llvm::isa<AffineConstantExpr>(sum.getRHS()) &&
              llvm::isa<AffineDimExpr>(e1))
            return 'x';
        }
      }
      // d0, d1 + c
      if (auto sum = llvm::dyn_cast<AffineBinaryOpExpr>(e1)) {
        if (sum.getKind() == AffineExprKind::Add) {
          if (llvm::isa<AffineDimExpr>(sum.getLHS()) &&
              llvm::isa<AffineConstantExpr>(sum.getRHS()) &&
              llvm::isa<AffineDimExpr>(e0))
            return 'y';
        }
      }
      return 0;
    };
    char cls = 0;
    if (results.size() >= 2)
      cls = classifyDim(results[0], results[1]);
    if (!cls)
      return;
    std::string name = "ic";
    if (auto sym = ic->getAttrOfType<StringAttr>("sym_name"))
      name = sym.getValue().str();
    if (cls == 'x')
      out.xICs.emplace_back(ic.getResult(), name);
    else if (cls == 'y')
      out.yICs.emplace_back(ic.getResult(), name);
  });

  return true;
}

static bool hasSpatialTotalReuse(DictionaryAttr reuse, StringRef dimName) {
  if (!reuse)
    return false;
  auto spatialAny = reuse.get("spatial");
  auto spatialArr =
      spatialAny ? llvm::dyn_cast<ArrayAttr>(spatialAny) : ArrayAttr();
  if (!spatialArr)
    return false;
  for (Attribute a : spatialArr) {
    auto entry = llvm::dyn_cast<DictionaryAttr>(a);
    if (!entry)
      continue;
    auto mappedAny = entry.get("mapped_to");
    auto rtAny = entry.get("reuse_type");
    auto mapped =
        mappedAny ? llvm::dyn_cast<StringAttr>(mappedAny) : StringAttr();
    auto rt = rtAny ? llvm::dyn_cast<StringAttr>(rtAny) : StringAttr();
    if (!mapped || !rt)
      continue;
    if (mapped.getValue() == dimName && rt.getValue() == "total_reuse")
      return true;
  }
  return false;
}

struct ExploreAllocCopyMappingPass
    : public PassWrapper<ExploreAllocCopyMappingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExploreAllocCopyMappingPass)

  ExploreAllocCopyMappingPass() = default;
  ExploreAllocCopyMappingPass(bool analysisOnly) : analysisOnly(analysisOnly) {}

  StringRef getArgument() const override {
    return "tmd-explore-alloc-copy-mapping";
  }
  StringRef getDescription() const override {
    return "Enumerate mapping choices for memref.alloc/copy using DF + reuse";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Collect DF context from the same module (DF ops are expected at top).
    DFContext df;
    if (!collectDFContext(module, df)) {
      signalPassFailure();
      return;
    }

    MLIRContext *ctx = module.getContext();

    // Helper to annotate memref.alloc.
    auto annotateAlloc = [&](memref::AllocOp alloc) {
      NamedAttrList nl;
      nl.append("local", BoolAttr::get(ctx, true));
      nl.append("memory_name", StringAttr::get(ctx, df.singleMemoryName));
      alloc->setAttr("tmd.alloc", DictionaryAttr::get(ctx, nl));
    };

    // Build list of target functions (all functions in the module).
    llvm::SmallVector<func::FuncOp> funcs;
    for (func::FuncOp f : module.getOps<func::FuncOp>())
      funcs.push_back(f);

    // Analysis-only: attach candidates in-place and annotate allocs.
    if (analysisOnly) {
      for (func::FuncOp f : funcs) {
        f.walk([&](memref::AllocOp a) { annotateAlloc(a); });
        f.walk([&](memref::CopyOp cpy) {
          SmallVector<Attribute> candidates;
          // Always include memory copy.
          {
            NamedAttrList m;
            m.append("kind", StringAttr::get(ctx, "mem"));
            m.append("memory_name", StringAttr::get(ctx, df.singleMemoryName));
            candidates.push_back(DictionaryAttr::get(ctx, m));
          }
          // Try to find source reinterpret_cast and its reuse.
          DictionaryAttr reuse;
          if (auto srcDef = cpy.getSource().getDefiningOp()) {
            if (auto rc = dyn_cast<memref::ReinterpretCastOp>(srcDef))
              reuse = rc->getAttrOfType<DictionaryAttr>("tmd.reuse");
          }
          // Broadcast candidates for x/y when reuse is total.
          if (hasSpatialTotalReuse(reuse, "x")) {
            for (auto &p : df.xICs) {
              NamedAttrList b;
              b.append("kind", StringAttr::get(ctx, "broadcast"));
              b.append("dim", StringAttr::get(ctx, "x"));
              b.append("interconnect_name", StringAttr::get(ctx, p.second));
              candidates.push_back(DictionaryAttr::get(ctx, b));
            }
          }
          if (hasSpatialTotalReuse(reuse, "y")) {
            for (auto &p : df.yICs) {
              NamedAttrList b;
              b.append("kind", StringAttr::get(ctx, "broadcast"));
              b.append("dim", StringAttr::get(ctx, "y"));
              b.append("interconnect_name", StringAttr::get(ctx, p.second));
              candidates.push_back(DictionaryAttr::get(ctx, b));
            }
          }
          cpy->setAttr("tmd.copy.candidates", ArrayAttr::get(ctx, candidates));
        });
      }
      return;
    }

    // Enumeration mode: clone per combination of choices (no limit).

    OpBuilder moduleBuilder(module.getBodyRegion());
    OwningOpRef<ModuleOp> out = ModuleOp::create(module.getLoc());
    OpBuilder outBuilder(out->getBodyRegion());
    llvm::SmallVector<func::FuncOp> originals(funcs.begin(), funcs.end());

    for (func::FuncOp f : originals) {
      // Gather copy sites and per-site candidate sets.
      SmallVector<memref::CopyOp> copies;
      SmallVector<SmallVector<DictionaryAttr>> perCopyCands;

      f.walk([&](memref::AllocOp a) { annotateAlloc(a); });

      f.walk([&](memref::CopyOp cpy) {
        copies.push_back(cpy);
        SmallVector<DictionaryAttr> cand;
        // Always include memory copy.
        {
          NamedAttrList m;
          m.append("kind", StringAttr::get(ctx, "mem"));
          m.append("memory_name", StringAttr::get(ctx, df.singleMemoryName));
          cand.push_back(DictionaryAttr::get(ctx, m));
        }
        DictionaryAttr reuse;
        if (auto srcDef = cpy.getSource().getDefiningOp()) {
          if (auto rc = dyn_cast<memref::ReinterpretCastOp>(srcDef))
            reuse = rc->getAttrOfType<DictionaryAttr>("tmd.reuse");
        }
        if (hasSpatialTotalReuse(reuse, "x")) {
          for (auto &p : df.xICs) {
            NamedAttrList b;
            b.append("kind", StringAttr::get(ctx, "broadcast"));
            b.append("dim", StringAttr::get(ctx, "x"));
            b.append("interconnect_name", StringAttr::get(ctx, p.second));
            cand.push_back(DictionaryAttr::get(ctx, b));
          }
        }
        if (hasSpatialTotalReuse(reuse, "y")) {
          for (auto &p : df.yICs) {
            NamedAttrList b;
            b.append("kind", StringAttr::get(ctx, "broadcast"));
            b.append("dim", StringAttr::get(ctx, "y"));
            b.append("interconnect_name", StringAttr::get(ctx, p.second));
            cand.push_back(DictionaryAttr::get(ctx, b));
          }
        }
        perCopyCands.push_back(std::move(cand));
      });

      // If no copies, just clone once to keep behavior consistent.
      if (copies.empty())
        perCopyCands.push_back({});

      // Enumerate cross product via mixed radix counters.
      SmallVector<size_t> idx(perCopyCands.size(), 0);
      auto bump = [&]() -> bool {
        for (size_t i = 0; i < idx.size(); ++i) {
          if (perCopyCands[i].empty())
            continue;
          idx[i]++;
          if (idx[i] < perCopyCands[i].size())
            return true;
          idx[i] = 0;
        }
        return false;
      };

      do {
        // Clone function.
        IRMapping map;
        func::FuncOp clone = cast<func::FuncOp>(outBuilder.clone(*f, map));
        // Re-find copies in the clone in the same order.
        SmallVector<memref::CopyOp> cloneCopies;
        clone.walk([&](memref::CopyOp c) { cloneCopies.push_back(c); });
        // Annotate allocs again in clone (attributes don't transfer to new ops
        // until we set them here because we annotated originals only).
        clone.walk([&](memref::AllocOp a) { annotateAlloc(a); });

        // Attach choices and build suffix.
        std::string suffix;
        for (size_t i = 0, k = 0;
             i < perCopyCands.size() && k < cloneCopies.size(); ++i) {
          if (perCopyCands[i].empty())
            continue;
          DictionaryAttr choice = perCopyCands[i][idx[i]];
          cloneCopies[k]->setAttr("tmd.copy.choice", choice);
          // Suffix token
          auto kindAny = choice.get("kind");
          auto kind =
              kindAny ? llvm::dyn_cast<StringAttr>(kindAny) : StringAttr();
          if (!suffix.empty())
            suffix += "_";
          if (kind && kind.getValue() == "mem") {
            suffix += ("c" + std::to_string(i) + "mem");
          } else if (kind && kind.getValue() == "broadcast") {
            auto dimAny = choice.get("dim");
            auto dim =
                dimAny ? llvm::dyn_cast<StringAttr>(dimAny) : StringAttr();
            std::string tag = "b";
            if (dim && dim.getValue() == "x")
              tag += "x";
            else if (dim && dim.getValue() == "y")
              tag += "y";
            else
              tag += "d";
            suffix += ("c" + std::to_string(i) + tag);
          } else {
            suffix += ("c" + std::to_string(i) + "unk");
          }
          ++k;
        }
        if (!suffix.empty())
          clone.setName((f.getSymName().str() + "__" + suffix).c_str());
      } while (bump());
    }

    // Drop all non-DF top-level ops (old functions etc.), keep DF ops only.
    SmallVector<Operation *, 16> toErase;
    for (Operation &op : *module.getBody()) {
      Dialect *dialect = op.getDialect();
      if (dialect && dialect->getNamespace() == StringRef("df"))
        continue; // keep df ops at the beginning
      toErase.push_back(&op);
    }
    for (auto it = toErase.rbegin(); it != toErase.rend(); ++it)
      (*it)->erase();

    // Materialize all clones into the original module after DF ops.
    for (Operation &op : *out->getBody()) {
      IRMapping m;
      moduleBuilder.clone(op, m);
    }
  }

  bool analysisOnly = false;
};

} // namespace

std::unique_ptr<mlir::Pass>
tmd::passes::createExploreAllocCopyMappingPass(bool analysisOnly) {
  return std::make_unique<ExploreAllocCopyMappingPass>(analysisOnly);
}

void tmd::passes::registerExploreAllocCopyMappingPass() {
  PassRegistration<ExploreAllocCopyMappingPass>();
}
