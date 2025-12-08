/**
 * @file explore_alloc_copy_mapping.cpp
 * @brief Implementation for alloc/copy mapping exploration.
 * @details
 * Dataflow assumptions
 * - Exactly one `df.memory` declares the local memory pool (`memory_name`).
 * - Optional `df.interconnects` define broadcast-capable links between tiles.
 *   We classify simple maps as horizontal (x) or vertical (y) heuristically.
 * - Prior reuse analysis must have attached `loom.reuse` on relevant
 *   `memref.reinterpret_cast` ops to expose spatial total-reuse.
 *
 * Behavior
 * - Always annotate `memref.alloc` with `loom.alloc = { local=true,
 *   memory_name=..., size=(bytes copied) }`.
 * - For each `memref.copy`, build a candidate set:
 *   - `kind=mem`: local memory copy into the single `df.memory`.
 *   - `kind=broadcast`, `dim=x|y`, `interconnect_name=...` for each eligible
 *     interconnect along a dimension with total-reuse.
 * - Analysis-only mode attaches `loom.copy.candidates` per site.
 * - Enumeration mode clones per cross-product of site candidates and attaches
 *   `loom.copy.choice` in each clone; function names are suffixed accordingly.
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
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/WithColor.h"
#include <optional>

using namespace mlir;

namespace {

struct DFContext {
  std::string singleMemoryName;
  // Map from dimension name (inferred from df.spatial_dim operands) to the
  // list of (interconnect handle value, interconnect label).
  llvm::StringMap<llvm::SmallVector<std::pair<Value, std::string>, 4>>
      icByDimName;
  llvm::SmallVector<std::string, 4> spatialDimNames;
};

static std::string getStringOr(mlir::StringAttr a, StringRef fallback) {
  return a ? a.getValue().str() : fallback.str();
}

static bool collectDFContext(ModuleOp module, DFContext &out) {
  // Find single df.memory and record its name.
  loom::df::MemoryOp singleMem;
  module.walk([&](loom::df::MemoryOp m) {
    if (!singleMem)
      singleMem = m;
  });
  if (!singleMem) {
    llvm::WithColor::error(llvm::errs()) << "No df.memory found in DF module\n";
    return false;
  }
  out.singleMemoryName = getStringOr(singleMem.getLabelAttr(), "mem0");

  // Collect spatial dim names.
  module.walk([&](loom::df::SpatialDimOp sd) {
    out.spatialDimNames.push_back(getStringOr(sd.getNameAttr(), "dim"));
  });

  // Collect memory-to-memory interconnects and classify by simple map shape.
  module.walk([&](loom::df::InterconnectsOp ic) {
    // Only consider memory-to-memory ICs.
    if (!llvm::isa<loom::df::MemoryHandleType>(ic.getSource().getType()) ||
        !llvm::isa<loom::df::MemoryHandleType>(ic.getTarget().getType()))
      return;
    // Heuristic: strictly accept +1 on exactly one axis.
    AffineMap map = ic.getMap();
    if (!map || map.getNumResults() < 1)
      return;
    auto results = map.getResults();
    auto isDim = [](AffineExpr e, unsigned p) -> bool {
      if (auto d = llvm::dyn_cast<AffineDimExpr>(e))
        return d.getPosition() == p;
      return false;
    };
    auto isDimPlusOne = [](AffineExpr e, unsigned p) -> bool {
      // Unwrap mod operations: ((d0 + 1) mod N) -> (d0 + 1)
      if (auto mod = llvm::dyn_cast<AffineBinaryOpExpr>(e)) {
        if (mod.getKind() == AffineExprKind::Mod)
          e = mod.getLHS();
      }
      
      // Check for (dim + 1) or (1 + dim) pattern
      auto add = llvm::dyn_cast<AffineBinaryOpExpr>(e);
      if (!add || add.getKind() != AffineExprKind::Add)
        return false;
      
      auto checkPair = [p](AffineExpr dimExpr, AffineExpr constExpr) -> bool {
        auto dim = llvm::dyn_cast<AffineDimExpr>(dimExpr);
        auto c = llvm::dyn_cast<AffineConstantExpr>(constExpr);
        return dim && dim.getPosition() == p && c && c.getValue() == 1;
      };
      
      return checkPair(add.getLHS(), add.getRHS()) ||
             checkPair(add.getRHS(), add.getLHS());
    };

    // Determine axis that advances by exactly +1.
    std::optional<unsigned> axis;
    unsigned numDims = map.getNumDims();
    if (numDims == 1 && results.size() >= 1 && isDimPlusOne(results[0], 0)) {
      axis = 0u;
    } else if (numDims >= 2 && results.size() >= 2) {
      if (isDimPlusOne(results[0], 0) && isDim(results[1], 1)) {
        axis = 0u;
      } else if (isDim(results[0], 0) && isDimPlusOne(results[1], 1)) {
        axis = 1u;
      }
    }
    if (!axis)
      return;
    
    // Get interconnect name (label or symbol name for backward compatibility).
    std::string name = "ic";
    if (auto label = ic.getLabelAttr()) {
      name = label.getValue().str();
    } else if (auto sym = ic->getAttrOfType<StringAttr>("sym_name")) {
      name = sym.getValue().str();
    }
    
    // Infer dimension name from source memory's scaleout.
    std::string dimName = "d" + std::to_string(*axis);
    Value source = ic.getSource();
    if (auto mem = source.getDefiningOp<loom::df::MemoryOp>()) {
      auto scaleout = mem.getScaleout();
      if (*axis < scaleout.size()) {
        if (auto sd = scaleout[*axis].getDefiningOp<loom::df::SpatialDimOp>()) {
          dimName = getStringOr(sd.getNameAttr(), dimName);
        }
      }
    }
    out.icByDimName[dimName].emplace_back(ic.getResult(), name);
  });

  return true;
}

static bool hasSpatialTotalReuse(DictionaryAttr reuse, StringRef dimName) {
  if (!reuse)
    return false;
  
  auto spatialArr = reuse.getAs<ArrayAttr>("spatial");
  if (!spatialArr)
    return false;
  
  for (Attribute a : spatialArr) {
    auto entry = llvm::dyn_cast<DictionaryAttr>(a);
    if (!entry)
      continue;
    
    auto mapped = entry.getAs<StringAttr>("mapped_to");
    auto rt = entry.getAs<StringAttr>("reuse_type");
    
    if (mapped && rt && mapped.getValue() == dimName && 
        rt.getValue() == "total_reuse")
      return true;
  }
  return false;
}

/// @brief Compute bytes from a memref type, or nullopt if not statically determinable.
static std::optional<int64_t> computeBytesFromType(Type t) {
  auto mr = llvm::dyn_cast<MemRefType>(t);
  if (!mr || !mr.hasStaticShape())
    return std::nullopt;
  
  int64_t numElems = 1;
  for (int64_t d : mr.getShape())
    numElems *= d;
  
  int64_t bitsPerElem = static_cast<int64_t>(mr.getElementTypeBitWidth());
  if (bitsPerElem <= 0)
    return std::nullopt;
  
  return numElems * ((bitsPerElem + 7) / 8);
}

// Try to compute the number of bytes copied for a given memref.copy.
// Returns std::nullopt if the copy size cannot be statically determined.
static std::optional<int64_t> getCopySizeBytes(memref::CopyOp cpy) {
  if (auto sz = computeBytesFromType(cpy.getSource().getType()))
    return sz;
  return computeBytesFromType(cpy.getTarget().getType());
}

/// @brief Helper to scan users of a value for copy operations and return max size.
static std::optional<int64_t> scanUsersForCopies(Value v, bool asTarget) {
  std::optional<int64_t> best;
  for (Operation *user : v.getUsers()) {
    if (auto c = llvm::dyn_cast<memref::CopyOp>(user)) {
      bool match = asTarget ? (c.getTarget() == v) : (c.getSource() == v);
      if (match) {
        if (auto sz = getCopySizeBytes(c)) {
          if (!best || *sz > *best)
            best = *sz;
        }
      }
    }
  }
  return best;
}

/// @brief Helper to scan cast-like operations (cast/reinterpret_cast) for copies.
static std::optional<int64_t> scanCastUsersForCopies(Value base, bool asTarget) {
  for (Operation *user : base.getUsers()) {
    Value castResult;
    if (auto castLike = llvm::dyn_cast<memref::CastOp>(user)) {
      castResult = castLike.getResult();
    } else if (auto rc = llvm::dyn_cast<memref::ReinterpretCastOp>(user)) {
      castResult = rc.getResult();
    } else {
      continue;
    }
    if (auto sz = scanUsersForCopies(castResult, asTarget))
      return sz;
  }
  return std::nullopt;
}

// Discover the copied size (in bytes) for a given memref.alloc by locating
// associated memref.copy operations. Preference order:
// 1) Copies writing into the alloc (alloc as target), directly or via one
//    level of memref.cast or memref.reinterpret_cast.
// 2) Copies reading from the alloc (alloc as source), with the same one-level
//    cast allowance.
// If multiple copies are found with different sizes, the maximum size is used.
static std::optional<int64_t> inferAllocCopiedSizeBytes(memref::AllocOp alloc) {
  Value base = alloc.getResult();

  // 1) Direct uses: alloc used as target of memref.copy
  if (auto sz = scanUsersForCopies(base, /*asTarget=*/true))
    return sz;

  // 1b) One-level cast/reinterpret_cast from alloc, then used as target
  if (auto sz = scanCastUsersForCopies(base, /*asTarget=*/true))
    return sz;

  // 2) Fallback: copies reading from the alloc
  if (auto sz = scanUsersForCopies(base, /*asTarget=*/false))
    return sz;

  return scanCastUsersForCopies(base, /*asTarget=*/false);
}

/**
 * @brief Information about copy operations and their candidates.
 * @details Encapsulates the result of collecting copy operations and building
 *          their candidate lists for enumeration.
 */
struct CopyCandidatesInfo {
  /// All memref.copy operations found in the function, in walk order.
  SmallVector<memref::CopyOp> copies;
  /// Candidate list for each copy operation, indexed by position in copies.
  SmallVector<SmallVector<DictionaryAttr>> perCopyCands;
};

/**
 * @brief Build candidate list for a single memref.copy operation.
 * @param cpy The copy operation to build candidates for.
 * @param df The dataflow context containing interconnect information.
 * @param ctx The MLIR context.
 * @return Vector of candidate dictionary attributes (mem + broadcast options).
 */
 static SmallVector<DictionaryAttr>
 BuildCandidatesForCopy(memref::CopyOp cpy, const DFContext &df,
                        MLIRContext *ctx) {
   SmallVector<DictionaryAttr> cand;
   
   // Always include memory copy candidate.
   NamedAttrList memAttrs;
   memAttrs.append("kind", StringAttr::get(ctx, "mem"));
   memAttrs.append("memory_name", StringAttr::get(ctx, df.singleMemoryName));
   cand.push_back(DictionaryAttr::get(ctx, memAttrs));
   
   // Try to find source reinterpret_cast and its reuse.
   DictionaryAttr reuse;
   if (auto srcDef = cpy.getSource().getDefiningOp()) {
     if (auto rc = dyn_cast<memref::ReinterpretCastOp>(srcDef))
       reuse = rc->getAttrOfType<DictionaryAttr>("loom.reuse");
   }
   
   // Build broadcast candidates for dimensions with total reuse.
   bool hasXReuse = false;
   bool hasYReuse = false;
   for (const auto &kv : df.icByDimName) {
     StringRef dimNameRef = kv.getKey();
     if (!hasSpatialTotalReuse(reuse, dimNameRef))
       continue;
     
     // Track x/y for all_links candidate
     if (dimNameRef == "x")
       hasXReuse = true;
     else if (dimNameRef == "y")
       hasYReuse = true;
     
     // Add broadcast candidates for each interconnect in this dimension
     for (const auto &[icHandle, icName] : kv.getValue()) {
       NamedAttrList broadcastAttrs;
       broadcastAttrs.append("kind", StringAttr::get(ctx, "broadcast"));
       broadcastAttrs.append("dim", StringAttr::get(ctx, dimNameRef));
       broadcastAttrs.append("interconnect_name", StringAttr::get(ctx, icName));
       cand.push_back(DictionaryAttr::get(ctx, broadcastAttrs));
     }
   }
   
   // If both x and y have total reuse, add all_links broadcast candidate.
   if (hasXReuse && hasYReuse) {
     NamedAttrList allLinksAttrs;
     allLinksAttrs.append("kind", StringAttr::get(ctx, "broadcast"));
     allLinksAttrs.append("dim", StringAttr::get(ctx, "xy"));
     allLinksAttrs.append("interconnect_name", StringAttr::get(ctx, "all_links"));
     cand.push_back(DictionaryAttr::get(ctx, allLinksAttrs));
   }
   
   return cand;
 }

/**
 * @brief Collect all copy operations and build their candidate lists.
 * @param f The function to analyze.
 * @param df The dataflow context containing interconnect information.
 * @param ctx The MLIR context.
 * @return CopyCandidatesInfo containing copies and their candidate lists.
 */
static CopyCandidatesInfo CollectCopyCandidates(func::FuncOp f,
                                                 const DFContext &df,
                                                 MLIRContext *ctx) {
  CopyCandidatesInfo info;
  f.walk([&](memref::CopyOp cpy) {
    info.copies.push_back(cpy);
    info.perCopyCands.push_back(BuildCandidatesForCopy(cpy, df, ctx));
  });
  return info;
}

/**
 * @brief Build function name suffix from selected candidates.
 * @param perCopyCands The candidate lists for each copy operation.
 * @param idx The current selection indices for each copy.
 * @return Function name suffix string (e.g., "c0mem_c1bx").
 */
static std::string
BuildFunctionNameSuffix(const SmallVector<SmallVector<DictionaryAttr>> &perCopyCands,
                       const SmallVector<size_t> &idx) {
  std::string suffix;
  for (size_t i = 0; i < perCopyCands.size(); ++i) {
    if (perCopyCands[i].empty() || idx[i] >= perCopyCands[i].size())
      continue;
    
    DictionaryAttr choice = perCopyCands[i][idx[i]];
    auto kind = choice.getAs<StringAttr>("kind");
    
    if (!suffix.empty())
      suffix += "_";
    
    if (kind && kind.getValue() == "mem") {
      suffix += ("c" + std::to_string(i) + "mem");
    } else if (kind && kind.getValue() == "broadcast") {
      auto dim = choice.getAs<StringAttr>("dim");
      std::string tag = "b";
      if (dim) {
        StringRef dimVal = dim.getValue();
        if (dimVal == "x")
          tag += "x";
        else if (dimVal == "y")
          tag += "y";
        else
          tag += "d";
      }
      suffix += ("c" + std::to_string(i) + tag);
    } else {
      suffix += ("c" + std::to_string(i) + "unk");
    }
  }
  return suffix;
}

/**
 * @brief Enumerate all candidate combinations and clone functions.
 * @param f The original function to clone.
 * @param info The copy candidates information.
 * @param outBuilder Builder for the output module.
 * @param annotateAlloc Function to annotate alloc operations.
 * @return Vector of all cloned functions with choices attached.
 */
static SmallVector<func::FuncOp>
EnumerateAndCloneFunctions(func::FuncOp f, const CopyCandidatesInfo &info,
                           OpBuilder &outBuilder,
                           std::function<void(memref::AllocOp)> annotateAlloc) {
  SmallVector<func::FuncOp> clones;

  // Enumerate cross product via mixed radix counters.
  SmallVector<size_t> idx(info.perCopyCands.size(), 0);
  auto bump = [&]() -> bool {
    for (size_t i = 0; i < idx.size(); ++i) {
      if (info.perCopyCands[i].empty())
        continue;
      if (++idx[i] < info.perCopyCands[i].size())
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
    std::string suffix = BuildFunctionNameSuffix(info.perCopyCands, idx);
    for (size_t i = 0, k = 0; i < info.perCopyCands.size() && k < cloneCopies.size(); ++i) {
      if (info.perCopyCands[i].empty() || idx[i] >= info.perCopyCands[i].size())
        continue;
      cloneCopies[k++]->setAttr("loom.copy.choice", info.perCopyCands[i][idx[i]]);
    }
    if (!suffix.empty())
      clone.setName((f.getSymName().str() + "__" + suffix).c_str());
    clones.push_back(clone);
  } while (bump());

  return clones;
}

struct ExploreAllocCopyMappingPass
    : public PassWrapper<ExploreAllocCopyMappingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExploreAllocCopyMappingPass)

  ExploreAllocCopyMappingPass() = default;
  ExploreAllocCopyMappingPass(bool analysisOnly) : analysisOnly(analysisOnly) {}

  StringRef getArgument() const override {
    return "loom-explore-alloc-copy-mapping";
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
      if (auto sz = inferAllocCopiedSizeBytes(alloc)) {
        nl.append("size", IntegerAttr::get(IntegerType::get(ctx, 64), *sz));
      }
      alloc->setAttr("loom.alloc", DictionaryAttr::get(ctx, nl));
    };

    // Build list of target functions (all functions in the module).
    llvm::SmallVector<func::FuncOp> funcs(module.getOps<func::FuncOp>().begin(),
                                           module.getOps<func::FuncOp>().end());

    // Analysis-only: attach candidates in-place and annotate allocs.
    if (analysisOnly) {
      for (func::FuncOp f : funcs) {
        f.walk([&](memref::AllocOp a) { annotateAlloc(a); });
        f.walk([&](memref::CopyOp cpy) {
          auto cand = BuildCandidatesForCopy(cpy, df, ctx);
          SmallVector<Attribute> candidates(cand.begin(), cand.end());
          cpy->setAttr("loom.copy.candidates", ArrayAttr::get(ctx, candidates));
        });
      }
      return;
    }

    // Enumeration mode: clone per combination of choices (no limit).
    OpBuilder moduleBuilder(module.getBodyRegion());
    OwningOpRef<ModuleOp> out = ModuleOp::create(module.getLoc());
    OpBuilder outBuilder(out->getBodyRegion());

    for (func::FuncOp f : funcs) {
      // Annotate allocs in the original function.
      f.walk([&](memref::AllocOp a) { annotateAlloc(a); });

      // Collect copy operations and their candidates, then enumerate all combinations.
      CopyCandidatesInfo info = CollectCopyCandidates(f, df, ctx);
      EnumerateAndCloneFunctions(f, info, outBuilder, annotateAlloc);
    }

    // Drop all non-DF top-level ops (old functions etc.), keep DF ops only.
    SmallVector<Operation *, 16> toErase;
    for (Operation &op : *module.getBody()) {
      if (auto dialect = op.getDialect();
          !dialect || dialect->getNamespace() != StringRef("df")) {
        toErase.push_back(&op);
      }
    }
    for (auto *op : llvm::reverse(toErase))
      op->erase();

    // Materialize all clones into the original module after DF ops.
    // Find the last DF op to insert after it, keeping DF decls first.
    Operation *lastDFOp = nullptr;
    for (Operation &op : *module.getBody()) {
      if (auto dialect = op.getDialect();
          dialect && dialect->getNamespace() == StringRef("df")) {
        lastDFOp = &op;
      }
    }
    
    if (lastDFOp) {
      moduleBuilder.setInsertionPointAfter(lastDFOp);
    } else {
      moduleBuilder.setInsertionPointToStart(module.getBody());
    }
    
    for (Operation &op : *out->getBody()) {
      IRMapping m;
      moduleBuilder.clone(op, m);
    }
  }

  bool analysisOnly = false;
};

} // namespace

std::unique_ptr<mlir::Pass>
loom::passes::createExploreAllocCopyMappingPass(bool analysisOnly) {
  return std::make_unique<ExploreAllocCopyMappingPass>(analysisOnly);
}

void loom::passes::registerExploreAllocCopyMappingPass() {
  PassRegistration<ExploreAllocCopyMappingPass>();
}
