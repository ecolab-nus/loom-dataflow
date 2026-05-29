#include "Passes.h"
#include "binary_scalar_chain.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace {

bool isScalarOrRankOne(Type type) {
  if (type.isIntOrIndexOrFloat())
    return true;

  auto shapedType = dyn_cast<ShapedType>(type);
  return shapedType && shapedType.hasRank() && shapedType.getRank() <= 1;
}

bool tracesToScalarOrRankOneInput(Value value, linalg::GenericOp genericOp,
                                  SmallPtrSetImpl<Operation *> &visitedOps) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (blockArg) {
    if (blockArg.getOwner() != genericOp.getBody())
      return false;

    unsigned argNumber = blockArg.getArgNumber();
    if (argNumber >= genericOp.getNumDpsInputs())
      return false;

    Value genericInput = genericOp.getDpsInputs()[argNumber];
    return isScalarOrRankOne(genericInput.getType());
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp || defOp->getBlock() != genericOp.getBody())
    return false;

  if (!visitedOps.insert(defOp).second)
    return false;

  for (Value operand : defOp->getOperands()) {
    if (tracesToScalarOrRankOneInput(operand, genericOp, visitedOps))
      return true;
  }
  return false;
}

bool binaryOpUsesScalarOrRankOneInput(Operation *op,
                                      linalg::GenericOp genericOp) {
  SmallPtrSet<Operation *, 8> visitedOps;
  for (Value operand : op->getOperands()) {
    if (tracesToScalarOrRankOneInput(operand, genericOp, visitedOps))
      return true;
  }
  return false;
}

bool matchHasUnsplittableScalarOrRankOneInput(
    const loom::utils::BinaryScalarChainMatch &match) {
  return binaryOpUsesScalarOrRankOneInput(match.intermediateOp,
                                          match.genericOp) ||
         binaryOpUsesScalarOrRankOneInput(match.splitOp, match.genericOp);
}

SmallVector<Value, 4> getInputsByIndex(linalg::GenericOp genericOp,
                                       ArrayRef<unsigned> indices) {
  SmallVector<Value, 4> inputs;
  ValueRange originalInputs = genericOp.getDpsInputs();
  for (unsigned index : indices)
    inputs.push_back(originalInputs[index]);
  return inputs;
}

SmallVector<AffineMap, 4> getInputMapsByIndex(linalg::GenericOp genericOp,
                                              ArrayRef<unsigned> indices) {
  SmallVector<AffineMap, 4> maps;
  SmallVector<AffineMap> originalMaps = genericOp.getIndexingMapsArray();
  for (unsigned index : indices)
    maps.push_back(originalMaps[index]);
  return maps;
}

void mapSelectedInputBlockArgs(IRMapping &mapping, linalg::GenericOp genericOp,
                               ArrayRef<unsigned> indices,
                               ValueRange newBlockArgs,
                               unsigned newBlockArgOffset) {
  Block *oldBody = genericOp.getBody();
  for (auto [newArgIndex, oldInputIndex] : llvm::enumerate(indices)) {
    mapping.map(oldBody->getArgument(oldInputIndex),
                newBlockArgs[newBlockArgOffset + newArgIndex]);
  }
}

Operation *clonePayloadOp(OpBuilder &builder, Operation *op,
                          IRMapping &mapping) {
  return builder.clone(*op, mapping);
}

void buildFirstBody(OpBuilder &builder, Location loc,
                    loom::utils::BinaryScalarChainMatch &match,
                    ValueRange blockArgs) {
  IRMapping mapping;
  linalg::GenericOp genericOp = match.genericOp;
  unsigned numInputs = genericOp.getNumDpsInputs();

  mapSelectedInputBlockArgs(mapping, genericOp, match.firstInputIndices,
                            blockArgs, 0);
  mapping.map(genericOp.getBody()->getArgument(numInputs),
              blockArgs[match.firstInputIndices.size()]);

  for (Operation *op : match.firstOps)
    clonePayloadOp(builder, op, mapping);

  Value yielded = mapping.lookup(match.intermediateOp->getResult(0));
  builder.create<linalg::YieldOp>(loc, yielded);
}

void buildSecondBody(OpBuilder &builder, Location loc,
                     loom::utils::BinaryScalarChainMatch &match,
                     ValueRange blockArgs) {
  IRMapping mapping;
  linalg::GenericOp genericOp = match.genericOp;
  unsigned numInputs = genericOp.getNumDpsInputs();

  mapping.map(match.intermediateOp->getResult(0), blockArgs[0]);
  mapSelectedInputBlockArgs(mapping, genericOp, match.secondInputIndices,
                            blockArgs, 1);
  mapping.map(genericOp.getBody()->getArgument(numInputs),
              blockArgs[1 + match.secondInputIndices.size()]);

  for (Operation *op : match.secondOps)
    clonePayloadOp(builder, op, mapping);

  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  Value yielded = mapping.lookupOrDefault(yieldOp.getOperand(0));
  builder.create<linalg::YieldOp>(loc, yielded);
}

bool splitBinaryScalarChain(loom::utils::BinaryScalarChainMatch match,
                            RewriterBase &rewriter) {
  linalg::GenericOp genericOp = match.genericOp;
  Value output = genericOp.getDpsInits()[0];
  SmallVector<AffineMap> originalMaps = genericOp.getIndexingMapsArray();
  unsigned numInputs = genericOp.getNumDpsInputs();
  AffineMap outputMap = originalMaps[numInputs];
  auto iteratorTypes = genericOp.getIteratorTypesArray();

  SmallVector<Value, 4> firstInputs =
      getInputsByIndex(genericOp, match.firstInputIndices);
  SmallVector<AffineMap, 4> firstMaps =
      getInputMapsByIndex(genericOp, match.firstInputIndices);
  firstMaps.push_back(outputMap);

  SmallVector<Value, 4> secondInputs;
  secondInputs.push_back(output);
  llvm::append_range(secondInputs,
                     getInputsByIndex(genericOp, match.secondInputIndices));

  SmallVector<AffineMap, 4> secondMaps;
  secondMaps.push_back(outputMap);
  llvm::append_range(secondMaps,
                     getInputMapsByIndex(genericOp, match.secondInputIndices));
  secondMaps.push_back(outputMap);

  Location loc = genericOp.getLoc();
  rewriter.setInsertionPoint(genericOp);
  rewriter.create<linalg::GenericOp>(
      loc, firstInputs, ValueRange(output), firstMaps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        buildFirstBody(nestedBuilder, nestedLoc, match, args);
      });

  rewriter.create<linalg::GenericOp>(
      loc, secondInputs, ValueRange(output), secondMaps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        buildSecondBody(nestedBuilder, nestedLoc, match, args);
      });

  rewriter.eraseOp(genericOp);
  return true;
}

struct SplitBinaryScalarChainPass
    : public PassWrapper<SplitBinaryScalarChainPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitBinaryScalarChainPass)

  StringRef getArgument() const override {
    return "tt-split-binary-scalar-chain";
  }

  StringRef getDescription() const override {
    return "Split safe fused binary scalar chains in linalg.generic ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    constexpr unsigned kMaxDepth = 3;
    MLIRContext *context = &getContext();
    loom::utils::BinaryScalarChainAnalyzer analyzer;

    for (unsigned depth = 0; depth < kMaxDepth; ++depth) {
      SmallVector<linalg::GenericOp, 16> generics;
      getOperation().walk(
          [&](linalg::GenericOp op) { generics.push_back(op); });

      bool changed = false;
      IRRewriter rewriter(context);
      for (linalg::GenericOp genericOp : generics) {
        if (!genericOp->getParentOp())
          continue;

        std::optional<loom::utils::BinaryScalarChainMatch> match =
            analyzer.findFirstMatch(genericOp);
        if (!match)
          continue;
        if (matchHasUnsplittableScalarOrRankOneInput(*match))
          continue;

        changed |= splitBinaryScalarChain(*match, rewriter);
      }

      if (!changed)
        break;
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
loom::passes::createSplitBinaryScalarChainPass() {
  return std::make_unique<SplitBinaryScalarChainPass>();
}
