#include "binary_scalar_chain.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace loom::utils {
namespace {

bool isScalarType(Type type) { return type.isIntOrIndexOrFloat(); }

bool isPayloadScalarOp(Operation *op) {
  if (!op || op->getNumResults() == 0 || op->getNumRegions() != 0 ||
      op->hasTrait<OpTrait::IsTerminator>())
    return false;

  if (!llvm::all_of(op->getOperandTypes(), isScalarType))
    return false;
  return llvm::all_of(op->getResultTypes(), isScalarType);
}

linalg::YieldOp getYieldOp(linalg::GenericOp genericOp) {
  if (genericOp.getRegion().empty())
    return nullptr;
  return dyn_cast<linalg::YieldOp>(genericOp.getRegion().front().getTerminator());
}

bool isInputBlockArgument(Value value, linalg::GenericOp genericOp,
                          unsigned &inputIndex) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg || blockArg.getOwner() != genericOp.getBody())
    return false;

  unsigned argNumber = blockArg.getArgNumber();
  if (argNumber >= genericOp.getNumDpsInputs())
    return false;

  inputIndex = argNumber;
  return true;
}

bool isInputAliasedWithOutput(linalg::GenericOp genericOp,
                              unsigned inputIndex) {
  Value input = genericOp.getDpsInputs()[inputIndex];
  return llvm::is_contained(genericOp.getDpsInits(), input);
}

bool isForbiddenOutputOperand(Value operand, linalg::GenericOp genericOp) {
  auto blockArg = dyn_cast<BlockArgument>(operand);
  if (!blockArg || blockArg.getOwner() != genericOp.getBody())
    return false;

  unsigned argNumber = blockArg.getArgNumber();
  unsigned numInputs = genericOp.getNumDpsInputs();
  if (argNumber >= numInputs)
    return true;

  return isInputAliasedWithOutput(genericOp, argNumber);
}

bool isEarlierBinaryResult(Value value, Operation *splitOp,
                           linalg::GenericOp genericOp,
                           Operation *&intermediateOp) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp || defOp->getBlock() != genericOp.getBody() ||
      !defOp->isBeforeInBlock(splitOp))
    return false;

  if (!isPureBinaryScalarOp(defOp) || hasControlSemantics(defOp))
    return false;

  intermediateOp = defOp;
  return true;
}

bool collectFirstSlice(Value value, linalg::GenericOp genericOp,
                       SmallPtrSetImpl<Operation *> &slice) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (blockArg)
    return true;

  Operation *defOp = value.getDefiningOp();
  if (!defOp || defOp->getBlock() != genericOp.getBody())
    return true;

  if (!isPayloadScalarOp(defOp))
    return false;

  for (Value operand : defOp->getOperands()) {
    if (!collectFirstSlice(operand, genericOp, slice))
      return false;
  }

  slice.insert(defOp);
  return true;
}

bool collectSecondSlice(Value value, linalg::GenericOp genericOp,
                        Operation *splitOp, Operation *intermediateOp,
                        SmallPtrSetImpl<Operation *> &slice) {
  if (value == intermediateOp->getResult(0))
    return true;

  auto blockArg = dyn_cast<BlockArgument>(value);
  if (blockArg)
    return true;

  Operation *defOp = value.getDefiningOp();
  if (!defOp || defOp->getBlock() != genericOp.getBody())
    return true;

  if (defOp->isBeforeInBlock(splitOp))
    return false;
  if (!isPayloadScalarOp(defOp))
    return false;

  for (Value operand : defOp->getOperands()) {
    if (!collectSecondSlice(operand, genericOp, splitOp, intermediateOp,
                            slice))
      return false;
  }

  slice.insert(defOp);
  return true;
}

SmallVector<Operation *, 4>
sortSliceInBlockOrder(linalg::GenericOp genericOp,
                      const SmallPtrSetImpl<Operation *> &slice) {
  SmallVector<Operation *, 4> ordered;
  for (Operation &op : genericOp.getBody()->without_terminator()) {
    if (slice.contains(&op))
      ordered.push_back(&op);
  }
  return ordered;
}

SmallVector<unsigned, 4>
collectInputIndices(linalg::GenericOp genericOp,
                    ArrayRef<Operation *> ops) {
  DenseSet<unsigned> used;
  for (Operation *op : ops) {
    for (Value operand : op->getOperands()) {
      unsigned inputIndex = 0;
      if (isInputBlockArgument(operand, genericOp, inputIndex))
        used.insert(inputIndex);
    }
  }

  SmallVector<unsigned, 4> ordered;
  for (unsigned i = 0, e = genericOp.getNumDpsInputs(); i < e; ++i) {
    if (used.contains(i))
      ordered.push_back(i);
  }
  return ordered;
}

bool chainHasForbiddenOutputOperands(ArrayRef<Operation *> ops,
                                     linalg::GenericOp genericOp) {
  for (Operation *op : ops) {
    for (Value operand : op->getOperands()) {
      if (isForbiddenOutputOperand(operand, genericOp))
        return true;
    }
  }
  return false;
}

} // namespace

bool isPureBinaryScalarOp(Operation *op) {
  if (!op || op->getNumOperands() != 2 || op->getNumResults() != 1)
    return false;

  if (!llvm::all_of(op->getOperandTypes(), isScalarType))
    return false;
  if (!isScalarType(op->getResult(0).getType()))
    return false;

  return OpTrait::hasElementwiseMappableTraits(op);
}

bool hasControlSemantics(Operation *op) {
  return isa_and_nonnull<arith::CmpFOp, arith::CmpIOp>(op);
}

std::optional<BinaryScalarChainMatch>
BinaryScalarChainAnalyzer::findFirstMatch(linalg::GenericOp genericOp) const {
  if (!genericOp || !genericOp.isAllParallelLoops() ||
      genericOp.getNumDpsInits() != 1 || genericOp->getNumResults() != 0)
    return std::nullopt;

  linalg::YieldOp yieldOp = getYieldOp(genericOp);
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return std::nullopt;

  Block *body = genericOp.getBody();
  for (Operation &op : body->without_terminator()) {
    Operation *splitOp = &op;
    if (!isPureBinaryScalarOp(splitOp) || hasControlSemantics(splitOp))
      continue;

    Operation *intermediateOp = nullptr;
    bool hasInputSide = false;
    for (unsigned operandIndex = 0; operandIndex < 2; ++operandIndex) {
      Operation *candidateIntermediate = nullptr;
      if (!isEarlierBinaryResult(splitOp->getOperand(operandIndex), splitOp,
                                 genericOp, candidateIntermediate))
        continue;

      unsigned inputIndex = 0;
      Value otherOperand = splitOp->getOperand(1 - operandIndex);
      if (!isInputBlockArgument(otherOperand, genericOp, inputIndex))
        continue;
      if (isInputAliasedWithOutput(genericOp, inputIndex))
        continue;

      intermediateOp = candidateIntermediate;
      hasInputSide = true;
      break;
    }
    if (!hasInputSide)
      continue;

    SmallPtrSet<Operation *, 8> firstSlice;
    if (!collectFirstSlice(intermediateOp->getResult(0), genericOp,
                           firstSlice))
      continue;

    SmallPtrSet<Operation *, 8> secondSlice;
    if (!collectSecondSlice(yieldOp.getOperand(0), genericOp, splitOp,
                            intermediateOp, secondSlice))
      continue;
    if (!secondSlice.contains(splitOp))
      continue;

    BinaryScalarChainMatch match;
    match.genericOp = genericOp;
    match.intermediateOp = intermediateOp;
    match.splitOp = splitOp;
    match.firstOps = sortSliceInBlockOrder(genericOp, firstSlice);
    match.secondOps = sortSliceInBlockOrder(genericOp, secondSlice);

    if (chainHasForbiddenOutputOperands(match.secondOps, genericOp))
      continue;

    match.firstInputIndices = collectInputIndices(genericOp, match.firstOps);
    match.secondInputIndices = collectInputIndices(genericOp, match.secondOps);
    return match;
  }

  return std::nullopt;
}

} // namespace loom::utils
