#include "ssa_utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace loom::utils {

bool dependsOn(Value value, Value target) {
  if (!value || value == target)
    return value == target;

  SmallPtrSet<Value, 16> visited;
  SmallVector<Value, 16> worklist = {value};

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    if (current == target)
      return true;

    // Block arguments stop the walk (treated as leaves).
    if (llvm::isa<BlockArgument>(current))
      continue;

    if (Operation *def = current.getDefiningOp()) {
      worklist.append(def->operand_begin(), def->operand_end());
    }
  }

  return false;
}

Value traceToRootAlloc(Value value) {
  if (!value)
    return nullptr;

  SmallPtrSet<Value, 32> visited;
  SmallVector<Value, 32> worklist = {value};

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!current || !visited.insert(current).second)
      continue;

    if (auto allocOp = current.getDefiningOp<loom::AllocOp>())
      return allocOp.getResult();

    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      Operation *parent = blockArg.getOwner()->getParentOp();

      if (auto scfFor = dyn_cast<scf::ForOp>(parent)) {
        unsigned argNum = blockArg.getArgNumber();
        if (argNum > 0 && argNum - 1 < scfFor.getInits().size())
          worklist.push_back(scfFor.getInits()[argNum - 1]);
      } else if (auto affFor = dyn_cast<affine::AffineForOp>(parent)) {
        unsigned bodyArgs = affFor.getBody()->getNumArguments();
        unsigned iterCount = affFor.getNumIterOperands();
        unsigned firstIterArg = bodyArgs - iterCount;
        unsigned argNum = blockArg.getArgNumber();
        if (argNum >= firstIterArg) {
          unsigned iterIdx = argNum - firstIterArg;
          if (iterIdx < affFor.getInits().size())
            worklist.push_back(affFor.getInits()[iterIdx]);
        }
      } else if (auto affPar = dyn_cast<affine::AffineParallelOp>(parent)) {
        unsigned argNum = blockArg.getArgNumber();
        if (argNum < affPar.getInits().size())
          worklist.push_back(affPar.getInits()[argNum]);
      }
      continue;
    }

    Operation *defOp = current.getDefiningOp();
    if (!defOp)
      continue;

    if (auto semTake = dyn_cast<loom::SemaphoreTakeOp>(defOp)) {
      worklist.push_back(semTake.getSource());
      continue;
    }
    if (auto init = dyn_cast<loom::InitTensorOp>(defOp)) {
      worklist.push_back(init.getBuffer());
      continue;
    }
    if (auto copyToTensor = dyn_cast<loom::CopyToTensorOp>(defOp)) {
      worklist.push_back(copyToTensor.getBuffer());
      continue;
    }
    if (auto toTensor = dyn_cast<loom::BufferizeToTensorOp>(defOp)) {
      worklist.push_back(toTensor.getSource());
      continue;
    }
    if (auto toMemref = dyn_cast<loom::BufferizeToMemrefOp>(defOp)) {
      worklist.push_back(toMemref.getSource());
      continue;
    }
    if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(defOp)) {
      worklist.push_back(toTensor.getBuffer());
      continue;
    }
    if (auto toBuffer = dyn_cast<bufferization::ToBufferOp>(defOp)) {
      worklist.push_back(toBuffer.getTensor());
      continue;
    }
    if (auto mcast = dyn_cast<memref::CastOp>(defOp)) {
      worklist.push_back(mcast.getSource());
      continue;
    }
    if (auto tcast = dyn_cast<tensor::CastOp>(defOp)) {
      worklist.push_back(tcast.getSource());
      continue;
    }

    if (auto dps = dyn_cast<DestinationStyleOpInterface>(defOp)) {
      if (auto res = dyn_cast<OpResult>(current)) {
        unsigned resIdx = res.getResultNumber();
        ValueRange inits = dps.getDpsInits();
        if (resIdx < inits.size()) {
          worklist.push_back(inits[resIdx]);
          continue;
        }
      }
    }

    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(defOp)) {
      if (auto res = dyn_cast<OpResult>(current)) {
        unsigned resIdx = res.getResultNumber();
        ValueRange inits = linalgOp.getDpsInits();
        if (resIdx < inits.size()) {
          worklist.push_back(inits[resIdx]);
          continue;
        }
      }
    }

    for (Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }

  return nullptr;
}

loom::AllocOp traceToRootAllocOp(Value value) {
  Value root = traceToRootAlloc(value);
  return root ? root.getDefiningOp<loom::AllocOp>() : nullptr;
}

} // namespace loom::utils
