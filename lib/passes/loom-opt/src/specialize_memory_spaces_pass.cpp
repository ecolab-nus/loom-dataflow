#include "Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"

#include "LoomDialect.h.inc"
#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace loom {
namespace passes {

#define GEN_PASS_DEF_SPECIALIZEMEMORYSPACESPASS
#include "Passes.h.inc"

namespace {

struct SpecializeMemorySpacesPass
    : public impl::SpecializeMemorySpacesPassBase<SpecializeMemorySpacesPass> {

  void runOnOperation() override {
    llvm::StringMap<int64_t> kinds;
    for (const std::string &binding : bindings) {
      StringRef text(binding);
      auto [port, kindText] = text.split('=');
      int64_t kind;
      if (port.empty() || kindText.empty() || kindText.getAsInteger(10, kind) ||
          kind < 0) {
        getOperation().emitError()
            << "invalid memory-space binding '" << text
            << "'; expected port=nonnegative-integer";
        return signalPassFailure();
      }
      auto [entry, inserted] = kinds.try_emplace(port, kind);
      if (!inserted && entry->second != kind) {
        getOperation().emitError()
            << "conflicting memory-space bindings for port @" << port;
        return signalPassFailure();
      }
    }

    bool failed = false;
    getOperation().walk([&](func::FuncOp function) {
      if (failed)
        return;
      llvm::DenseMap<Value, StringRef> valuePorts;
      function.walk([&](loom::BindMemOp bind) {
        Value value = bind.getMemref();
        StringRef port = bind.getMemory();
        auto [entry, inserted] = valuePorts.try_emplace(value, port);
        if (!inserted && entry->second != port) {
          bind.emitError() << "memref has conflicting loom.bind_mem ports @"
                           << entry->second << " and @" << port;
          failed = true;
        }
      });
      if (failed)
        return;

      if (llvm::any_of(function.getResultTypes(),
                       [](Type type) { return isa<MemRefType>(type); })) {
        function.emitError()
            << "memref function results are unsupported in frontend native "
               "memory-space specialization";
        failed = true;
        return;
      }

      SmallVector<Type> inputs(function.getArgumentTypes());
      for (auto [index, argument] : llvm::enumerate(function.getArguments())) {
        auto memref = dyn_cast<MemRefType>(argument.getType());
        if (!memref)
          continue;
        if (memref.getMemorySpace()) {
          function.emitError()
              << "argument " << index
              << " has an authored memory space; frontend native MLIR must "
                 "leave memory spaces unspecified";
          failed = true;
          continue;
        }
        auto port = valuePorts.find(argument);
        if (port == valuePorts.end()) {
          function.emitError() << "memref argument " << index
                               << " has no loom.bind_mem";
          failed = true;
          continue;
        }
        auto kind = kinds.find(port->second);
        if (kind == kinds.end()) {
          function.emitError() << "port @" << port->second
                               << " has no supplied memory-space binding";
          failed = true;
          continue;
        }
        Attribute space = IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                           kind->second);
        Type specialized = MemRefType::get(memref.getShape(),
                                           memref.getElementType(),
                                           memref.getLayout(), space);
        argument.setType(specialized);
        inputs[index] = specialized;
      }
      if (failed)
        return;
      function.setType(FunctionType::get(&getContext(), inputs,
                                         function.getResultTypes()));

      function.walk([&](Operation *operation) {
        if (operation == function.getOperation())
          return;
        if (llvm::any_of(operation->getResultTypes(),
                         [](Type type) { return isa<MemRefType>(type); })) {
          operation->emitError()
              << "memref-producing operations are unsupported in frontend "
                 "native memory-space specialization";
          failed = true;
          return;
        }
        bool consumesMemref = llvm::any_of(operation->getOperandTypes(),
                                           [](Type type) {
                                             return isa<MemRefType>(type);
                                           });
        if (!consumesMemref)
          return;
        StringRef name = operation->getName().getStringRef();
        bool supported = name == "loom.bind_mem" || name == "loom.bind_shape" ||
                         name == "loom.copy" || name == "loom.gather" ||
                         name == "memref.copy" || name == "memref.dim" ||
                         isa<linalg::LinalgOp>(operation);
        if (!supported) {
          operation->emitError()
              << "operation consuming a specialized memref is unsupported";
          failed = true;
        }
      });
      if (failed)
        return;

      function.walk([&](loom::CopyOp copy) {
        if (copy.getSrcMemKind() || copy.getDstMemKind()) {
          copy.emitError() << "frontend native MLIR must not author copy "
                              "memory-kind attributes";
          failed = true;
          return;
        }
        auto sourcePort = copy.getSrcMemSpaceAttr();
        auto destinationPort = copy.getDstMemSpaceAttr();
        if (!sourcePort || !destinationPort) {
          copy.emitError() << "copy must name both logical memory ports";
          failed = true;
          return;
        }
        StringRef boundSource = valuePorts.lookup(copy.getSource());
        StringRef boundDestination = valuePorts.lookup(copy.getDestination());
        if (boundSource != sourcePort.getRootReference().getValue() ||
            boundDestination !=
                destinationPort.getRootReference().getValue()) {
          copy.emitError()
              << "copy endpoint ports must match the operands' loom.bind_mem "
                 "ports";
          failed = true;
          return;
        }
        auto sourceKind = kinds.find(sourcePort.getRootReference().getValue());
        auto destinationKind =
            kinds.find(destinationPort.getRootReference().getValue());
        if (sourceKind == kinds.end() || destinationKind == kinds.end()) {
          copy.emitError() << "copy port has no supplied memory-space binding";
          failed = true;
          return;
        }
        copy.setSrcMemKind(sourceKind->second);
        copy.setDstMemKind(destinationKind->second);
      });
      if (failed)
        return;

      function.walk([&](loom::GatherOp gather) {
        auto sourcePort = gather.getSrcMemSpaceAttr();
        auto destinationPort = gather.getDstMemSpaceAttr();
        if (!sourcePort || !destinationPort) {
          gather.emitError() << "gather must name both logical memory ports";
          failed = true;
          return;
        }
        StringRef boundSource = valuePorts.lookup(gather.getSource());
        StringRef boundDestination = valuePorts.lookup(gather.getDestination());
        if (boundSource != sourcePort.getRootReference().getValue() ||
            boundDestination !=
                destinationPort.getRootReference().getValue()) {
          gather.emitError()
              << "gather endpoint ports must match the operands' "
                 "loom.bind_mem ports";
          failed = true;
        }
      });

      function.walk([&](Operation *operation) {
        if (auto symbol = dyn_cast<loom::SymOp>(operation)) {
          operation->setLoc(NameLoc::get(StringAttr::get(
              &getContext(),
              symbol.getSymbolRef().getRootReference().getValue())));
        } else if (operation->getNumResults() != 0) {
          operation->setLoc(NameLoc::get(
              StringAttr::get(&getContext(), "native")));
        }
      });
    });
    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSpecializeMemorySpacesPass() {
  return std::make_unique<SpecializeMemorySpacesPass>();
}

} // namespace passes
} // namespace loom
