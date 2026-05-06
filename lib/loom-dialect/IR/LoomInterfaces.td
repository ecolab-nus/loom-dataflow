//===- LoomInterfaces.td -----------------------------------*- tablegen -*-===//
//
// Operation interfaces for the LOOM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_INTERFACES
#define LOOM_INTERFACES

include "mlir/IR/OpBase.td"

def ComputeMemoryHandoffOpInterface
    : OpInterface<"ComputeMemoryHandoffOpInterface"> {
  let cppNamespace = "::loom";
  let description = [{
    Interface for operations that mark tensor SSA values crossing a
    compute-memory handoff boundary.  Returned values are the ranked tensor
    values that memory analysis must treat as exclusive targets.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return tensor SSA values that are compute-memory handoff targets.",
      /*retTy=*/"::mlir::SmallVector<::mlir::Value>",
      /*methodName=*/"getHandoffTargetTensors",
      /*args=*/(ins)
    >
  ];
}

#endif // LOOM_INTERFACES
