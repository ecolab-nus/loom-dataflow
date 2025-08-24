#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "DataflowDialect.h.inc"
// Generated type declarations
#define GET_TYPEDEF_CLASSES
#include "DataflowTypes.h.inc"
// Generated type definitions (TypeID, printers/parsers)
#define GET_TYPEDEF_CLASSES
#include "DataflowTypes.cpp.inc"

using namespace mlir;
using namespace tmd::df;

#include "DataflowDialect.cpp.inc"
// Bring in op class declarations for registration below.
#define GET_OP_CLASSES
#include "DataflowOps.h.inc"

void DataflowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "DataflowOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "DataflowTypes.cpp.inc"
      >();
}

// Custom assembly for df.chained_load
ParseResult ChainedLoadOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  // Parse: %memref [ affine-map-of-ssa-ids ]
  OpAsmParser::UnresolvedOperand memrefInfo;
  if (parser.parseOperand(memrefInfo))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 4> mapOperands;
  AffineMapAttr mapAttr;
  if (parser.parseAffineMapOfSSAIds(mapOperands, mapAttr, /*attrName=*/"map",
                                    result.attributes))
    return failure();

  // Optional attrs
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse ": memref-type"
  if (parser.parseColon())
    return failure();
  MemRefType memrefTy;
  if (parser.parseType(memrefTy))
    return failure();

  // Optionally require a comma before trailing clause, then: over %chain
  if (parser.parseComma() || parser.parseKeyword("over"))
    return failure();
  OpAsmParser::UnresolvedOperand chainInfo;
  if (parser.parseOperand(chainInfo))
    return failure();

  // Resolve operands
  if (parser.resolveOperand(memrefInfo, memrefTy, result.operands))
    return failure();
  if (parser.resolveOperands(mapOperands, builder.getIndexType(),
                             result.operands))
    return failure();
  if (parser.resolveOperand(chainInfo,
                            InterconnectHandleType::get(builder.getContext()),
                            result.operands))
    return failure();

  // Types and properties
  result.addTypes(memrefTy.getElementType());
  if (mapAttr)
    result.getOrAddProperties<ChainedLoadOp::Properties>().map = mapAttr;
  return success();
}

void ChainedLoadOp::print(OpAsmPrinter &p) {
  // Print: %memref[affine-of-ssa-ids]
  p << ' ' << getMemref() << '[';
  if (AffineMapAttr mapAttr = getMapAttr())
    p.printAffineMapOfSSAIds(mapAttr, getIndices());
  p << ']';
  // Elide inherent attrs
  p.printOptionalAttrDict((*this)->getAttrs(), /*elided=*/{"map"});
  // Print: ": memref-type, over %chain"
  p << " : " << getMemref().getType() << ", over "
    << (*this)->getOperand((*this)->getNumOperands() - 1);
}

#define GET_OP_CLASSES
#include "DataflowOps.cpp.inc"
