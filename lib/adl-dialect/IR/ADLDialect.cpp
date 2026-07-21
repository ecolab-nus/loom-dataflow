#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ADLDialect.h.inc"
// Generated type declarations
#define GET_TYPEDEF_CLASSES
#include "ADLTypes.h.inc"
// Generated type definitions (TypeID, printers/parsers)
#define GET_TYPEDEF_CLASSES
#include "ADLTypes.cpp.inc"

using namespace mlir;
using namespace adl;

#include "ADLDialect.cpp.inc"
// Bring in op class declarations for registration below.
#define GET_OP_CLASSES
#include "ADLOps.h.inc"

void ADLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ADLOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "ADLTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Custom parser/printer for ProcessorComputeOp & ProcessorDMoverOp
//===----------------------------------------------------------------------===//

/// Finish building a processor-like op after name, route and resources have been parsed.
static ParseResult resolveProcessorOperands(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand &source,
    OpAsmParser::UnresolvedOperand &destination,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resources) {
  auto memType = MemHandleType::get(parser.getContext());
  auto resourceType = ResourceHandleType::get(parser.getContext());
  if (parser.resolveOperand(source, memType, result.operands) ||
      parser.resolveOperand(destination, memType, result.operands) ||
      parser.resolveOperands(resources, resourceType, result.operands))
    return failure();
  result.addTypes(ArchHandleType::get(parser.getContext()));
  return success();
}

//--- ProcessorComputeOp ---------------------------------------------------
// Name is a FlatSymbolRefAttr (prints/parses as @symbol).

ParseResult ProcessorComputeOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  FlatSymbolRefAttr symNameAttr;
  OpAsmParser::UnresolvedOperand source, destination;
  SmallVector<OpAsmParser::UnresolvedOperand> resources;
  if (parser.parseAttribute(symNameAttr) || parser.parseComma() ||
      parser.parseKeyword("from") || parser.parseOperand(source) ||
      parser.parseKeyword("to") || parser.parseOperand(destination))
    return failure();

  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseKeyword("with") || parser.parseLSquare() ||
        parser.parseCommaSeparatedList([&]() {
          OpAsmParser::UnresolvedOperand res;
          if (parser.parseOperand(res))
            return failure();
          resources.push_back(res);
          return success();
        }) ||
        parser.parseRSquare())
      return failure();
  }

  result.addAttribute("sym_name", symNameAttr);
  return resolveProcessorOperands(parser, result, source, destination, resources);
}

void ProcessorComputeOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printAttribute(getSymNameAttr()); // prints @name
  p << ", from ";
  p.printOperand(getSource());
  p << " to ";
  p.printOperand(getDestination());
  if (!getResources().empty()) {
    p << ", with [";
    llvm::interleaveComma(getResources(), p,
                          [&](Value v) { p.printOperand(v); });
    p << "]";
  }
}


//--- ProcessorDMoverOp ----------------------------------------------------
// Name is a FlatSymbolRefAttr (prints/parses as @symbol).

ParseResult ProcessorDMoverOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  FlatSymbolRefAttr symNameAttr;
  OpAsmParser::UnresolvedOperand source, destination;
  SmallVector<OpAsmParser::UnresolvedOperand> resources;
  if (parser.parseAttribute(symNameAttr) || parser.parseComma() ||
      parser.parseKeyword("from") || parser.parseOperand(source) ||
      parser.parseKeyword("to") || parser.parseOperand(destination))
    return failure();

  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseKeyword("with") || parser.parseLSquare() ||
        parser.parseCommaSeparatedList([&]() {
          OpAsmParser::UnresolvedOperand res;
          if (parser.parseOperand(res))
            return failure();
          resources.push_back(res);
          return success();
        }) ||
        parser.parseRSquare())
      return failure();
  }

  result.addAttribute("sym_name", symNameAttr);
  return resolveProcessorOperands(parser, result, source, destination, resources);
}

void ProcessorDMoverOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printAttribute(getSymNameAttr()); // prints @name
  p << ", from ";
  p.printOperand(getSource());
  p << " to ";
  p.printOperand(getDestination());
  if (!getResources().empty()) {
    p << ", with [";
    llvm::interleaveComma(getResources(), p,
                          [&](Value v) { p.printOperand(v); });
    p << "]";
  }
}


#define GET_OP_CLASSES
#include "ADLOps.cpp.inc"
