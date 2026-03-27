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
// Custom parser/printer/verifier for ProcessorComputeOp & ProcessorDMoverOp
//===----------------------------------------------------------------------===//

/// Parse the mem-pair list: `[` `(` %src, %dst `)` , ... `]`
static ParseResult parseMemPairs(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &srcMems,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dstMems) {
  if (parser.parseLSquare())
    return failure();
  if (succeeded(parser.parseOptionalRSquare()))
    return success();
  do {
    OpAsmParser::UnresolvedOperand src, dst;
    if (parser.parseLParen() || parser.parseOperand(src) ||
        parser.parseComma() || parser.parseOperand(dst) ||
        parser.parseRParen())
      return failure();
    srcMems.push_back(src);
    dstMems.push_back(dst);
  } while (succeeded(parser.parseOptionalComma()));
  return parser.parseRSquare();
}

/// Finish building a processor-like op after name and pairs have been parsed.
static ParseResult resolveProcessorOperands(
    OpAsmParser &parser, OperationState &result,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &srcMems,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dstMems) {
  auto memType = MemHandleType::get(parser.getContext());
  if (parser.resolveOperands(srcMems, memType, result.operands) ||
      parser.resolveOperands(dstMems, memType, result.operands))
    return failure();
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {static_cast<int32_t>(srcMems.size()),
           static_cast<int32_t>(dstMems.size())}));
  result.addTypes(ArchHandleType::get(parser.getContext()));
  return success();
}

/// Print the mem-pair list: `[` `(` %src, %dst `)` , ... `]`
static void printMemPairs(OpAsmPrinter &p, OperandRange srcMems,
                          OperandRange dstMems) {
  p << "[";
  for (size_t i = 0, e = srcMems.size(); i < e; ++i) {
    if (i > 0)
      p << ", ";
    p << "(";
    p.printOperand(srcMems[i]);
    p << ", ";
    p.printOperand(dstMems[i]);
    p << ")";
  }
  p << "]";
}

//--- ProcessorComputeOp ---------------------------------------------------
// Name is a FlatSymbolRefAttr (prints/parses as @symbol).

ParseResult ProcessorComputeOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  FlatSymbolRefAttr symNameAttr;
  SmallVector<OpAsmParser::UnresolvedOperand> srcMems, dstMems;
  if (parser.parseAttribute(symNameAttr) || parser.parseComma() ||
      parseMemPairs(parser, srcMems, dstMems))
    return failure();
  result.addAttribute("sym_name", symNameAttr);
  return resolveProcessorOperands(parser, result, srcMems, dstMems);
}

void ProcessorComputeOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printAttribute(getSymNameAttr()); // prints @name
  p << ", ";
  printMemPairs(p, getSrcMems(), getDstMems());
}

LogicalResult ProcessorComputeOp::verify() {
  if (getSrcMems().size() != getDstMems().size())
    return emitOpError("expected equal number of source and destination "
                       "memory operands in region pairs");
  return success();
}

//--- ProcessorDMoverOp ----------------------------------------------------
// Name is a StrAttr + Symbol trait (prints/parses as "name").

ParseResult ProcessorDMoverOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  StringAttr symNameAttr;
  SmallVector<OpAsmParser::UnresolvedOperand> srcMems, dstMems;
  if (parser.parseAttribute(symNameAttr) || parser.parseComma() ||
      parseMemPairs(parser, srcMems, dstMems))
    return failure();
  result.addAttribute("sym_name", symNameAttr);
  return resolveProcessorOperands(parser, result, srcMems, dstMems);
}

void ProcessorDMoverOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printAttributeWithoutType(getSymNameAttr()); // prints "name"
  p << ", ";
  printMemPairs(p, getSrcMems(), getDstMems());
}

LogicalResult ProcessorDMoverOp::verify() {
  if (getSrcMems().size() != getDstMems().size())
    return emitOpError("expected equal number of source and destination "
                       "memory operands in region pairs");
  return success();
}

#define GET_OP_CLASSES
#include "ADLOps.cpp.inc"
