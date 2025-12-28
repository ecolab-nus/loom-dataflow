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
using namespace loom::df;

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

#define GET_OP_CLASSES
#include "DataflowOps.cpp.inc"

//===----------------------------------------------------------------------===//
// CoreOp Custom Assembly Format
//===----------------------------------------------------------------------===//

/// Parse CoreOp with custom syntax:
///   df.core "label" {scaleout=(%x, %y), scalein=(%mat_unit, %vec_unit, [8,1])}
ParseResult CoreOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse label attribute
  StringAttr label;
  if (parser.parseAttribute(label, "label", result.attributes))
    return failure();

  // Parse attributes dictionary which may contain scaleout and scalein
  SmallVector<OpAsmParser::UnresolvedOperand> scaleoutOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> scaleinOperands;
  DenseI64ArrayAttr scaleinCounts;
  
  if (succeeded(parser.parseOptionalLBrace())) {
    // Parse scaleout=(%x, %y, ...)
    if (succeeded(parser.parseOptionalKeyword("scaleout"))) {
      if (parser.parseEqual() || parser.parseLParen())
        return failure();
      
      if (parser.parseOperandList(scaleoutOperands))
        return failure();
      
      if (parser.parseRParen())
        return failure();
    }

    // Parse optional comma before scalein (only if scaleout exists)
    if (!scaleoutOperands.empty()) {
      if (parser.parseOptionalComma().succeeded()) {
        // Comma was found, continue
      }
      // If comma not found, it's okay - might be directly followed by scalein or closing brace
    }

    // Parse scalein=(%mat_unit, %vec_unit, [8,1])
    if (succeeded(parser.parseOptionalKeyword("scalein"))) {
      if (parser.parseEqual() || parser.parseLParen())
        return failure();
      
      // Parse functional unit operands - need to handle mixed syntax with array literal
      // We parse until we see a '[' which indicates the start of counts array
      while (true) {
        // Check if next token is '[' (start of counts array)
        if (succeeded(parser.parseOptionalLSquare())) {
          // Parse counts array
          SmallVector<int64_t> counts;
          if (parser.parseCommaSeparatedList([&]() -> ParseResult {
                int64_t count;
                if (parser.parseInteger(count))
                  return failure();
                counts.push_back(count);
                return success();
              }) || parser.parseRSquare()) {
            return failure();
          }
          scaleinCounts = parser.getBuilder().getDenseI64ArrayAttr(counts);
          break;
        }
        
        // Try to parse an operand (with or without % prefix)
        OpAsmParser::UnresolvedOperand operand;
        if (parser.parseOperand(operand).failed()) {
          // If parsing operand failed, we might have reached the end
          // Check if we can parse ')' or if there's an error
          break;
        }
        scaleinOperands.push_back(operand);
        
        // Check if there's a comma (more operands or counts array coming)
        if (parser.parseOptionalComma().failed()) {
          // No comma, we're done with operands
          break;
        }
      }
      
      if (parser.parseRParen())
        return failure();
    }

    // Parse other attributes if any
    if (parser.parseOptionalComma().succeeded()) {
      if (parser.parseOptionalAttrDict(result.attributes).failed())
        return failure();
    }

    if (parser.parseRBrace())
      return failure();
  } else {
    // No braces, just parse optional attributes
    if (parser.parseOptionalAttrDict(result.attributes).failed())
      return failure();
  }

  // Track the starting size before adding operands
  unsigned numOperandsBefore = result.operands.size();

  // Resolve types for scaleout operands (all should be index type)
  SmallVector<Type> scaleoutTypes(scaleoutOperands.size(),
                                  parser.getBuilder().getIndexType());
  if (!scaleoutOperands.empty() &&
      parser.resolveOperands(scaleoutOperands, scaleoutTypes, parser.getNameLoc(),
                             result.operands))
    return failure();
  unsigned scaleoutSize = result.operands.size() - numOperandsBefore;

  // Resolve types for scalein operands (all should be functional_unit type)
  auto funcUnitType = FunctionalUnitHandleType::get(parser.getBuilder().getContext());
  SmallVector<Type> scaleinTypes(scaleinOperands.size(), funcUnitType);
  if (!scaleinOperands.empty() &&
      parser.resolveOperands(scaleinOperands, scaleinTypes, parser.getNameLoc(),
                             result.operands))
    return failure();
  unsigned scaleinSize = result.operands.size() - numOperandsBefore - scaleoutSize;

  // Set operand segment sizes attribute (required for multiple variadic operands)
  Builder &builder = parser.getBuilder();
  auto segmentSizes = builder.getDenseI32ArrayAttr({
      static_cast<int32_t>(scaleoutSize),
      static_cast<int32_t>(scaleinSize)
  });
  result.addAttribute("operand_segment_sizes", segmentSizes);

  // Set scalein_counts attribute if we parsed it
  if (scaleinCounts)
    result.addAttribute("scalein_counts", scaleinCounts);

  // Set result type
  result.addTypes(ComputeHandleType::get(parser.getBuilder().getContext()));

  return success();
}

/// Print CoreOp with custom syntax
void CoreOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getLabelAttr());
  
  bool hasScaleout = !getScaleout().empty();
  bool hasScalein = !getScaleinUnits().empty();
  
  if (hasScaleout || hasScalein) {
    p << " {";
    
    // Print scaleout if present
    if (hasScaleout) {
      p << "scaleout=(";
      p.printOperands(getScaleout());
      p << ")";
    }
    
    // Print scalein if present
    if (hasScalein) {
      if (hasScaleout)
        p << " ,";
      p << " scalein=(";
      p.printOperands(getScaleinUnits());
      
      // Print counts array if present
      if (auto countsAttr = getScaleinCounts()) {
        p << ", [";
        // DenseI64ArrayAttr can be converted to ArrayRef, access values directly
        ArrayRef<int64_t> counts = *countsAttr;
        for (size_t i = 0, e = counts.size(); i < e; ++i) {
          if (i > 0)
            p << ", ";
          p << counts[i];
        }
        p << "]";
      }
      p << ")";
    }
    
    p << "}";
  }
  
}

/// Verify CoreOp invariants
LogicalResult CoreOp::verify() {
  // Verify that if scalein_counts is provided, its size matches scalein_units count
  if (auto counts = getScaleinCounts()) {
    size_t unitsCount = getScaleinUnits().size();
    size_t countsSize = counts->size();
    
    if (unitsCount != countsSize) {
      return emitOpError() << "scalein_counts array size (" << countsSize
                           << ") must match the number of scalein_units (" 
                           << unitsCount << ")";
    }
  }
  
  // Verify operand types are correct (this is typically checked by TableGen,
  // but we can add additional checks if needed)
  
  return success();
}

//===----------------------------------------------------------------------===//
// MatOp Custom Assembly Format
//===----------------------------------------------------------------------===//

/// Parse MatOp with custom syntax:
///   df.mat "FPU" {shape = [32, 32, 32]}
ParseResult MatOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse name attribute
  StringAttr name;
  if (parser.parseAttribute(name, "name", result.attributes))
    return failure();

  // Parse attributes dictionary - handle shape = [32, 32, 32] format
  if (succeeded(parser.parseOptionalLBrace())) {
    // Check if shape attribute is present
    if (succeeded(parser.parseOptionalKeyword("shape"))) {
      if (parser.parseEqual())
        return failure();
      
      // Parse [32, 32, 32] format
      if (parser.parseLSquare())
        return failure();
      
      SmallVector<int64_t> shape;
      if (parser.parseCommaSeparatedList([&]() -> ParseResult {
            int64_t dim;
            if (parser.parseInteger(dim))
              return failure();
            shape.push_back(dim);
            return success();
          }) || parser.parseRSquare()) {
        return failure();
      }
      
      // Create DenseI64ArrayAttr
      result.addAttribute("shape", 
                         parser.getBuilder().getDenseI64ArrayAttr(shape));
    }

    // Optional comma before throughput if both are present.
    (void)parser.parseOptionalComma();

    // Optional throughput attribute: throughput = <int>
    if (succeeded(parser.parseOptionalKeyword("throughput"))) {
      if (parser.parseEqual())
        return failure();

      int64_t throughputValue;
      if (parser.parseInteger(throughputValue))
        return failure();

      result.addAttribute("throughput",
                          parser.getBuilder().getI64IntegerAttr(throughputValue));
    }

    if (parser.parseRBrace())
      return failure();
  }

  // Set result type
  result.addTypes(FunctionalUnitHandleType::get(parser.getBuilder().getContext()));

  return success();
}

/// Print MatOp with custom syntax
void MatOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getNameAttr());
  p << " {shape = [";
  auto shape = getShape();
  ArrayRef<int64_t> values = shape;
  for (size_t i = 0, e = values.size(); i < e; ++i) {
    if (i > 0)
      p << ", ";
    p << values[i];
  }
  p << "]";
  if (auto throughput = getThroughput()) {
    p << ", throughput = " << throughput.value();
  }
  p << "}";
}

//===----------------------------------------------------------------------===//
// VecOp Custom Assembly Format
//===----------------------------------------------------------------------===//

/// Parse VecOp with custom syntax:
///   df.vec "SFPU" {shape = [32]}
ParseResult VecOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse name attribute
  StringAttr name;
  if (parser.parseAttribute(name, "name", result.attributes))
    return failure();

  // Parse attributes dictionary - handle shape = [32] format
  if (succeeded(parser.parseOptionalLBrace())) {
    // Check if shape attribute is present
    if (succeeded(parser.parseOptionalKeyword("shape"))) {
      if (parser.parseEqual())
        return failure();
      
      // Parse [32] format
      if (parser.parseLSquare())
        return failure();
      
      SmallVector<int64_t> shape;
      if (parser.parseCommaSeparatedList([&]() -> ParseResult {
            int64_t dim;
            if (parser.parseInteger(dim))
              return failure();
            shape.push_back(dim);
            return success();
          }) || parser.parseRSquare()) {
        return failure();
      }
      
      // Create DenseI64ArrayAttr
      result.addAttribute("shape", 
                         parser.getBuilder().getDenseI64ArrayAttr(shape));
    }

    // Optional comma before throughput if both are present.
    (void)parser.parseOptionalComma();

    // Optional throughput attribute: throughput = <int>
    if (succeeded(parser.parseOptionalKeyword("throughput"))) {
      if (parser.parseEqual())
        return failure();

      int64_t throughputValue;
      if (parser.parseInteger(throughputValue))
        return failure();

      result.addAttribute("throughput",
                          parser.getBuilder().getI64IntegerAttr(throughputValue));
    }

    if (parser.parseRBrace())
      return failure();
  }

  // Set result type
  result.addTypes(FunctionalUnitHandleType::get(parser.getBuilder().getContext()));

  return success();
}

/// Print VecOp with custom syntax
void VecOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getNameAttr());
  p << " {shape = [";
  auto shape = getShape();
  ArrayRef<int64_t> values = shape;
  for (size_t i = 0, e = values.size(); i < e; ++i) {
    if (i > 0)
      p << ", ";
    p << values[i];
  }
  p << "]";
  if (auto throughput = getThroughput()) {
    p << ", throughput = " << throughput.value();
  }
  p << "}";
}

//===----------------------------------------------------------------------===//
// MemoryOp Custom Assembly Format
//===----------------------------------------------------------------------===//

/// Parse MemoryOp with custom syntax:
///   df.memory "L1" {scaleout=(%x, %y), size = 32768, bandwidth = 64}
ParseResult MemoryOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse label attribute
  StringAttr label;
  if (parser.parseAttribute(label, "label", result.attributes))
    return failure();

  // Parse attributes dictionary which may contain scaleout, size, and bandwidth
  SmallVector<OpAsmParser::UnresolvedOperand> scaleoutOperands;
  
  if (succeeded(parser.parseOptionalLBrace())) {
    // Parse scaleout=(%x, %y, ...) if present
    if (succeeded(parser.parseOptionalKeyword("scaleout"))) {
      if (parser.parseEqual() || parser.parseLParen())
        return failure();
      
      if (parser.parseOperandList(scaleoutOperands))
        return failure();
      
      if (parser.parseRParen())
        return failure();
    }

    // Parse optional comma after scaleout before other attributes
    bool hasCommaAfterScaleout = false;
    if (!scaleoutOperands.empty()) {
      hasCommaAfterScaleout = parser.parseOptionalComma().succeeded();
    }

    // Manually parse size and bandwidth attributes
    // Parse size = <value> if present
    if (succeeded(parser.parseOptionalKeyword("size"))) {
      if (parser.parseEqual())
        return failure();
      
      int64_t sizeValue;
      if (parser.parseInteger(sizeValue))
        return failure();
      
      result.addAttribute("size", 
                         parser.getBuilder().getI64IntegerAttr(sizeValue));
      
      // Parse optional comma before bandwidth
      (void)parser.parseOptionalComma();
    } else if (hasCommaAfterScaleout) {
      // If we had a comma after scaleout but no size keyword, 
      // we need to consume it or it will cause parsing issues
      // Actually, the comma was already consumed, so we're fine
    }

    // Parse bandwidth = <value> if present
    if (succeeded(parser.parseOptionalKeyword("bandwidth"))) {
      if (parser.parseEqual())
        return failure();
      
      int64_t bandwidthValue;
      if (parser.parseInteger(bandwidthValue))
        return failure();
      
      result.addAttribute("bandwidth", 
                         parser.getBuilder().getI64IntegerAttr(bandwidthValue));
      
      // Parse optional comma before other attributes
      (void)parser.parseOptionalComma();
    }

    // Parse any remaining attributes (for backward compatibility)
    // Only parse if we haven't already consumed all attributes
    // parseOptionalAttrDict will handle the case where there are no more attributes
    (void)parser.parseOptionalAttrDict(result.attributes);

    if (parser.parseRBrace())
      return failure();
  } else {
    // No braces, just parse optional attributes
    if (parser.parseOptionalAttrDict(result.attributes).failed())
      return failure();
  }

  // Resolve types for scaleout operands (all should be index type)
  SmallVector<Type> scaleoutTypes(scaleoutOperands.size(),
                                  parser.getBuilder().getIndexType());
  if (!scaleoutOperands.empty() &&
      parser.resolveOperands(scaleoutOperands, scaleoutTypes, parser.getNameLoc(),
                             result.operands))
    return failure();

  // Set result type
  result.addTypes(MemoryHandleType::get(parser.getBuilder().getContext()));

  return success();
}

/// Print MemoryOp with custom syntax
void MemoryOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getLabelAttr());
  
  bool hasScaleout = !getScaleout().empty();
  bool hasSize = static_cast<bool>(getSizeAttr());
  bool hasBandwidth = static_cast<bool>(getBandwidthAttr());
  
  // Only print braces if there are attributes to print
  if (hasScaleout || hasSize || hasBandwidth) {
    p << " {";
    
    // Print scaleout if present
    if (hasScaleout) {
      p << "scaleout=(";
      p.printOperands(getScaleout());
      p << ")";
    }
    
    // Print size if present
    if (hasSize) {
      if (hasScaleout)
        p << " ,";
      p << " size = ";
      p << getSize();
    }
    
    // Print bandwidth if present
    if (hasBandwidth) {
      if (hasScaleout || hasSize)
        p << ", ";
      p << "bandwidth = ";
      p << getBandwidth();
    }
    
    // Print any other attributes (excluding label, size, bandwidth)
    SmallVector<StringRef> elidedAttrs = {"label", "size", "bandwidth"};
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
    
    p << "}";
  } else {
    // No attributes, just print empty dict or other attributes
    p.printOptionalAttrDict((*this)->getAttrs(), {"label"});
  }
}

//===----------------------------------------------------------------------===//
// MuxOp Custom Assembly Format
//===----------------------------------------------------------------------===//

/// Parse MuxOp with custom syntax:
///   df.mux %cores, %memories, {map = affine_map<(d0, d1) -> (d0, d1)>}
///   or: df.mux %cores: !df.compute, %memories: !df.memory, %x, %y {map = ...}
ParseResult MuxOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse source operand
  OpAsmParser::UnresolvedOperand source;
  if (parser.parseOperand(source))
    return failure();

  // Parse optional type annotation
  Type sourceType;
  bool hasSourceType = false;
  if (succeeded(parser.parseOptionalColon())) {
    if (parser.parseType(sourceType).failed())
      return failure();
    hasSourceType = true;
  }

  // Parse comma
  if (parser.parseComma())
    return failure();

  // Parse target operand
  OpAsmParser::UnresolvedOperand target;
  if (parser.parseOperand(target))
    return failure();

  // Parse optional type annotation
  Type targetType;
  bool hasTargetType = false;
  if (succeeded(parser.parseOptionalColon())) {
    if (parser.parseType(targetType).failed())
      return failure();
    hasTargetType = true;
  }

  // Parse optional indices and comma
  SmallVector<OpAsmParser::UnresolvedOperand> indices;
  if (parser.parseOptionalComma().succeeded()) {
    // Try to parse indices - if next token is '{', parseOperandList will fail
    // and we'll just continue to parse attributes
    (void)parser.parseOperandList(indices);
  }

  // Parse attributes dictionary
  if (parser.parseOptionalAttrDict(result.attributes).failed())
    return failure();

  // Resolve operand types
  // If types were not provided, we need to use placeholder types
  // The actual types will be inferred from the operands during resolution
  if (!hasSourceType) {
    // Use a placeholder - we'll infer from context
    // In practice, we can try to get the type from the SSA value if it's already defined
    sourceType = ComputeHandleType::get(parser.getBuilder().getContext());
  }
  if (!hasTargetType) {
    targetType = MemoryHandleType::get(parser.getBuilder().getContext());
  }

  SmallVector<Type> operandTypes = {sourceType, targetType};
  SmallVector<OpAsmParser::UnresolvedOperand> allOperands = {source, target};
  allOperands.append(indices.begin(), indices.end());

  // Resolve types for indices (all should be index type)
  SmallVector<Type> indexTypes(indices.size(), parser.getBuilder().getIndexType());
  operandTypes.append(indexTypes.begin(), indexTypes.end());

  if (parser.resolveOperands(allOperands, operandTypes, parser.getNameLoc(),
                             result.operands))
    return failure();

  // Set result type
  result.addTypes(MuxHandleType::get(parser.getBuilder().getContext()));

  return success();
}

/// Print MuxOp with custom syntax
void MuxOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printOperand(getSource());
  p << " : ";
  p.printType(getSource().getType());
  p << ", ";
  p.printOperand(getTarget());
  p << " : ";
  p.printType(getTarget().getType());
  
  if (!getIndices().empty()) {
    p << ", ";
    p.printOperands(getIndices());
  }
  
  p << " ";
  p.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// InterconnectsOp Custom Assembly Format
//===----------------------------------------------------------------------===//

/// Parse InterconnectsOp with custom syntax:
///   df.interconnects "horizontal_links" %memories, %memories, {map = ..., bandwidth = 128}
///   or: df.interconnects %memories: !df.memory, %drams : !df.memory, {map = ...}
/// 
/// This parser uses a flexible function-object based approach to handle various
/// format combinations (with/without sym_name, type annotations, indices, etc.)
ParseResult InterconnectsOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();
  MLIRContext *context = builder.getContext();

  // Parse symbol name (required for Symbol trait, but allow optional for backward compatibility)
  StringAttr symName;
  std::string symNameStr;
  if (succeeded(parser.parseOptionalString(&symNameStr))) {
    symName = builder.getStringAttr(symNameStr);
  } else {
    // Generate a default name if not provided (for backward compatibility)
    // Use a simple default name - if there are conflicts, MLIR's Symbol system will report an error
    // In practice, users should provide explicit names for interconnects
    symName = builder.getStringAttr("interconnect");
  }
  result.addAttribute("sym_name", symName);

  // Helper function to parse an operand with optional type annotation
  // Returns the operand and whether a type was provided
  auto parseOperandWithType = [&](OpAsmParser::UnresolvedOperand &operand,
                                  Type &type, bool &hasType) -> ParseResult {
    if (parser.parseOperand(operand))
      return failure();
    
    hasType = false;
    if (succeeded(parser.parseOptionalColon())) {
      if (parser.parseType(type).failed())
        return failure();
      hasType = true;
    }
    return success();
  };

  // Helper function to validate that a type is a valid df handle type
  auto isValidHandleType = [](Type type) -> bool {
    return llvm::isa<ComputeHandleType>(type) ||
           llvm::isa<MemoryHandleType>(type);
  };

  // Parse source operand with optional type
  OpAsmParser::UnresolvedOperand source;
  Type sourceType;
  bool hasSourceType = false;
  if (parseOperandWithType(source, sourceType, hasSourceType).failed())
    return failure();

  // Validate source type if provided
  if (hasSourceType && !isValidHandleType(sourceType)) {
    return parser.emitError(parser.getNameLoc(),
                           "source type must be a df handle type (!df.compute, "
                           "!df.memory, etc.)");
  }

  // Parse comma separator
  if (parser.parseComma())
    return failure();

  // Parse target operand with optional type
  OpAsmParser::UnresolvedOperand target;
  Type targetType;
  bool hasTargetType = false;
  if (parseOperandWithType(target, targetType, hasTargetType).failed())
    return failure();

  // Validate target type if provided
  if (hasTargetType && !isValidHandleType(targetType)) {
    return parser.emitError(parser.getNameLoc(),
                           "target type must be a df handle type (!df.compute, "
                           "!df.memory, etc.)");
  }

  // Parse optional indices (comma-separated list after target)
  SmallVector<OpAsmParser::UnresolvedOperand> indices;
  if (parser.parseOptionalComma().succeeded()) {
    // Try to parse indices - if next token is '{', parseOperandList will fail
    // gracefully and we'll continue to parse attributes
    (void)parser.parseOperandList(indices);
  }

  // Parse attributes dictionary (map, bandwidth, etc.)
  if (parser.parseOptionalAttrDict(result.attributes).failed())
    return failure();

  // Parse optional result type annotation
  Type resultType;
  if (succeeded(parser.parseOptionalColon())) {
    if (parser.parseType(resultType).failed())
      return failure();
  } else {
    // No result type provided, use default
    resultType = InterconnectHandleType::get(context);
  }

  // Resolve operand types
  // If types were not provided, use default placeholder types
  if (!hasSourceType) {
    sourceType = MemoryHandleType::get(context);
  }
  if (!hasTargetType) {
    targetType = MemoryHandleType::get(context);
  }

  // Build operand types list: source, target, then indices
  SmallVector<Type> operandTypes = {sourceType, targetType};
  SmallVector<OpAsmParser::UnresolvedOperand> allOperands = {source, target};
  allOperands.append(indices.begin(), indices.end());

  // Resolve types for indices (all should be index type)
  SmallVector<Type> indexTypes(indices.size(), builder.getIndexType());
  operandTypes.append(indexTypes.begin(), indexTypes.end());

  // Resolve all operands
  if (parser.resolveOperands(allOperands, operandTypes, parser.getNameLoc(),
                             result.operands))
    return failure();

  // Set result type
  result.addTypes(resultType);

  return success();
}

/// Print InterconnectsOp with custom syntax
void InterconnectsOp::print(OpAsmPrinter &p) {
  p << " ";
  
  // Print symbol name
  p << "\"";
  p << getSymName();
  p << "\" ";
  
  p.printOperand(getSource());
  p << " : ";
  p.printType(getSource().getType());
  p << ", ";
  p.printOperand(getTarget());
  p << " : ";
  p.printType(getTarget().getType());
  
  if (!getIndices().empty()) {
    p << ", ";
    p.printOperands(getIndices());
  }
  
  p << " ";
  // Elide sym_name from attribute dict since it's printed separately
  // spatial_dims will be printed in the attribute dict automatically
  SmallVector<StringRef> elidedAttrs = {"sym_name"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  
  p << " : ";
  p.printType(getHandle().getType());
}
