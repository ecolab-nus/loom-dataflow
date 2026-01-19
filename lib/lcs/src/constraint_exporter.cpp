//===- constraint_exporter.cpp - Loom Constraint JSON Exporter ------------===//
//
// Implementation of the ConstraintExporter class.
//
//===----------------------------------------------------------------------===//

#include "constraint_exporter.h"
#include "analysis_engine.h"
#include "constraint_space_utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

// Include Loom dialect headers
#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

namespace loom {
namespace lcs {

ConstraintExporter::ConstraintExporter(mlir::Operation *csOp,
                                       llvm::StringRef passName)
    : csOp_(csOp), passName_(passName.str()) {
  buildVariableMapping();
}

void ConstraintExporter::buildVariableMapping() {
  unsigned sIdx = 0;
  csOp_->walk([&](mlir::Operation *op) {
    llvm::TypeSwitch<mlir::Operation *>(op)
        .Case<loom::SymbolicVarOp>([&](loom::SymbolicVarOp varOp) {
          std::string name = varOp.getName().str();
          valueToName_[varOp.getResult()] = name;
        })
        .Case<loom::IntermediateVarOp>([&](loom::IntermediateVarOp varOp) {
          // Default naming for intermediate variables if not already named
          std::string name = "S" + std::to_string(sIdx++);
          valueToName_[varOp.getResult()] = name;
        });
  });
}

std::string ConstraintExporter::getValueName(mlir::Value val) {
  auto it = valueToName_.find(val);
  if (it != valueToName_.end())
    return it->second;
  return "unknown";
}

ConstraintExport
ConstraintExporter::visitPolynomial(loom::PolynomialConstraintOp op) {
  ConstraintExport result;
  result.type = "polynomial";
  result.upperBound = op.getUpperBound();

  auto monomials = loom::lcs::parseMonomials(op.getMonomials());
  for (const auto &m : monomials) {
    TermExport term;
    term.coefficient = m.coeff;
    for (int64_t vIdx : m.varIndices) {
      if (vIdx >= 0 && static_cast<size_t>(vIdx) < op.getNumOperands()) {
        term.variables.push_back(getValueName(op.getOperand(vIdx)));
      }
    }
    result.terms.push_back(term);
  }
  return result;
}

ConstraintExport ConstraintExporter::visitLinear(loom::LinearConstraintOp op) {
  ConstraintExport result;
  result.type = "linear";
  result.isEquality = false; // LinearConstraintOp results are usually >= 0

  mlir::AffineMap map = op.getMap();
  unsigned numDims = op.getNumOperands();

  for (mlir::AffineExpr expr : map.getResults()) {
    llvm::SmallVector<int64_t, 8> coeffs;
    int64_t constant;
    lcs::CoefficientExtractor extractor(numDims, coeffs, constant);
    extractor.visit(expr);

    if (extractor.succeeded()) {
      ConstraintExport subResult;
      subResult.type = "linear";
      subResult.constant = constant;
      for (unsigned i = 0; i < coeffs.size(); ++i) {
        if (coeffs[i] != 0) {
          TermExport term;
          term.coefficient = coeffs[i];
          term.variables.push_back(getValueName(op.getOperand(i)));
          subResult.terms.push_back(term);
        }
      }
      // Note: If multiple results, they should probably be separate constraints
      // in JSON But for simplicity in this implementation, we return one at a
      // time if called correctly. Actually, we'll return the first one or we
      // should refactor to return a list. Let's just return the first one for
      // now, but handle multiple in exportToStruct.
      return subResult;
    }
  }
  return result;
}

ConstraintSpaceExport ConstraintExporter::exportToStruct() {
  ConstraintSpaceExport res;
  res.passName = passName_;

  // Try to find related function name
  if (auto funcOp = csOp_->getParentOfType<mlir::func::FuncOp>()) {
    res.functionName = funcOp.getName().str();
  } else if (auto parentModule = csOp_->getParentOfType<mlir::ModuleOp>()) {
    // If inside a wrapper module, find the first function
    parentModule.walk([&](mlir::func::FuncOp func) {
      res.functionName = func.getName().str();
      return mlir::WalkResult::interrupt();
    });
  }

  // Fallback to internal name if function not found
  if (res.functionName.empty()) {
    if (auto csOp = llvm::dyn_cast<loom::ConstraintSpaceOp>(csOp_))
      res.functionName = csOp.getSymName().str();
    else
      res.functionName = "unknown";
  }

  // Collect variables
  csOp_->walk([&](mlir::Operation *op) {
    llvm::TypeSwitch<mlir::Operation *>(op)
        .Case<loom::SymbolicVarOp>([&](loom::SymbolicVarOp varOp) {
          res.variables.push_back(
              {varOp.getName().str(), 0, /*isIntermediate=*/false});
        })
        .Case<loom::IntermediateVarOp>([&](loom::IntermediateVarOp varOp) {
          res.variables.push_back(
              {getValueName(varOp.getResult()), 0, /*isIntermediate=*/true});
        });
  });

  // Assign indices based on appearance
  for (unsigned i = 0; i < res.variables.size(); ++i) {
    res.variables[i].index = i;
  }

  // Collect metadata
  csOp_->walk([&](mlir::Operation *op) {
    llvm::TypeSwitch<mlir::Operation *>(op)
        .Case<loom::RangeOp>([&](loom::RangeOp rangeOp) {
          MetadataExport m;
          m.variable = getValueName(rangeOp.getVariable());
          m.type = "range";
          m.lowerBound = rangeOp.getLowerBound();
          m.upperBound = rangeOp.getUpperBound();
          res.metadata.push_back(m);
        })
        .Case<loom::AlignOp>([&](loom::AlignOp alignOp) {
          MetadataExport m;
          m.variable = getValueName(alignOp.getVariable());
          m.type = "align";
          m.alignment = alignOp.getAlignment();
          res.metadata.push_back(m);
        });
  });

  // Collect constraints
  csOp_->walk([&](mlir::Operation *op) {
    if (auto polyOp = llvm::dyn_cast<loom::PolynomialConstraintOp>(op)) {
      res.constraints.push_back(visitPolynomial(polyOp));
    } else if (auto linearOp = llvm::dyn_cast<loom::LinearConstraintOp>(op)) {
      // Handle multiple results in linear map
      mlir::AffineMap map = linearOp.getMap();
      unsigned numOperands = linearOp.getNumOperands();
      for (mlir::AffineExpr expr : map.getResults()) {
        llvm::SmallVector<int64_t, 8> coeffs;
        int64_t constant;
        lcs::CoefficientExtractor extractor(numOperands, coeffs, constant);
        extractor.visit(expr);
        if (extractor.succeeded()) {
          ConstraintExport ce;
          ce.type = "linear";
          ce.constant = constant;
          for (unsigned i = 0; i < coeffs.size(); ++i) {
            if (coeffs[i] != 0) {
              TermExport term;
              term.coefficient = coeffs[i];
              term.variables.push_back(getValueName(linearOp.getOperand(i)));
              ce.terms.push_back(term);
            }
          }
          res.constraints.push_back(ce);
        }
      }
    }
  });

  return res;
}

llvm::json::Value ConstraintExporter::exportToJson() {
  ConstraintSpaceExport data = exportToStruct();

  llvm::json::Array vars;
  for (const auto &v : data.variables) {
    vars.push_back(llvm::json::Object{
        {"name", v.name},
        {"index", v.index},
        {"is_intermediate", v.isIntermediate},
    });
  }

  llvm::json::Array meta;
  for (const auto &m : data.metadata) {
    llvm::json::Object mObj{
        {"variable", m.variable},
        {"type", m.type},
    };
    if (m.lowerBound)
      mObj["lower_bound"] = *m.lowerBound;
    if (m.upperBound)
      mObj["upper_bound"] = *m.upperBound;
    if (m.alignment)
      mObj["alignment"] = *m.alignment;
    meta.push_back(std::move(mObj));
  }

  llvm::json::Array constraints;
  for (const auto &c : data.constraints) {
    llvm::json::Array terms;
    for (const auto &t : c.terms) {
      terms.push_back(llvm::json::Object{
          {"coefficient", t.coefficient},
          {"variables", llvm::json::Array(t.variables)},
      });
    }

    llvm::json::Object cObj{
        {"type", c.type},
        {"is_equality", c.isEquality},
        {"terms", std::move(terms)},
    };
    if (c.constant)
      cObj["constant"] = *c.constant;
    if (c.upperBound)
      cObj["upper_bound"] = *c.upperBound;

    constraints.push_back(std::move(cObj));
  }

  // Requested order: func_name, pass_name, variables, metadata,
  // constraints
  return llvm::json::Object{
      {"func_name", data.functionName},        {"pass_name", data.passName},
      {"variables", std::move(vars)},          {"metadata", std::move(meta)},
      {"constraints", std::move(constraints)},
  };
}

std::string ConstraintExporter::toJsonString() {
  ConstraintSpaceExport data = exportToStruct();

  // We manually build the top-level object to enforce the order requested by
  // the user: constraint_space_name, pass_name, variables, metadata,
  // constraints. We still use the JSON library for escaping and nested
  // structures.

  llvm::json::Array vars;
  for (const auto &v : data.variables) {
    vars.push_back(llvm::json::Object{
        {"name", v.name},
        {"index", v.index},
        {"is_intermediate", v.isIntermediate},
    });
  }

  llvm::json::Array meta;
  for (const auto &m : data.metadata) {
    llvm::json::Object mObj{
        {"variable", m.variable},
        {"type", m.type},
    };
    if (m.lowerBound)
      mObj["lower_bound"] = *m.lowerBound;
    if (m.upperBound)
      mObj["upper_bound"] = *m.upperBound;
    if (m.alignment)
      mObj["alignment"] = *m.alignment;
    meta.push_back(std::move(mObj));
  }

  llvm::json::Array constraints;
  for (const auto &c : data.constraints) {
    llvm::json::Array terms;
    for (const auto &t : c.terms) {
      terms.push_back(llvm::json::Object{
          {"coefficient", t.coefficient},
          {"variables", llvm::json::Array(t.variables)},
      });
    }

    llvm::json::Object cObj{
        {"type", c.type},
        {"is_equality", c.isEquality},
        {"terms", std::move(terms)},
    };
    if (c.constant)
      cObj["constant"] = *c.constant;
    if (c.upperBound)
      cObj["upper_bound"] = *c.upperBound;
    constraints.push_back(std::move(cObj));
  }

  std::string s;
  llvm::raw_string_ostream os(s);
  os << "{\n";
  os << "  \"func_name\": "
     << llvm::formatv("{0}", llvm::json::Value(data.functionName)) << ",\n";
  os << "  \"pass_name\": "
     << llvm::formatv("{0}", llvm::json::Value(data.passName)) << ",\n";
  os << "  \"variables\": "
     << llvm::formatv("{0:2}", llvm::json::Value(std::move(vars))) << ",\n";
  os << "  \"metadata\": "
     << llvm::formatv("{0:2}", llvm::json::Value(std::move(meta))) << ",\n";
  os << "  \"constraints\": "
     << llvm::formatv("{0:2}", llvm::json::Value(std::move(constraints)))
     << "\n";
  os << "}";

  return s;
}

mlir::LogicalResult ConstraintExporter::writeToFile(llvm::StringRef path) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
  if (ec)
    return mlir::failure();
  os << toJsonString();
  return mlir::success();
}

std::string exportConstraintSpaceToJson(mlir::Operation *csOp,
                                        llvm::StringRef passName) {
  ConstraintExporter exporter(csOp, passName);
  return exporter.toJsonString();
}

llvm::json::Value exportConstraintSpaceToValue(mlir::Operation *csOp,
                                               llvm::StringRef passName) {
  ConstraintExporter exporter(csOp, passName);
  return exporter.exportToJson();
}

} // namespace lcs
} // namespace loom
