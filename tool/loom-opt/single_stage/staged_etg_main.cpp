#include "staged_etg_builder.h"
#include "compute_op_registry.h"
#include "driver_utils.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
using namespace loom::lcs;

static llvm::cl::opt<std::string>
    clInput("input", llvm::cl::desc("Path to input MLIR file"),
            llvm::cl::value_desc("filename"), llvm::cl::Required);

static llvm::cl::opt<std::string>
    clOutput("output", llvm::cl::desc("Path to output JSON file"),
             llvm::cl::value_desc("filename"),
             llvm::cl::init("staged_etg_dump.json"));

static llvm::cl::opt<std::string>
    clHWPlatformFile("hw_spec",
                     llvm::cl::desc("Hardware platform MLIR file"),
                     llvm::cl::value_desc("filename"), llvm::cl::Required);

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                     "LOOM Staged-ETG Analysis Tool\n");

  MLIRContext context;
  loom::driver::registerLoomAndADLDialects(context);
  context.loadDialect<mlir::math::MathDialect>();

  loom::lcs::HWOpRegistry registry;
  if (mlir::failed(registry.loadFromPlatformFile(clHWPlatformFile, context))) {
    llvm::errs() << "Failed to load platform IR from: " << clHWPlatformFile << "\n";
    return 1;
  }

  auto module = loom::driver::parseMLIRFile(clInput, context);
  if (!module) return 1;

  llvm::json::Array json_etgs;

  // Iterate over all functions and variants effectively
  module->walk([&](mlir::func::FuncOp func_op) {
    mlir::scf::ForOp target_loop = nullptr;

    // Find the scf.for loop with loom.iter_type = sequential
    func_op.walk([&](mlir::scf::ForOp for_op) {
      if (for_op->hasAttr("loom.iter_type")) {
        std::string attr_str;
        llvm::raw_string_ostream os(attr_str);
        for_op->getAttr("loom.iter_type").print(os);
        if (attr_str.find("sequential") != std::string::npos) {
          target_loop = for_op;
        }
      }
    });

    if (target_loop) {
      VariantETG etg(func_op.getName(), &registry);
      etg.buildFromSCFFor(target_loop);
      etg.buildConstraintScope(func_op);
      etg.buildL1FootprintConstraint();
      json_etgs.push_back(etg.toJSON());
    }
  });

  std::error_code ec;
  llvm::raw_fd_ostream output(clOutput, ec);
  if (ec) {
    llvm::errs() << "Failed to open output file: " << clOutput << "\n";
    return 1;
  }

  // Custom pretty-printer: collapses Expr / ConstraintExpr nodes
  auto isExprKind = [](llvm::StringRef key) -> bool {
    static const llvm::StringRef kKinds[] = {
        "Const", "Sym", "Add", "Sub", "Mul", "Div", "Min", "Max", "IfElse",
        "And", "Or", "Not", "Eq", "Le", "Lt", "Ge", "Gt",
        "Divisible", "InRange",
    };
    for (auto k : kKinds)
      if (k == key)
        return true;
    return false;
  };

  auto isExprNode = [&](const llvm::json::Value &val) -> bool {
    if (const auto *obj = val.getAsObject())
      if (obj->size() == 1)
        return isExprKind(obj->begin()->first);
    return false;
  };

  auto isStringArray = [](const llvm::json::Value &val) -> bool {
    const auto *arr = val.getAsArray();
    if (!arr || arr->empty()) return false;
    for (const auto &elem : *arr)
      if (!elem.getAsString()) return false;
    return true;
  };

  auto isSymMapEntry = [&](const llvm::json::Value &val) -> bool {
    const auto *arr = val.getAsArray();
    if (!arr || arr->size() != 2) return false;
    return (*arr)[0].getAsString().has_value() && isExprNode((*arr)[1]);
  };

  std::function<void(llvm::raw_ostream &, const llvm::json::Value &, int)>
      writeJSON = [&](llvm::raw_ostream &os, const llvm::json::Value &val,
                      int indent) {
        if (isExprNode(val)) {
          os << llvm::formatv("{0}", val);
          return;
        }
        if (isStringArray(val)) {
          os << llvm::formatv("{0}", val);
          return;
        }
        if (isSymMapEntry(val)) {
          os << llvm::formatv("{0}", val);
          return;
        }
        if (const auto *arr = val.getAsArray()) {
          if (arr->empty()) { os << "[]"; return; }
          os << "[\n";
          for (size_t i = 0; i < arr->size(); ++i) {
            os.indent(indent + 2);
            writeJSON(os, (*arr)[i], indent + 2);
            if (i + 1 < arr->size()) os << ",";
            os << "\n";
          }
          os.indent(indent) << "]";
          return;
        }
        if (const auto *obj = val.getAsObject()) {
          if (obj->empty()) { os << "{}"; return; }
          os << "{\n";
          size_t i = 0, size = obj->size();
          for (const auto &kv : *obj) {
            os.indent(indent + 2);
            os << "\"" << kv.first << "\": ";
            writeJSON(os, kv.second, indent + 2);
            if (++i < size) os << ",";
            os << "\n";
          }
          os.indent(indent) << "}";
          return;
        }
        os << llvm::formatv("{0}", val);
      };

  llvm::json::Value root(std::move(json_etgs));
  writeJSON(output, root, 0);
  output << "\n";

  return 0;
}
