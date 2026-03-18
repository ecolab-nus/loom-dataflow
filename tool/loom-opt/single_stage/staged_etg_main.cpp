#include "staged_etg_builder.h"
#include "compute_op_registry.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"

// Assuming LoomDialect and DataflowDialect are available
#include "DataflowDialect.h.inc"
#include "LoomDialect.h.inc"

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
    clHWComputeDir("hw-compute-dir",
                   llvm::cl::desc("Hardware compute IR directory"),
                   llvm::cl::value_desc("directory"), llvm::cl::Required);

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "LOOM Staged-ETG Analysis Tool\n");

  MLIRContext context;
  context.loadDialect<mlir::BuiltinDialect, mlir::func::FuncDialect,
                      mlir::arith::ArithDialect, mlir::affine::AffineDialect,
                      mlir::tensor::TensorDialect, mlir::linalg::LinalgDialect,
                      mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                      mlir::math::MathDialect, loom::LoomDialect,
                      loom::df::DataflowDialect>();

  loom::lcs::ComputeOpRegistry registry;
  if (mlir::failed(registry.loadFromDirectory(clHWComputeDir, context))) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to load hw compute IR from: " << clHWComputeDir << "\n";
    return 1;
  }

  llvm::SourceMgr sm;
  auto file = mlir::openInputFile(clInput);
  if (!file) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to open input file: " << clInput << "\n";
    return 1;
  }
  sm.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sm, &context);
  if (!module) {
    llvm::WithColor::error(llvm::errs()) << "Failed to parse MLIR module\n";
    return 1;
  }

  llvm::json::Array json_etgs;

  // Iterate over all functions andvariants effectively
  module->walk([&](mlir::func::FuncOp func_op) {
    mlir::affine::AffineForOp target_loop = nullptr;

    // Find the unique loop with loom.iter_type = sequential
    func_op.walk([&](mlir::affine::AffineForOp for_op) {
      auto iter_type_attr = for_op->getAttrOfType<StringAttr>("loom.iter_type");
      if (!iter_type_attr && for_op->getAttr("loom.iter_type")) {
        // In some cases it may be a custom attribute or just an enum-like
        // String. Let's check both string and custom attr if needed. Given the
        // plan says loom.iter_type = #loom.iter_type<sequential> We should
        // check the string value of the attribute if possible.
      }
      // For now, let's just dump any loop that has this attribute if the
      // specific enum check is complex. But usually it's handled via the Loom
      // dialect's IterTypeAttr.
      if (for_op->hasAttr("loom.iter_type")) {
        // Simple heuristic: search for "sequential" in the attribute string
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
      etg.buildFromAffineFor(target_loop);
      etg.buildConstraintScope(func_op);
      json_etgs.push_back(etg.toJSON());
    }
  });

  std::error_code ec;
  llvm::raw_fd_ostream output(clOutput, ec);
  if (ec) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to open output file: " << clOutput << "\n";
    return 1;
  }

  // Custom pretty-printer: collapses Expr / ConstraintExpr nodes
  // (single key ∈ known kind set) onto one line.
  auto isExprKind = [](llvm::StringRef key) -> bool {
    static const llvm::StringRef kKinds[] = {
        // Expr variants
        "Const", "Sym", "Add", "Sub", "Mul", "Div", "Min", "Max", "IfElse",
        // ConstraintExpr variants (True/False are bare strings, not objects)
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

  // True for arrays whose every element is a plain string.
  auto isStringArray = [](const llvm::json::Value &val) -> bool {
    const auto *arr = val.getAsArray();
    if (!arr || arr->empty())
      return false;
    for (const auto &elem : *arr)
      if (!elem.getAsString())
        return false;
    return true;
  };

  // True for a sym_map entry: a 2-element array [string, ExprNode].
  auto isSymMapEntry = [&](const llvm::json::Value &val) -> bool {
    const auto *arr = val.getAsArray();
    if (!arr || arr->size() != 2)
      return false;
    return (*arr)[0].getAsString().has_value() && isExprNode((*arr)[1]);
  };

  // Recursive pretty-printer; indent is the current indentation level.
  std::function<void(llvm::raw_ostream &, const llvm::json::Value &, int)>
      writeJSON =
          [&](llvm::raw_ostream &os, const llvm::json::Value &val,
              int indent) {
            // Expr nodes: compact one-liner using llvm's built-in formatter.
            if (isExprNode(val)) {
              os << llvm::formatv("{0}", val);
              return;
            }

            // String arrays (e.g. symbols): compact one-liner.
            if (isStringArray(val)) {
              os << llvm::formatv("{0}", val);
              return;
            }

            // sym_map entries [string, ExprNode]: compact one-liner.
            if (isSymMapEntry(val)) {
              os << llvm::formatv("{0}", val);
              return;
            }

            if (const auto *arr = val.getAsArray()) {
              if (arr->empty()) {
                os << "[]";
                return;
              }
              os << "[\n";
              for (size_t i = 0; i < arr->size(); ++i) {
                os.indent(indent + 2);
                writeJSON(os, (*arr)[i], indent + 2);
                if (i + 1 < arr->size())
                  os << ",";
                os << "\n";
              }
              os.indent(indent) << "]";
              return;
            }

            if (const auto *obj = val.getAsObject()) {
              if (obj->empty()) {
                os << "{}";
                return;
              }
              os << "{\n";
              size_t i = 0, size = obj->size();
              for (const auto &kv : *obj) {
                os.indent(indent + 2);
                llvm::StringRef key = kv.first;
                os << "\"" << key << "\": ";
                writeJSON(os, kv.second, indent + 2);
                if (++i < size)
                  os << ",";
                os << "\n";
              }
              os.indent(indent) << "}";
              return;
            }

            // Scalars (null, bool, number, string): use llvm's formatter.
            os << llvm::formatv("{0}", val);
          };

  llvm::json::Value root(std::move(json_etgs));
  writeJSON(output, root, 0);
  output << "\n";

  return 0;
}
