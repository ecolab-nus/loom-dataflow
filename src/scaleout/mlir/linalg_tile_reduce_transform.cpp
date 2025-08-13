#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/Utils.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include <iostream>

using namespace mlir;

int main(int argc, char **argv) {
  // Parse args: prog <input.mlir> [-o output.xform.mlir]
  if (argc < 2) {
    llvm::WithColor::error(llvm::errs())
        << "Usage: " << argv[0] << " <input.mlir> [-o output.mlir]\n";
    return 1;
  }
  std::string inputFilename = argv[1];
  std::string outputFilename;
  for (int i = 2; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "-o") {
      outputFilename = argv[i + 1];
      ++i;
    }
  }

  MLIRContext context;
  (void)context.getOrLoadDialect<mlir::BuiltinDialect>();
  context.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                      mlir::memref::MemRefDialect, mlir::arith::ArithDialect,
                      mlir::tensor::TensorDialect, mlir::scf::SCFDialect>();

  // Load file
  llvm::SourceMgr sourceMgr;
  auto file = mlir::openInputFile(inputFilename);
  if (!file) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to open input file: " << inputFilename << "\n";
    return 1;
  }
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  // Parse
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::WithColor::error(llvm::errs()) << "Failed to parse MLIR module\n";
    return 1;
  }

  // Find linalg.generic ops.
  llvm::SmallVector<mlir::linalg::GenericOp, 1> generics;
  module->walk([&](mlir::linalg::GenericOp g) { generics.push_back(g); });

  if (generics.size() != 1) {
    llvm::WithColor::error(llvm::errs())
        << "Expected exactly one linalg.generic in the module, found "
        << generics.size() << "\n";
    return 1;
  }

  mlir::linalg::GenericOp generic = generics.front();

  // Get iterator_types attribute and locate first reduction.
  ArrayAttr iteratorTypesAttr =
      generic->getAttrOfType<ArrayAttr>("iterator_types");
  if (!iteratorTypesAttr) {
    llvm::WithColor::error(llvm::errs())
        << "linalg.generic missing iterator_types attribute\n";
    return 1;
  }

  int64_t numIters = static_cast<int64_t>(iteratorTypesAttr.size());
  int64_t reductionDim = -1;
  for (int64_t i = 0; i < numIters; ++i) {
    Attribute itAttr = iteratorTypesAttr[i];
    if (auto strAttr = llvm::dyn_cast<StringAttr>(itAttr)) {
      if (strAttr.getValue() == "reduction") {
        reductionDim = i;
        break;
      }
      continue;
    }
    if (auto iterAttr =
            llvm::dyn_cast<mlir::linalg::IteratorTypeAttr>(itAttr)) {
      if (iterAttr.getValue() == mlir::utils::IteratorType::reduction) {
        reductionDim = i;
        break;
      }
      continue;
    }
    llvm::WithColor::error(llvm::errs())
        << "Invalid iterator_types entry at index " << i << "\n";
    return 1;
  }

  if (reductionDim < 0) {
    llvm::WithColor::error(llvm::errs())
        << "No reduction iterator found in linalg.generic\n";
    return 1;
  }

  // Build tile_sizes list with 8 at the first reduction dim, 0 elsewhere.
  std::string tileSizes;
  tileSizes.push_back('[');
  for (int64_t i = 0; i < numIters; ++i) {
    if (i)
      tileSizes += ", ";
    if (i == reductionDim)
      tileSizes += "8";
    else
      tileSizes += "0";
  }
  tileSizes.push_back(']');

  // Emit Transform dialect module as text.
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "module attributes {transform.with_named_sequence} {\n";
  os << "  transform.named_sequence @__transform_main(%arg1: !transform.any_op "
        "{transform.readonly}) {\n";
  os << "    %0 = transform.structured.match ops{[\"linalg.generic\"]} in "
        "%arg1 "
        ": (!transform.any_op) -> !transform.any_op\n";
  os << "    %tiled, %loop = transform.structured.tile_using_for %0 tile_sizes "
     << tileSizes
     << " : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n";
  os << "    transform.yield\n";
  os << "  }\n";
  os << "}\n";
  os.flush();

  // Determine output directory alongside the input file.
  llvm::SmallString<256> inputPath(inputFilename);
  llvm::sys::path::remove_dots(inputPath, /*remove_dot_dot=*/true);
  llvm::StringRef parentDir = llvm::sys::path::parent_path(inputPath);

  // Write generated Transform IR to requested file if provided, and also
  // always emit alongside the input file with the requested name.
  {
    if (!outputFilename.empty()) {
      std::string errorMessage;
      auto outFile = mlir::openOutputFile(outputFilename, &errorMessage);
      if (!outFile) {
        llvm::WithColor::error(llvm::errs())
            << "Failed to open output file: " << outputFilename << ": "
            << errorMessage << "\n";
        return 1;
      }
      outFile->os() << out;
      outFile->keep();
    }

    llvm::SmallString<256> genTransformPath(parentDir);
    llvm::sys::path::append(genTransformPath,
                            "linalg_example_generated_transforms.mlir");
    std::string errorMessage;
    auto outFile =
        mlir::openOutputFile(genTransformPath.str().str(), &errorMessage);
    if (!outFile) {
      llvm::WithColor::error(llvm::errs())
          << "Failed to open output file: " << genTransformPath << ": "
          << errorMessage << "\n";
      return 1;
    }
    outFile->os() << out;
    outFile->keep();
  }

  // Apply the generated Transform IR to the input payload and print the result.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();
  registry.insert<mlir::transform::TransformDialect>();

  mlir::MLIRContext tContext(registry, mlir::MLIRContext::Threading::DISABLED);
  mlir::ParserConfig parseConfig(&tContext);

  // Parse payload module again in the transform context.
  std::string payloadErr;
  auto payloadFile = mlir::openInputFile(inputFilename, &payloadErr);
  if (!payloadFile) {
    llvm::WithColor::error(llvm::errs()) << payloadErr << "\n";
    return 1;
  }
  llvm::SourceMgr payloadSM;
  payloadSM.AddNewSourceBuffer(std::move(payloadFile), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> payloadModule =
      parseSourceFile<mlir::ModuleOp>(payloadSM, parseConfig);
  if (!payloadModule) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to parse payload module for transform application\n";
    return 1;
  }

  // Parse transform module from the generated string.
  auto transformBuf =
      llvm::MemoryBuffer::getMemBuffer(out, "generated_transform", false);
  llvm::SourceMgr transformSM;
  transformSM.AddNewSourceBuffer(std::move(transformBuf), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> transformModule =
      parseSourceFile<mlir::ModuleOp>(transformSM, parseConfig);
  if (!transformModule) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to parse generated Transform IR\n";
    return 1;
  }

  // Find entry point and apply transforms.
  mlir::transform::TransformOpInterface entryPoint =
      mlir::transform::detail::findTransformEntryPoint(
          *transformModule, mlir::ModuleOp(), "__transform_main");
  if (!entryPoint) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to find transform entry point __transform_main\n";
    return 1;
  }

  mlir::transform::TransformOptions tOpts;
  tOpts.enableExpensiveChecks(true);
  if (mlir::failed(mlir::transform::applyTransforms(
          *payloadModule, entryPoint, {}, tOpts,
          /*enforceToplevelTransformOp=*/false))) {
    llvm::WithColor::error(llvm::errs())
        << "Transform interpreter failed to apply transforms\n";
    return 1;
  }

  // Print to stdout.
  payloadModule->print(llvm::outs());
  llvm::outs() << "\n";

  // Also write the transformed payload alongside the input file with the
  // requested filename.
  {
    std::string transformedIR;
    llvm::raw_string_ostream pos(transformedIR);
    payloadModule->print(pos);
    pos.flush();

    llvm::SmallString<256> transformedPath(parentDir);
    llvm::sys::path::append(transformedPath, "lialg_example_transformed.mlir");
    std::string errorMessage;
    auto outFile =
        mlir::openOutputFile(transformedPath.str().str(), &errorMessage);
    if (!outFile) {
      llvm::WithColor::error(llvm::errs())
          << "Failed to open output file: " << transformedPath << ": "
          << errorMessage << "\n";
      return 1;
    }
    outFile->os() << transformedIR;
    outFile->keep();
  }

  return 0;
}
