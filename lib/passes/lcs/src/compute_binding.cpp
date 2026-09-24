#include "compute_binding.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/ADT/STLExtras.h"

namespace loom::lcs {

bool isComputeLinalg(mlir::Operation *op) {
  return llvm::isa<mlir::linalg::LinalgOp>(op) &&
         !llvm::isa<mlir::linalg::FillOp, mlir::linalg::CopyOp>(op);
}

static bool isPayload(mlir::Operation *op) {
  mlir::Dialect *dialect = op->getDialect();
  if (!dialect)
    return false;
  if (op->hasTrait<mlir::OpTrait::ConstantLike>())
    return false;
  llvm::StringRef ns = dialect->getNamespace();
  return ns == "arith" || ns == "math";
}

static BindingSiteValue classifyValue(mlir::Value value,
                                      mlir::Block &body) {
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value))
    if (argument.getOwner() == &body)
      return {BindingSiteValue::ExternalOperand, value,
              argument.getArgNumber()};
  if (value.getDefiningOp() && isPayload(value.getDefiningOp()))
    return {BindingSiteValue::InternalValue, value, 0};
  return {BindingSiteValue::Constant, value, 0};
}

mlir::FailureOr<std::vector<ComputeBindingSite>>
analyzeComputeBindingSites(mlir::func::FuncOp function) {
  std::vector<ComputeBindingSite> sites;
  bool failed = false;
  function.walk([&](mlir::Operation *op) {
    if (failed || !isComputeLinalg(op))
      return;
    auto linalgOp = llvm::cast<mlir::linalg::LinalgOp>(op);
    auto match = getComputeOpMatchInfo(linalgOp);
    if (mlir::failed(match)) {
      failed = true;
      return;
    }
    if (!llvm::isa<mlir::linalg::GenericOp>(op)) {
      ComputeBindingSite site;
      site.enclosing_op = op;
      site.payload_op = op;
      site.id = sites.size();
      site.is_named = true;
      site.semantic_key =
          HWOpKey::named(op->getName().getStringRef().str(), *match);
      unsigned index = 0;
      for (mlir::Value value : linalgOp.getDpsInputs())
        site.inputs.push_back(
            {BindingSiteValue::ExternalOperand, value, index++});
      for (mlir::Value value : linalgOp.getDpsInits())
        site.outputs.push_back(
            {BindingSiteValue::ExternalOperand, value, index++});
      sites.push_back(std::move(site));
      return;
    }

    mlir::Block &body = op->getRegion(0).front();
    GenericDimAnalysis dims = analyzeGenericDims(linalgOp);
    auto yield = llvm::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
    unsigned payloadCount = 0;
    for (mlir::Operation &bodyOp : body) {
      if (isPayload(&bodyOp))
        ++payloadCount;
      else if (!llvm::isa<mlir::linalg::YieldOp>(&bodyOp) &&
               !bodyOp.hasTrait<mlir::OpTrait::ConstantLike>()) {
        bodyOp.emitError("unsupported generic body operation in processor binding");
        failed = true;
        return;
      }
    }
    if (payloadCount > 1 &&
        dims.generic_class != GenericClass::Parallel) {
      op->emitError("compound reduction generic requires explicit reduction semantics");
      failed = true;
      return;
    }
    for (mlir::Operation &bodyOp : body) {
      if (!isPayload(&bodyOp))
        continue;
      ComputeBindingSite site;
      site.enclosing_op = op;
      site.payload_op = &bodyOp;
      site.id = sites.size();
      site.semantic_key = HWOpKey::generic(
          bodyOp.getName().getStringRef().str(), dims.generic_class, *match);
      for (mlir::Value operand : bodyOp.getOperands())
        site.inputs.push_back(classifyValue(operand, body));
      for (mlir::Value result : bodyOp.getResults()) {
        BindingSiteValue output{BindingSiteValue::InternalValue, result, 0};
        if (yield)
          for (auto [index, yielded] : llvm::enumerate(yield.getValues()))
            if (yielded == result) {
              unsigned dpsIndex = linalgOp.getNumDpsInputs() + index;
              output.external_dps_indices.push_back(dpsIndex);
            }
        site.outputs.push_back(std::move(output));
      }
      sites.push_back(std::move(site));
    }
  });
  if (failed)
    return mlir::failure();
  return sites;
}

bool candidateMatchesSite(const HWComputeFunc &candidate,
                          const ComputeBindingSite &site) {
  if (site.is_named)
    return candidate.linalg_op_name ==
               site.enclosing_op->getName().getStringRef() &&
           candidate.operand_types.size() ==
               site.inputs.size() + site.outputs.size();
  auto semanticAttributesMatch = [&]() {
    auto isBindingAttribute = [](mlir::NamedAttribute attribute) {
      return attribute.getName().getValue().starts_with("loom.");
    };
    for (mlir::NamedAttribute attribute : candidate.body_attributes)
      if (!isBindingAttribute(attribute) &&
          site.payload_op->getAttr(attribute.getName()) !=
              attribute.getValue())
        return false;
    for (mlir::NamedAttribute attribute :
         site.payload_op->getAttrDictionary())
      if (!isBindingAttribute(attribute) &&
          candidate.body_attributes.get(attribute.getName()) !=
              attribute.getValue())
        return false;
    return true;
  };
  if (candidate.body_op_name !=
          site.payload_op->getName().getStringRef() ||
      candidate.generic_class != site.semantic_key.generic_class ||
      candidate.scalar_operand_types.size() != site.inputs.size() ||
      candidate.scalar_result_types.size() != site.outputs.size() ||
      !semanticAttributesMatch())
    return false;
  for (auto [type, input] :
       llvm::zip(candidate.scalar_operand_types, site.inputs))
    if (type != input.value.getType())
      return false;
  for (auto [type, output] :
       llvm::zip(candidate.scalar_result_types, site.outputs))
    if (type != output.value.getType())
      return false;
  return true;
}

} // namespace loom::lcs
