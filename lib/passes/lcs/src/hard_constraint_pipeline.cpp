#include "hard_constraint_pipeline.h"
#include "hw_op_registry.h"
#include "staged_etg_builder.h"
#include "ADLDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "ADLTypes.h.inc"
#define GET_OP_CLASSES
#include "ADLOps.h.inc"
#include <memory>

namespace loom {
namespace lcs {
namespace {

int64_t extractL1SizeFromPlatform(const HWOpRegistry *registry) {
  if (!registry)
    return 0;
  mlir::ModuleOp platformModule = registry->getPlatformModule();
  if (!platformModule)
    return 0;

  int64_t l1_size = 0;
  platformModule.walk([&](adl::MemoryArrayOp arrayOp) {
    if (arrayOp->getParentOp() != platformModule.getOperation())
      return mlir::WalkResult::skip();
    if (arrayOp.getSymName() != "mem_L1")
      return mlir::WalkResult::advance();

    int64_t spatial_product = 1;
    for (mlir::Value spatialVal : arrayOp.getSpatialDims())
      if (auto dimOp = spatialVal.getDefiningOp<adl::SpatialDimOp>())
        spatial_product *= static_cast<int64_t>(dimOp.getSize());

    if (auto bankOp = arrayOp.getBank().getDefiningOp<adl::MemoryBankOp>()) {
      int64_t bsize = static_cast<int64_t>(bankOp.getBsize());
      int64_t nblk = static_cast<int64_t>(bankOp.getNblk());
      l1_size = spatial_product * bsize * nblk;
    }
    return mlir::WalkResult::advance();
  });
  return l1_size;
}

void pushL1SizeConstraint(const HWOpRegistry *registry, ConstraintScope &scope) {
  if (scope.l1_footprint.empty())
    return;

  int64_t l1_size = extractL1SizeFromPlatform(registry);
  if (l1_size == 0)
    return;

  Expr footprint_sum = Expr::con(0);
  for (const Expr &term : scope.l1_footprint)
    footprint_sum = footprint_sum + term;

  int64_t elem_bytes = 2;
  auto db_cond = std::make_shared<ConstraintExpr>(
      ConstraintExpr::eq(Expr::sym("is_double_buffer"), Expr::con(1)));
  Expr multiplier =
      Expr::ifelse(db_cond, Expr::con(elem_bytes * 2), Expr::con(elem_bytes));

  scope.pushHardConstraint(
      ConstraintExpr::le(footprint_sum * multiplier, Expr::con(l1_size)));
}

} // namespace

void HardConstraintPipeline::pushAll(const HWOpRegistry *registry,
                                     ConstraintScope &scope) {
  pushL1SizeConstraint(registry, scope);
}

} // namespace lcs
} // namespace loom
