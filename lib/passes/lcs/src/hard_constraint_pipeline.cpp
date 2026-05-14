#include "hard_constraint_pipeline.h"
#include "hw_op_registry.h"
#include "loop_iv_dependency.h"
#include "staged_etg_builder.h"
#include "utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "ADLDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "ADLTypes.h.inc"
#define GET_OP_CLASSES
#include "ADLOps.h.inc"
#include "LoomInterfaces.h.inc"
#include <memory>
#include <cassert>
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

namespace loom {
namespace lcs {
namespace {

[[noreturn]] void failGatherConstraint(mlir::Operation *op, llvm::StringRef msg) {
  if (op) {
    op->emitError() << msg;
    std::string loc;
    llvm::raw_string_ostream os(loc);
    op->getLoc().print(os);
    os.flush();
    llvm::errs() << "gather hard-constraint error at " << loc << ": " << msg
                 << "\n";
  } else {
    llvm::errs() << "gather hard-constraint error: " << msg << "\n";
  }
  assert(false && "gather hard-constraint construction failed");
  llvm::report_fatal_error("gather hard-constraint construction failed");
}

Expr traceIndexValueToExpr(mlir::Value val);

Expr traceAffineExprToExpr(mlir::AffineExpr affineExpr, mlir::ValueRange operands,
                           unsigned numDims) {
  if (auto dim = llvm::dyn_cast<mlir::AffineDimExpr>(affineExpr)) {
    unsigned pos = dim.getPosition();
    if (pos >= numDims || pos >= operands.size())
      return Expr::none();
    return traceIndexValueToExpr(operands[pos]);
  }

  if (auto sym = llvm::dyn_cast<mlir::AffineSymbolExpr>(affineExpr)) {
    unsigned pos = numDims + sym.getPosition();
    if (pos >= operands.size())
      return Expr::none();
    return traceIndexValueToExpr(operands[pos]);
  }

  if (auto cst = llvm::dyn_cast<mlir::AffineConstantExpr>(affineExpr))
    return Expr::con(cst.getValue());

  if (auto bin = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(affineExpr)) {
    Expr lhs = traceAffineExprToExpr(bin.getLHS(), operands, numDims);
    Expr rhs = traceAffineExprToExpr(bin.getRHS(), operands, numDims);
    switch (bin.getKind()) {
    case mlir::AffineExprKind::Add:
      return lhs + rhs;
    case mlir::AffineExprKind::Mul:
      return lhs * rhs;
    case mlir::AffineExprKind::Mod:
    case mlir::AffineExprKind::FloorDiv:
    case mlir::AffineExprKind::CeilDiv:
      return lhs / rhs;
    default:
      return Expr::none();
    }
  }

  return Expr::none();
}

Expr traceIndexValueToExpr(mlir::Value val) {
  using namespace mlir;
  if (!val)
    return Expr::none();

  llvm::StringRef symName = loom::utils::traceToSymbolicVar(val);
  if (!symName.empty())
    return Expr::sym(symName.str());

  Operation *op = val.getDefiningOp();
  if (!op)
    return Expr::none();

  if (auto cst = dyn_cast<arith::ConstantOp>(op))
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return Expr::con(intAttr.getInt());

  if (auto add = dyn_cast<arith::AddIOp>(op))
    return traceIndexValueToExpr(add.getLhs()) + traceIndexValueToExpr(add.getRhs());
  if (auto sub = dyn_cast<arith::SubIOp>(op))
    return traceIndexValueToExpr(sub.getLhs()) - traceIndexValueToExpr(sub.getRhs());
  if (auto mul = dyn_cast<arith::MulIOp>(op))
    return traceIndexValueToExpr(mul.getLhs()) * traceIndexValueToExpr(mul.getRhs());
  if (auto div = dyn_cast<arith::DivUIOp>(op))
    return traceIndexValueToExpr(div.getLhs()) / traceIndexValueToExpr(div.getRhs());
  if (auto div = dyn_cast<arith::DivSIOp>(op))
    return traceIndexValueToExpr(div.getLhs()) / traceIndexValueToExpr(div.getRhs());
  if (auto cdiv = dyn_cast<arith::CeilDivUIOp>(op))
    return traceIndexValueToExpr(cdiv.getLhs()) / traceIndexValueToExpr(cdiv.getRhs());
  if (auto cdiv = dyn_cast<arith::CeilDivSIOp>(op))
    return traceIndexValueToExpr(cdiv.getLhs()) / traceIndexValueToExpr(cdiv.getRhs());
  if (auto fdiv = dyn_cast<arith::FloorDivSIOp>(op))
    return traceIndexValueToExpr(fdiv.getLhs()) / traceIndexValueToExpr(fdiv.getRhs());

  if (auto cast = dyn_cast<arith::IndexCastOp>(op))
    return traceIndexValueToExpr(cast.getIn());
  if (auto cast = dyn_cast<arith::IndexCastUIOp>(op))
    return traceIndexValueToExpr(cast.getIn());
  if (auto ext = dyn_cast<arith::ExtSIOp>(op))
    return traceIndexValueToExpr(ext.getIn());
  if (auto ext = dyn_cast<arith::ExtUIOp>(op))
    return traceIndexValueToExpr(ext.getIn());
  if (auto trunc = dyn_cast<arith::TruncIOp>(op))
    return traceIndexValueToExpr(trunc.getIn());

  if (auto apply = dyn_cast<affine::AffineApplyOp>(op)) {
    mlir::AffineExpr resultExpr = apply.getAffineMap().getResult(0);
    return traceAffineExprToExpr(resultExpr, apply.getOperands(),
                                 apply.getAffineMap().getNumDims());
  }

  return Expr::none();
}

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

ConstraintExpr makeScfForStepCountEqOne(mlir::scf::ForOp forOp) {
  Expr ub = traceIndexValueToExpr(forOp.getUpperBound());
  Expr lb = traceIndexValueToExpr(forOp.getLowerBound());
  Expr step = traceIndexValueToExpr(forOp.getStep());
  if (ub.isNone() || lb.isNone() || step.isNone())
    failGatherConstraint(forOp, "cannot express scf.for bounds/step to Expr");
  Expr trip = (ub - lb) / step;
  return ConstraintExpr::eq(trip, Expr::con(1));
}

ConstraintExpr makeAffineParallelStepCountEqOne(mlir::affine::AffineParallelOp par,
                                                unsigned dimIdx) {
  if (dimIdx >= par.getNumDims())
    failGatherConstraint(par, "invalid affine.parallel IV dimension index");
  mlir::AffineExpr lbExpr = par.getLowerBoundMap(dimIdx).getResult(0);
  mlir::AffineExpr ubExpr = par.getUpperBoundMap(dimIdx).getResult(0);
  Expr lb = traceAffineExprToExpr(lbExpr, par.getLowerBoundsOperands(),
                                  par.getLowerBoundMap(dimIdx).getNumDims());
  Expr ub = traceAffineExprToExpr(ubExpr, par.getUpperBoundsOperands(),
                                  par.getUpperBoundMap(dimIdx).getNumDims());
  if (ub.isNone() || lb.isNone())
    failGatherConstraint(par, "cannot express affine.parallel bound to Expr");
  Expr trip = (ub - lb) / Expr::con(1);
  return ConstraintExpr::eq(trip, Expr::con(1));
}

void pushGatherTemporalAcrossConstraints(mlir::func::FuncOp funcOp,
                                         ConstraintScope &scope) {
  if (!funcOp)
    return;

  llvm::SmallPtrSet<mlir::Value, 16> constrainedIVs;

  funcOp.walk([&](loom::GatherOp gatherOp) {
    mlir::Value across = gatherOp.getAcross();
    auto temporalDeps = loom::utils::collectTemporalIVDependencies(across);
    if (temporalDeps.empty())
      failGatherConstraint(gatherOp, "across does not depend on any temporal loop IV");

    for (const auto &dep : temporalDeps) {
      if (!constrainedIVs.insert(dep.iv).second)
        continue;

      mlir::Operation *loopOp = dep.loop;
      if (auto scfFor = llvm::dyn_cast<mlir::scf::ForOp>(loopOp)) {
        scope.pushHardConstraint(makeScfForStepCountEqOne(scfFor));
        continue;
      }

      if (auto par = llvm::dyn_cast<mlir::affine::AffineParallelOp>(loopOp)) {
        if (dep.ivIndex >= par.getNumDims())
          failGatherConstraint(gatherOp, "cannot resolve affine.parallel IV dimension for across");
        scope.pushHardConstraint(makeAffineParallelStepCountEqOne(par, dep.ivIndex));
        continue;
      }

      failGatherConstraint(gatherOp, "temporal loop kind is unsupported for gather across");
    }
  });
}

} // namespace

void HardConstraintPipeline::pushAll(mlir::func::FuncOp funcOp,
                                     const HWOpRegistry *registry,
                                     ConstraintScope &scope) {
  pushL1SizeConstraint(registry, scope);
  pushGatherTemporalAcrossConstraints(funcOp, scope);
}

} // namespace lcs
} // namespace loom
