#include "constraint_expr.h"
#include "expr.h"
#include <functional>

namespace loom {
namespace lcs {

// ── Factories ────────────────────────────────────────────────────────────────

Expr Expr::con(int64_t val) {
  auto n = std::make_shared<Node>();
  n->kind = Kind::Const;
  n->val = val;
  return Expr{std::move(n)};
}

Expr Expr::sym(std::string name) {
  auto n = std::make_shared<Node>();
  n->kind = Kind::Sym;
  n->name = std::move(name);
  return Expr{std::move(n)};
}

Expr Expr::ifelse(std::shared_ptr<ConstraintExpr> cond, Expr then_expr,
                  Expr else_expr) {
  auto n = std::make_shared<Node>();
  n->kind = Kind::IfElse;
  n->cond = std::move(cond);
  n->lhs = then_expr.node_;
  n->rhs = else_expr.node_;
  return Expr{std::move(n)};
}

// ── Binary ops ───────────────────────────────────────────────────────────────

Expr Expr::binop(Kind k, Expr lhs, Expr rhs) {
  auto n = std::make_shared<Node>();
  n->kind = k;
  n->lhs = lhs.node_;
  n->rhs = rhs.node_;
  return Expr{std::move(n)};
}

Expr operator+(Expr lhs, Expr rhs) {
  return Expr::binop(Expr::Kind::Add, std::move(lhs), std::move(rhs));
}
Expr operator-(Expr lhs, Expr rhs) {
  return Expr::binop(Expr::Kind::Sub, std::move(lhs), std::move(rhs));
}
Expr operator*(Expr lhs, Expr rhs) {
  return Expr::binop(Expr::Kind::Mul, std::move(lhs), std::move(rhs));
}
Expr operator/(Expr lhs, Expr rhs) {
  return Expr::binop(Expr::Kind::Div, std::move(lhs), std::move(rhs));
}
Expr min_expr(Expr lhs, Expr rhs) {
  return Expr::binop(Expr::Kind::Min, std::move(lhs), std::move(rhs));
}
Expr max_expr(Expr lhs, Expr rhs) {
  return Expr::binop(Expr::Kind::Max, std::move(lhs), std::move(rhs));
}

// ── JSON serialization ────────────────────────────────────────────────────────
//
// Mirrors Rust serde externally-tagged enum format:
//   Tuple variants  -> {"Tag": [field0, field1]}
//   Struct variants -> {"Tag": {"field": value, ...}}

llvm::json::Value Expr::toJSON() const {
  if (!node_)
    return nullptr;

  // Helper: build a two-element array from lhs/rhs for tuple variants.
  auto binaryArr = [&]() {
    llvm::json::Array arr;
    arr.push_back(Expr{node_->lhs}.toJSON());
    arr.push_back(Expr{node_->rhs}.toJSON());
    return arr;
  };

  switch (node_->kind) {
  case Kind::None:
    return nullptr;
  case Kind::Const:
    return llvm::json::Object{{"Const", node_->val}};
  case Kind::Sym:
    return llvm::json::Object{{"Sym", node_->name}};
  case Kind::Add:
    return llvm::json::Object{{"Add", binaryArr()}};
  case Kind::Sub:
    return llvm::json::Object{{"Sub", binaryArr()}};
  case Kind::Mul:
    return llvm::json::Object{{"Mul", binaryArr()}};
  case Kind::Div:
    return llvm::json::Object{{"Div", binaryArr()}};
  case Kind::Min:
    return llvm::json::Object{{"Min", binaryArr()}};
  case Kind::Max:
    return llvm::json::Object{{"Max", binaryArr()}};
  case Kind::IfElse: {
    // Struct variant: {"IfElse": {"cond": ..., "then_expr": ..., "else_expr": ...}}
    llvm::json::Object body;
    body["cond"] = node_->cond ? node_->cond->toJSON() : nullptr;
    body["then_expr"] = Expr{node_->lhs}.toJSON();
    body["else_expr"] = Expr{node_->rhs}.toJSON();
    return llvm::json::Object{{"IfElse", std::move(body)}};
  }
  }
  return nullptr;
}

// ── toString (human-readable infix for debug dumps) ───────────────────────────

std::string Expr::toString() const {
  using NodePtr = std::shared_ptr<Node>;
  std::function<std::string(const NodePtr &, Kind, bool)> fmt =
      [&](const NodePtr &n, Kind parentKind, bool isRhs) -> std::string {
    if (!n)
      return "<none>";

    std::string s;
    switch (n->kind) {
    case Kind::None:
      s = "<none>";
      break;
    case Kind::Const:
      s = std::to_string(n->val);
      break;
    case Kind::Sym:
      s = n->name;
      break;
    case Kind::Add:
      s = fmt(n->lhs, Kind::Add, false) + " + " +
          fmt(n->rhs, Kind::Add, true);
      break;
    case Kind::Sub:
      s = fmt(n->lhs, Kind::Sub, false) + " - " +
          fmt(n->rhs, Kind::Sub, true);
      break;
    case Kind::Mul:
      s = fmt(n->lhs, Kind::Mul, false) + " * " +
          fmt(n->rhs, Kind::Mul, true);
      break;
    case Kind::Div:
      s = fmt(n->lhs, Kind::Div, false) + " / " +
          fmt(n->rhs, Kind::Div, true);
      break;
    case Kind::Min:
      s = "min(" + fmt(n->lhs, Kind::None, false) + ", " +
          fmt(n->rhs, Kind::None, false) + ")";
      break;
    case Kind::Max:
      s = "max(" + fmt(n->lhs, Kind::None, false) + ", " +
          fmt(n->rhs, Kind::None, false) + ")";
      break;
    case Kind::IfElse: {
      std::string condStr = n->cond ? n->cond->toString() : "<cond>";
      s = "(" + condStr + " ? " + fmt(n->lhs, Kind::None, false) + " : " +
          fmt(n->rhs, Kind::None, false) + ")";
      break;
    }
    }

    // Parenthesize for display readability where precedence demands it.
    bool wrap = false;
    if ((parentKind == Kind::Mul || parentKind == Kind::Div) &&
        (n->kind == Kind::Add || n->kind == Kind::Sub))
      wrap = true;
    if (isRhs && parentKind == Kind::Sub &&
        (n->kind == Kind::Add || n->kind == Kind::Sub))
      wrap = true;
    if (isRhs && parentKind == Kind::Div &&
        (n->kind == Kind::Mul || n->kind == Kind::Div ||
         n->kind == Kind::Add || n->kind == Kind::Sub))
      wrap = true;

    return wrap ? "(" + s + ")" : s;
  };

  return fmt(node_, Kind::None, false);
}

} // namespace lcs
} // namespace loom
