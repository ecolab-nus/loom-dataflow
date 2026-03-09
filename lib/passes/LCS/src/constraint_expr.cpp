#include "constraint_expr.h"
#include <functional>
#include <numeric>

namespace loom {
namespace lcs {

// ── Unit variants ─────────────────────────────────────────────────────────────

ConstraintExpr ConstraintExpr::true_val() {
  auto n = std::make_shared<Node>();
  n->kind = Kind::True;
  return ConstraintExpr{std::move(n)};
}

ConstraintExpr ConstraintExpr::false_val() {
  auto n = std::make_shared<Node>();
  n->kind = Kind::False;
  return ConstraintExpr{std::move(n)};
}

// ── Logical connectives ───────────────────────────────────────────────────────

ConstraintExpr ConstraintExpr::and_(std::vector<ConstraintExpr> operands) {
  auto n = std::make_shared<Node>();
  n->kind = Kind::And;
  for (auto &op : operands)
    n->children.push_back(op.node_);
  return ConstraintExpr{std::move(n)};
}

ConstraintExpr ConstraintExpr::or_(std::vector<ConstraintExpr> operands) {
  auto n = std::make_shared<Node>();
  n->kind = Kind::Or;
  for (auto &op : operands)
    n->children.push_back(op.node_);
  return ConstraintExpr{std::move(n)};
}

ConstraintExpr ConstraintExpr::not_(ConstraintExpr operand) {
  auto n = std::make_shared<Node>();
  n->kind = Kind::Not;
  n->child = operand.node_;
  return ConstraintExpr{std::move(n)};
}

// ── Comparisons ───────────────────────────────────────────────────────────────

ConstraintExpr ConstraintExpr::cmp(Kind k, Expr lhs, Expr rhs) {
  auto n = std::make_shared<Node>();
  n->kind = k;
  n->a = std::move(lhs);
  n->b = std::move(rhs);
  return ConstraintExpr{std::move(n)};
}

ConstraintExpr ConstraintExpr::eq(Expr lhs, Expr rhs) {
  return cmp(Kind::Eq, std::move(lhs), std::move(rhs));
}
ConstraintExpr ConstraintExpr::le(Expr lhs, Expr rhs) {
  return cmp(Kind::Le, std::move(lhs), std::move(rhs));
}
ConstraintExpr ConstraintExpr::lt(Expr lhs, Expr rhs) {
  return cmp(Kind::Lt, std::move(lhs), std::move(rhs));
}
ConstraintExpr ConstraintExpr::ge(Expr lhs, Expr rhs) {
  return cmp(Kind::Ge, std::move(lhs), std::move(rhs));
}
ConstraintExpr ConstraintExpr::gt(Expr lhs, Expr rhs) {
  return cmp(Kind::Gt, std::move(lhs), std::move(rhs));
}

// ── Convenience predicates ────────────────────────────────────────────────────

ConstraintExpr ConstraintExpr::divisible(Expr x, Expr by) {
  auto n = std::make_shared<Node>();
  n->kind = Kind::Divisible;
  n->a = std::move(x);
  n->b = std::move(by);
  return ConstraintExpr{std::move(n)};
}

ConstraintExpr ConstraintExpr::in_range(Expr x, Expr lo, Expr hi) {
  auto n = std::make_shared<Node>();
  n->kind = Kind::InRange;
  n->a = std::move(x);
  n->b = std::move(lo);
  n->c = std::move(hi);
  return ConstraintExpr{std::move(n)};
}

// ── JSON serialization ────────────────────────────────────────────────────────
//
// Rust serde externally-tagged enum format:
//   unit variants   -> "True" / "False"   (bare strings)
//   tuple variants  -> {"Tag": value}     (single: value; multiple: array)
//   struct variants -> {"Tag": {fields}}

llvm::json::Value ConstraintExpr::toJSON() const {
  if (!node_)
    return nullptr;

  // Helper: build a JSON array from a child-node vector.
  auto childArray = [&](const std::vector<std::shared_ptr<Node>> &kids) {
    llvm::json::Array arr;
    for (const auto &k : kids)
      arr.push_back(ConstraintExpr{k}.toJSON());
    return arr;
  };

  switch (node_->kind) {
  case Kind::None:
    return nullptr;

  // Unit variants → bare string.
  case Kind::True:
    return llvm::json::Value("True");
  case Kind::False:
    return llvm::json::Value("False");

  // And / Or: tuple variant with Vec<ConstraintExpr> → array.
  case Kind::And:
    return llvm::json::Object{{"And", childArray(node_->children)}};
  case Kind::Or:
    return llvm::json::Object{{"Or", childArray(node_->children)}};

  // Not: tuple variant with single Box → value (not array).
  case Kind::Not:
    return llvm::json::Object{
        {"Not", ConstraintExpr{node_->child}.toJSON()}};

  // Comparisons: tuple variant with two Exprs → two-element array.
  case Kind::Eq:
    return llvm::json::Object{
        {"Eq", llvm::json::Array{node_->a.toJSON(), node_->b.toJSON()}}};
  case Kind::Le:
    return llvm::json::Object{
        {"Le", llvm::json::Array{node_->a.toJSON(), node_->b.toJSON()}}};
  case Kind::Lt:
    return llvm::json::Object{
        {"Lt", llvm::json::Array{node_->a.toJSON(), node_->b.toJSON()}}};
  case Kind::Ge:
    return llvm::json::Object{
        {"Ge", llvm::json::Array{node_->a.toJSON(), node_->b.toJSON()}}};
  case Kind::Gt:
    return llvm::json::Object{
        {"Gt", llvm::json::Array{node_->a.toJSON(), node_->b.toJSON()}}};

  // Divisible: struct variant → {"Divisible": {"x":..., "by":...}}.
  case Kind::Divisible: {
    llvm::json::Object body;
    body["x"] = node_->a.toJSON();
    body["by"] = node_->b.toJSON();
    return llvm::json::Object{{"Divisible", std::move(body)}};
  }

  // InRange: struct variant → {"InRange": {"x":..., "lo":..., "hi":...}}.
  case Kind::InRange: {
    llvm::json::Object body;
    body["x"] = node_->a.toJSON();
    body["lo"] = node_->b.toJSON();
    body["hi"] = node_->c.toJSON();
    return llvm::json::Object{{"InRange", std::move(body)}};
  }
  }
  return nullptr;
}

// ── toString (human-readable for debug dumps) ─────────────────────────────────

std::string ConstraintExpr::toString() const {
  using NodePtr = std::shared_ptr<Node>;
  std::function<std::string(const NodePtr &)> fmt =
      [&](const NodePtr &n) -> std::string {
    if (!n)
      return "<none>";
    switch (n->kind) {
    case Kind::None:
      return "<none>";
    case Kind::True:
      return "true";
    case Kind::False:
      return "false";
    case Kind::And: {
      if (n->children.empty())
        return "true";
      std::string s = fmt(n->children[0]);
      for (size_t i = 1; i < n->children.size(); ++i)
        s += " && " + fmt(n->children[i]);
      return "(" + s + ")";
    }
    case Kind::Or: {
      if (n->children.empty())
        return "false";
      std::string s = fmt(n->children[0]);
      for (size_t i = 1; i < n->children.size(); ++i)
        s += " || " + fmt(n->children[i]);
      return "(" + s + ")";
    }
    case Kind::Not:
      return "!" + fmt(n->child);
    case Kind::Eq:
      return n->a.toString() + " == " + n->b.toString();
    case Kind::Le:
      return n->a.toString() + " <= " + n->b.toString();
    case Kind::Lt:
      return n->a.toString() + " < " + n->b.toString();
    case Kind::Ge:
      return n->a.toString() + " >= " + n->b.toString();
    case Kind::Gt:
      return n->a.toString() + " > " + n->b.toString();
    case Kind::Divisible:
      return n->a.toString() + " % " + n->b.toString() + " == 0";
    case Kind::InRange:
      return n->b.toString() + " <= " + n->a.toString() +
             " <= " + n->c.toString();
    }
    return "<none>";
  };

  return fmt(node_);
}

} // namespace lcs
} // namespace loom
