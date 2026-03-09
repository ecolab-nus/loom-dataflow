#pragma once

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <memory>
#include <string>

namespace loom {
namespace lcs {

/// Forward declaration — full definition in constraint_expr.h.
class ConstraintExpr;

/// Lightweight algebraic expression ADT.
///
/// Mirrors the Rust Expr enum and serializes to matching serde JSON:
///   Const(128)          -> {"Const": 128}
///   Sym("BB")           -> {"Sym": "BB"}
///   Mul(a, b)           -> {"Mul": [a, b]}          (tuple variant)
///   IfElse{cond,t,e}    -> {"IfElse": {"cond":..., "then_expr":..., "else_expr":...}}
///
/// Value-semantic: cheap to copy due to shared_ptr<Node> internals.
/// Immutable after construction.
class Expr {
public:
  enum class Kind { None, Const, Sym, Add, Sub, Mul, Div, Min, Max, IfElse };

  // ── Factories ─────────────────────────────────────────────────────────────

  /// Default-constructed Expr is the none/sentinel value.
  Expr() = default;

  /// Sentinel: failed trace, missing information.
  static Expr none() { return Expr{}; }

  /// Integer constant.
  static Expr con(int64_t val);

  /// Symbolic variable (e.g., "BB", "BM").
  static Expr sym(std::string name);

  /// Conditional expression: if cond then then_expr else else_expr.
  static Expr ifelse(std::shared_ptr<ConstraintExpr> cond, Expr then_expr,
                     Expr else_expr);

  // ── Queries ───────────────────────────────────────────────────────────────

  bool isNone() const { return !node_; }
  Kind kind() const { return node_ ? node_->kind : Kind::None; }

  /// For binary ops: returns the left / right sub-expression.
  /// Returns Expr::none() if this node is not a binary op.
  Expr lhs() const { return node_ ? Expr{node_->lhs} : Expr{}; }
  Expr rhs() const { return node_ ? Expr{node_->rhs} : Expr{}; }

  // ── Operators ─────────────────────────────────────────────────────────────

  friend Expr operator+(Expr lhs, Expr rhs);
  friend Expr operator-(Expr lhs, Expr rhs);
  friend Expr operator*(Expr lhs, Expr rhs);
  friend Expr operator/(Expr lhs, Expr rhs);

  // ── Named binary constructors (no natural operator spelling) ───────────────

  friend Expr min_expr(Expr lhs, Expr rhs);
  friend Expr max_expr(Expr lhs, Expr rhs);

  // ── Serialization ─────────────────────────────────────────────────────────

  /// Rust serde-compatible JSON (externally tagged enum format).
  llvm::json::Value toJSON() const;

  /// Human-readable infix string for debug dumps.
  std::string toString() const;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Expr &e) {
    return os << e.toString();
  }

private:
  struct Node {
    Kind kind;
    int64_t val{0};                          // Const
    std::string name;                        // Sym
    std::shared_ptr<Node> lhs;              // binary ops / IfElse then_expr
    std::shared_ptr<Node> rhs;              // binary ops / IfElse else_expr
    std::shared_ptr<ConstraintExpr> cond;   // IfElse condition
  };

  explicit Expr(std::shared_ptr<Node> n) : node_(std::move(n)) {}

  static Expr binop(Kind k, Expr lhs, Expr rhs);

  std::shared_ptr<Node> node_;
};

} // namespace lcs
} // namespace loom
