#pragma once

#include "expr.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <vector>

namespace loom {
namespace lcs {

/// Boolean constraint expression ADT.
///
/// Mirrors the Rust ConstraintExpr enum and serializes to matching serde JSON:
///   True/False               -> "True" / "False"         (unit variants)
///   And([a,b,...])           -> {"And": [a, b, ...]}     (tuple, Vec)
///   Or([a,b,...])            -> {"Or":  [a, b, ...]}
///   Not(c)                   -> {"Not": c}               (tuple, single)
///   Eq(lhs, rhs)             -> {"Eq":  [lhs, rhs]}      (tuple, pair)
///   Divisible{x, by}         -> {"Divisible": {"x":..., "by":...}}
///   InRange{x, lo, hi}       -> {"InRange": {"x":..., "lo":..., "hi":...}}
///
/// Value-semantic via shared_ptr<Node>; cheap to copy and store.
class ConstraintExpr {
public:
  enum class Kind {
    None,
    True, False,
    And, Or, Not,
    Eq, Le, Lt, Ge, Gt,
    Divisible,
    InRange,
  };

  // ── Default / sentinel ────────────────────────────────────────────────────
  ConstraintExpr() = default;
  bool isNone() const { return !node_; }

  // ── Unit variants ─────────────────────────────────────────────────────────
  static ConstraintExpr true_val();
  static ConstraintExpr false_val();

  // ── Logical connectives ───────────────────────────────────────────────────
  static ConstraintExpr and_(std::vector<ConstraintExpr> operands);
  static ConstraintExpr or_(std::vector<ConstraintExpr> operands);
  static ConstraintExpr not_(ConstraintExpr operand);

  // ── Comparisons over Expr ─────────────────────────────────────────────────
  static ConstraintExpr eq(Expr lhs, Expr rhs);
  static ConstraintExpr le(Expr lhs, Expr rhs);
  static ConstraintExpr lt(Expr lhs, Expr rhs);
  static ConstraintExpr ge(Expr lhs, Expr rhs);
  static ConstraintExpr gt(Expr lhs, Expr rhs);

  // ── Convenience predicates ────────────────────────────────────────────────
  /// x % by == 0
  static ConstraintExpr divisible(Expr x, Expr by);
  /// lo <= x <= hi
  static ConstraintExpr in_range(Expr x, Expr lo, Expr hi);

  // ── Serialization ─────────────────────────────────────────────────────────
  llvm::json::Value toJSON() const;
  std::string toString() const;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const ConstraintExpr &c) {
    return os << c.toString();
  }

private:
  struct Node {
    Kind kind;
    // And, Or: children are sub-constraint nodes.
    std::vector<std::shared_ptr<Node>> children;
    // Not: single child.
    std::shared_ptr<Node> child;
    // Comparison / predicate Expr operands:
    //   Eq/Le/Lt/Ge/Gt : a = lhs, b = rhs
    //   Divisible       : a = x,   b = by
    //   InRange         : a = x,   b = lo,  c = hi
    Expr a, b, c;
  };

  explicit ConstraintExpr(std::shared_ptr<Node> n) : node_(std::move(n)) {}

  static ConstraintExpr cmp(Kind k, Expr lhs, Expr rhs);

  std::shared_ptr<Node> node_;
};

} // namespace lcs
} // namespace loom
