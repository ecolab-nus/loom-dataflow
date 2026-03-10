"""Convert JSON Expr trees (Rust-style tagged unions) into Z3 AST nodes.

This module is a pure "compiler back-end": it has no side effects, holds no
state, and does not know anything about the surrounding solver or pipeline.

Supported arithmetic tags:
    Const, Sym, Mul, Add, Div, Min, IfElse

Supported constraint/boolean tags (used in IfElse conditions and hard_constraints):
    Eq, Ge, Divisible
"""

import functools
import operator
from typing import Union

import z3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_expr(
    expr: dict,
    symbol_map: dict[str, z3.ArithRef],
) -> z3.ArithRef:
    """Recursively convert a JSON Expr dict into a Z3 integer expression.

    Args:
        expr:       A dict with exactly one key identifying the operator.
        symbol_map: Mapping from symbol name to the corresponding z3.Int variable.

    Returns:
        A Z3 ArithRef representing the expression.

    Raises:
        ValueError: On unknown tags or malformed structure.
        KeyError:   When a Sym references a name not in symbol_map.
    """
    if not isinstance(expr, dict) or len(expr) != 1:
        raise ValueError(f"Expected a single-key dict for Expr, got: {expr!r}")

    tag, payload = next(iter(expr.items()))

    if tag == "Const":
        return z3.IntVal(int(payload))

    if tag == "Sym":
        if payload not in symbol_map:
            raise KeyError(f"Symbol '{payload}' not found in symbol_map")
        return symbol_map[payload]

    if tag == "Mul":
        _check_binary(tag, payload)
        return resolve_expr(payload[0], symbol_map) * resolve_expr(payload[1], symbol_map)

    if tag == "Add":
        _check_variadic(tag, payload)
        args = [resolve_expr(a, symbol_map) for a in payload]
        return functools.reduce(operator.add, args)

    if tag == "Div":
        _check_binary(tag, payload)
        return resolve_expr(payload[0], symbol_map) / resolve_expr(payload[1], symbol_map)

    if tag == "Min":
        _check_variadic(tag, payload)
        args = [resolve_expr(a, symbol_map) for a in payload]
        return functools.reduce(lambda x, y: z3.If(x <= y, x, y), args)

    if tag == "IfElse":
        cond_z3 = resolve_constraint(payload["cond"], symbol_map)
        then_z3 = resolve_expr(payload["then_expr"], symbol_map)
        else_z3 = resolve_expr(payload["else_expr"], symbol_map)
        return z3.If(cond_z3, then_z3, else_z3)

    raise ValueError(f"Unknown Expr tag: '{tag}'")


def resolve_constraint(
    expr: dict,
    symbol_map: dict[str, z3.ArithRef],
) -> z3.BoolRef:
    """Convert a JSON constraint/boolean Expr dict into a Z3 BoolRef.

    Args:
        expr:       A dict with exactly one key identifying the boolean operator.
        symbol_map: Mapping from symbol name to the corresponding z3.Int variable.

    Returns:
        A Z3 BoolRef representing the boolean expression.

    Raises:
        ValueError: On unknown tags or malformed structure.
    """
    if not isinstance(expr, dict) or len(expr) != 1:
        raise ValueError(f"Expected a single-key dict for constraint, got: {expr!r}")

    tag, payload = next(iter(expr.items()))

    if tag == "Eq":
        _check_binary(tag, payload)
        return resolve_expr(payload[0], symbol_map) == resolve_expr(payload[1], symbol_map)

    if tag == "Ge":
        _check_binary(tag, payload)
        return resolve_expr(payload[0], symbol_map) >= resolve_expr(payload[1], symbol_map)

    if tag == "Divisible":
        x_z3 = resolve_expr(payload["x"], symbol_map)
        by_z3 = resolve_expr(payload["by"], symbol_map)
        return x_z3 % by_z3 == 0

    raise ValueError(f"Unknown constraint tag: '{tag}'")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_binary(tag: str, payload) -> None:
    if not isinstance(payload, list) or len(payload) != 2:
        raise ValueError(f"'{tag}' requires exactly 2 operands, got: {payload!r}")


def _check_variadic(tag: str, payload) -> None:
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError(f"'{tag}' requires at least 2 operands, got: {payload!r}")
