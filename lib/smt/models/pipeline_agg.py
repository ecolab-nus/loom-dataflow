"""Aggregate resolved_time expressions across the ETG hierarchy.

Aggregation rules (from spec):
  - Within a stage:  queues execute in PARALLEL → Max of queue resolved_times.
  - Across stages:   stages execute in SERIAL   → Sum of per-stage maxima.
  - Across scopes:   T_stage = max(T_comp, T_mem).
  - Total:           T_total = T_stage * seq_iter * product(temp_iter).

This module only builds Z3 expression trees; it does not interact with the
solver directly.
"""

import functools
import operator

import z3

from core.expr_resolver import resolve_expr


def compute_total_time(
    variant: dict,
    symbol_map: dict[str, z3.ArithRef],
) -> z3.ArithRef:
    """Build the Z3 expression for the total pipeline execution time.

    Args:
        variant:    A single variant dict from the staged ETG JSON.
        symbol_map: Z3 integer variables keyed by symbol name.

    Returns:
        A Z3 ArithRef for T_total = max(T_comp, T_mem) * seq_iter * Πtemp_iter.
    """
    t_comp = _aggregate_scope(variant["compute_scope"], symbol_map)
    t_mem = _aggregate_scope(variant["memory_scope"], symbol_map)
    t_stage = z3.If(t_comp >= t_mem, t_comp, t_mem)

    iter_num = variant["constraint_scope"]["metadata"]["iter_num"]
    seq_iter = resolve_expr(iter_num["seq_iter"], symbol_map)
    temp_iter_exprs = [resolve_expr(t, symbol_map) for t in iter_num["temp_iter"]]
    temp_iter_product = functools.reduce(operator.mul, temp_iter_exprs)

    return t_stage * seq_iter * temp_iter_product


def _aggregate_scope(
    scope: dict,
    symbol_map: dict[str, z3.ArithRef],
) -> z3.ArithRef:
    """Compute the total time for a single scope (compute or memory).

    Stages are serial (sum); queues within a stage are parallel (max).
    Queues with a null resolved_time are skipped.

    Args:
        scope:      A compute_scope or memory_scope dict.
        symbol_map: Z3 integer variables keyed by symbol name.

    Returns:
        A Z3 ArithRef for the total scope time.

    Raises:
        ValueError: If the scope has no stages with resolvable queue times.
    """
    stage_times: list[z3.ArithRef] = []

    for stage in scope["stages"]:
        queue_times: list[z3.ArithRef] = []
        for queue in stage["queues"].values():
            rt = queue.get("resolved_time")
            if rt is None:
                continue
            queue_times.append(resolve_expr(rt, symbol_map))

        if not queue_times:
            continue

        # Parallel queues → max
        stage_max = functools.reduce(
            lambda a, b: z3.If(a >= b, a, b),
            queue_times,
        )
        stage_times.append(stage_max)

    if not stage_times:
        raise ValueError(
            f"Scope '{scope.get('scope_name', '?')}' has no queues with resolved_time"
        )

    # Serial stages → sum
    return functools.reduce(operator.add, stage_times)
