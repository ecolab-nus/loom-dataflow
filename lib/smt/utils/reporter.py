"""Console reporting utilities for SMT solver results.

Responsible for formatting and printing the per-scope / per-stage / per-queue
timing breakdown after optimal symbol values have been determined.
"""

import sys
from typing import TextIO

import z3

from core.expr_resolver import resolve_expr


def eval_expr(expr_dict: dict, concrete_map: dict[str, z3.ArithRef]) -> int:
    """Evaluate a JSON Expr to a concrete integer given a concrete symbol map."""
    return z3.simplify(resolve_expr(expr_dict, concrete_map)).as_long()


def print_breakdown(
    variant: dict,
    assignments: dict[str, int],
    min_val: int,
    index: int,
    total: int,
    file: TextIO = None,
) -> None:
    """Write a hierarchical timing breakdown for the given symbol assignments.

    Substitutes concrete integer values for all symbols, then writes every
    scope → stage → queue resolved_time, scope totals, and the final T_total
    derivation to *file* (defaults to sys.stdout).

    Args:
        variant:     A single variant dict from the staged ETG JSON.
        assignments: Mapping of symbol name → concrete integer value.
        min_val:     The optimal T_total value.
        index:       0-based index of this variant in the full list.
        total:       Total number of variants.
        file:        Output stream. Defaults to sys.stdout.
    """
    if file is None:
        file = sys.stdout

    def p(*args, **kwargs):
        print(*args, **kwargs, file=file)

    variant_name = variant.get("variant_name", "unknown")
    p(f"Variant [{index}/{total - 1}]: {variant_name}")
    p(f"Optimal T_total: {min_val:,} cycles")
    for sym, val in sorted(assignments.items()):
        p(f"  {sym} = {val}")
    p()

    concrete_map: dict[str, z3.ArithRef] = {
        name: z3.IntVal(val) for name, val in assignments.items()
    }

    iter_num = variant["constraint_scope"]["metadata"]["iter_num"]
    seq_val = eval_expr(iter_num["seq_iter"], concrete_map)
    temp_vals = [eval_expr(t, concrete_map) for t in iter_num["temp_iter"]]
    temp_product = 1
    for v in temp_vals:
        temp_product *= v

    temp_str = " × ".join(str(v) for v in temp_vals)
    p(f"Iteration factors:  seq_iter={seq_val},  temp_iter=[{temp_str}]  (product={temp_product})")
    p()

    scope_totals: list[tuple[str, int]] = []

    for scope_key in ("compute_scope", "memory_scope"):
        scope = variant[scope_key]
        scope_name = scope.get("scope_name", scope_key)
        p(f"  [{scope_name}]")

        scope_total = 0
        for stage in scope["stages"]:
            stage_id = stage.get("stage_id", "?")
            p(f"    Stage {stage_id}:")

            queue_vals: dict[str, int] = {}
            for q_name, queue in stage["queues"].items():
                rt = queue.get("resolved_time")
                if rt is None:
                    p(f"      {q_name:12s}  (no resolved_time)")
                    continue
                queue_vals[q_name] = eval_expr(rt, concrete_map)

            if queue_vals:
                stage_max = max(queue_vals.values())
                for q_name, val in queue_vals.items():
                    marker = " ← bottleneck" if val == stage_max else ""
                    p(f"      {q_name:12s}  {val:>10,} cycles{marker}")
                p(f"      {'(stage max)':12s}  {stage_max:>10,} cycles")
                scope_total += stage_max

        p(f"  {'→ scope total':16s}  {scope_total:>10,} cycles")
        p()
        scope_totals.append((scope_name, scope_total))

    t_comp = next(v for n, v in scope_totals if "compute" in n.lower())
    t_mem  = next(v for n, v in scope_totals if "memory"  in n.lower())
    t_stage = max(t_comp, t_mem)

    p(f"  T_comp  = {t_comp:>10,} cycles")
    p(f"  T_mem   = {t_mem:>10,} cycles")
    p(f"  T_stage = max(T_comp, T_mem) = {t_stage:,} cycles")
    p(f"  T_total = {t_stage:,} × {seq_val} × {temp_product} = {t_stage * seq_val * temp_product:,} cycles")
    p()
