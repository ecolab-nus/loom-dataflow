"""SMT-based block-size optimizer for the Loom dataflow pipeline.

Reads a staged ETG JSON file, builds Z3 constraints and a symbolic timing
model, and finds the block-size combination (BM, BN, BK) that minimizes the
total pipeline execution time T_total.

Usage:
    python main.py --input PATH [--variant-index N] [--enumerate]

Arguments:
    --input           Path to staged_etg_dump.json (required).
    --variant-index   Which variant to optimize (default: 0).
    --enumerate       Use brute-force enumeration instead of Z3 Optimize.
                      Enumeration is always fast for the default 3-symbol
                      domain (~256 combinations) and is guaranteed to find
                      the global optimum.
"""

import argparse
import sys
from itertools import product as iter_product

import z3

from utils.json_loader import load_variants
from utils.utils import default_symbol_domains
from utils.reporter import print_breakdown
from core.solver_context import SolverContext
from models.pipeline_agg import compute_total_time


# ---------------------------------------------------------------------------
# Optimization strategies
# ---------------------------------------------------------------------------

def optimize_z3(ctx: SolverContext, t_total: z3.ArithRef):
    """Run Z3 Optimize to minimize t_total.

    Returns (min_val, assignments) on success, or None if z3 returns
    unsat/unknown (in which case the caller should fall back to enumeration).
    """
    result = ctx.minimize(t_total)
    return result


def optimize_enumerate(
    ctx: SolverContext,
    t_total: z3.ArithRef,
    domains: dict[str, list[int]],
) -> tuple[int, dict[str, int]] | None:
    """Enumerate all domain combinations and find the minimum feasible T_total.

    For each candidate (BM, BN, BK), a fresh Z3 Solver checks feasibility
    (hard constraints + concrete assignments), then evaluates T_total.

    Returns (min_val, assignments) or None if no feasible point exists.
    """
    base_constraints = ctx.get_constraints()
    sym_names = list(ctx.symbol_map.keys())

    # Only iterate over symbols that are in both the domain dict and symbol_map.
    active = [(name, domains[name]) for name in sym_names if name in domains]
    if not active:
        return None

    names, value_lists = zip(*active)
    best: tuple[int, dict[str, int]] | None = None

    for combo in iter_product(*value_lists):
        s = z3.Solver()
        s.add(base_constraints)
        for name, val in zip(names, combo):
            s.add(ctx.symbol_map[name] == val)

        if s.check() != z3.sat:
            continue

        model = s.model()
        val = model.eval(t_total, model_completion=True).as_long()

        if best is None or val < best[0]:
            best = (val, dict(zip(names, combo)))

    return best


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # 1. Load and validate the JSON.
    variants = load_variants(args.input)
    if args.variant_index >= len(variants):
        print(
            f"Error: --variant-index {args.variant_index} out of range "
            f"(file has {len(variants)} variants).",
            file=sys.stderr,
        )
        sys.exit(1)

    variant = variants[args.variant_index]
    print(f"Variant: {variant['variant_name']}")
    print(f"Index:   {args.variant_index} / {len(variants) - 1}")

    # 2. Build symbol table and add constraints.
    ctx = SolverContext()
    ctx.load_symbols(variant["constraint_scope"]["metadata"]["symbols"])
    ctx.add_hard_constraints(variant["constraint_scope"]["hard_constraints"])

    domains = default_symbol_domains()
    ctx.add_domain_constraints(domains)

    # 3. Build the symbolic timing objective.
    t_total = compute_total_time(variant, ctx.symbol_map)

    # 4. Optimize.
    result = None

    if not args.enumerate:
        print("Strategy: Z3 Optimize")
        result = optimize_z3(ctx, t_total)
        if result is None:
            print("  Z3 Optimize returned unsat/unknown — falling back to enumeration.")
            args.enumerate = True

    if args.enumerate:
        print("Strategy: Brute-force enumeration")
        result = optimize_enumerate(ctx, t_total, domains)

    # 5. Output.
    print()
    if result is None:
        print("Result: UNSAT — no feasible block-size combination found.")
        sys.exit(2)

    min_val, assignments = result
    print(f"Optimal T_total: {min_val:,} cycles")
    for sym, val in sorted(assignments.items()):
        print(f"  {sym} = {val}")

    print()
    print_breakdown(variant, assignments)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the optimal block sizes (BM, BN, BK) for a Loom ETG variant."
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="JSON",
        help="Path to staged_etg_dump.json",
    )
    parser.add_argument(
        "--variant-index",
        type=int,
        default=0,
        metavar="N",
        help="Which variant to optimize (default: 0)",
    )
    parser.add_argument(
        "--enumerate",
        action="store_true",
        help="Force brute-force enumeration (skips Z3 Optimize)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
