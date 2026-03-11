"""SMT-based block-size optimizer for the Loom dataflow pipeline.

Solves all variants in the staged ETG JSON in parallel, writes per-variant
detailed results to a log file, and prints the global best variant to the
console.

NOTE: Z3's global AST context is not thread-safe. Workers run in separate
processes (ProcessPoolExecutor) so each gets its own Z3 context.

Usage:
    python main.py --input PATH [--njobs N] [--output PATH] [--enumerate]

Arguments:
    --input      Path to staged_etg_dump.json (required).
    --njobs      Number of parallel worker processes (default: 1).
    --output     Path for the per-variant log file (optional).
    --enumerate  Use brute-force enumeration instead of Z3 Optimize.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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

def _optimize_z3(ctx: SolverContext, t_total: z3.ArithRef):
    """Run Z3 Optimize to minimize t_total.

    Returns (min_val, assignments) on success, or None on unsat/unknown.
    """
    return ctx.minimize(t_total)


def _optimize_enumerate(
    ctx: SolverContext,
    t_total: z3.ArithRef,
    domains: dict[str, list[int]],
) -> tuple[int, dict[str, int]] | None:
    """Enumerate all domain combinations and find the minimum feasible T_total.

    Returns (min_val, assignments) or None if no feasible point exists.
    """
    base_constraints = ctx.get_constraints()
    sym_names = list(ctx.symbol_map.keys())

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
# Single-variant solver (runs in a worker process)
# ---------------------------------------------------------------------------

def solve_variant(
    variant: dict,
    index: int,
    total: int,
    use_enumerate: bool,
    domains: dict[str, list[int]],
) -> dict:
    """Solve one variant and return a result dict.

    Runs in its own process, so Z3 state is fully isolated.

    Returns:
        {
            "variant":     variant dict,
            "index":       int,
            "total":       int,
            "min_val":     int | None,   # None means UNSAT
            "assignments": dict | None,
        }
    """
    ctx = SolverContext()
    ctx.load_symbols(variant["constraint_scope"]["metadata"]["symbols"])
    ctx.add_hard_constraints(variant["constraint_scope"]["hard_constraints"])
    ctx.add_domain_constraints(domains)

    t_total = compute_total_time(variant, ctx.symbol_map)

    result = None
    if not use_enumerate:
        result = _optimize_z3(ctx, t_total)
        if result is None:
            use_enumerate = True

    if use_enumerate:
        result = _optimize_enumerate(ctx, t_total, domains)

    min_val, assignments = result if result is not None else (None, None)

    return {
        "variant":     variant,
        "index":       index,
        "total":       total,
        "min_val":     min_val,
        "assignments": assignments,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    variants = load_variants(args.input)
    total = len(variants)
    domains = default_symbol_domains()

    print(f"Solving {total} variants with {args.njobs} process(es)...")
    print()

    # Dispatch all variants to a process pool.
    # Progress is printed by the main process as each future completes.
    results: list[dict] = [None] * total
    with ProcessPoolExecutor(max_workers=args.njobs) as pool:
        futures = {
            pool.submit(solve_variant, v, i, total, args.enumerate, domains): i
            for i, v in enumerate(variants)
        }
        completed = 0
        for future in as_completed(futures):
            r = future.result()
            results[r["index"]] = r
            completed += 1
            variant_name = r["variant"].get("variant_name", f"variant_{r['index']}")
            if r["min_val"] is None:
                print(f"[{completed:3d}/{total}] {variant_name}  UNSAT")
            else:
                print(f"[{completed:3d}/{total}] {variant_name}  T={r['min_val']:,} cycles")

    # Write per-variant details to log file, ordered by original index.
    if args.output:
        with open(args.output, "w", encoding="utf-8") as log:
            for r in results:
                if r["min_val"] is None:
                    print(
                        f"Variant [{r['index']}/{total - 1}]: "
                        f"{r['variant'].get('variant_name', '?')}  UNSAT\n",
                        file=log,
                    )
                else:
                    print_breakdown(
                        r["variant"], r["assignments"],
                        r["min_val"], r["index"], total,
                        file=log,
                    )
                    print("-" * 72, file=log)
        print(f"\nPer-variant log written to: {args.output}")

    # Find the global best (minimum T_total across all feasible variants).
    feasible = [r for r in results if r["min_val"] is not None]
    if not feasible:
        print("\nResult: UNSAT — no variant has a feasible solution.")
        sys.exit(2)

    best = min(feasible, key=lambda r: r["min_val"])

    print()
    print("=" * 72)
    print("GLOBAL BEST")
    print("=" * 72)
    print_breakdown(
        best["variant"], best["assignments"],
        best["min_val"], best["index"], total,
    )

    # Build and return the consolidated block size map for programmatic use.
    block_sizes: dict[str, dict[str, int]] = {}
    for r in results:
        if r["min_val"] is not None:
            vname = r["variant"].get("variant_name", f"variant_{r['index']}")
            block_sizes[vname] = dict(r["assignments"])
    return block_sizes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find optimal block sizes across all Loom ETG variants."
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="JSON",
        help="Path to staged_etg_dump.json",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel worker processes (default: 1)",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Log file for per-variant results (optional)",
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
