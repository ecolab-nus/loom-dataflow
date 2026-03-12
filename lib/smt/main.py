"""SMT-based block-size optimizer for the Loom dataflow pipeline.

Solves all variants in the staged ETG JSON in parallel, writes per-variant
detailed results to a log file, and prints the global best variant to the
console.

NOTE: Z3's global AST context is not thread-safe. Workers run in separate
processes (ProcessPoolExecutor) so each gets its own Z3 context.

Usage:
    python main.py --input PATH [--njobs N] [--output PATH]

Arguments:
    --input      Path to staged_etg_dump.json (required).
    --njobs      Number of parallel worker processes (default: 1).
    --output     Path for the per-variant log file (optional).
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.json_loader import load_variants
from utils.utils import default_symbol_domains
from utils.reporter import print_breakdown, print_unsat_core
from core.solver_context import SolverContext
from models.pipeline_agg import compute_total_time


# ---------------------------------------------------------------------------
# Single-variant solver (runs in a worker process)
# ---------------------------------------------------------------------------

def solve_variant(
    variant: dict,
    index: int,
    total: int,
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
    result = ctx.find_optimum(t_total, domains)

    min_val, assignments = result if result is not None else (None, None)

    return {
        "variant":     variant,
        "index":       index,
        "total":       total,
        "min_val":     min_val,
        "assignments": assignments,
        "unsat_core":  ctx.last_unsat_core_info,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, dict[str, int] | None]:
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
            pool.submit(solve_variant, v, i, total, domains): i
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
                vname = r["variant"].get("variant_name", "?")
                if r["min_val"] is None:
                    print(
                        f"Variant [{r['index']}/{total - 1}]: {vname}  UNSAT\n",
                        file=log,
                    )
                    if r.get("unsat_core"):
                        print_unsat_core(
                            vname, r["unsat_core"],
                            context="Infeasible", file=log,
                        )
                else:
                    print_breakdown(
                        r["variant"], r["assignments"],
                        r["min_val"], r["index"], total,
                        file=log,
                    )
                    if r.get("unsat_core"):
                        print_unsat_core(
                            vname, r["unsat_core"],
                            context=f"Optimum bound T>={r['min_val']}",
                            file=log,
                        )
                    print("-" * 72, file=log)
        print(f"\nPer-variant log written to: {args.output}")

    # Build the consolidated block size map (None for UNSAT variants).
    block_sizes: dict[str, dict[str, int] | None] = {}
    for r in results:
        vname = r["variant"].get("variant_name", f"variant_{r['index']}")
        if r["min_val"] is not None:
            block_sizes[vname] = dict(r["assignments"])
        else:
            block_sizes[vname] = None

    # Find the global best (minimum T_total across all feasible variants).
    feasible = [r for r in results if r["min_val"] is not None]
    if not feasible:
        print("\nResult: UNSAT — no variant has a feasible solution.")
        return block_sizes

    best = min(feasible, key=lambda r: r["min_val"])

    print()
    print("=" * 72)
    print("GLOBAL BEST")
    print("=" * 72)
    print_breakdown(
        best["variant"], best["assignments"],
        best["min_val"], best["index"], total,
    )

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
    args = parser.parse_args()
    block_sizes = run(args)
    if not any(v is not None for v in block_sizes.values()):
        sys.exit(2)


if __name__ == "__main__":
    main()
