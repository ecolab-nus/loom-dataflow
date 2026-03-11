"""Manage Z3 solver state, symbol table, and constraint accumulation.

SolverContext acts as the 'virtual symbol table' for the SMT problem.
It owns a Z3 Solver and the symbol_map, and provides methods for loading
symbols, adding constraints, and finding the optimum via binary search.

NOTE: We deliberately use z3.Solver (not z3.Optimize) because the T_total
objective is non-linear (nested If/max, symbolic multiplication, integer
division).  z3.Optimize (nu-Z) is incomplete for NIA and can return
sub-optimal solutions without reporting UNKNOWN.  z3.Solver.check() on
bounded integer domains is decidable and complete, so a binary search on
the objective value guarantees the true global minimum.
"""

from itertools import product as iter_product

import z3

from .expr_resolver import resolve_constraint


class SolverContext:
    """Holds the Z3 solver, symbol table, and all accumulated constraints.

    Usage:
        ctx = SolverContext()
        ctx.load_symbols(variant["constraint_scope"]["metadata"]["symbols"])
        ctx.add_hard_constraints(variant["constraint_scope"]["hard_constraints"])
        ctx.add_domain_constraints(default_symbol_domains())
        result = ctx.find_optimum(t_total_expr, domains)
    """

    def __init__(self) -> None:
        self.solver = z3.Solver()
        self.symbol_map: dict[str, z3.ArithRef] = {}
        self._constraints: list[z3.BoolRef] = []

    def load_symbols(self, metadata_symbols: dict[str, str]) -> None:
        """Declare Z3 integer variables for each symbol in the metadata.

        Args:
            metadata_symbols: e.g. {"BM": "int", "BN": "int", "BK": "int"}

        Raises:
            ValueError: If a symbol type other than "int" is encountered.
        """
        for name, typ in metadata_symbols.items():
            if typ != "int":
                raise ValueError(f"Unsupported symbol type '{typ}' for symbol '{name}'")
            var = z3.Int(name)
            self.symbol_map[name] = var
            guard = var > 0
            self.solver.add(guard)
            self._constraints.append(guard)

    def add_hard_constraints(self, constraints: list[dict]) -> None:
        """Resolve and add every hard constraint from the ETG JSON.

        Args:
            constraints: List of constraint dicts (Ge, Divisible, …).
        """
        for c in constraints:
            z3_c = resolve_constraint(c, self.symbol_map)
            self.solver.add(z3_c)
            self._constraints.append(z3_c)

    def add_domain_constraints(self, domains: dict[str, list[int]]) -> None:
        """Restrict each symbol to a finite set of allowed values.

        Args:
            domains: e.g. {"BM": [32, 64, 128, …], "BK": [32, 64, …, 512]}
        """
        for name, values in domains.items():
            if name not in self.symbol_map:
                continue
            sym = self.symbol_map[name]
            domain_c = z3.Or([sym == v for v in values])
            self.solver.add(domain_c)
            self._constraints.append(domain_c)

    def get_constraints(self) -> list[z3.BoolRef]:
        """Return all accumulated constraints (for use in enumeration mode)."""
        return list(self._constraints)

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def find_optimum(
        self,
        objective: z3.ArithRef,
        domains: dict[str, list[int]],
    ) -> tuple[int, dict[str, int]] | None:
        """Find the minimum value of *objective* via binary search.

        Phase 1 – obtain an initial feasible upper bound by calling
        ``solver.check()`` with no objective constraint.
        Phase 2 – binary-search: add ``objective <= mid``, check SAT/UNSAT.
        Fallback – if Z3 ever returns UNKNOWN, fall back to full enumeration.

        Returns:
            ``(min_value, {sym: value, …})`` on success, or ``None`` when the
            constraint system is infeasible (UNSAT).
        """
        # Phase 1: any feasible point → upper bound
        bound = self._find_initial_bound(objective)
        if bound is None:
            return None
        upper, best_assignments = bound

        # Phase 2: binary search
        lower = 0
        while lower < upper:
            mid = (lower + upper) // 2
            self.solver.push()
            self.solver.add(objective <= mid)
            result = self.solver.check()

            if result == z3.sat:
                model = self.solver.model()
                val = model.eval(objective, model_completion=True).as_long()
                upper = val
                best_assignments = {
                    name: model.eval(var, model_completion=True).as_long()
                    for name, var in self.symbol_map.items()
                }
                self.solver.pop()
            elif result == z3.unsat:
                self.solver.pop()
                lower = mid + 1
            else:
                self.solver.pop()
                return self._enumerate_all(objective, domains)

        return upper, best_assignments

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_initial_bound(
        self, objective: z3.ArithRef,
    ) -> tuple[int, dict[str, int]] | None:
        """Return *(upper_bound, assignments)* from any feasible point, or None."""
        result = self.solver.check()
        if result == z3.sat:
            model = self.solver.model()
            val = model.eval(objective, model_completion=True).as_long()
            assignments = {
                name: model.eval(var, model_completion=True).as_long()
                for name, var in self.symbol_map.items()
            }
            return val, assignments
        # unsat or unknown — caller treats both as "no bound found"
        return None

    def _enumerate_all(
        self,
        objective: z3.ArithRef,
        domains: dict[str, list[int]],
    ) -> tuple[int, dict[str, int]] | None:
        """Brute-force all domain combinations (UNKNOWN fallback)."""
        sym_names = list(self.symbol_map.keys())
        active = [(n, domains[n]) for n in sym_names if n in domains]
        if not active:
            return None

        names, value_lists = zip(*active)
        best: tuple[int, dict[str, int]] | None = None

        for combo in iter_product(*value_lists):
            s = z3.Solver()
            s.add(self._constraints)
            for name, val in zip(names, combo):
                s.add(self.symbol_map[name] == val)

            if s.check() != z3.sat:
                continue

            model = s.model()
            val = model.eval(objective, model_completion=True).as_long()
            if best is None or val < best[0]:
                best = (val, dict(zip(names, combo)))

        return best
