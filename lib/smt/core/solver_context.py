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

from .expr_resolver import resolve_constraint, resolve_expr


class SolverContext:
    """Holds the Z3 solver, symbol table, and all accumulated constraints.

    Usage:
        ctx = SolverContext()
        ctx.load_symbols(variant["constraint_scope"]["metadata"]["symbols"])
        ctx.add_hard_constraints(variant["constraint_scope"]["hard_constraints"])
        ctx.add_domain_constraints(default_symbol_domains())
        result = ctx.find_optimum(t_total_expr, domains)
    """

    def __init__(self, debug: bool = False) -> None:
        self.solver = z3.Solver()
        self.symbol_map: dict[str, z3.ArithRef] = {}
        self._constraints: list[z3.BoolRef] = []
        self._tracking_vars: dict[str, tuple[z3.BoolRef, z3.BoolRef]] = {}
        self.last_unsat_core_info: list[tuple[str, str]] | None = None
        self.debug = debug

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

        Phase 0 – simplify constraints and objective via Z3 tactics.
        Phase 1 – obtain an initial feasible upper bound by calling
        ``solver.check()`` with no objective constraint.
        Phase 2 – binary-search: add ``obj_var <= mid``, check SAT/UNSAT.
        Fallback – if Z3 ever returns UNKNOWN in Phase 1 or Phase 2,
        fall back to full enumeration.

        Returns:
            ``(min_value, {sym: value, …})`` on success, or ``None`` when the
            constraint system is infeasible (UNSAT).
        """
        # Phase 0: canonicalize constraints + objective
        obj_var = self._simplify(objective)

        # Phase 1: any feasible point → upper bound
        status, upper, best_assignments = self._find_initial_bound(obj_var)
        if status == z3.unsat:
            if self.debug:
                self._capture_unsat_core()
            return None
        if status == z3.unknown:
            return self._enumerate_all(objective, domains)

        # Phase 2: binary search
        lower = 0
        while lower < upper:
            mid = (lower + upper) // 2
            self.solver.push()
            if self.debug:
                bound_label = f"__obj_le_{mid}"
                bound_var = z3.Bool(bound_label)
                self.solver.assert_and_track(obj_var <= mid, bound_var)
            else:
                self.solver.add(obj_var <= mid)
            result = self.solver.check()

            if result == z3.sat:
                model = self.solver.model()
                val = model.eval(obj_var, model_completion=True).as_long()
                upper = val
                best_assignments = {
                    name: model.eval(var, model_completion=True).as_long()
                    for name, var in self.symbol_map.items()
                }
                self.solver.pop()
            elif result == z3.unsat:
                if self.debug:
                    self._capture_unsat_core()
                self.solver.pop()
                lower = mid + 1
            else:
                self.solver.pop()
                return self._enumerate_all(objective, domains)

        return upper, best_assignments

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simplify(self, objective: z3.ArithRef) -> z3.ArithRef:
        """Run Z3 tactic pipeline to canonicalize constraints and objective.

        Introduces an auxiliary variable ``__obj == objective``, runs the
        tactic chain, and rebuilds the solver with simplified constraints.
        When ``self.debug`` is True, constraints are added via
        ``assert_and_track()`` so that UNSAT cores can be extracted later.

        Returns:
            The ``__obj`` auxiliary variable (bound to the simplified
            objective inside the solver).
        """
        tactic = z3.Then(
            'simplify',           # constant folding, flatten, basic arith
            'propagate-values',   # substitute known equalities
            'ctx-simplify',       # context-dependent simplification
            'elim-term-ite',      # lift if-then-else out of terms
        )

        obj_var = z3.Int('__obj')

        goal = z3.Goal()
        for c in self.solver.assertions():
            goal.add(c)
        goal.add(obj_var == objective)

        simplified = tactic(goal)

        self.solver.reset()
        self._constraints = []
        self._tracking_vars = {}
        for i, subgoal in enumerate(simplified):
            for j, c in enumerate(subgoal):
                if self.debug:
                    track_name = f"sc_{i}_{j}"
                    track_var = z3.Bool(track_name)
                    self.solver.assert_and_track(c, track_var)
                    self._tracking_vars[track_name] = (track_var, c)
                else:
                    self.solver.add(c)
                self._constraints.append(c)

        return obj_var

    def _find_initial_bound(
        self, objective: z3.ArithRef,
    ) -> tuple[z3.CheckSatResult, int | None, dict[str, int] | None]:
        """Check feasibility and return an initial upper bound if SAT.

        Returns:
            ``(z3.sat, upper_bound, assignments)`` on success,
            ``(z3.unsat, None, None)`` when infeasible, or
            ``(z3.unknown, None, None)`` when the solver cannot decide.
        """
        result = self.solver.check()
        if result == z3.sat:
            model = self.solver.model()
            val = model.eval(objective, model_completion=True).as_long()
            assignments = {
                name: model.eval(var, model_completion=True).as_long()
                for name, var in self.symbol_map.items()
            }
            return z3.sat, val, assignments
        if result == z3.unsat:
            return z3.unsat, None, None
        return z3.unknown, None, None

    def _capture_unsat_core(self) -> None:
        """Store formatted UNSAT core from the most recent UNSAT check."""
        core = self.solver.unsat_core()
        info = []
        for v in core:
            name = str(v)
            if name in self._tracking_vars:
                info.append((name, str(self._tracking_vars[name][1])))
            else:
                info.append((name, "(bound constraint)"))
        self.last_unsat_core_info = info

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

        if best is None:
            self.last_unsat_core_info = [
                ("enumerate_fallback", "no feasible combination found in domain")
            ]

        return best

    # ------------------------------------------------------------------
    # Debug analysis
    # ------------------------------------------------------------------

    def find_active_constraints(
        self,
        hard_constraints: list[dict],
        assignments: dict[str, int],
    ) -> list[tuple[int, dict, str]]:
        """Identify hard constraints that are tight at the given assignments.

        A Ge constraint ``lhs >= rhs`` is "active" when ``lhs_val == rhs_val``.
        Eq constraints are always active at any feasible point.

        Returns:
            List of ``(index, constraint_json, description_str)`` for tight
            constraints.
        """
        concrete_map = {name: z3.IntVal(val) for name, val in assignments.items()}
        active: list[tuple[int, dict, str]] = []
        for i, c in enumerate(hard_constraints):
            tag, payload = next(iter(c.items()))
            if tag == "Ge":
                lhs_val = z3.simplify(resolve_expr(payload[0], concrete_map)).as_long()
                rhs_val = z3.simplify(resolve_expr(payload[1], concrete_map)).as_long()
                if lhs_val == rhs_val:
                    desc = f"Ge: {lhs_val} >= {rhs_val} (tight, slack=0)"
                    active.append((i, c, desc))
            elif tag == "Eq":
                lhs_val = z3.simplify(resolve_expr(payload[0], concrete_map)).as_long()
                rhs_val = z3.simplify(resolve_expr(payload[1], concrete_map)).as_long()
                desc = f"Eq: {lhs_val} == {rhs_val}"
                active.append((i, c, desc))
        return active

    def find_mus(
        self,
        hard_constraints: list[dict],
        domains: dict[str, list[int]],
    ) -> list[tuple[int, str, str]]:
        """Find a Minimum Unsatisfiable Subset via deletion-based algorithm.

        Builds a fresh solver for each removal attempt so the main solver
        state is not disturbed.  Operates on original (pre-simplification)
        constraints for interpretability.

        Returns:
            List of ``(index, label, z3_expr_str)`` for each MUS member.
        """
        labeled: list[tuple[str, z3.BoolRef]] = []

        # Positivity guards
        for name, var in self.symbol_map.items():
            labeled.append((f"positivity({name})", var > 0))

        # Domain constraints
        for name, values in domains.items():
            if name not in self.symbol_map:
                continue
            sym = self.symbol_map[name]
            labeled.append((f"domain({name})", z3.Or([sym == v for v in values])))

        # Hard constraints
        for i, c in enumerate(hard_constraints):
            tag = next(iter(c.keys()))
            z3_c = resolve_constraint(c, self.symbol_map)
            labeled.append((f"hard[{i}]({tag})", z3_c))

        # Deletion-based MUS
        remaining = list(range(len(labeled)))
        for idx in list(remaining):
            candidate = [i for i in remaining if i != idx]
            s = z3.Solver()
            for i in candidate:
                s.add(labeled[i][1])
            if s.check() != z3.sat:
                # Still UNSAT without this constraint → not needed
                remaining.remove(idx)

        return [(i, labeled[i][0], str(labeled[i][1])) for i in remaining]
