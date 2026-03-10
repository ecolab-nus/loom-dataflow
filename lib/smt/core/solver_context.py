"""Manage Z3 solver state, symbol table, and constraint accumulation.

SolverContext acts as the 'virtual symbol table' for the SMT problem.
It owns the Z3 Optimize object and the symbol_map, and provides methods
for loading symbols, adding constraints, and running the optimizer.
"""

import z3
from .expr_resolver import resolve_constraint


class SolverContext:
    """Holds the Z3 solver, symbol table, and all accumulated constraints.

    Usage:
        ctx = SolverContext()
        ctx.load_symbols(variant["constraint_scope"]["metadata"]["symbols"])
        ctx.add_hard_constraints(variant["constraint_scope"]["hard_constraints"])
        ctx.add_domain_constraints(default_symbol_domains())
    """

    def __init__(self) -> None:
        self.optimize = z3.Optimize()
        self.symbol_map: dict[str, z3.ArithRef] = {}
        # Keep a plain Solver copy of all constraints for enumeration mode.
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
            # Safety guard: prevent division-by-zero in expressions.
            guard = var > 0
            self.optimize.add(guard)
            self._constraints.append(guard)

    def add_hard_constraints(self, constraints: list[dict]) -> None:
        """Resolve and add every hard constraint from the ETG JSON.

        Args:
            constraints: List of constraint dicts (Ge, Divisible, …).
        """
        for c in constraints:
            z3_c = resolve_constraint(c, self.symbol_map)
            self.optimize.add(z3_c)
            self._constraints.append(z3_c)

    def add_domain_constraints(self, domains: dict[str, list[int]]) -> None:
        """Restrict each symbol to a finite set of allowed values.

        Adds an Or(sym == v1, sym == v2, …) clause for each symbol.
        Symbols in domains that are not in symbol_map are silently skipped
        (the variant may not use all three block dimensions).

        Args:
            domains: e.g. {"BM": [32, 64, 128, …], "BK": [32, 64, …, 512]}
        """
        for name, values in domains.items():
            if name not in self.symbol_map:
                continue
            sym = self.symbol_map[name]
            domain_c = z3.Or([sym == v for v in values])
            self.optimize.add(domain_c)
            self._constraints.append(domain_c)

    def get_constraints(self) -> list[z3.BoolRef]:
        """Return all accumulated constraints (for use in enumeration mode)."""
        return list(self._constraints)

    def minimize(self, objective: z3.ArithRef) -> tuple[int, dict[str, int]] | None:
        """Minimize objective using Z3 Optimize.

        Args:
            objective: The Z3 expression to minimize.

        Returns:
            (min_value, {sym_name: concrete_value, …}) on sat, or None on unsat/unknown.
        """
        self.optimize.minimize(objective)
        result = self.optimize.check()
        if result == z3.sat:
            model = self.optimize.model()
            val = model.eval(objective, model_completion=True).as_long()
            assignments = {
                name: model.eval(var, model_completion=True).as_long()
                for name, var in self.symbol_map.items()
            }
            return val, assignments
        return None
