import numpy as np
from typing import List, Dict, Tuple, Callable, Any
from .parser import Constraint, Term, ConstraintSpace

class FourierMotzkinProjector:
    """
    Projects out intermediate variables from linear constraints using 
    Fourier-Motzkin elimination logic.
    """
    def __init__(self, constraints: List[Constraint], primary_vars: List[str], intermediate_vars: List[str]):
        self.constraints = constraints
        self.primary_vars = primary_vars
        self.intermediate_vars = intermediate_vars

    def compile_projected_field_function(self) -> Callable:
        """
        Returns a function f(M, N, K) that computes the feasibility boundary.
        Calculates the feasibility condition after eliminating all intermediate variables.
        """
        
        # We start with the original constraints
        current_constraints = self.simplify_and_flatten(self.constraints)
        
        # Eliminate each intermediate variable one by one
        for var in self.intermediate_vars:
            current_constraints = self._project_variable(current_constraints, var)
        
        # Now current_constraints contains only primary variables
        params = ", ".join(self.primary_vars)
        
        if not current_constraints:
            # If no constraints left, everything is valid (field = 0.5 < 1.0)
            return eval(f"lambda {params}: np.zeros_like({self.primary_vars[0]}) + 0.5", {"np": np})
            
        combined_expr = f"np.maximum.reduce([{', '.join(current_constraints)}])"
        # Boundary at 1.0
        final_code = f"lambda {params}: {combined_expr} + 1.0"
        
        return eval(final_code, {"np": np})

    def simplify_and_flatten(self, constraints: List[Constraint]) -> List[str]:
        """Converts Constraints to string expressions of the form LHS <= 0."""
        exprs = []
        for c in constraints:
            if c.type != 'linear':
                continue
            
            term_parts = []
            for t in c.terms:
                v_part = " * ".join(t.variables) if t.variables else "1"
                term_parts.append(f"({t.coefficient} * {v_part})")
            
            lhs = " + ".join(term_parts) if term_parts else "0"
            const = c.constant if c.constant is not None else 0
            
            if c.is_equality:
                # Equality S=L is two inequalities S <= L and S >= L
                # (S - L <= 0) and (L - S <= 0)
                exprs.append(f"(({lhs}) + ({const}))")
                exprs.append(f"(-(({lhs}) + ({const})))")
            else:
                exprs.append(f"(({lhs}) + ({const}))")
        return exprs

    def _project_variable(self, current_exprs: List[str], target_var: str) -> List[str]:
        """Eliminates target_var from the list of linear expressions."""
        lowers = [] # L - S <= 0  => L <= S
        uppers = [] # S - U <= 0  => S <= U
        neutrals = [] # No S
        
        for expr in current_exprs:
            # This is a bit hacky as we are using raw strings. 
            # For robust elimination, we'd want a proper symbolic representation.
            # But since we are generating these strings ourselves, we can track coefficients.
            
            # Since we can't easily parse coefficients back from strings reliably without a parser,
            # let's re-implement the extraction logic using the original Constraint objects 
            # but only for the first pass, or keep tracked coefficients.
            pass

        # RE-IMPLEMENTATION using original data structures for better robustness
        # Let's perform the projection on the Constraint objects themselves.
        return self._project_on_constraints(target_var)

    def _project_on_constraints(self, target_var: str) -> List[str]:
        # This is hard because recursion creates new virtual constraints.
        # Let's use a simpler approach: 
        # For each constraint, represent it as Σ a_i * x_i + c <= 0.
        
        # Initial representation: List of dict {var_name: coeff} + 'const'
        vectors = []
        for c in self.constraints:
            if c.type != 'linear': continue
            v = {name: 0 for name in self.primary_vars + self.intermediate_vars}
            for t in c.terms:
                for var in t.variables:
                    v[var] += t.coefficient
            v['const'] = c.constant if c.constant is not None else 0
            
            if c.is_equality:
                vectors.append(v)
                inv_v = {k: -val for k, val in v.items()}
                vectors.append(inv_v)
            else:
                vectors.append(v)
        
        # Fourier-Motzkin elimination on vectors
        for sv in self.intermediate_vars:
            new_vectors = []
            pos = []
            neg = []
            zero = []
            for v in vectors:
                coeff = v.get(sv, 0)
                if coeff > 1e-9:
                    # Normalize s.t. coeff of sv is 1
                    pos.append({k: val/coeff for k, val in v.items()})
                elif coeff < -1e-9:
                    # Normalize s.t. coeff of sv is -1
                    neg.append({k: val/abs(coeff) for k, val in v.items()})
                else:
                    zero.append(v)
            
            # Pair pos (S - U <= 0) and neg (L - S <= 0)
            # Result: L - U <= 0
            for p in pos:
                for n in neg:
                    # L - U is n + p (since n is L - S and p is S - U)
                    combined = {k: p.get(k, 0) + n.get(k, 0) for k in self.primary_vars + self.intermediate_vars + ['const']}
                    if sv in combined: del combined[sv]
                    new_vectors.append(combined)
            
            new_vectors.extend(zero)
            vectors = new_vectors
            
        # Convert back to strings
        final_exprs = []
        for v in vectors:
            terms = []
            for name in self.primary_vars:
                coeff = v.get(name, 0)
                if abs(coeff) > 1e-9:
                    terms.append(f"({coeff} * {name})")
            const = v.get('const', 0)
            expr = "(" + (" + ".join(terms) if terms else "0") + f" + {const})"
            final_exprs.append(expr)
            
        return final_exprs
