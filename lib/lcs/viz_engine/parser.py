import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import numpy as np

@dataclass
class Variable:
    index: int
    name: str
    is_intermediate: bool

@dataclass
class RangeMetadata:
    variable: str
    lower_bound: int
    upper_bound: int

@dataclass
class AlignMetadata:
    variable: str
    alignment: int

@dataclass
class Term:
    coefficient: int
    variables: List[str]

@dataclass
class Constraint:
    type: str  # 'polynomial' or 'linear'
    terms: List[Term]
    upper_bound: Optional[int] = None
    constant: Optional[int] = None
    is_equality: bool = False

@dataclass
class ConstraintSpace:
    func_name: str
    pass_name: str
    variables: List[Variable]
    metadata: List[Any]
    constraints: List[Constraint]

def load_constraint_spaces(json_paths: List[str]) -> List[ConstraintSpace]:
    all_spaces = []
    for path in json_paths:
        with open(path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            for item in data:
                variables = [Variable(**v) for v in item.get('variables', [])]
                
                metadata = []
                for m in item.get('metadata', []):
                    if m['type'] == 'range':
                        metadata.append(RangeMetadata(m['variable'], m['lower_bound'], m['upper_bound']))
                    elif m['type'] == 'align':
                        metadata.append(AlignMetadata(m['variable'], m['alignment']))
                
                constraints = []
                for c in item.get('constraints', []):
                    terms = [Term(t['coefficient'], t['variables']) for t in c.get('terms', [])]
                    constraints.append(Constraint(
                        type=c['type'],
                        terms=terms,
                        upper_bound=c.get('upper_bound'),
                        constant=c.get('constant'),
                        is_equality=c.get('is_equality', False)
                    ))
                
                all_spaces.append(ConstraintSpace(
                    func_name=item['func_name'],
                    pass_name=item['pass_name'],
                    variables=variables,
                    metadata=metadata,
                    constraints=constraints
                ))
    return all_spaces

def find_by_func_name(spaces: List[ConstraintSpace], func_name: str) -> Optional[ConstraintSpace]:
    for space in spaces:
        if space.func_name == func_name:
            return space
    return None

def build_var_index_map(space: ConstraintSpace) -> Dict[str, int]:
    return {v.name: i for i, v in enumerate(space.variables)}

def compile_constraint_function(constraint: Constraint, var_names: List[str]) -> Callable:
    """
    Returns a boolean-returning function for mask evaluation.
    """
    term_exprs = [f"{t.coefficient}* {'*'.join(t.variables)}" if t.variables else str(t.coefficient) for t in constraint.terms]
    lhs = " + ".join(term_exprs)
    
    if constraint.type == 'linear' and constraint.constant is not None:
        expr = f"(({lhs}) + {constraint.constant}) <= 0"
    else:
        rhs = constraint.upper_bound if constraint.upper_bound is not None else 0
        expr = f"({lhs}) <= {rhs}"
    
    params = ", ".join(var_names)
    return eval(f"lambda {params}: {expr}", {"np": np})

def compile_expression_function(constraint: Constraint, var_names: List[str]) -> Callable:
    """
    Returns a function that evaluates (LHS, RHS) for continuous rendering.
    """
    term_exprs = [f"{t.coefficient}* {'*'.join(t.variables)}" if t.variables else str(t.coefficient) for t in constraint.terms]
    lhs = " + ".join(term_exprs)
    
    if constraint.type == 'linear' and constraint.constant is not None:
        # For linear f(x) + C <= 0, we can use score = (f(x) + C) / |-C| or similar
        # But simpler: score = f(x) + C. Boundary is 0.
        expr = f"({lhs}) + {constraint.constant}"
        rhs = 0.0
    else:
        expr = lhs
        rhs = constraint.upper_bound if constraint.upper_bound is not None else 1.0
    
    params = ", ".join(var_names)
    return eval(f"lambda {params}: ({expr}, {rhs})", {"np": np})

def get_ranges(space: ConstraintSpace) -> Dict[str, tuple]:
    ranges = {}
    for m in space.metadata:
        if isinstance(m, RangeMetadata):
            ranges[m.variable] = (m.lower_bound, m.upper_bound)
    return ranges

def get_alignments(space: ConstraintSpace) -> Dict[str, int]:
    aligns = {}
    for m in space.metadata:
        if isinstance(m, AlignMetadata):
            aligns[m.variable] = m.alignment
    return aligns
