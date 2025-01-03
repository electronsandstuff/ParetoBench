from .problem import Problem, ProblemWithFixedPF, ProblemWithPF
from .factory import register_problem, get_problem_names, create_problem
from .containers import Population, History, Experiment
from .metrics import InverseGenerationalDistance, eval_metrics_experiments
from .analyze_metrics import (
    aggregate_metrics_feval_budget,
    construct_metric_comparison_table,
    comparison_table_to_latex,
)
from .exceptions import (
    SerializationError,
    DeserializationError,
    UnknownProblemError,
    InputError,
    EmptyPopulationError,
    NoObjectivesError,
    NoConstraintsError,
    NoDecisionVarsError,
)

# Test problems
from .cf import CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9, CF10
from .ctp import CTP1, CTP2, CTP3, CTP4, CTP5, CTP6, CTP7
from .dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7, DTLZ8, DTLZ9
from .misc import SCH, FON, POL, KUR, CONSTR, SRN, TNK, WATER
from .wfg import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9
from .zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6

_PROBLEM_CLASSES = [
    # DTLZ suite
    *[DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7, DTLZ8, DTLZ9],
    # ZDT suite
    *[ZDT1, ZDT2, ZDT3, ZDT4, ZDT6],
    # WFG suite
    *[WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9],
    # Classical problems
    *[SCH, FON, POL, KUR, CONSTR, SRN, TNK, WATER],
    # CTP suite
    *[CTP1, CTP2, CTP3, CTP4, CTP5, CTP6, CTP7],
    # CF suite
    *[CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9, CF10],
]

for problem_class in _PROBLEM_CLASSES:
    register_problem(problem_class)

__all__ = [
    "Problem",
    "ProblemWithFixedPF",
    "ProblemWithPF",
    "register_problem",
    "get_problem_names",
    "create_problem",
    "SerializationError",
    "DeserializationError",
    "UnknownProblemError",
    "InputError",
    "Population",
    "History",
    "Experiment",
    "InverseGenerationalDistance",
    "eval_metrics_experiments",
    "aggregate_metrics_feval_budget",
    "construct_metric_comparison_table",
    "comparison_table_to_latex",
    "EmptyPopulationError",
    "NoObjectivesError",
    "NoConstraintsError",
    "NoDecisionVarsError",
] + [cls.__name__ for cls in _PROBLEM_CLASSES]
