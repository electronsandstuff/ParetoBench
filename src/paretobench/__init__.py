from .problem import Problem, ProblemWithFixedPF, ProblemWithPF
from .factory import register_problem, get_problem_names, create_problem

from .dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7, DTLZ8, DTLZ9
for p in [DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7, DTLZ8, DTLZ9]:
    register_problem(p)

from .zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
for p in [ZDT1, ZDT2, ZDT3, ZDT4, ZDT6]:
    register_problem(p)

# from .wfg import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9
# for p in [WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9]:
#     register_problem(p)
    
# from .misc import SCH, FON, POL, KUR, CONSTR, SRN, TNK, WATER
# for p in [SCH, FON, POL, KUR, CONSTR, SRN, TNK, WATER]:
#     register_problem(p)

from .ctp import CTP1, CTP2, CTP3, CTP4, CTP5, CTP6, CTP7
for p in [CTP1, CTP2, CTP3, CTP4, CTP5, CTP6, CTP7]:
    register_problem(p)
    
from .cf import CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9, CF10
for p in [CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9, CF10]:
    register_problem(p)
