from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Union


@dataclass
class Population:
    x: np.ndarray
    f: np.ndarray
    g: np.ndarray
    feval: int
    

@dataclass
class History:
    reports: List[Population]
    problem: str
    metadata: Dict[str, Union[str, int, float, bool]]


@dataclass
class Experiment:
    runs: List[History]
    identifier: str
