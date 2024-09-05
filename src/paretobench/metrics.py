import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
import pandas as pd
import concurrent.futures
from operator import methodcaller
from typing import List, Union

from .containers import Experiment, History, Population
from .problem import Problem, ProblemWithPF, ProblemWithFixedPF


@dataclass
class EvalMetricsJob:
    run_idx: int
    metrics: Dict[str, Any]
    run: History

    def __call__(self):
        # Run through the evaluations, keeping only the nondominated solutions up to this point
        pfs = self.run.to_nondominated()
        
        # For each population, evaluate the metrics and return a new row in the table
        rows = []
        for idx, pop in enumerate(pfs.reports):
            # Copy information to the row from the job
            row = {
                'problem': self.run.problem,
                'fevals': pop.feval,
                'run_idx': self.run_idx,
                'pop_idx': idx,
            }
            
            # Evaluate the metrics
            row.update({name: f(pop, self.run.problem) for name, f in self.metrics.items()})
            
            # Add to the list of rows
            rows.append(row)
        return rows


def eval_metrics_experiments(fnames: List[str], metrics: Dict[str, Any], n_procs=1):
    # Handle case of a function being passed for `metrics`
    if callable(metrics):
        metrics = {'metric': metrics}

    # Load each of the experiments and analyze
    dfs = []
    for exp_idx, fname in enumerate(fnames):
        # Load the experiment
        exp = Experiment.load(fname)
        
        # Construct a series of "jobs" over each evaluation of the optimizer contained in the file
        jobs = []
        for run_idx, run in enumerate(exp.runs):
            jobs.append(EvalMetricsJob(run=run, run_idx=run_idx, metrics=metrics.copy()))
        
        # Run each of the jobs (potentially in parallel)
        if n_procs == 1:
            results = map(methodcaller('__call__'), jobs)
        else:
            with concurrent.futures.ProcessPoolExecutor(n_procs) as ex:
                results = ex.map(methodcaller('__call__'), jobs)
            
        # Construct the dataframe for this experiment
        df = pd.DataFrame(sum(results, []))
        df['exp_name'] = exp.name
        df['exp_idx'] = exp_idx
        df['fname'] = fname
        
        # Add to the list of dataframes
        dfs.append(df)
        
    # Combine and return
    return pd.concat(dfs)


class InverseGenerationalDistance:
    def __init__(self, n_pf=1000):
        """
        Parameters
        ----------
        n_pf : int, optional
            Number of points to calculate on the Pareto front, by default 1000
        """
        self.n_pf = n_pf

    def __call__(self, pop: Population, problem: Union[Problem, str]):
        # Handle the problem
        if isinstance(problem, str):
            prob = Problem.from_line_fmt(problem)
        elif isinstance(problem, Problem):
            prob = problem
        else:
            raise ValueError('Function must be passed problem object or description in single line format.')
        
        # Get the Pareto front
        if isinstance(prob, ProblemWithPF):
            pf = prob.get_pareto_front(self.n_pf)
        elif isinstance(prob, ProblemWithFixedPF):
            pf = prob.get_pareto_front()
        else:
            raise ValueError(f'Could not load Pareto front from object of type "{type(prob)}"')
    
        # Calculate the IGD metric
        # Compute pairwise distance between every point in the front and reference
        d = np.sqrt(np.sum((pop.f[None, :, :] - pf[:, None, :]) ** 2, axis=2))

        # Find the minimum distance for each point and average it
        d_min = np.min(d, axis=1)
        return np.mean(d_min)
