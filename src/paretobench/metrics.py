from dataclasses import dataclass
from operator import methodcaller
from typing import Dict, Any, List, Union
import concurrent.futures
import numpy as np
import os
import pandas as pd

from .containers import Experiment, History, Population
from .problem import Problem, ProblemWithPF, ProblemWithFixedPF


@dataclass
class EvalMetricsJob:
    """
    Represents a "job" in the batch calculation of metrics from experiment objects. One job will evaluate all populations within
    a single `History` object and return a list of rows (as dictionaries) with the metric values, one for each population.
    """
    exp_idx: int
    exp_name: str
    fname: str
    metrics: Dict[str, Any]
    run_idx: int
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
                'exp_name': self.exp_name,
                'exp_idx': self.exp_idx,
                'fname': self.fname,
            }
            
            # Evaluate the metrics
            row.update({name: f(pop, self.run.problem) for name, f in self.metrics.items()})
            
            # Add to the list of rows
            rows.append(row)
        return rows


def eval_metrics_experiments(
        experiments: Union[Union[Experiment, str], List[Union[Experiment, str]]],
        metrics: Dict[str, Any],
        n_procs=1
    ):
    """
    Evaluates a set of metrics on all of the nondominated solutions in the populations contained in `experiments`.  Calculation
    includes some basic parallelism using the `multiprocessing` library and can be enabled with 
    the `n_procs` parameter.
    
    The metrics should have the following signature.
    `metric(pop: Population, problem: str)`
    The problem will be a definition of a problem in "single line" format and the population will contain only nondominated
    solutions.
    
    The resulting dataframe will have the following columns.
    * `problem`: The problem (in single line format)
    * `exp_idx`: The index of this experiment in the list
    * `exp_name`: The attribute `Experiment.name` for this evaluation
    * `run_idx`: The index of the run within the experiment
    * `pop_idx`: An index for the Population object within the run
    * `fevals`: The number of function evaluations used to achieve this result
    * `fname`: The filename the experiment was loaded from. Empty string if not loaded from file.
    * One column with the name of each metric and the values stored in it

    Parameters
    ----------
    experiments : Union[Union[Experiment, str], List[Union[Experiment, str]]]
        The experiment or list of experiments. May either be loaded experiment objects or filenames.
    metrics : Dict[str, Any]
        Map from metric names to functions which will evaluate them. Passing a function here will give it the name `metric`.
    n_procs : int, optional
        Number of processes to use in `multiprocessing.Pool`. Set to one to disable, by default 1

    Returns
    -------
    pd.DataFrame
        Pandas dataframe with the results of evaluating the metrics.
    """
    # Handle case of a function being passed for `metrics`
    if callable(metrics):
        metrics = {'metric': metrics}

    # Handle single valued experiments
    if not isinstance(experiments, list):
        experiments = [experiments]

    # Load each of the experiments and analyze
    dfs = []
    for exp_idx, exp_in in enumerate(experiments):
        # Load the experiment if it's a file
        if isinstance(exp_in, (str, os.PathLike)):
            exp = Experiment.load(exp_in)
            fname = exp_in
        elif isinstance(exp_in, Experiment):
            exp = exp_in
            fname = ''
        else:
            ValueError(f'Incompatible experiment type: idx={exp_idx}, type={type(exp_in)}')
        
        # Construct a series of "jobs" over each evaluation of the optimizer contained in the file
        jobs = []
        for run_idx, run in enumerate(exp.runs):
            jobs.append(EvalMetricsJob(
                run=run, 
                run_idx=run_idx,
                metrics=metrics.copy(), 
                exp_name=exp.name, 
                exp_idx=exp_idx, 
                fname=fname
            ))
        
        # Run each of the jobs (potentially in parallel)
        # Note: the map happens inside of the loop over experiments because the files must be opened before constructing the
        # jobs. Performing the batching in a way that includes all experiments would incur the (potentially large) memory hit
        # of having all experiment objects loaded in memory at the same time.
        if n_procs == 1:
            results = map(methodcaller('__call__'), jobs)
        else:
            with concurrent.futures.ProcessPoolExecutor(n_procs) as ex:
                results = ex.map(methodcaller('__call__'), jobs)
            
        # Construct the dataframe for this experiment (results is list of list of rows)
        dfs.append(pd.DataFrame(sum(results, [])))
        
    # Combine and return
    return pd.concat(dfs)


class InverseGenerationalDistance:
    """
    Calculates the inverse generational distance for the population to the Pareto front in the named problem.
    """
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
