import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
import pandas as pd

from .containers import Experiment, History, Population


@dataclass
class EvalMetricsJob:
    run_idx: int
    metrics: Dict[str, Any]
    run: History

    def __eval__(self):
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


def eval_metrics_experiment(exp: Experiment, metrics: Dict[str, Any]):
    # Handle case of a function being passed for `metrics`
    if callable(metrics):
        metrics = {'metric': metrics}
    
    # Construct a series of "jobs" over each evaluation of the optimizer contained in the file
    jobs = []
    for idx, run in enumerate(exp.runs):
        jobs.append(EvalMetricsJob(hist=run, run_idx=idx, metrics=metrics.copy()))
    
    results = map(lambda x: x(), jobs)
    return  pd.DataFrame(sum(results, []))


def get_inverse_generational_distance(O, ref):
    """
    Calculates convergence metric between a pareto front O and a reference set points ref.  This is the mean of minimum
    distances between points on the front and the reference as described in the NSGA-II paper.

    :param O: (M,N) numpy array where N is the number of individuals and M is the number of objectives
    :param ref: (M,L) numpy array of reference points on the Pareto front
    :return: T, the convergence metric
    """
    # Compute pairwise distance between every point in the front and reference
    d = np.sqrt(np.sum((ref[:, :, None] - O[:, None, :]) ** 2, axis=0))

    # Find the minimum distance for each point and average it
    d_min = np.min(d, axis=1)
    return np.mean(d_min)


def get_generational_distance(O, ref):
    """
    Calculates convergence metric between a pareto front O and a reference set points ref.  This is the mean of minimum
    distances between points on the front and the reference as described in the NSGA-II paper.

    :param O: (M,N) numpy array where N is the number of individuals and M is the number of objectives
    :param ref: (M,L) numpy array of reference points on the Pareto front
    :return: T, the convergence metric
    """
    # Compute pairwise distance between every point in the front and reference
    d = np.sqrt(np.sum((ref[:, :, None] - O[:, None, :]) ** 2, axis=0))

    # Find the minimum distance for each point and average it
    d_min = np.min(d, axis=0)
    return np.mean(d_min)