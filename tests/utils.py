import numpy as np

from paretobench import Experiment


def example_metric(pop, prob):
    """
    An example metric for testing functions.

    Parameters
    ----------
    pop : Population
        The population object
    prob : str
        Problem name

    Returns
    -------
    float
        An example metric value
    """
    return np.mean(pop.f) + sum(ord(x) for x in prob)


def generate_moga_runs(n_runs=4, names=None):
    """
    Helper function to generate multiple randomized Experiment objects for testing the metric analysis functions. Forces them
    to all have the same problems in them.

    Parameters
    ----------
    n_runs : int, optional
        The number of runs to make, by default 4

    Returns
    -------
    List[Experiment]
        The runs
    """
    if names is None:
        names = ['' for _ in range(n_runs)]
    
    # Make each run
    experiments = []
    for name in names:
        # Create an experiment
        experiment = Experiment.from_random(16, 30, 2, 20, 0, 50)
        experiment.name = name
        
        # Force the problem names to be the same
        for idx, a in enumerate(experiment.runs):
            a.problem = f'ZDT1 (n={idx+1})'
        experiments.append(experiment)
    
    # Return it
    return experiments

