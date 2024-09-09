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


def generate_moga_experiments(names=None):
    """
    Helper function to generate multiple randomized Experiment objects for testing the metric analysis functions. Forces them
    to all have the same problems in them.

    Parameters
    ----------
    names: str
        Names to use for the experiments

    Returns
    -------
    List[Experiment]
        The runs
    """
    # Handle default name
    if names is None:
        names = ['', '', '', '']
    
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
