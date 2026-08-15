import hashlib
import numpy as np

from paretobench import Experiment


def array_digest(arr):
    """
    Hash a numeric array in a byte order independent way so digests are comparable across platforms.

    Parameters
    ----------
    arr : array-like
        The array to hash

    Returns
    -------
    str
        Hex digest of the array contents
    """
    return hashlib.sha256(np.ascontiguousarray(arr, dtype="<f8").tobytes()).hexdigest()


def to_builtin(value):
    """
    Convert a numpy scalar to the equivalent python type. Values loaded from HDF5 attributes are often numpy scalars which
    are not JSON serializable.

    Parameters
    ----------
    value : Any
        The value to convert

    Returns
    -------
    Any
        The value as a builtin python type
    """
    return value.item() if isinstance(value, np.generic) else value


def experiment_to_manifest(exp):
    """
    Describe the contents of an experiment as a JSON serializable dict. Used to record what a saved file should contain so
    that regression tests can confirm later versions of the library still read the same data back out of it.

    Parameters
    ----------
    exp : Experiment
        The experiment to describe

    Returns
    -------
    dict
        The description of the experiment
    """
    return {
        "file_version": exp.file_version,
        "name": exp.name,
        "author": exp.author,
        "software": exp.software,
        "software_version": exp.software_version,
        "comment": exp.comment,
        "creation_time": exp.creation_time.isoformat(),
        "runs": [
            {
                "problem": run.problem,
                "metadata": {k: to_builtin(v) for k, v in run.metadata.items()},
                "reports": [
                    {
                        "pop_size": len(report),
                        "fevals": int(report.fevals),
                        "n": report.n,
                        "m": report.m,
                        "n_constraints": report.n_constraints,
                        "names_x": list(report.names_x) if report.names_x is not None else None,
                        "names_f": list(report.names_f) if report.names_f is not None else None,
                        "names_g": list(report.names_g) if report.names_g is not None else None,
                        "obj_directions": report.obj_directions,
                        "constraint_directions": report.constraint_directions,
                        "constraint_targets": report.constraint_targets.tolist(),
                        "x_sha256": array_digest(report.x),
                        "f_sha256": array_digest(report.f),
                        "g_sha256": array_digest(report.g),
                    }
                    for report in run.reports
                ],
            }
            for run in exp.runs
        ],
    }


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
        names = ["", "", "", ""]

    # Make each run
    experiments = []
    for name in names:
        # Create an experiment
        experiment = Experiment.from_random(16, 30, 2, 20, 0, 50)
        experiment.name = name

        # Force the problem names to be the same
        for idx, a in enumerate(experiment.runs):
            a.problem = f"ZDT1 (n={idx+1})"
        experiments.append(experiment)

    # Return it
    return experiments
