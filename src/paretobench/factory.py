registered_probs = {}


def register_problem(cls):
    registered_probs[cls.__name__] = cls


def get_problem_names():
    return list(registered_probs.keys())


def create_problem(name: str, **kwargs):
    """Generates problem object from string name of the problem. Keyword arguments get passed to the object being created.

    Parameters
    ----------
    name : str
        The registered name of the problem (same as class name)

    Returns
    -------
    Problem
        The instantiated problem object
    """
    return registered_probs[name](**kwargs)
