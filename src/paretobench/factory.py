import re

from .simple_serialize import loads


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


def from_line(s: str):
    """Create a problem object from the "single line" format.

    Parameters
    ----------
    s : str
        The single line describing the problem object

    Returns
    -------
    Problem
        The instantiated problem
    """
    # Find the section of the string corresponding to serialized parameters
    serialization_beg = s.find('(')
    serialization_end = s.find(')')
    
    # Get the object name and deserialized args
    name = s[:serialization_beg].strip()
    kwargs = loads(s[serialization_beg+1:serialization_end])
    
    # Create the problem and return
    return registered_probs[name](**kwargs)
