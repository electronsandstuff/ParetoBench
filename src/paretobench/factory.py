from .exceptions import DeserializationError
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


def from_line_fmt(s: str):
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
    
    # No parameters were passed
    if (serialization_beg == -1) and (serialization_end == -1):
        name = s.strip()
        kwargs = {}
    elif (serialization_beg != -1) and (serialization_end != -1):
        name = s[:serialization_beg].strip()
        kwargs = loads(s[serialization_beg+1:serialization_end])
    else:
        raise DeserializationError('could not interpret line "s"')
    
    # Create the problem and return
    return registered_probs[name](**kwargs)
