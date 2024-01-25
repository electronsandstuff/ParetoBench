import re


registered_probs = {}


def register_problem(cls):
    registered_probs[cls.__name__] = cls


def get_problem_names():
    return list(registered_probs.keys())


def create_problem(s: str, **kwargs):
    """
    Generates problem object from string name of the problem
    :param s:
    :param kwargs:
    :return:
    """
    return registered_probs[s](**kwargs)
