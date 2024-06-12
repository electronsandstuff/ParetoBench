import pytest
import numpy as np
import random

import paretobench as pb
from paretobench.simple_serialize import split_unquoted, dumps, loads


@pytest.mark.parametrize("problem_name", pb.get_problem_names())
def test_evaluate(problem_name, n_eval = 64):
    """
    Try creating each registered problem w/ default parameters and then call it.
    """
    p = pb.create_problem(problem_name)
    bnd = p.var_bounds
    x = np.random.random((bnd.shape[1], n_eval))*(bnd[1, :] - bnd[0, :])[:, None] + bnd[0, :][:, None]
    f, g = p(x)

    assert isinstance(f, np.ndarray)
    assert isinstance(g, np.ndarray)
    assert not np.isnan(f).any()
    assert not np.isnan(g).any()


@pytest.mark.parametrize("problem_name", pb.get_problem_names())
def test_get_params(problem_name):
    """
    Checks all parameters like number of decision variables, objectives, and constraints are set w/ right type and that when you 
    call the problem, those values are consistant with what comes out.
    """
    p = pb.create_problem(problem_name)
    
    # Check the properties themselves for the right type
    assert isinstance(p.n_vars, int)
    assert isinstance(p.n, int)
    assert isinstance(p.n_objs, int)
    assert isinstance(p.n_constraints, int)
    assert isinstance(p.var_bounds, np.ndarray)
    
    # Check that if you actually call the values, you get the right sized objects (everything is consistent)
    bnd = p.var_bounds
    x = np.random.random((bnd.shape[1], 1))*(bnd[1, :] - bnd[0, :])[:, None] + bnd[0, :][:, None]
    f, g = p(x)
    assert p.n_vars == x.shape[0]
    assert p.n_objs == f.shape[0]
    assert p.n_constraints == g.shape[0]
    assert p.n_vars == p.n


@pytest.mark.parametrize("problem_name", pb.get_problem_names())
def test_pareto_front(problem_name, npoints=1000):
    """
    Try getting a pareto front for each of the registered test problems.
    """
    p = pb.create_problem(problem_name)

    if not isinstance(p, (pb.ProblemWithPF, pb.ProblemWithFixedPF)):
        return
    if isinstance(p, pb.ProblemWithPF):  # If we can choose number of points, check at lesat that many are returned
        f = p.get_pareto_front(npoints)
        assert f.shape[1] >= npoints
    else:  # If it's the fixed PF case
        f = p.get_pareto_front()

    # Make sure the right size array is returned and it doesn't give bad values
    assert p.n_objs == f.shape[0]
    assert not np.isnan(f).any()


@pytest.mark.parametrize("problem_name", pb.get_problem_names())
def test_refs(problem_name):
    p = pb.create_problem(problem_name)
    assert isinstance(p.get_reference(), str)
