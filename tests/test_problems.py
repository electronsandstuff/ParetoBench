import pytest
import numpy as np

import paretobench as pb


@pytest.mark.parametrize("problem_name", pb.get_problem_names())
def test_evaluate(problem_name, n_eval=64):
    """
    Try creating each registered problem w/ default parameters and then call it.
    """
    # Create the problem object (default parameters)
    p = pb.create_problem(problem_name)

    # Create a set of points to evaluate the problem on
    bnd = p.var_bounds
    x = np.random.random((n_eval, bnd.shape[1])) * (bnd[1, :] - bnd[0, :])[None, :] + bnd[0, :][None, :]

    # Evaluate on batched data
    res = p(x)
    assert isinstance(res, pb.Population)
    assert isinstance(res.f, np.ndarray)
    assert res.f.shape[0] == n_eval
    assert res.f.shape[1] == p.n_objs
    assert isinstance(res.g, np.ndarray)
    assert res.g.shape[0] == n_eval
    assert res.g.shape[1] == p.n_constraints
    np.testing.assert_array_equal(x, res.x)
    assert not np.isnan(res.f).any()
    assert not np.isnan(res.g).any()

    # Evaluate on a single value
    res = p(x[0])
    assert isinstance(res, pb.Population)
    assert isinstance(res.f, np.ndarray)
    assert res.f.shape[0] == 1
    assert res.f.shape[1] == p.n_objs
    assert len(res.f.shape) == 2
    assert isinstance(res.g, np.ndarray)
    assert res.g.shape[0] == 1
    assert res.g.shape[1] == p.n_constraints
    assert len(res.g.shape) == 2
    assert not np.isnan(res.f).any()
    assert not np.isnan(res.g).any()

    # Check that an exception is triggered on invalid input
    with pytest.raises(pb.InputError):
        p(x[:, 1:])
    with pytest.raises(pb.InputError):
        p(x[0, 1:])
    with pytest.raises(pb.InputError):
        p(x + (p.var_upper_bounds - p.var_lower_bounds + 1))
    with pytest.raises(pb.InputError):
        p(x[0] + (p.var_upper_bounds - p.var_lower_bounds + 1))


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
    assert isinstance(p.m, int)
    assert isinstance(p.n_constraints, int)
    assert isinstance(p.var_bounds, np.ndarray)
    assert isinstance(p.reference, str)

    # Check that if you actually call the values, you get the right sized objects (everything is consistent)
    bnd = p.var_bounds
    x = np.random.random((1, bnd.shape[1])) * (bnd[1, :] - bnd[0, :])[None, :] + bnd[0, :][None, :]
    res = p(x)
    assert p.n_vars == x.shape[1]
    assert p.n_objs == res.f.shape[1]
    assert p.n_constraints == res.g.shape[1]
    assert p.n_vars == p.n
    assert p.n_objs == p.m


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
        assert f.shape[0] >= npoints
    else:  # If it's the fixed PF case
        f = p.get_pareto_front()

    # Make sure the right size array is returned and it doesn't give bad values
    assert p.n_objs == f.shape[1]
    assert not np.isnan(f).any()
