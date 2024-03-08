import numpy as np
import random

import paretobench as pb
from paretobench.simple_serialize import split_unquoted, dumps, loads


def test_evaluate(self, n_eval = 64):
    """
    Try creating each registered problem w/ default parameters and then call it.
    """
    for name in pb.get_problem_names():
        with self.subTest(name=name):
            p = pb.create_problem(name)
            bnd = p.decision_var_bounds
            x = np.random.random((bnd.shape[1], n_eval))*(bnd[1, :] - bnd[0, :])[:, None] + bnd[0, :][:, None]
            f, g = p(x)

            assert isinstance(f, np.ndarray)
            assert isinstance(g, np.ndarray)
            assert np.isfinite(f).all()
            assert np.isfinite(g).all()

def test_get_params(self):
    """
    Checks all parameters like numbe rof decision variables, objectives, and constraints are set w/ right type and that when you 
    call the problem, those values are consistant with what comes out.
    """
    for name in pb.get_problem_names():
        with self.subTest(name=name):
            p = pb.create_problem(name)
            
            # Check the properties themselves for the right type
            assert isinstance(p.n_decision_vars, int)
            assert isinstance(p.n_objectives, int)
            assert isinstance(p.n_constraints, int)
            assert isinstance(p.decision_var_bounds, np.ndarray)
            
            # Check that if you actually call the values, you get the right sized objects (everything is consistent)
            bnd = p.decision_var_bounds
            x = np.random.random((bnd.shape[1], 1))*(bnd[1, :] - bnd[0, :])[:, None] + bnd[0, :][:, None]
            f, g = p(x)
            assert p.n_decision_vars == x.shape[0]
            assert p.n_objectives == f.shape[0]
            assert p.n_constraints == g.shape[0]

def test_pareto_front(self, npoints=1000):
    """
    Try getting a pareto front for each of the registered test problems.
    """
    for name in pb.get_problem_names():
        with self.subTest(name=name):
            p = pb.create_problem(name)
        
            if not isinstance(p, (pb.ProblemWithPF, pb.ProblemWithFixedPF)):
                continue
            if isinstance(p, pb.ProblemWithPF):  # If we can choose number of points, check at lesat that many are returned
                f = p.get_pareto_front(npoints)
                assert f.shape[1] >= npoints
            else:  # If it's the fixed PF case
                f = p.get_pareto_front()

            # Make sure the right size array is returned and it doesn't give bad values
            assert p.n_objectives == f.shape[0]
            assert np.isfinite(f).all()

def test_refs(self):
    for name in pb.get_problem_names():
        with self.subTest(name=name):
            p = pb.create_problem(name)
            assert isinstance(p.get_reference(), str)
