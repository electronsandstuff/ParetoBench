import numpy as np
import pytest

from paretobench.ext.pymoo import PymooProblemWrapper


@pytest.mark.parametrize("prob_name", ["CTP1", "ZDT1", "WFG1", "CF1"])
def test_pymoo_problem_wrapper(prob_name):
    wrapper = PymooProblemWrapper.from_line_fmt(prob_name)
    prob = wrapper.prob

    # Check metadata matches
    assert wrapper.n_var == prob.n_vars
    assert wrapper.n_obj == prob.n_objs
    assert wrapper.n_ieq_constr == prob.n_constraints
    np.testing.assert_allclose(wrapper.xl, prob.var_lower_bounds)
    np.testing.assert_allclose(wrapper.xu, prob.var_upper_bounds)

    rng = np.random.default_rng(42)
    lbs, ubs = prob.var_lower_bounds, prob.var_upper_bounds

    # Test with a batch of inputs
    X = rng.uniform(lbs, ubs, size=(8, prob.n_vars))
    pop = prob(X)

    out = wrapper.evaluate(X, return_as_dictionary=True)

    np.testing.assert_allclose(out["F"], pop.f)

    if prob.n_constraints > 0:
        np.testing.assert_allclose(out["G"], pop.g)
