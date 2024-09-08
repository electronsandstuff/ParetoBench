import numpy as np
import os
import paretobench as pb
import pytest
import tempfile

from .utils import generate_moga_runs, example_metric


class ProblemExample(pb.Problem, pb.ProblemWithFixedPF):
    """
    Problem with specific Pareto front for `test_inverse_generational_distance`
    """
    def get_pareto_front(self):
        return np.array([
            [0, 1],
            [0.5, 0.5],
            [1, 0],
        ])


def test_inverse_generational_distance():
    """
    Make sure IGD calculation works on analytical cases
    """
    # Create the metric
    igd = pb.InverseGenerationalDistance()
    
    # Get the IGD of a test population and compare with analytical value
    test_pop = pb.Population(f=np.array([[0.0, 0.0]]))
    val = igd(test_pop, ProblemExample())
    actual1 = np.mean([1, np.sqrt(0.5**2 + 0.5**2), 1])
    assert val == actual1

    # Another point
    test_pop = pb.Population(f=np.array([[0.0, 1.0]]))
    val = igd(test_pop, ProblemExample())
    actual2 = np.mean([0, np.sqrt(0.5**2 + 0.5**2), np.sqrt(1**2 + 1**2)])
    assert val == actual2
    
    # Do multiple points
    test_pop = pb.Population(f=np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    val = igd(test_pop, ProblemExample())
    assert val == np.mean([0, 0, np.sqrt(0.5**2 + 0.5**2)])


@pytest.mark.parametrize('input_type', ['MOGARun', 'file', 'single'])
def test_eval_metrics_experiments(input_type):
    """
    This test generates some MOGARun objects and uses eval_metrics_experiments to evaluate them with a test metric. It confirms that
    the right fields are generated and that the results line up with the individual eval_metrics method in MOGARun.

    Parameters
    ----------
    input_type : str
        What type of input to use (MOGARun or file)
    """
    # Create some test objects
    if input_type == 'single':
        runs = generate_moga_runs(1)
    else:
        runs = generate_moga_runs()
    
    
    with tempfile.TemporaryDirectory() as dir:
        # Handle creating the input (files or moga run objects)
        if input_type == 'file':
            fun_ins = []
            for idx, run in enumerate(runs):
                fname = os.path.join(dir, f'run-{idx}.h5')
                run.save(fname)
                fun_ins.append(fname)
        elif input_type == 'MOGARun':
            fun_ins = runs
        elif input_type == 'single':
            fun_ins = runs[0]
        else:
            raise ValueError(f'Unrecognized input_type: "{ input_type }"')
        
        # Try running a metric calc
        df = pb.eval_metrics_experiments(fun_ins, metrics={'test': example_metric})

    # Make sure we get the expected number of rows
    assert len(df) == sum(sum(len(evl.reports) for evl in run.runs) for run in runs)

    # Check that the filename field works correctly
    if input_type == 'file':
        actual_fnames = df.apply(lambda x: fun_ins[x['exp_idx']], axis=1)
        assert (df['fname'] == actual_fnames).all()
    elif input_type == 'MOGARun':
        assert (df['fname'] == '').all()
