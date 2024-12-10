import numpy as np
import os
import paretobench as pb
import pytest
import tempfile

from .utils import generate_moga_experiments, example_metric


class ProblemExample(pb.Problem, pb.ProblemWithFixedPF):
    """
    Problem with specific Pareto front for `test_inverse_generational_distance`
    """

    def get_pareto_front(self):
        return np.array(
            [
                [0, 1],
                [0.5, 0.5],
                [1, 0],
            ]
        )


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


@pytest.mark.parametrize("input_type", ["Experiment", "file", "single"])
def test_eval_metrics_experiments(input_type):
    """
    This test generates some Experiment objects and uses eval_metrics_experiments to evaluate them with a test metric. It
    confirms that the right fields are generated.

    Parameters
    ----------
    input_type : str
        What type of input to use (Experiments, files, or a single object)
    """
    # Create some test objects
    if input_type == "single":
        runs = generate_moga_experiments(names=["test"])
    else:
        runs = generate_moga_experiments()

    with tempfile.TemporaryDirectory() as dir:
        # Handle creating the input (files or moga run objects)
        if input_type == "file":
            fun_ins = []
            for idx, run in enumerate(runs):
                fname = os.path.join(dir, f"run-{idx}.h5")
                run.save(fname)
                fun_ins.append(fname)
        elif input_type == "Experiment":
            fun_ins = runs
        elif input_type == "single":
            fun_ins = runs[0]
        else:
            raise ValueError(f'Unrecognized input_type: "{ input_type }"')

        # Try running a metric calc
        df = pb.eval_metrics_experiments(fun_ins, metrics=("test", example_metric))

    # Make sure we get the expected number of rows
    assert len(df) == sum(sum(len(evl.reports) for evl in run.runs) for run in runs)

    # Check that the filename field works correctly
    if input_type == "file":
        actual_fnames = df.apply(lambda x: fun_ins[x["exp_idx"]], axis=1)
        assert (df["fname"] == actual_fnames).all()
    elif input_type == "Experiment":
        assert (df["fname"] == "").all()


def test_eval_metrics_experiments_invalid_metric_type():
    # Test unrecognized `metrics` type
    with pytest.raises(TypeError, match="Unrecognized type for `metrics`"):
        pb.eval_metrics_experiments(experiments=[], metrics={"test": 1234})
    with pytest.raises(TypeError, match="Unrecognized type for `metrics`"):
        pb.eval_metrics_experiments(experiments=[], metrics=1234)


def test_eval_metrics_experiments_invalid_tuple_type():
    # Test if first element of the tuple in metrics is not a string
    with pytest.raises(TypeError, match="Unrecognized type for `metrics"):
        pb.eval_metrics_experiments(experiments=[], metrics=[(123, lambda x: x)])


def test_eval_metrics_experiments_invalid_callable_in_tuple():
    # Test if the second element of the tuple is not callable
    with pytest.raises(TypeError, match="`metrics\\[0\\]\\[1\\]` is not callable"):
        pb.eval_metrics_experiments(experiments=[], metrics=[("valid_name", 123)])


def test_eval_metrics_experiments_invalid_experiment_type():
    # Test unrecognized `experiments` type (e.g., int)
    with pytest.raises(ValueError, match="Incompatible experiment type: idx=0"):
        pb.eval_metrics_experiments(experiments=123, metrics=lambda pop, problem: None)

    # Test list with an invalid experiment type inside
    with pytest.raises(ValueError, match="Incompatible experiment type: idx=1"):
        pb.eval_metrics_experiments(
            experiments=[pb.Experiment(runs=[], name=""), 123],
            metrics=lambda pop, problem: None,
        )


def test_eval_metrics_experiments_duplicate_metric_name():
    # Make a mock metric
    class DummyMetric:
        def __init__(self, name):
            self.name = name

    # Test for duplicate metric name error
    with pytest.raises(ValueError, match=r'Duplicate name for `metrics\[1\]`: "metric1"'):
        metric1 = ("metric1", lambda pop, problem: None)
        metric2 = ("metric1", lambda pop, problem: None)
        pb.eval_metrics_experiments(experiments=[], metrics=[metric1, metric2])


def test_eval_metrics_experiments_unrecognized_metric_type_in_list():
    # Test for unrecognized type in the list of metrics
    with pytest.raises(TypeError, match=r"Unrecognized type for `metrics\[0\]`"):
        pb.eval_metrics_experiments(experiments=[], metrics=[123])

    with pytest.raises(TypeError, match=r"Unrecognized type for `metrics\[1\]`"):
        pb.eval_metrics_experiments(experiments=[], metrics=[("valid_metric", lambda pop, problem: None), 123])
