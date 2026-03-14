import numpy as np
import os
import paretobench as pb
import pytest
import tempfile
import moocore

from .utils import generate_moga_experiments, example_metric


class ProblemExample(pb.Problem, pb.ProblemWithFixedPF):
    """
    Problem with specific Pareto front for `test_inverted_generational_distance`
    """

    def get_pareto_front(self):
        return np.array(
            [
                [0, 1],
                [0.5, 0.5],
                [1, 0],
            ]
        )


def test_inverted_generational_distance():
    """
    Make sure IGD calculation works on analytical cases
    """
    # Create the metric
    igd = pb.InvertedGenerationalDistance()

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
    test_pop = pb.Population(f=np.array([[0.0, 1.0], [1.0, 0.0]]))
    val = igd(test_pop, ProblemExample())
    assert val == np.mean([0, 0, np.sqrt(0.5**2 + 0.5**2)])


@pytest.mark.parametrize(
    "ref_point, f, expected",
    [
        # Single point: area of rectangle (2-1)*(2-1) = 1.0
        ([2.0, 2.0], [[1.0, 1.0]], 1.0),
        # Two non-dominated points: L-shaped area = 3.0
        ([3.0, 3.0], [[1.0, 2.0], [2.0, 1.0]], 3.0),
        # All points outside ref point: zero hypervolume
        ([1.0, 1.0], [[1.0, 2.0], [2.0, 1.0]], 0.0),
        # Single point: box volume (2-1)^3 = 1.0
        ([2.0, 2.0, 2.0], [[1.0, 1.0, 1.0]], 1.0),
        # All points outside ref point: zero hypervolume
        ([1.0, 1.0, 1.0], [[2.0, 2.0, 2.0]], 0.0),
        # Two non-dominated points (inclusion-exclusion):
        ([3.0, 3.0, 3.0], [[1.0, 2.0, 2.0], [2.0, 1.0, 2.0]], 3.0),
        ([2.0, 2.0, 2.0], [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], 4.0),
    ],
)
def test_hypervolume(ref_point, f, expected):
    """Hypervolume with analytically computed expected values."""
    hv = pb.Hypervolume(ref_point=np.array(ref_point))
    pop = pb.Population(f=np.array(f))
    assert hv(pop, None) == pytest.approx(expected)


@pytest.mark.parametrize(
    "n_dim, n_points, seed",
    [
        (2, 20, 42),
        (3, 20, 42),
        (4, 50, 42),
        (5, 50, 42),
    ],
)
def test_hypervolume_vs_moocore(n_dim, n_points, seed):
    """Compare hypervolume against moocore on random data."""
    rng = np.random.default_rng(seed)
    f = rng.random((n_points, n_dim))
    ref_point = np.ones(n_dim) * 1.1
    hv = pb.Hypervolume(ref_point=ref_point)
    pop = pb.Population(f=f).get_nondominated_set()
    assert hv(pop, None) == pytest.approx(moocore.hypervolume(f, ref=ref_point))


def test_hypervolume_obj_directions():
    # Mixed directions: minimize obj0, maximize obj1
    pop_mixed = pb.Population(f=np.array([[1.0, 3.0]]), obj_directions="-+")
    hv_mixed = pb.Hypervolume(ref_point=np.array([2.0, 2.0]))

    # Equivalent all-minimize: negate obj1 and ref1
    pop_min = pb.Population(f=np.array([[1.0, -3.0]]))
    hv_min = pb.Hypervolume(ref_point=np.array([2.0, -2.0]))

    assert hv_mixed(pop_mixed, None) == pytest.approx(hv_min(pop_min, None))


def test_eval_metrics_list():
    """Passing a list of Experiment objects produces the right number of rows with empty fname."""
    runs = generate_moga_experiments()
    df = pb.eval_metrics(runs, metrics=("test", example_metric))
    assert len(df) == sum(sum(len(evl.reports) for evl in run.runs) for run in runs)
    assert (df["fname"] == "").all()


def test_eval_metrics_files():
    """Passing a list of file paths produces the right number of rows with correct fnames."""
    runs = generate_moga_experiments()
    with tempfile.TemporaryDirectory() as dir:
        fnames = []
        for idx, run in enumerate(runs):
            fname = os.path.join(dir, f"run-{idx}.h5")
            run.save(fname)
            fnames.append(fname)
        df = pb.eval_metrics(fnames, metrics=("test", example_metric))
    assert len(df) == sum(sum(len(evl.reports) for evl in run.runs) for run in runs)
    actual_fnames = df.apply(lambda x: fnames[x["exp_idx"]], axis=1)
    assert (df["fname"] == actual_fnames).all()


def test_eval_metrics_single_experiment():
    """Passing a single Experiment object produces the right number of rows."""
    runs = generate_moga_experiments(names=["test"])
    df = pb.eval_metrics(runs[0], metrics=("test", example_metric))
    assert len(df) == sum(sum(len(evl.reports) for evl in run.runs) for run in runs)


def test_eval_metrics_single_file():
    """Passing a single file path produces the right number of rows with a correct fname."""
    runs = generate_moga_experiments(names=["test"])
    with tempfile.TemporaryDirectory() as dir:
        fname = os.path.join(dir, "run.h5")
        runs[0].save(fname)
        df = pb.eval_metrics(fname, metrics=("test", example_metric))
    assert len(df) == sum(sum(len(evl.reports) for evl in run.runs) for run in runs)
    assert (df["fname"] == fname).all()


def test_eval_metrics_population():
    """Passing a Population directly produces one row with default experiment/problem names."""
    pop = pb.Population.from_random(n_objectives=2, n_decision_vars=3, n_constraints=0, pop_size=10)
    df = pb.eval_metrics(pop, metrics=("test", example_metric))
    assert len(df) == 1
    assert (df["exp_name"] == "default_experiment").all()
    assert (df["problem"] == "default_problem").all()


def test_eval_metrics_history():
    """Passing a History directly produces one row per population with the history's problem name."""
    n_populations = 5
    history = pb.History.from_random(
        n_populations=n_populations, n_objectives=2, n_decision_vars=3, n_constraints=0, pop_size=10
    )
    df = pb.eval_metrics(history, metrics=("test", example_metric))
    assert len(df) == n_populations
    assert (df["exp_name"] == "default_experiment").all()
    assert (df["problem"] == history.problem).all()


def test_eval_metrics_empty_list():
    df = pb.eval_metrics([], metrics=("test", example_metric))
    assert len(df) == 0
    assert "problem" in df.columns
    assert "fevals" in df.columns
    assert "run_idx" in df.columns


def test_eval_metrics_invalid_runs_type():
    """Passing an unrecognized type for `runs` raises ValueError."""
    with pytest.raises(ValueError, match="Unrecognized type for `runs`"):
        pb.eval_metrics(runs=12345, metrics=example_metric)


def test_eval_metrics_invalid_runs_list_type():
    """Passing a list with a non-Experiment element raises ValueError."""
    with pytest.raises(ValueError, match="All runs must have type `Experiment`"):
        pb.eval_metrics(runs=[pb.Experiment(runs=[], name=""), 123], metrics=example_metric)


def test_eval_metrics_experiments_deprecation_warning():
    """Calling eval_metrics_experiments raises a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="eval_metrics_experiments is deprecated"):
        pb.eval_metrics_experiments(experiments=[], metrics=example_metric)


def test_eval_metrics_invalid_metric_type():
    # Test unrecognized `metrics` type
    with pytest.raises(TypeError, match="Unrecognized type for `metrics`"):
        pb.eval_metrics(runs=[], metrics={"test": 1234})
    with pytest.raises(TypeError, match="Unrecognized type for `metrics`"):
        pb.eval_metrics(runs=[], metrics=1234)


def test_eval_metrics_invalid_tuple_type():
    # Test if first element of the tuple in metrics is not a string
    with pytest.raises(TypeError, match="Unrecognized type for `metrics"):
        pb.eval_metrics(runs=[], metrics=[(123, lambda x: x)])


def test_eval_metrics_invalid_callable_in_tuple():
    # Test if the second element of the tuple is not callable
    with pytest.raises(TypeError, match="`metrics\\[0\\]\\[1\\]` is not callable"):
        pb.eval_metrics(runs=[], metrics=[("valid_name", 123)])


def test_eval_metrics_duplicate_metric_name():
    # Make a mock metric
    class DummyMetric:
        def __init__(self, name):
            self.name = name

    # Test for duplicate metric name error
    with pytest.raises(ValueError, match=r'Duplicate name for `metrics\[1\]`: "metric1"'):
        metric1 = ("metric1", lambda pop, problem: None)
        metric2 = ("metric1", lambda pop, problem: None)
        pb.eval_metrics(runs=[], metrics=[metric1, metric2])


def test_eval_metrics_unrecognized_metric_type_in_list():
    # Test for unrecognized type in the list of metrics
    with pytest.raises(TypeError, match=r"Unrecognized type for `metrics\[0\]`"):
        pb.eval_metrics(runs=[], metrics=[123])

    with pytest.raises(TypeError, match=r"Unrecognized type for `metrics\[1\]`"):
        pb.eval_metrics(runs=[], metrics=[("valid_metric", lambda pop, problem: None), 123])
