from pydantic import ValidationError
import numpy as np
import os
import pytest
import tempfile

from paretobench.containers import Experiment, Population, History


@pytest.mark.parametrize("generate_names", [False, True])
def test_experiment_save_load(generate_names):
    """
    Make a randomized experiment, save it to disk, load it, and then confirm everything matches.
    """
    # Create a randomized Experiment object
    experiment = Experiment.from_random(
        n_histories=32,
        n_populations=10,
        n_objectives=5,
        n_decision_vars=30,
        n_constraints=2,
        pop_size=50,
        generate_names=generate_names,
    )

    # Use a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define the file path
        file_path = os.path.join(tmpdir, "test.h5")
        experiment.save(file_path)

        # Load the experiment from the file and compare with original
        loaded_experiment = Experiment.load(file_path)
        assert experiment == loaded_experiment, "The loaded experiment is not equal to the original experiment."


def test_empty_history():
    """
    Confirms save/load methods work correctly whan an empty history object is saved.
    """
    # Create an Experiment object
    experiment = Experiment(name="", runs=[History(reports=[], problem="")])

    # Use a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define the file path
        file_path = os.path.join(tmpdir, "test.h5")
        experiment.save(file_path)

        # Load the experiment from the file and compare with original
        loaded_experiment = Experiment.load(file_path)
        assert len(experiment.runs[0]) == 0
        assert experiment == loaded_experiment, "The loaded experiment is not equal to the original experiment."


def test_generate_population():
    pop = Population(f=np.random.random((128, 3)), fevals=0)
    assert pop.x.shape[0] == 128
    assert pop.g.shape[0] == 128


@pytest.mark.parametrize(
    "f, g, idx",
    [
        (np.array([[1.0, 0.0], [0.0, 1.0]]).T, None, [True, True]),
        (np.array([[1.0, 0.0], [1.0, 1.0]]).T, None, [False, True]),
        (np.array([[0.0, 1.0], [1.0, 1.0]]).T, None, [True, False]),
        (np.array([[0.0, 1.0], [1.0, 0.0]]).T, np.array([[1.0, 0.1]]).T, [True, True]),
        (
            np.array([[0.0, 1.0], [1.0, 0.0]]).T,
            np.array([[1.0, -0.1]]).T,
            [True, False],
        ),
        (
            np.array([[0.0, 1.0], [1.0, 0.0]]).T,
            np.array([[-1.0, -0.1]]).T,
            [False, True],
        ),
        (
            np.array([[0.0, 1.0], [1.0, 0.0]]).T,
            np.array([[-1.0, -0.1], [100.0, -0.1], [100.0, 3e3]]).T,
            [False, True],
        ),
    ],
)
def test_get_nondominated_indices(f, g, idx):
    """
    Check domination function for Population on some simple examples.
    """
    pop = Population(f=f, g=g, fevals=0)
    assert all(pop.get_nondominated_indices() == idx)


def test_to_nondominated():
    """
    Confirm that History.to_nondominated produces the nondominated individuals up to each point in the runs.
    """
    # Make a mock history object
    hist = History.from_random(10, 3, 4, 0, 50)
    hist_nd = hist.to_nondominated()

    for report_idx, report_nd in enumerate(hist_nd.reports):
        # Calculate the nondominated individuals ourself
        ref_nd = sum(hist.reports[1 : report_idx + 1], hist.reports[0]).get_nondominated_set()

        # Sort test and reference set so they are comparable
        ref_idx = np.lexsort(np.concatenate((ref_nd.x, ref_nd.f), axis=1).T)
        test_idx = np.lexsort(np.concatenate((report_nd.x, report_nd.f), axis=1).T)

        # Make sure the two are the same
        np.testing.assert_array_equal(ref_nd.x[ref_idx, :], report_nd.x[test_idx, :])

        # Double check that fevals hasn't changed
        assert report_nd.fevals == hist.reports[report_idx].fevals


def test_population_batch_dimension():
    """
    Confirm validation of batch dimension size works correctly.
    """
    # Create valid arrays with matching batch dimensions
    valid_x = np.random.rand(10, 5)
    valid_f = np.random.rand(10, 3)
    valid_g = np.random.rand(10, 2)

    # Create invalid arrays with different batch dimensions
    invalid_x = np.random.rand(10, 5)
    invalid_f = np.random.rand(8, 3)
    invalid_g = np.random.rand(10, 2)

    # Test that creating a valid Population instance does not raise an error
    try:
        Population(x=valid_x, f=valid_f, g=valid_g, fevals=1)
    except ValidationError:
        pytest.fail("Population creation with valid batch dimensions raised ValidationError unexpectedly!")

    # Test that creating an invalid Population instance raises a ValidationError
    with pytest.raises(
        ValidationError,
        match=r".*Batch dimensions do not match \(len\(x\)=10, len\(f\)=8, len\(g\)=10\).*",
    ):
        Population(x=invalid_x, f=invalid_f, g=invalid_g, fevals=1)


def test_history_validation():
    """
    Make sure that consistency checks for history objects work.
    """
    # Create valid populations with consistent decision variables, objectives, and constraints
    valid_population_1 = Population.from_random(
        n_objectives=3, n_decision_vars=5, n_constraints=2, pop_size=10, fevals=1
    )
    valid_population_2 = Population.from_random(
        n_objectives=3, n_decision_vars=5, n_constraints=2, pop_size=10, fevals=2
    )

    # Create invalid populations
    invalid_population_decision_vars = Population.from_random(
        n_objectives=3, n_decision_vars=6, n_constraints=2, pop_size=10, fevals=3
    )
    invalid_population_objectives = Population.from_random(
        n_objectives=4, n_decision_vars=5, n_constraints=2, pop_size=10, fevals=4
    )
    invalid_population_constraints = Population.from_random(
        n_objectives=3, n_decision_vars=5, n_constraints=3, pop_size=10, fevals=5
    )

    # Test that creating a valid History instance does not raise an error
    try:
        History(
            reports=[valid_population_1, valid_population_2],
            problem="Test Problem",
            metadata={"description": "A valid test case"},
        )
    except ValidationError:
        pytest.fail("History creation with consistent populations raised ValidationError unexpectedly!")

    # Test that creating an invalid History instance raises a ValidationError due to inconsistent decision variables
    with pytest.raises(ValidationError, match="Inconsistent number of decision variables in reports"):
        History(
            reports=[valid_population_1, invalid_population_decision_vars],
            problem="Test Problem",
            metadata={"description": "An invalid test case with inconsistent decision variables"},
        )

    # Test that creating an invalid History instance raises a ValidationError due to inconsistent objectives
    with pytest.raises(ValidationError, match="Inconsistent number of objectives in reports"):
        History(
            reports=[valid_population_1, invalid_population_objectives],
            problem="Test Problem",
            metadata={"description": "An invalid test case with inconsistent objectives"},
        )

    # Test that creating an invalid History instance raises a ValidationError due to inconsistent constraints
    with pytest.raises(ValidationError, match="Inconsistent number of constraints in reports"):
        History(
            reports=[valid_population_1, invalid_population_constraints],
            problem="Test Problem",
            metadata={"description": "An invalid test case with inconsistent constraints"},
        )

    # Test for inconsistent names - case where some have names and others don't
    population_with_names = Population.from_random(
        n_objectives=3,
        n_decision_vars=5,
        n_constraints=2,
        pop_size=10,
        fevals=6,
    )
    population_with_names.names_x = ["var1", "var2", "var3", "var4", "var5"]
    population_with_names.names_f = ["obj1", "obj2", "obj3"]
    population_with_names.names_g = ["con1", "con2"]
    population_without_names = Population.from_random(
        n_objectives=3,
        n_decision_vars=5,
        n_constraints=2,
        pop_size=10,
        fevals=7,
    )

    with pytest.raises(ValidationError, match="Inconsistent names for decision variables in reports"):
        History(
            reports=[population_with_names, population_without_names],
            problem="Test Problem",
            metadata={"description": "An invalid test case with inconsistent names"},
        )

    # Test for inconsistent names - case where names are different across populations
    population_with_different_names = Population.from_random(
        n_objectives=3,
        n_decision_vars=5,
        n_constraints=2,
        pop_size=10,
        fevals=8,
    )
    population_with_different_names.names_x = ["varA", "varB", "varC", "varD", "varE"]
    population_with_different_names.names_f = ["obj1", "obj2", "obj3"]
    population_with_different_names.names_g = ["con1", "con2"]

    with pytest.raises(ValidationError, match="Inconsistent names for decision variables in reports"):
        History(
            reports=[population_with_names, population_with_different_names],
            problem="Test Problem",
            metadata={"description": "An invalid test case with inconsistent names"},
        )


def test_population_invalid_dimensions():
    batch_size = 5
    num_variables = 3

    # Invalid dimension (3D instead of 2D)
    x = np.random.rand(batch_size, num_variables, 2)
    f = np.random.rand(batch_size, 2)
    g = np.random.rand(batch_size, 1)

    # Expect ValueError for incorrect number of dimensions
    with pytest.raises(ValueError, match="Expected array with 2 dimensions for field 'x'"):
        Population(x=x, f=f, g=g, fevals=5)


def test_overwrite():
    # Create a randomized Experiment object
    experiment1 = Experiment.from_random(
        n_histories=32,
        n_populations=10,
        n_objectives=5,
        n_decision_vars=30,
        n_constraints=2,
        pop_size=50,
    )

    # A second experiment
    experiment2 = Experiment.from_random(
        n_histories=32,
        n_populations=10,
        n_objectives=5,
        n_decision_vars=30,
        n_constraints=2,
        pop_size=50,
    )

    with tempfile.TemporaryDirectory() as dir:
        # Overwrite the file
        experiment1.save(os.path.join(dir, "experiment.h5"))
        experiment2.save(os.path.join(dir, "experiment.h5"))

        # Load the experiment from the file and compare with original
        loaded_experiment = Experiment.load(os.path.join(dir, "experiment.h5"))
        assert experiment2 == loaded_experiment, "The loaded experiment is not equal to the original experiment."


def test_count_unique_individuals():
    # Test empty population
    pop_empty = Population(x=np.empty((0, 2)), f=np.empty((0, 1)), g=np.empty((0, 1)))
    assert pop_empty.count_unique_individuals() == 0

    # Test all identical individuals
    pop_identical = Population(
        x=np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]),
        f=np.array([[3.0], [3.0], [3.0]]),
        g=np.array([[4.0], [4.0], [4.0]]),
    )
    assert pop_identical.count_unique_individuals() == 1

    # Test all unique individuals
    pop_unique = Population(
        x=np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
        f=np.array([[3.0], [4.0], [5.0]]),
        g=np.array([[4.0], [5.0], [6.0]]),
    )
    assert pop_unique.count_unique_individuals() == 3

    # Test floating point tolerance
    pop_tolerance = Population(
        x=np.array([[1.0, 2.0], [1.0 + 1e-7, 2.0 - 1e-7], [1.5, 2.5]]),
        f=np.array([[3.0], [3.0 + 1e-7], [3.5]]),
        g=np.array([[4.0], [4.0 - 1e-7], [4.5]]),
    )
    assert pop_tolerance.count_unique_individuals(decimals=4) == 2
    assert pop_tolerance.count_unique_individuals() == 3

    # Test with empty dimensions
    pop_empty_dims = Population(
        x=np.empty((3, 0)),
        f=np.array([[1.0], [2.0], [1.0]]),
        g=np.array([[3.0], [4.0], [3.0]]),
    )
    assert pop_empty_dims.count_unique_individuals() == 2
