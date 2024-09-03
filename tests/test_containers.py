import pytest
import tempfile
import os
import numpy as np
from pydantic import ValidationError

from paretobench.containers import Experiment, Population, History


@pytest.mark.parametrize('generate_names', [False, True])
def test_experiment_save_load(generate_names):
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
        file_path = os.path.join(tmpdir, 'test.h5')
        experiment.save(file_path)
        
        # Load the experiment from the file and compare with original
        loaded_experiment = Experiment.load(file_path)
        assert experiment == loaded_experiment, "The loaded experiment is not equal to the original experiment."


def test_generate_population():
    pop = Population(
        f=np.random.random((128, 3)),
        feval=0
    )
    assert pop.x.shape[0] == 128
    assert pop.g.shape[0] == 128


@pytest.mark.parametrize("f, g, idx", [
    (np.array([[1.0, 0.0], [0.0, 1.0]]).T, None, [True, True]),
    (np.array([[1.0, 0.0], [1.0, 1.0]]).T, None, [False, True]),
    (np.array([[0.0, 1.0], [1.0, 1.0]]).T, None, [True, False]),
    (np.array([[0.0, 1.0], [1.0, 0.0]]).T, np.array([[1.0, 0.1]]).T, [True, True]),
    (np.array([[0.0, 1.0], [1.0, 0.0]]).T, np.array([[1.0, -0.1]]).T, [True, False]),
    (np.array([[0.0, 1.0], [1.0, 0.0]]).T, np.array([[-1.0, -0.1]]).T, [False, True]),
    (np.array([[0.0, 1.0], [1.0, 0.0]]).T, np.array([[-1.0, -0.1], [100.0, -0.1], [100.0, 3e3]]).T, [False, True]),
])
def test_get_nondominated_indices(f, g, idx):
    """
    Check domination function for Population on some simple examples. 
    """
    pop = Population(f=f, g=g, feval=0)
    assert all(pop.get_nondominated_indices() == idx)


def test_population_batch_dimension():
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
        Population(x=valid_x, f=valid_f, g=valid_g, feval=1)
    except ValidationError:
        pytest.fail("Population creation with valid batch dimensions raised ValidationError unexpectedly!")

    # Test that creating an invalid Population instance raises a ValidationError
    with pytest.raises(ValidationError, match=r".*Batch dimensions do not match \(len\(x\)=10, len\(f\)=8, len\(g\)=10\).*"):
        Population(x=invalid_x, f=invalid_f, g=invalid_g, feval=1)


def test_history_validation():
    # Create valid populations with consistent decision variables, objectives, and constraints
    valid_population_1 = Population.from_random(n_objectives=3, n_decision_vars=5, n_constraints=2, pop_size=10, feval=1)
    valid_population_2 = Population.from_random(n_objectives=3, n_decision_vars=5, n_constraints=2, pop_size=10, feval=2)

    # Create invalid populations
    invalid_population_decision_vars = Population.from_random(n_objectives=3, n_decision_vars=6, n_constraints=2, pop_size=10, 
                                                              feval=3)
    invalid_population_objectives = Population.from_random(n_objectives=4, n_decision_vars=5, n_constraints=2, pop_size=10, 
                                                           feval=4)
    invalid_population_constraints = Population.from_random(n_objectives=3, n_decision_vars=5, n_constraints=3, pop_size=10, 
                                                            feval=5)

    # Test that creating a valid History instance does not raise an error
    try:
        History(
            reports=[valid_population_1, valid_population_2],
            problem="Test Problem",
            metadata={"description": "A valid test case"}
        )
    except ValidationError:
        pytest.fail("History creation with consistent populations raised ValidationError unexpectedly!")

    # Test that creating an invalid History instance raises a ValidationError due to inconsistent decision variables
    with pytest.raises(ValidationError, match="Inconsistent number of decision variables in reports"):
        History(
            reports=[valid_population_1, invalid_population_decision_vars],
            problem="Test Problem",
            metadata={"description": "An invalid test case with inconsistent decision variables"}
        )

    # Test that creating an invalid History instance raises a ValidationError due to inconsistent objectives
    with pytest.raises(ValidationError, match="Inconsistent number of objectives in reports"):
        History(
            reports=[valid_population_1, invalid_population_objectives],
            problem="Test Problem",
            metadata={"description": "An invalid test case with inconsistent objectives"}
        )

    # Test that creating an invalid History instance raises a ValidationError due to inconsistent constraints
    with pytest.raises(ValidationError, match="Inconsistent number of constraints in reports"):
        History(
            reports=[valid_population_1, invalid_population_constraints],
            problem="Test Problem",
            metadata={"description": "An invalid test case with inconsistent constraints"}
        )

    # Test for inconsistent names - case where some have names and others don't
    population_with_names = Population.from_random(
        n_objectives=3, n_decision_vars=5, n_constraints=2, pop_size=10, feval=6,
    )
    population_with_names.names_x = ["var1", "var2", "var3", "var4", "var5"]
    population_with_names.names_f = ["obj1", "obj2", "obj3"]
    population_with_names.names_g = ["con1", "con2"]
    population_without_names = Population.from_random(
        n_objectives=3, n_decision_vars=5, n_constraints=2, pop_size=10, feval=7,
    )

    with pytest.raises(ValidationError, match="Inconsistent names for decision variables in reports"):
        History(
            reports=[population_with_names, population_without_names],
            problem="Test Problem",
            metadata={"description": "An invalid test case with inconsistent names"}
        )

    # Test for inconsistent names - case where names are different across populations
    population_with_different_names = Population.from_random(
        n_objectives=3, n_decision_vars=5, n_constraints=2, pop_size=10, feval=8,
    )
    population_with_different_names.names_x = ["varA", "varB", "varC", "varD", "varE"]
    population_with_different_names.names_f = ["obj1", "obj2", "obj3"]
    population_with_different_names.names_g = ["con1", "con2"]

    with pytest.raises(ValidationError, match="Inconsistent names for decision variables in reports"):
        History(
            reports=[population_with_names, population_with_different_names],
            problem="Test Problem",
            metadata={"description": "An invalid test case with inconsistent names"}
        )
