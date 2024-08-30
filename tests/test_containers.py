import pytest
import tempfile
import os
import numpy as np
from pydantic import ValidationError

from paretobench.containers import Experiment, Population


def test_experiment_save_load():
    # Create a randomized Experiment object
    experiment = Experiment.from_random(
        n_histories=32,
        n_populations=10,
        n_objectives=5,
        n_decision_vars=30,
        n_constraints=2,
        pop_size=50
    )
    
    # Use a temporary directory to save the file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define the file path
        file_path = os.path.join(tmpdir, 'test.h5')
        experiment.save(file_path)
        
        # Load the experiment from the file and compare with original
        loaded_experiment = Experiment.load(file_path)
        assert experiment == loaded_experiment, "The loaded experiment is not equal to the original experiment."


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
