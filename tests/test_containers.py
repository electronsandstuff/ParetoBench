import pytest
import tempfile
import os

from paretobench.containers import Experiment


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
