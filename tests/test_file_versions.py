import h5py
import json
import pytest
import re
from pathlib import Path

from paretobench import Experiment

from .utils import experiment_to_manifest

FILE_VERSION_DIR = Path(__file__).parent / "test_data" / "file_versions"
GENERATOR = "tests/test_data/file_versions/generate_file_version_data.py"


def get_file_version_files():
    """
    Returns the saved files of each version of the file format which are used for regression testing.
    """
    return sorted(FILE_VERSION_DIR.glob("*.h5"))


def version_from_filename(path):
    """
    Extract the file format version a saved file is named after.

    Parameters
    ----------
    path : Path
        Path to the saved file

    Returns
    -------
    str
        The version in the filename
    """
    match = re.search(r"_v(\d+\.\d+\.\d+)\.h5$", path.name)
    if match is None:
        raise ValueError(f'Could not get a file format version out of the filename "{path.name}"')
    return match.group(1)


def test_file_version_files_exist():
    """
    Confirms the regression files are where the tests expect them. Without this, a missing or moved directory would
    leave the test below parametrized over nothing and silently passing.
    """
    assert get_file_version_files(), f"No files found in {FILE_VERSION_DIR}"


@pytest.mark.parametrize("path", get_file_version_files(), ids=lambda p: p.stem)
def test_load_file_version(path):
    """
    Load a file saved by an earlier version of the library and confirm it still reads back the data which was recorded
    in its manifest when it was generated.
    """
    exp = Experiment.load(path)
    assert exp.file_version == version_from_filename(path)

    with open(path.with_suffix(".json")) as fd:
        assert experiment_to_manifest(exp) == json.load(fd)


def test_current_file_version_has_file(tmp_path):
    # Get the version we are writing by saving a file and reading the version back out of it
    probe_path = tmp_path / "probe.h5"
    Experiment(name="", runs=[]).save(probe_path)
    with h5py.File(probe_path) as fd:
        current_version = fd.attrs["file_version"]

    versions = set(map(version_from_filename, get_file_version_files()))
    assert current_version in versions, (
        f"No saved file for file format version {current_version}. Files for old versions of the format cannot be "
        f"created later, so save one now by running: python {GENERATOR}"
    )
