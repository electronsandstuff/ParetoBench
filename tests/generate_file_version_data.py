import argparse
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from paretobench import Experiment, History
from utils import experiment_to_manifest

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "test_data" / "file_versions"
FILENAME_FMT = "paretobench_file_format_v{version}.h5"


def make_experiment():
    """
    Build the synthetic experiment which gets saved into the fixtures. The contents are deterministic so that
    running this script twice with the same version of the library produces the same file. Names and the
    objective / constraint settings are included so that every serialized field is covered.

    Returns
    -------
    Experiment
        The experiment to save
    """
    random.seed(0)
    np.random.seed(0)

    runs = []
    for problem in ["ZDT1 (n=4)", "CTP1 (n=4)", "TNK"]:
        run = History.from_random(
            n_populations=4,
            n_objectives=3,
            n_decision_vars=4,
            n_constraints=2,
            pop_size=25,
            generate_names=True,
            generate_obj_constraint_settings=True,
            generate_bounds=True,
        )
        run.problem = problem
        runs.append(run)

    return Experiment(
        runs=runs,
        name="file_format_regression",
        author="ParetoBench test suite",
        software="ParetoBench",
        software_version="0.0.0",
        comment="Synthetic data used by the file format regression tests",
        creation_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


def write_manifest(path):
    """
    Record what a saved file contains in a JSON file next to it. The description is taken from the file as it
    reads back off of disk so that it captures the output of the reader and not the object which was saved.

    Parameters
    ----------
    path : Path
        The saved HDF5 file to describe

    Returns
    -------
    Path
        Path of the manifest which was written
    """
    manifest_path = path.with_suffix(".json")
    with open(manifest_path, "w") as fd:
        json.dump(experiment_to_manifest(Experiment.load(path)), fd, indent=2, sort_keys=True)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Save a file of synthetic data in the version of the ParetoBench file format the library currently "
            "writes."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="directory to save the file into")
    parser.add_argument("--force", action="store_true", help="overwrite an existing file for this version")
    parser.add_argument(
        "--refresh-manifests",
        type=Path,
        nargs="*",
        help="rewrite the manifests of files which already exist, defaulting to every file in the output directory",
    )
    args = parser.parse_args()

    # Rewrite the manifests of files which already exist. Needed whenever the contents of a manifest change, such as
    # when a new field is added to the containers, and to bootstrap the manifests of files saved by older versions.
    if args.refresh_manifests is not None:
        for path in args.refresh_manifests or sorted(args.out_dir.glob("*.h5")):
            print(f"Wrote {write_manifest(path)}")
        return

    # Save the data, then name the file after the version which actually ended up in it
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = args.out_dir / "_generate_file_version_data.h5"
    make_experiment().save(tmp_path)
    with h5py.File(tmp_path) as fd:
        version = fd.attrs["file_version"]
    path = args.out_dir / FILENAME_FMT.format(version=version)

    # Regenerating the file of an already released version would destroy the record of how that version wrote data
    if path.exists() and not args.force:
        tmp_path.unlink()
        raise SystemExit(
            f"{path} already exists, refusing to regenerate the fixture of a released format version. "
            "Pass --force to overwrite it anyway."
        )

    shutil.move(tmp_path, path)
    print(f"Wrote {path} (file version {version})")
    print(f"Wrote {write_manifest(path)}")


if __name__ == "__main__":
    main()
