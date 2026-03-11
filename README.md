[![](https://img.shields.io/pypi/v/paretobench.svg)](https://pypi.org/pypi/paretobench/)
[![](https://anaconda.org/conda-forge/paretobench/badges/version.svg)](https://anaconda.org/channels/conda-forge/packages/paretobench/overview)
[![](https://img.shields.io/pypi/pyversions/paretobench.svg)](https://pypi.org/pypi/paretobench/)
[![](https://img.shields.io/pypi/l/paretobench.svg)](https://pypi.org/pypi/paretobench/)

# ParetoBench
ParetoBench is a Python library that provides a collection of tools for the benchmarking of multi-objective optimization algorithms. It includes the following.
- Multi-objective benchmark problems including analytical Pareto fronts when available
- Container objects for storing and manipulating data from optimization algorithms
- A standardized file format for saving the results of optimizations on benchmark problems
- Tools for calculating convergence metrics on results and running statistical analyses on them to compare algorithms
- Plotting utilities for objectives/decision variables and for both populations and series of populations (history objects)

## Installation
ParetoBench is available from pip and conda.
```
pip install paretobench
```
or
```
conda install paretobench
```

## Installation for Developers
1) Install the development conda environment.
```
conda create env -f environment.yml
```
2) Install the package in editing mode (with dependencies required for running tests)
```
pip install -e .[test]
```

## Usage
Please see the code in `example_notebooks` for usage instructions.

# Testing
Tests are written with the pytest framework. They can be run by calling `pytest` from the base of this repo with the package installed.

# Parameter Naming Conventions
To help standardize the code in this package, the following naming convention is used throughout for parameters.

Some names are reserved for specific purposes. These are the following.
 - `n`: The dimension of the input vector to the problem, ie the number of decision variables.
 - `m`: The number of objectives.

All parameters should follow the PEP 8 naming scheme for variables. Whenever this leads to a parameter being named something different than what it was called in the problem's defining paper, this change must be documented in the class.
