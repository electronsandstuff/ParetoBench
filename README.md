[![](https://img.shields.io/pypi/v/paretobench.svg)](https://pypi.org/pypi/paretobench/)
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
Please install our package from pip.
```
pip install paretobench
```

## Installation for Developers
1) Install the development conda environment.
```
conda create env -f environment.yml
```
2) Install the package in editing mode
```
pip install -e .
```

## Usage
Please see the code in `example_notebooks` for usage instructions.

# Testing
Tests are written with the pytest framework. They can be run by calling `pytest` from the base of this repo with the package installed.

# Parameter Naming Conventions
To help standardize the code in this package, the following naming convention is used for parameters.

Some names are reserved for specific purposes. These are the following.
 - `n`: The dimension of the input vector to the problem, ie the number of decision variables.
 - `m`: The number of objectives.

All parameters should follow the PEP 8 naming scheme for variables. Whenever this leads to a parameter being named something different than what it was called in the problem's defining paper, this change must be documented in the class.

# "Single Line" Serialization and Deserialization of Problems
Sometimes it is useful to be able to specify problems with all of their parameters in a short single line string. Examples of applications include defining problems for testing optimizers in a config file, referring to problems in logging, and saving problems along with all parameters in file formats. To support this need, ParetoBench includes a simple serialization/deserialization format for problem parameters as well as a standard format for writing the problem's name and list of parameters into a single line string.

## The Serialization Format
The scope of this format is restricted to dicts containing floats, strings, ints, and bools. This is enough to allow the saving / loading of parameters for all of the problem objects without having to define a more complicated standard.

All of the keys and values are written in pairs of the form `<key>=<value>` and are combined into a single string with the pairs separated by commas. White space is allowed around all of the elements (the keys, values, and pairs) and will be removed on deserialization. The keys may only be alphanumeric strings plus the underscore character. For int and float values, they can take on any value which is correctly interpreted by the respective python conversion functions. Floats must contain the decimal point character. Strings must be contained between two double quotes. The double quote character and backslash character must both be escaped with a single backslash character.

## The "Single Line" Format for Problem Definition
Problems along with their parameters are serialized in the format `NAME (<SERIALIZED PARAMETERS>)`. The name will always be the name of the python class in this library which should also match the literature name for the problem. It must be an alphanumeric string and can include the underscore character. The string "serialized parameters" is generated from the class's parameters using the above described serialization format. White space may be included around and in between any of these objects and will be parsed out.

Not all parameters need to be defined. Default values in the classes will be used for any parameters not specified. Problems can also be specified by name only (ie `NAME`) and this corresponds to the problem with all default parameters. The standard also allows for the format `NAME ()` for objects without parameters or all default parameters.

It should be noted that many single line format strings can describe same problem object. For instance, by rearranging the parameters. This means that the line format should never be used to compare problems or be used as the key to a dictionary for instance as you will end up with duplicates. Additionally, the default values of parameters are not guaranteed to remain constant and cannot be relied on to define a problem. For the purposes of saving problems, all parameters should be defined.
