from datetime import datetime
from functools import partial
from typing import Union, Optional, TextIO
from xopt import VOCS, Xopt
import logging
import numpy as np
import os
import pandas as pd
import re
import time
import warnings

from paretobench import Problem, Population, History


logger = logging.getLogger(__name__)


class XoptProblemWrapper:
    def __init__(self, problem: Problem):
        """
        This class wraps a ParetoBench problem for use with xopt. After creation of the wrapper object from
        the problem, the Xopt VOCS object can be accessed through a class property. The wrapper object is
        also a callable and may be directly passed to the Xopt evaluator object.

        Example
        -------
        > import paretobench as pb
        > from xopt import Xopt, Evaluator
        > from xopt.generators.ga.cnsga import CNSGAGenerator
        >
        > prob = XoptProblemWrapper(pb.Problem.from_line_fmt('WFG1'))
        > population_size = 50
        > ev = Evaluator(function=prob, vectorized=True, max_workers=population_size)
        > X = Xopt(
        >        generator=CNSGAGenerator(vocs=prob.vocs, population_size=population_size),
        >        evaluator=ev,
        >        vocs=prob.vocs,
        >    )


        Parameters
        ----------
        problem : Problem
            A problem object that follows the Problem class interface.
        """
        self.prob = problem

    @property
    def vocs(self) -> VOCS:
        """Return the VOCS object."""
        # Construct the decision variables
        lbs = self.prob.var_lower_bounds
        ubs = self.prob.var_upper_bounds
        vars = {f"x{i}": [lb, ub] for i, (lb, ub) in enumerate(zip(lbs, ubs))}

        # Construct the objectives
        objs = {f"f{i}": "MINIMIZE" for i in range(self.prob.n_objs)}

        # The constraints
        constraints = {f"g{i}": ["LESS_THAN", 0] for i in range(self.prob.n_constraints)}

        # Construct VOCS object
        return VOCS(variables=vars, objectives=objs, constraints=constraints)

    def __call__(self, input_dict: dict) -> dict:
        """
        Evaluate the problem using the dict -> dict convention for xopt.

        Parameters
        ----------
        input_dict : dict
            A dictionary containing the decision variables

        Returns
        -------
        dict
            A dictionary with the objectives and constraints
        """
        # Convert the input dictionary to a NumPy array of decision variables
        x = np.array([input_dict[f"x{i}"] for i in range(self.prob.n_vars)]).T

        # Evaluate the problem
        pop = self.prob(x)  # Pass single batch

        # Convert the result to the format expected by Xopt
        ret = {}
        ret.update({f"f{i}": pop.f[:, i] for i in range(self.prob.n_objs)})
        ret.update({f"g{i}": pop.g[:, i] for i in range(self.prob.n_constraints)})
        return ret

    def __repr__(self):
        return f"XoptProblemWrapper({self.prob.to_line_fmt()})"


def population_from_dataframe(df: pd.DataFrame, vocs: VOCS, errors_as_constraints: bool = False):
    """
    Import a population file from Xopt's CNSGA geneator into a ParetoBench Population object.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing this population's data.
    vocs : VOCS
        VOCS object defining the variables, objectives, and constraints.
    errors_as_constraints : bool, optional
        If True, imports the 'xopt_error' column as an additional constraint,
        where True maps to -1 (violated) and False maps to +1 (satisfied), by default False.

    Returns
    -------
    Population
        Population object with the loaded data
    """
    # Get base constraints if they exist
    g = df[vocs.constraint_names].to_numpy() if vocs.constraints else None
    names_g = vocs.constraint_names
    constraint_targets = [vocs.constraints[name][1] for name in vocs.constraint_names]
    constraint_directions = [
        ">" if vocs.constraints[name][0] == "GREATER_THAN" else "<" for name in vocs.constraint_names
    ]

    # Handle error column if requested
    if errors_as_constraints:
        # Convert boolean strings to +/-1, reshape to 2D array
        error_constraints = np.where(df["xopt_error"].astype(str).str.lower() == "true", -1.0, 1.0)[:, np.newaxis]

        # Combine with existing constraints if present
        if g is not None:
            g = np.hstack([g, error_constraints])
        else:
            g = error_constraints
        names_g.append("xopt_error")
        constraint_targets.append(0.0)
        constraint_directions.append(">")

    return Population(
        x=df[vocs.variable_names].to_numpy(),
        f=df[vocs.objective_names].to_numpy(),
        g=g,
        names_x=vocs.variable_names,
        names_f=vocs.objective_names,
        names_g=names_g,
        obj_directions="".join(["+" if vocs.objectives[name] == "MAXIMIZE" else "-" for name in vocs.objective_names]),
        constraint_directions="".join(constraint_directions),
        constraint_targets=np.array(constraint_targets),
    )


def import_cnsga_population(
    path: Union[str, os.PathLike[str]],
    vocs: VOCS | str | os.PathLike[str] | None = None,
    errors_as_constraints: bool = False,
):
    """
    Import a population file from Xopt's CNSGA geneator into a ParetoBench Population object.

    Parameters
    ----------
    path : Union[str, os.PathLike[str]]
        Path to the CSV file containing the population data.
    vocs : VOCS
        VOCS object defining the variables, objectives, and constraints.
    errors_as_constraints : bool, optional
        If True, imports the 'xopt_error' column as an additional constraint,
        where True maps to -1 (violated) and False maps to +1 (satisfied), by default False.

    Returns
    -------
    Population
        Population object with the loaded data
    """
    return population_from_dataframe(
        pd.read_csv(path),
        vocs,
        errors_as_constraints,
    )


def import_cnsga_history(
    output_path: Union[str, os.PathLike[str]],
    vocs: Optional[VOCS] = None,
    config: Union[None, str, TextIO] = None,
    problem: str = "",
    errors_as_constraints: bool = False,
):
    """
    Import all population files in output_path from Xopt's CNSGA generator
    into a ParetoBench History object.

    Parameters
    ----------
    output_path : Union[str, os.PathLike[str]]
        Directory containing the CNSGA population CSV files.
    vocs : VOCS | str | os.PathLike[str], optional
        VOCS object defining the variables, objectives, and constraints. Either python object or path to json file.
        If None, must provide config, by default None.
    config : Union[None, str, os.PathLike[str]], optional
        YAML config file or open file object with the information (passed to `Xopt.from_yaml`)
        If None, must provide vocs, by default None.
    problem : str, optional
        Name of the optimization problem, by default "".
    errors_as_constraints : bool, optional
        If True, imports the 'xopt_error' column as an additional constraint
        for each population, by default False.

    Returns
    -------
    History
        History object containing the population data

    Notes
    -----
    Population files must be named following the pattern:
    'cnsga_population_YYYY-MM-DDThh:mm:ss.ffffff+ZZ:ZZ.csv'
    or 'cnsga_population_YYYY-MM-DDThh_mm_ss.ffffff+ZZ_ZZ.csv'
    """
    start_t = time.perf_counter()
    # Handle vocs in json file (ie output from NSGA2Generator)
    if isinstance(vocs, (str, os.PathLike)):
        with open(vocs) as f:
            _vocs = VOCS.model_validate_json(f.read())

    # Passed an actual VOCS object
    elif isinstance(vocs, VOCS):
        _vocs = vocs

    # Get vocs from config file
    elif config is not None:
        xx = Xopt.from_yaml(config)
        _vocs = xx.vocs

    else:
        raise ValueError("Must specify one of vocs or config")

    # Get list of population files and their datetimes
    population_files = []
    population_datetimes = []

    # Regex pattern to match both datetime formats
    datetime_pattern = r"cnsga_population_(\d{4}-\d{2}-\d{2}T\d{2}[_:]\d{2}[_:]\d{2}\.\d+[-+]\d{2}[_:]\d{2})\.csv"

    # Walk through all files in directory
    for filename in os.listdir(output_path):
        match = re.match(datetime_pattern, filename)
        if match:
            # Get datetime string and convert _ to : if needed
            dt_str = match.group(1).replace("_", ":")

            # Parse datetime string
            dt = datetime.fromisoformat(dt_str)

            # Store filename and datetime
            population_files.append(filename)
            population_datetimes.append(dt)

    # Sort based on datetime
    population_files = [os.path.join(output_path, x) for _, x in sorted(zip(population_datetimes, population_files))]
    logger.info(
        f'Detected {len(population_files)} population files out of {len(os.listdir(output_path))} total files at "{output_path}"'
    )

    # Import files as populations
    pops = list(
        map(
            partial(
                import_cnsga_population,
                vocs=_vocs,
                errors_as_constraints=errors_as_constraints,
            ),
            population_files,
        )
    )

    # Update fevals
    fevals = 0
    for pop in pops:
        fevals += len(pop)
        pop.fevals = fevals

    hist = History(reports=pops, problem=problem)
    logger.info(f"Successfully loaded History object in {time.perf_counter()-start_t:.2f}s: {hist}")
    return hist


def import_nsga2_history(
    populations_path: str | os.PathLike[str],
    vocs: VOCS | str | os.PathLike[str] | None = None,
    config: Union[None, str, TextIO] = None,
    problem: str = "",
    errors_as_constraints: bool = False,
):
    """
    Import all populations from the output of NSGA2Generator.

    Parameters
    ----------
    populations_path : Union[str, os.PathLike[str]]
        Path to the populations file.
    vocs : VOCS | str | os.PathLike[str], optional
        VOCS object defining the variables, objectives, and constraints. Either python object or path to json file.
        If None, must provide config, by default None.
    config : Union[None, str, os.PathLike[str]], optional
        YAML config file or open file object with the information (passed to `Xopt.from_yaml`)
        If None, must provide vocs, by default None.
    problem : str, optional
        Name of the optimization problem, by default "".
    errors_as_constraints : bool, optional
        If True, imports the 'xopt_error' column as an additional constraint
        for each population, by default False.
    """
    start_t = time.perf_counter()

    # Handle vocs in json file (ie output from NSGA2Generator)
    if isinstance(vocs, (str, os.PathLike)):
        with open(vocs) as f:
            _vocs = VOCS.model_validate_json(f.read())

    # Passed an actual VOCS object
    elif isinstance(vocs, VOCS):
        _vocs = vocs

    # Get vocs from config file
    elif config is not None:
        xx = Xopt.from_yaml(config)
        _vocs = xx.vocs

    else:
        raise ValueError("Must specify one of vocs or config")

    # Load all population data
    pop_df = pd.read_csv(populations_path)

    pops = []
    for idx, (gen, grp) in enumerate(pop_df.groupby(by="xopt_generation", sort=True)):
        # Sanity check that we are at the expected generation
        if idx + 1 != gen:
            warnings.warn(
                f"Non-consecutive groups detected - populations file is possibly corrupt. (generation at index {idx} is {gen})"
            )

        # Convert to population
        pops.append(population_from_dataframe(grp, _vocs, errors_as_constraints=errors_as_constraints))

    # Update fevals
    fevals = 0
    for pop in pops:
        fevals += len(pop)
        pop.fevals = fevals

    hist = History(reports=pops, problem=problem)
    logger.info(f"Successfully loaded History object in {time.perf_counter()-start_t:.2f}s: {hist}")
    return hist
