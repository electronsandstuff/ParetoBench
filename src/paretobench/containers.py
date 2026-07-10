from datetime import datetime, timezone
from functools import reduce
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from typing import List, Dict, Union, Optional, Literal, Tuple, TYPE_CHECKING
import h5py
import numpy as np
import random
import re
import string

from .utils import get_domination, binary_str_to_numpy

if TYPE_CHECKING:
    from .metrics import Metric
    from .problem import Problem


class Population(BaseModel):
    """
    Stores the individuals in a population for one reporting interval in a genetic algorithm. Conventional names are used for
    the decision variables (x), the objectives (f), and inequality constraints (g). The first dimension of each array is the
    batch dimension. The number of evaluations of the objective functions performed to reach this state is also recorded.

    All arrays must have the same size batch dimension even if they are empty. In this case the non-batch dimension will be
    zero length. Names may be associated with decision variables, objectives, or constraints in the form of lists.

    Whether each objectives is being minimized or maximized is set by the string obj_directions where each character corresponds
    to an objectve and '+' means maximize with '-' meaning minimize. The constraints are configured by the string of directions
    `constraint_directions` and the numpy array of targets `constraint_targets`. The string should contain either the '<' or '>'
    character for the constraint at that index to be satisfied when it is less than or greater than the target respectively.
    """

    # The decision vars, objectives, and constraints
    x: np.ndarray
    f: np.ndarray
    g: np.ndarray

    # Total number of function evaluations performed during optimization after this population was completed
    fevals: int
    # Optional lists of names for decision variables, objectives, and constraints
    names_x: Optional[List[str]] = None
    names_f: Optional[List[str]] = None
    names_g: Optional[List[str]] = None

    # Configuration of objectives/constraints (minimization or maximization problem, direction of and target of constraint)
    obj_directions: str  # '+' means maximize, '-' means minimize
    constraint_directions: (
        str  # '<' means satisfied when less-than target, '>' means satisfied when greater-than target
    )
    constraint_targets: np.ndarray

    # Pydantic config
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def set_default_vals(cls, values):
        """
        Handles automatic setting of `x`, `f`,  `g`, `fevals` when some are not specified. The arrays are set to an empty array
        with a zero length non-batch dimension. The number of function evaluations (`fevals`) is set to the number of individuals
        in the population (assuming here that each was evaluated to get to this point).
        """
        # Determine the batch size from the first non-None array
        batch_size = next(
            (arr.shape[0] for arr in [values.get("x"), values.get("f"), values.get("g")] if arr is not None),
            None,
        )
        if batch_size is None:
            raise ValueError("Must specify one of x, f, or g")

        # Set empty arrays for unspecified fields
        if values.get("x") is None:
            values["x"] = np.empty((batch_size, 0), dtype=np.float64)
        if values.get("f") is None:
            values["f"] = np.empty((batch_size, 0), dtype=np.float64)
        if values.get("g") is None:
            values["g"] = np.empty((batch_size, 0), dtype=np.float64)

        # Handle objectives / constraints settings (default to canonical problem)
        if values.get("obj_directions") is None:
            values["obj_directions"] = "-" * values["f"].shape[1]
        if values.get("constraint_directions") is None:
            values["constraint_directions"] = "<" * values["g"].shape[1]
        if values.get("constraint_targets") is None:
            values["constraint_targets"] = np.zeros(values["g"].shape[1], dtype=float)

        # Set fevals to number of individuals if not included
        if values.get("fevals") is None:
            values["fevals"] = batch_size
        return values

    @model_validator(mode="after")
    def validate_batch_dimensions(self):
        """
        Confirms that the arrays have the same length batch dimension.
        """
        # Validate batch dimensions
        x_size, f_size, g_size = self.x.shape[0], self.f.shape[0], self.g.shape[0]
        if len(set([x_size, f_size, g_size])) != 1:
            raise ValueError(f"Batch dimensions do not match (len(x)={x_size}, len(f)={f_size}, len(g)={g_size})")
        return self

    @model_validator(mode="after")
    def validate_names(self):
        """
        Checks that the name lists, if used, are correctly sized to the number of decision variables, objectives, or
        constraints.
        """
        if self.names_x and len(self.names_x) != self.x.shape[1]:
            raise ValueError("Length of names_x must match the number of decision variables in x.")
        if self.names_f and len(self.names_f) != self.f.shape[1]:
            raise ValueError("Length of names_f must match the number of objectives in f.")
        if self.names_g and len(self.names_g) != self.g.shape[1]:
            raise ValueError("Length of names_g must match the number of constraints in g.")
        return self

    @model_validator(mode="after")
    def validate_obj_directions(self):
        if len(self.obj_directions) != self.m:
            raise ValueError(
                "Length of obj_directions must match number of objectives, got"
                f" {len(self.obj_directions)} chars but we have {self.m} objectives"
            )
        if not all(c in "+-" for c in self.obj_directions):
            raise ValueError(f'obj_directions must contain only + or - characters. Got: "{self.obj_directions}"')
        return self

    @model_validator(mode="after")
    def validate_constraint_directions(self):
        if len(self.constraint_directions) != self.n_constraints:
            raise ValueError(
                "Length of constraint_directions must match number of constraints, got"
                f" {len(self.constraint_directions)} chars but we have {self.n_constraints} constraints"
            )
        if not all(c in "<>" for c in self.constraint_directions):
            raise ValueError(
                f'constraint_directions must contain only < or > characters. Got: "{self.constraint_directions}"'
            )
        return self

    @model_validator(mode="after")
    def validate_constraint_targets(self):
        # Check the targets
        attr = "constraint_targets"
        if not isinstance(getattr(self, attr), np.ndarray):
            raise ValueError(f"{attr} must be a numpy array, got: {type(getattr(self, attr))}")
        if len(getattr(self, attr).shape) != 1:
            raise ValueError(f"{attr} must be 1D, shape was: {getattr(self, attr).shape}")
        if len(getattr(self, attr)) != self.g.shape[1]:
            raise ValueError(
                f"Length of {attr} must match number of constraints in g. Got {len(getattr(self, attr))} "
                f"elements and {self.g.shape[1]} constraints from g"
            )
        if getattr(self, attr).dtype != np.float64:
            raise ValueError(f"{attr} dtype must be {np.float64}. Got {getattr(self, attr).dtype}")
        return self

    @field_validator("x", "f", "g")
    @classmethod
    def validate_numpy_arrays(cls, value: np.ndarray, info) -> np.ndarray:
        """
        Double checks that the arrays have the right number of dimensions and datatype.
        """
        if value.dtype != np.float64:
            raise TypeError(f"Expected array of type { np.float64} for field '{info.field_name}', got {value.dtype}")
        if value.ndim != 2:
            raise ValueError(f"Expected array with 2 dimensions for field '{info.field_name}', got {value.ndim}")

        return value

    @field_validator("fevals")
    @classmethod
    def validate_feval(cls, v):
        if v < 0:
            raise ValueError("fevals must be a non-negative integer")
        return v

    def __eq__(self, other):
        if not isinstance(other, Population):
            return False
        return (
            np.array_equal(self.x, other.x)
            and np.array_equal(self.f, other.f)
            and np.array_equal(self.g, other.g)
            and self.fevals == other.fevals
            and self.names_x == other.names_x
            and self.names_f == other.names_f
            and self.names_g == other.names_g
            and self.obj_directions == other.obj_directions
            and self.constraint_directions == other.constraint_directions
            and np.array_equal(self.constraint_targets, other.constraint_targets)
        )

    @property
    def f_canonical(self):
        """
        Return the objectives transformed so that we are a minimization problem.
        """
        return binary_str_to_numpy(self.obj_directions, "-", "+")[None, :] * self.f

    @property
    def g_canonical(self):
        """
        Return constraints transformed such that g[...] >= 0 are the feasible solutions.
        """
        gc = binary_str_to_numpy(self.constraint_directions, "<", ">")[None, :] * self.g
        gc += binary_str_to_numpy(self.constraint_directions, ">", "<")[None, :] * self.constraint_targets[None, :]
        return gc

    def __add__(self, other: "Population") -> "Population":
        """
        The sum of two populations is defined here as the population containing all unique individuals from both (set union).
        The number of function evaluations is by default set to the sum of the function evaluations in each input population.
        """
        if not isinstance(other, Population):
            raise TypeError("Operands must be instances of Population")

        # Check that the names/settings are consistent
        if self.names_x != other.names_x:
            raise ValueError("names_x are inconsistent between populations")
        if self.names_f != other.names_f:
            raise ValueError("names_f are inconsistent between populations")
        if self.names_g != other.names_g:
            raise ValueError("names_g are inconsistent between populations")
        if self.obj_directions != other.obj_directions:
            raise ValueError("obj_directions are inconsistent between populations")
        if self.constraint_directions != other.constraint_directions:
            raise ValueError("constraint_directions are inconsistent between populations")
        if not np.array_equal(self.constraint_targets, other.constraint_targets):
            raise ValueError("constraint_targets are inconsistent between populations")

        # Concatenate the arrays along the batch dimension (axis=0)
        new_x = np.concatenate((self.x, other.x), axis=0)
        new_f = np.concatenate((self.f, other.f), axis=0)
        new_g = np.concatenate((self.g, other.g), axis=0)

        # Unique the arrays
        _, indices = np.unique(np.concatenate([new_x, new_f, new_g], axis=1), return_index=True, axis=0)
        new_x = new_x[indices, :]
        new_f = new_f[indices, :]
        new_g = new_g[indices, :]

        # Set fevals to the maximum of the two fevals values
        new_feval = self.fevals + other.fevals

        # Return a new Population instance
        return Population(
            x=new_x,
            f=new_f,
            g=new_g,
            fevals=new_feval,
            names_x=self.names_x,
            names_f=self.names_f,
            names_g=self.names_g,
            obj_directions=self.obj_directions,
            constraint_directions=self.constraint_directions,
            constraint_targets=self.constraint_targets,
        )

    def __getitem__(self, idx: Union[slice, np.ndarray, List[int]]) -> "Population":
        """
        Indexing operator to select along the batch dimension in the arrays.

        Parameters
        ----------
        idx : slice, np.ndarray, or list of ints
            The indices used to select along the batch dimension.

        Returns
        -------
        Population
            A new Population instance containing the selected individuals.
        """
        return Population(
            x=self.x[idx],
            f=self.f[idx],
            g=self.g[idx],
            fevals=self.fevals,
            names_x=self.names_x,
            names_f=self.names_f,
            names_g=self.names_g,
            obj_directions=self.obj_directions,
            constraint_directions=self.constraint_directions,
            constraint_targets=self.constraint_targets,
        )

    def get_nondominated_indices(self):
        """
        Returns a boolean array of whether or not an individual is non-dominated.
        """
        return np.sum(get_domination(self.f_canonical, self.g_canonical), axis=0) == 0

    def get_feasible_indices(self):
        if self.g.shape[1] == 0:
            return np.ones((len(self)), dtype=bool)
        return np.all(self.g_canonical <= 0.0, axis=1)

    def get_nondominated_set(self):
        return self[self.get_nondominated_indices()]

    @classmethod
    def from_random(
        cls,
        n_objectives: int,
        n_decision_vars: int,
        n_constraints: int,
        pop_size: int,
        fevals: int = 0,
        generate_names: bool = False,
        generate_obj_constraint_settings: bool = False,
    ) -> "Population":
        """
        Generate a randomized instance of the Population class.

        Parameters
        ----------
        n_objectives : int
            The number of objectives for each individual.
        n_decision_vars : int
            The number of decision variables for each individual.
        n_constraints : int
            The number of inequality constraints for each individual.
        pop_size : int
            The number of individuals in the population.
        fevals : int, optional
            The number of evaluations of the objective functions performed to reach this state, by default 0.
        generate_names : bool, optional
            Whether to include names for the decision variables, objectives, and constraints, by default False.
        generate_obj_constraint_settings : bool, optional
            Randomize the objective and constraint settings, default to minimization problem and g >= 0 constraint

        Returns
        -------
        Population
            An instance of the Population class with random values for decision variables (`x`), objectives (`f`),
            and inequality constraints (`g`). Optionally, names for these components can be included.

        Examples
        --------
        >>> random_population = Population.from_random(n_objectives=3, n_decision_vars=5, n_constraints=2, pop_size=10)
        >>> print(random_population.x.shape)
        (10, 5)
        >>> print(random_population.f.shape)
        (10, 3)
        >>> print(random_population.g.shape)
        (10, 2)
        >>> print(random_population.fevals)
        0
        """
        x = np.random.rand(pop_size, n_decision_vars)
        f = np.random.rand(pop_size, n_objectives)
        g = np.random.rand(pop_size, n_constraints) - 0.5 if n_constraints > 0 else np.empty((pop_size, 0))

        # Optionally generate names if generate_names is True
        if generate_names:
            names_x = [f"x{i+1}" for i in range(n_decision_vars)]
            names_f = [f"f{i+1}" for i in range(n_objectives)]
            names_g = [f"g{i+1}" for i in range(n_constraints)]
        else:
            names_x = None
            names_f = None
            names_g = None

        # Create randomized settings for objectives/constraints
        if generate_obj_constraint_settings:
            obj_directions = "".join(np.random.choice(["+", "-"], size=n_objectives))
            constraint_directions = "".join(np.random.choice([">", "<"], size=n_constraints))
            constraint_targets = np.random.rand(n_constraints) - 0.5
        else:
            obj_directions = None
            constraint_directions = None
            constraint_targets = None

        return cls(
            x=x,
            f=f,
            g=g,
            fevals=fevals,
            names_x=names_x,
            names_f=names_f,
            names_g=names_g,
            obj_directions=obj_directions,
            constraint_directions=constraint_directions,
            constraint_targets=constraint_targets,
        )

    def __len__(self):
        return self.x.shape[0]

    def _get_constraint_direction_str(self) -> str:
        # Create a list of >/< symbols with their targets
        return "[" + ",".join(f"{d}{t:.1e}" for d, t in zip(self.constraint_directions, self.constraint_targets)) + "]"

    def __repr__(self) -> str:
        return (
            f"Population(size={len(self)}, "
            f"vars={self.x.shape[1]}, "
            f"objs=[{self.obj_directions}], "
            f"cons={self._get_constraint_direction_str()}, "
            f"fevals={self.fevals})"
        )

    def __str__(self):
        return self.__repr__()

    def count_unique_individuals(self, decimals=13):
        """
        Calculates the number of unique individuals in the population. Uses `np.round` to avoid floating point accuracy issues.

        Parameters
        ----------
        decimals : int, optional
            Number of digits to which we will compare the values, by default 13

        Returns
        -------
        int
            The number of unique individuals
        """
        features = np.concatenate((self.x, self.f, self.g), axis=1)
        return np.unique(features.round(decimals=decimals), axis=0).shape[0]

    @property
    def n(self):
        """
        The number of decision variables.
        """
        return self.x.shape[1]

    @property
    def m(self):
        """
        The number of objectives.
        """
        return self.f.shape[1]

    @property
    def n_constraints(self):
        """
        The number of constraints.
        """
        return self.g.shape[1]

    def plot_obj_scatter(
        self,
        fig=None,
        ax=None,
        domination_filt: Literal["all", "dominated", "non-dominated"] = "all",
        feasibility_filt: Literal["all", "feasible", "infeasible"] = "all",
        show_points: bool = True,
        problem: Optional[Union[str, "Problem"]] = None,
        n_pf: int = 1000,
        pf_objectives: Optional[np.ndarray] = None,
        show_attainment: bool = False,
        show_dominated_area: bool = False,
        dominated_area_zorder: Optional[int] = -2,
        ref_point: Optional[Tuple[float, float]] = None,
        ref_point_padding: float = 0.05,
        label: Optional[str] = None,
        legend_loc: Optional[str] = None,
        show_names: bool = True,
        color: Optional[str] = None,
        scale: Optional[np.ndarray] = None,
        flip_objs: bool = False,
    ):
        """
        Plot the objectives in 2D and 3D.

        Parameters
        ----------
        fig : matplotlib figure, optional
            Figure to plot on, by default None
        ax : matplotlib axis, optional
            Axis to plot on, by default None
        domination_filt : Literal["all", "dominated", "non-dominated"], optional
            Plot only the dominated/non-dominated solutions, or all. Defaults to all
        feasibility_filt : Literal['all', 'feasible', 'infeasible'], optional
            Plot only the feasible/infeasible solutions, or all. Defaults to all
        show_points : bool
            Whether to actually show the points (useful for only showing attainment surface or dominated region)
        problem : str/Problem, optional
            Name of the problem for Pareto front plotting, by default None
        n_pf : int, optional
            The number of points used for plotting the Pareto front (when problem allows user selectable number of points)
        pf_objectives : array-like, optional
            User-specified Pareto front objectives. Should be a 2D array where each row represents a point
            on the Pareto front and each column represents an objective value.
        show_attainment : bool, optional
            Whether to plot the attainment surface, by default False
        show_dominated_area : bool, optional
            Plots the dominated region towards the larger values of each decision var
        dominated_area_zorder : int, optional
            What "zorder" to draw dominated region at. Mostly used internally to correctly show dominated area in history plots.
        ref_point : Union[str, Tuple[float, float]], optional
            Where to stop plotting the dominated region / attainment surface. Must be a point to the upper right (increasing
            value of objectives in 3D) of all plotted points. By default, will set to right of max of each objective plus
            padding.
        ref_point_padding : float
            Amount of padding to apply to the automatic reference point calculation.
        label : str, optional
            The label for these points, if shown in a legend
        legend_loc : str, optional
            Passed to `loc` argument of plt.legend
        show_names : bool, optional
            Whether to show the names of the objectives if provided by population
        color : str, optional
            What color should we use for the points. Defaults to selecting from matplotlib color cycler
        scale : array-like, optional
            Scale factors for each objective. Must have the same length as the number of objectives.
            If None, no scaling is applied.
        flip_objs : bool, optional
            Flips the order the objectives are plotted in. IE swaps axes in 2D, reverse them in 3D.

        Returns
        -------
        matplotlib figure and matplotlib axis
            The figure and axis containing the objectives plot
        """
        from .plotting import population_obj_scatter

        return population_obj_scatter(
            self,
            fig=fig,
            ax=ax,
            domination_filt=domination_filt,
            feasibility_filt=feasibility_filt,
            show_points=show_points,
            problem=problem,
            n_pf=n_pf,
            pf_objectives=pf_objectives,
            show_attainment=show_attainment,
            show_dominated_area=show_dominated_area,
            dominated_area_zorder=dominated_area_zorder,
            ref_point=ref_point,
            ref_point_padding=ref_point_padding,
            label=label,
            legend_loc=legend_loc,
            show_names=show_names,
            color=color,
            scale=scale,
            flip_objs=flip_objs,
        )

    def plot_dvar_pairs(
        self,
        dvars: Optional[Union[int, slice, List[int], Tuple[int, int]]] = None,
        fig=None,
        axes=None,
        domination_filt: Literal["all", "dominated", "non-dominated"] = "all",
        feasibility_filt: Literal["all", "feasible", "infeasible"] = "all",
        hist_bins: Optional[int] = None,
        show_names: bool = True,
        problem: Optional[Union[str, "Problem"]] = None,
        lower_bounds: Optional[np.ndarray] = None,
        upper_bounds: Optional[np.ndarray] = None,
        color: Optional[str] = None,
        scale: Optional[np.ndarray] = None,
    ):
        """
        Creates a pairs plot (scatter matrix) showing correlations between decision variables
        and their distributions.

        Parameters
        ----------
        dvars : int, slice, List[int], or Tuple[int, int], optional
            Specifies which decision variables to plot. See `selection_to_indices` for more details.
        fig : matplotlib.figure.Figure, optional
            Figure to plot on. If None and axes is None, creates a new figure.
        axes : numpy.ndarray of matplotlib.axes.Axes, optional
            2D array of axes to plot on. If None and fig is None, creates new axes.
            Must be provided if fig is provided and vice versa.
        domination_filt : Literal["all", "dominated", "non-dominated"], optional
            Plot only the dominated/non-dominated solutions, or all. Defaults to all
        feasibility_filt : Literal['all', 'feasible', 'infeasible'], optional
            Plot only the feasible/infeasible solutions, or all. Defaults to all
        hist_bins : int, optional
            Number of bins for histograms on the diagonal, default is to let matplotlib choose
        show_names : bool, optional
            Whether to include variable names on the axes if they exist, default is True
        problem : str/Problem, optional
            The problem for plotting decision variable bounds
        lower_bounds : array-like, optional
            Lower bounds for each decision variable
        upper_bounds : array-like, optional
            Upper bounds for each decision variable
        color : str, optional
            What color should we use for the points. Defaults to selecting from matplotlib color cycler
        scale : array-like, optional
            Scale factors for each variable. Must have the same length as the number of decision vars.
            If None, no scaling is applied.

        Returns
        -------
        tuple
            (matplotlib.figure.Figure, numpy.ndarray of matplotlib.axes.Axes)
            The figure and axes containing the pairs plot
        """
        from .plotting import population_dvar_pairs

        return population_dvar_pairs(
            self,
            dvars=dvars,
            fig=fig,
            axes=axes,
            domination_filt=domination_filt,
            feasibility_filt=feasibility_filt,
            hist_bins=hist_bins,
            show_names=show_names,
            problem=problem,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            color=color,
            scale=scale,
        )


class History(BaseModel):
    """
    Contains populations output from a multiobjective genetic algorithm at some reporting interval in order to track
    its history as it converges to a solution.

    Assumptions:
     - All reports must have a consistent number of objectives, decision variables, and constraints.
     - Objective/constraint settings and names, if used, must be consistent across populations
    """

    reports: List[Population]
    problem: str
    metadata: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_consistent_populations(self):
        """
        Makes sure that all populations have the same number of objectives, constraints, and decision variables.
        """
        n_decision_vars = [x.x.shape[1] for x in self.reports]
        n_objectives = [x.f.shape[1] for x in self.reports]
        n_constraints = [x.g.shape[1] for x in self.reports]

        if n_decision_vars and len(set(n_decision_vars)) != 1:
            raise ValueError(f"Inconsistent number of decision variables in reports: {n_decision_vars}")
        if n_objectives and len(set(n_objectives)) != 1:
            raise ValueError(f"Inconsistent number of objectives in reports: {n_objectives}")
        if n_constraints and len(set(n_constraints)) != 1:
            raise ValueError(f"Inconsistent number of constraints in reports: {n_constraints}")

        # Validate consistency of names across populations
        names_x = [tuple(x.names_x) if x.names_x is not None else None for x in self.reports]
        names_f = [tuple(x.names_f) if x.names_f is not None else None for x in self.reports]
        names_g = [tuple(x.names_g) if x.names_g is not None else None for x in self.reports]

        # If names are provided, check consistency
        if names_x and len(set(names_x)) != 1:
            raise ValueError(f"Inconsistent names for decision variables in reports: {names_x}")
        if names_f and len(set(names_f)) != 1:
            raise ValueError(f"Inconsistent names for objectives in reports: {names_f}")
        if names_g and len(set(names_g)) != 1:
            raise ValueError(f"Inconsistent names for constraints in reports: {names_g}")

        # Check settings for objectives and constraints
        obj_directions = [x.obj_directions for x in self.reports]
        constraint_directions = [x.constraint_directions for x in self.reports]
        constraint_targets = [tuple(x.constraint_targets) for x in self.reports]
        if obj_directions and len(set(obj_directions)) != 1:
            raise ValueError(f"Inconsistent obj_directions in reports: {obj_directions}")
        if constraint_directions and len(set(constraint_directions)) != 1:
            raise ValueError(f"Inconsistent constraint_directions in reports: {constraint_directions}")
        if constraint_targets and len(set(constraint_targets)) != 1:
            raise ValueError(f"Inconsistent constraint_targets in reports: {constraint_targets}")

        return self

    def __eq__(self, other):
        if not isinstance(other, History):
            return False
        return self.reports == other.reports and self.problem == other.problem and self.metadata == other.metadata

    @classmethod
    def from_random(
        cls,
        n_populations: int,
        n_objectives: int,
        n_decision_vars: int,
        n_constraints: int,
        pop_size: int,
        generate_names: bool = False,
        generate_obj_constraint_settings: bool = False,
    ) -> "History":
        """
        Generate a randomized instance of the History class, including random problem name and metadata.

        Parameters
        ----------
        n_populations : int
            The number of populations (reports) to generate.
        n_objectives : int
            The number of objectives for each individual in each population.
        n_decision_vars : int
            The number of decision variables for each individual in each population.
        n_constraints : int
            The number of inequality constraints for each individual in each population.
        pop_size : int
            The number of individuals in each population.
        generate_names : bool, optional
            Whether to include names for the decision variables, objectives, and constraints, by default False.
        generate_obj_constraint_settings : bool, optional
            Randomize the objective and constraint settings, default to minimization problem and g >= 0 constraint

        Returns
        -------
        History
            An instance of the History class with a list of randomly generated Population instances, a random problem name, and
            random metadata.
        """
        # Randomly generate a problem name
        problem = f"Optimization_Problem_{random.randint(1000, 9999)}"

        # Randomly generate metadata
        metadata_keys = ["author", "version", "description", "date"]
        metadata_values = [
            "".join(random.choices(string.ascii_uppercase + string.digits, k=5)),
            random.uniform(1.0, 3.0),
            "Randomly generated metadata",
            f"{random.randint(2020, 2024)}-0{random.randint(1, 9)}-{random.randint(10, 28)}",
        ]
        metadata = dict(zip(metadata_keys, metadata_values))

        # Generate populations with increasing fevals values
        reports = [
            Population.from_random(
                n_objectives,
                n_decision_vars,
                n_constraints,
                pop_size,
                fevals=(i + 1) * pop_size,
                generate_names=generate_names,
            )
            for i in range(n_populations)
        ]

        # Create randomized settings for objectives/constraints (must be consistent between objects)
        if generate_obj_constraint_settings:
            obj_directions = "".join(np.random.choice(["+", "-"], size=n_objectives))
            constraint_directions = "".join(np.random.choice([">", "<"], size=n_constraints))
            constraint_targets = np.random.rand(n_constraints)
            for report in reports:
                report.obj_directions = obj_directions
                report.constraint_directions = constraint_directions
                report.constraint_targets = constraint_targets

        return cls(reports=reports, problem=problem, metadata=metadata)

    def _to_h5py_group(self, g: h5py.Group):
        """
        Store the history object in an HDF5 group. The populations are concatenated together and stored together as a few small
        arrays rather than their own groups to limit the number of python calls to the HDF5 API. This is done for performance
        reasons and the speed of both variants (concatenated ararys and populations in groups) were benchmarked with the
        concatenated arrays performing up to 10x faster on combined write/read tests.

        Parameters
        ----------
        g : h5py.Group
            The h5py group to write our data to.
        """
        # Save the metadata
        g.attrs["problem"] = self.problem
        g_md = g.create_group("metadata")
        for k, v in self.metadata.items():
            g_md.attrs[k] = v

        # Save data from each population into one bigger dataset to reduce API calls to HDF5 file reader
        g.attrs["pop_sizes"] = [len(r) for r in self.reports]
        g.attrs["fevals"] = [r.fevals for r in self.reports]
        if self.reports:
            g["x"] = np.concatenate([r.x for r in self.reports], axis=0)
            g["f"] = np.concatenate([r.f for r in self.reports], axis=0)
            g["g"] = np.concatenate([r.g for r in self.reports], axis=0)
        else:
            g["x"] = np.empty(())
            g["f"] = np.empty(())
            g["g"] = np.empty(())

        # Save names
        if self.reports and self.reports[0].names_x is not None:
            g["x"].attrs["names"] = self.reports[0].names_x
        if self.reports and self.reports[0].names_f is not None:
            g["f"].attrs["names"] = self.reports[0].names_f
        if self.reports and self.reports[0].names_g is not None:
            g["g"].attrs["names"] = self.reports[0].names_g

        # Save the configuration data
        if self.reports:
            g["f"].attrs["directions"] = self.reports[0].obj_directions
            g["g"].attrs["directions"] = self.reports[0].constraint_directions
            g["g"].attrs["targets"] = self.reports[0].constraint_targets

    @classmethod
    def _from_h5py_group(cls, grp: h5py.Group, file_version: str):
        """
        Construct a new History object from data in an HDF5 group.

        Parameters
        ----------
        grp : h5py.Group
            The group containing the history data.

        Returns
        -------
        History
            The loaded history object
        """
        # Get the decision vars, objectives, and constraints
        x = grp["x"][()]
        f = grp["f"][()]
        g = grp["g"][()]

        # Get the objective / constraint settings
        obj_directions = grp["f"].attrs.get("directions", None)
        constraint_directions = grp["g"].attrs.get("directions", None)
        constraint_targets = grp["g"].attrs.get("targets", None)

        # Before file version 1.1.0 which introduced explicit constraint directions,
        # the default constraint type was g(x) >= 0.0
        if file_version == "1.0.0":
            constraint_directions = ">" * g.shape[1]

        # Get the names
        names_x = grp["x"].attrs.get("names", None)
        names_f = grp["f"].attrs.get("names", None)
        names_g = grp["g"].attrs.get("names", None)

        # Create the population objects
        start_idx = 0
        reports = []
        for pop_size, fevals in zip(grp.attrs["pop_sizes"], grp.attrs["fevals"]):
            reports.append(
                Population(
                    x=x[start_idx : start_idx + pop_size],
                    f=f[start_idx : start_idx + pop_size],
                    g=g[start_idx : start_idx + pop_size],
                    fevals=fevals,
                    names_x=names_x,
                    names_f=names_f,
                    names_g=names_g,
                    obj_directions=obj_directions,
                    constraint_directions=constraint_directions,
                    constraint_targets=constraint_targets,
                )
            )
            start_idx += pop_size

        # Return as a history object
        return cls(
            problem=grp.attrs["problem"],
            reports=reports,
            metadata={k: v for k, v in grp["metadata"].attrs.items()},
        )

    def to_nondominated(self):
        """
        Returns a history object with the same number of population objects, but the individuals in each generation are the
        nondominated solutions seen up to this point. The function evaluation count is unchanged.

        Returns
        -------
        History
            History object containing the nondominated solution
        """
        if len(self.reports) < 2:
            return self

        def pf_reduce(a, b):
            return a + [(a[-1] + b).get_nondominated_set()]

        # Get the nondominated objectives
        new_reports = reduce(pf_reduce, self.reports[1:], [self.reports[0].get_nondominated_set()])

        # Make sure fevals carries over
        for n, o in zip(new_reports, self.reports):
            n.fevals = o.fevals

        return History(reports=new_reports, problem=self.problem, metadata=self.metadata.copy())

    def plot_obj_scatter(
        self,
        reports: Optional[Union[int, slice, List[int], Tuple[int, int]]] = None,
        fig=None,
        ax=None,
        domination_filt: Literal["all", "dominated", "non-dominated"] = "all",
        feasibility_filt: Literal["all", "feasible", "infeasible"] = "all",
        show_points: bool = True,
        n_pf: int = 1000,
        pf_objectives: Optional[np.ndarray] = None,
        show_attainment: bool = False,
        show_dominated_area: bool = False,
        ref_point: Optional[Tuple[float, float]] = None,
        ref_point_padding: float = 0.05,
        legend_loc: Optional[str] = None,
        scale: Optional[np.ndarray] = None,
        flip_objs: bool = False,
        show_names: bool = True,
        show_pf: bool = False,
        colormap: str = "viridis",
        cmap_label: Optional[str] = None,
        generation_mode: Literal["cmap", "cumulative"] = "cmap",
        single_color: Optional[str] = None,
        label_mode: Literal["index", "fevals"] = "index",
    ):
        """
        Plot the objectives from this history of populations, using either a colormap for generations
        or merging all generations into a single population.

        Parameters
        ----------
        reports : int, slice, List[int], or Tuple[int, int], optional
            Specifies which generations to plot. See `selection_to_indices` for more details.
        fig : matplotlib figure, optional
            Figure to plot on, by default None
        ax : matplotlib axis, optional
            Axis to plot on, by default None
        domination_filt : Literal["all", "dominated", "non-dominated"], optional
            Plot only the dominated/non-dominated solutions, or all. Defaults to all
        feasibility_filt : Literal['all', 'feasible', 'infeasible'], optional
            Plot only the feasible/infeasible solutions, or all. Defaults to all
        show_points : bool
            Whether to actually show the points (useful for only showing attainment surface or dominated region)
        n_pf : int, optional
            The number of points used for plotting the Pareto front (when problem allows user selectable number of points)
        pf_objectives : array-like, optional
            User-specified Pareto front objectives. Should be a 2D array where each row represents a point
            on the Pareto front and each column represents an objective value.
        show_attainment : bool, optional
            Whether to plot the attainment surface, by default False
        show_dominated_area : bool, optional
            Plots the dominated region towards the larger values of each decision var
        ref_point : Union[str, Tuple[float, float]], optional
            Where to stop plotting the dominated region / attainment surface. Must be a point to the upper right (increasing
            value of objectives in 3D) of all plotted points. By default, will set to right of max of each objective plus
            padding.
        ref_point_padding : float
            Amount of padding to apply to the automatic reference point calculation.
        legend_loc : str, optional
            Passed to `loc` argument of plt.legend
        scale : array-like, optional
            Scale factors for each objective. Must have the same length as the number of objectives.
            If None, no scaling is applied.
        flip_objs : bool, optional
            Flips the order the objectives are plotted in. IE swaps axes in 2D, reverse them in 3D.
        show_names : bool, optional
            Whether to show the names of the objectives if provided by population
        show_pf : bool, optional
            Whether to plot the Pareto front, by default True
        colormap : str, optional
            Name of the colormap to use for generation colors, by default 'viridis'
        cmap_label: Optional[str] = "Generation"
            Label for colorbar (only used when generation_mode is 'cmap')
        generation_mode: Literal['cmap', 'cumulative'] = 'cmap'
            How to handle multiple generations:
            'cmap': Plot each generation separately with colors from colormap
            'cumulative': Merge all selected generations into single population
        single_color: Optional[str] = None
            Color to use when generation_mode is 'cumulative'. If None, uses default color from matplotlib.
        label_mode: Literal['index', 'fevals'] = 'index'
            Whether to use report index or function evaluations (fevals) for labels

        Returns
        -------
        matplotlib figure and matplotlib axis
            The figure and axis containing the history plot
        """
        from .plotting import history_obj_scatter

        return history_obj_scatter(
            self,
            reports=reports,
            fig=fig,
            ax=ax,
            domination_filt=domination_filt,
            feasibility_filt=feasibility_filt,
            show_points=show_points,
            n_pf=n_pf,
            pf_objectives=pf_objectives,
            show_attainment=show_attainment,
            show_dominated_area=show_dominated_area,
            ref_point=ref_point,
            ref_point_padding=ref_point_padding,
            legend_loc=legend_loc,
            scale=scale,
            flip_objs=flip_objs,
            show_names=show_names,
            show_pf=show_pf,
            colormap=colormap,
            cmap_label=cmap_label,
            generation_mode=generation_mode,
            single_color=single_color,
            label_mode=label_mode,
        )

    def plot_dvar_pairs(
        self,
        reports: Optional[Union[int, slice, List[int], Tuple[int, int]]] = None,
        dvars: Optional[Union[int, slice, List[int], Tuple[int, int]]] = None,
        fig=None,
        axes=None,
        domination_filt: Literal["all", "dominated", "non-dominated"] = "all",
        feasibility_filt: Literal["all", "feasible", "infeasible"] = "all",
        hist_bins: Optional[int] = None,
        show_names: bool = True,
        lower_bounds: Optional[np.ndarray] = None,
        upper_bounds: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        colormap: str = "viridis",
        cmap_label: Optional[str] = None,
        generation_mode: Literal["cmap", "cumulative"] = "cmap",
        single_color: Optional[str] = None,
        plot_bounds: bool = False,
        label_mode: Literal["index", "fevals"] = "index",
    ):
        """
        Plot the decision variables from this history of populations, using either a colormap
        for generations or merging all generations into a single population.

        Parameters
        ----------
        reports : int, slice, List[int], or Tuple[int, int], optional
            Specifies which generations to plot. See `selection_to_indices` for more details.
        dvars : int, slice, List[int], or Tuple[int, int], optional
            Which decision vars to plot. See `population_dvar_pairs` docstring for more details.
        fig : matplotlib figure, optional
            Figure to plot on, by default None
        axes : array of matplotlib axes, optional
            Axes to plot on, by default None
        domination_filt : Literal["all", "dominated", "non-dominated"], optional
            Plot only the dominated/non-dominated solutions, or all. Defaults to all
        feasibility_filt : Literal['all', 'feasible', 'infeasible'], optional
            Plot only the feasible/infeasible solutions, or all. Defaults to all
        hist_bins : int, optional
            Number of bins for histograms on the diagonal, default is to let matplotlib choose
        show_names : bool, optional
            Whether to include variable names on the axes if they exist, default is True
        lower_bounds : array-like, optional
            Lower bounds for each decision variable
        upper_bounds : array-like, optional
            Upper bounds for each decision variable
        scale : array-like, optional
            Scale factors for each variable. Must have the same length as the number of decision vars.
            If None, no scaling is applied.
        colormap : str, optional
            Name of the colormap to use for generation colors, by default 'viridis'
        cmap_label: Optional[str] = "Generation"
            Label for colorbar (only used when generation_mode is 'cmap')
        generation_mode: Literal['cmap', 'cumulative'] = 'cmap'
            How to handle multiple generations:
            'cmap': Plot each generation separately with colors from colormap
            'cumulative': Merge all selected generations into single population
        single_color: Optional[str] = None
            Color to use when generation_mode is 'cumulative'. If None, uses default color from matplotlib.
        plot_bounds: bool = False
            Whether to plot bounds for the problem
        label_mode: Literal['index', 'fevals'] = 'index'
            Whether to use report index or function evaluations (fevals) for labels

        Returns
        -------
        matplotlib figure and array of matplotlib axes
            The figure and axes containing the history plot
        """
        from .plotting import history_dvar_pairs

        return history_dvar_pairs(
            self,
            reports=reports,
            dvars=dvars,
            fig=fig,
            axes=axes,
            domination_filt=domination_filt,
            feasibility_filt=feasibility_filt,
            hist_bins=hist_bins,
            show_names=show_names,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            scale=scale,
            colormap=colormap,
            cmap_label=cmap_label,
            generation_mode=generation_mode,
            single_color=single_color,
            plot_bounds=plot_bounds,
            label_mode=label_mode,
        )

    def plot_obj_animation(
        self,
        reports: Optional[Union[int, slice, List[int], Tuple[int, int]]] = None,
        interval: int = 200,
        domination_filt: Literal["all", "dominated", "non-dominated"] = "all",
        feasibility_filt: Literal["all", "feasible", "infeasible"] = "all",
        show_points: bool = True,
        n_pf: int = 1000,
        pf_objectives: Optional[np.ndarray] = None,
        show_attainment: bool = False,
        show_dominated_area: bool = False,
        ref_point: Optional[Tuple[float, float]] = None,
        ref_point_padding: float = 0.05,
        legend_loc: Optional[str] = "upper right",
        scale: Optional[np.ndarray] = None,
        flip_objs: bool = False,
        show_names: bool = True,
        show_pf: bool = False,
        single_color: Optional[str] = None,
        dynamic_scaling: bool = False,
        cumulative: bool = False,
        scale_padding: float = 0.05,
    ):
        """
        Creates an animated visualization of how the Pareto front evolves across generations.

        Parameters
        ----------
        reports : int, slice, List[int], or Tuple[int, int], optional
            Specifies which generations to animate. See `selection_to_indices` for more details.
        interval : int, optional
            Delay between frames in milliseconds, by default 200
        domination_filt : Literal["all", "dominated", "non-dominated"], optional
            Plot only the dominated/non-dominated solutions, or all. Defaults to all
        feasibility_filt : Literal['all', 'feasible', 'infeasible'], optional
            Plot only the feasible/infeasible solutions, or all. Defaults to all
        show_points : bool
            Whether to actually show the points (useful for only showing attainment surface or dominated region)
        n_pf : int, optional
            The number of points used for plotting the Pareto front (when problem allows user selectable number of points)
        pf_objectives : array-like, optional
            User-specified Pareto front objectives. Should be a 2D array where each row represents a point
            on the Pareto front and each column represents an objective value.
        show_attainment : bool, optional
            Whether to plot the attainment surface, by default False
        show_dominated_area : bool, optional
            Plots the dominated region towards the larger values of each decision var
        ref_point : Union[str, Tuple[float, float]], optional
            Where to stop plotting the dominated region / attainment surface. Must be a point to the upper right (increasing
            value of objectives in 3D) of all plotted points. By default, will set to right of max of each objective plus
            padding.
        ref_point_padding : float
            Amount of padding to apply to the automatic reference point calculation.
        legend_loc : str, optional
            Passed to `loc` argument of plt.legend
        scale : array-like, optional
            Scale factors for each objective. Must have the same length as the number of objectives.
            If None, no scaling is applied.
        flip_objs : bool, optional
            Flips the order the objectives are plotted in. IE swaps axes in 2D, reverse them in 3D.
        show_names : bool, optional
            Whether to show the names of the objectives if provided by population
        show_pf : bool, optional
            Whether to plot the Pareto front, by default True
        single_color: Optional[str] = None
            Color to use when generation_mode is 'cumulative'. If None, uses default color from matplotlib.
        dynamic_scaling : bool, optional
            If True, axes limits will update based on each frame's data.
            If False, axes limits will be fixed based on all data, by default False
        cumulative : bool, optional
            If True, shows all points seen up to current frame.
            If False, shows only current frame's points, by default False
        scale_padding : float, optional
            Padding used when calculating axis limits w/ dynamic limits

        Returns
        -------
        animation.Animation
            The animation object that can be displayed in notebooks or saved to file
        """
        from .plotting import history_obj_animation

        return history_obj_animation(
            self,
            reports=reports,
            interval=interval,
            domination_filt=domination_filt,
            feasibility_filt=feasibility_filt,
            show_points=show_points,
            n_pf=n_pf,
            pf_objectives=pf_objectives,
            show_attainment=show_attainment,
            show_dominated_area=show_dominated_area,
            ref_point=ref_point,
            ref_point_padding=ref_point_padding,
            legend_loc=legend_loc,
            scale=scale,
            flip_objs=flip_objs,
            show_names=show_names,
            show_pf=show_pf,
            single_color=single_color,
            dynamic_scaling=dynamic_scaling,
            cumulative=cumulative,
            scale_padding=scale_padding,
        )

    def plot_dvar_animation(
        self,
        reports: Optional[Union[int, slice, List[int], Tuple[int, int]]] = None,
        dvars: Optional[Union[int, slice, List[int], Tuple[int, int]]] = None,
        interval: int = 200,
        domination_filt: Literal["all", "dominated", "non-dominated"] = "all",
        feasibility_filt: Literal["all", "feasible", "infeasible"] = "all",
        hist_bins: Optional[int] = None,
        show_names: bool = True,
        lower_bounds: Optional[np.ndarray] = None,
        upper_bounds: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        single_color: Optional[str] = None,
        plot_bounds: bool = False,
        dynamic_scaling: bool = False,
        cumulative: bool = False,
        scale_padding=0.05,
    ):
        """
        Creates an animated visualization of how the decision variables evolve across generations.

        Parameters
        ----------
        reports : int, slice, List[int], or Tuple[int, int], optional
            Specifies which generations to animate. See `selection_to_indices` for more details.
        dvars : int, slice, List[int], or Tuple[int, int], optional
            Which decision vars to plot. See `population_dvar_pairs` docstring for more details.
        interval : int, optional
            Delay between frames in milliseconds, by default 200
        domination_filt : Literal["all", "dominated", "non-dominated"], optional
            Plot only the dominated/non-dominated solutions, or all. Defaults to all
        feasibility_filt : Literal['all', 'feasible', 'infeasible'], optional
            Plot only the feasible/infeasible solutions, or all. Defaults to all
        hist_bins : int, optional
            Number of bins for histograms on the diagonal, default is to let matplotlib choose
        show_names : bool, optional
            Whether to include variable names on the axes if they exist, default is True
        lower_bounds : array-like, optional
            Lower bounds for each decision variable
        upper_bounds : array-like, optional
            Upper bounds for each decision variable
        scale : array-like, optional
            Scale factors for each variable. Must have the same length as the number of decision vars.
            If None, no scaling is applied.
        single_color: Optional[str] = None
            Color to use when generation_mode is 'cumulative'. If None, uses default color from matplotlib.
        plot_bounds: bool = False
            Whether to plot bounds for the problem
        dynamic_scaling : bool, optional
            If True, axes limits will update based on each frame's data.
            If False, axes limits will be fixed based on all data, by default False
        cumulative : bool, optional
            If True, shows all points seen up to current frame.
            If False, shows only current frame's points, by default False
        scale_padding : float, optional
            Padding used when calculating axis limits w/ dynamic limits

        Returns
        -------
        animation.Animation
            The animation object that can be displayed in notebooks or saved to file
        """
        from .plotting import history_dvar_animation

        return history_dvar_animation(
            self,
            reports=reports,
            dvars=dvars,
            interval=interval,
            domination_filt=domination_filt,
            feasibility_filt=feasibility_filt,
            hist_bins=hist_bins,
            show_names=show_names,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            scale=scale,
            single_color=single_color,
            plot_bounds=plot_bounds,
            dynamic_scaling=dynamic_scaling,
            cumulative=cumulative,
            scale_padding=scale_padding,
        )

    def plot_metric(
        self,
        metric: "Metric",
        x_axis: Literal["fevals", "generation"] = "fevals",
        fig=None,
        ax=None,
    ):
        """
        Evaluate and plot evolution of a metric across the populations within this history.

        Parameters
        ----------
        metric : Metric
            The metric to evaluate and plot
        x_axis : Literal["fevals", "generation"]
            What value to use for x-axis in plot
        fig : matplotlib figure, optional
            Matplotlib figure if plotting to user-provided figure (must also specify axis)
        ax : matplotlib axis, optional
            Matplotlib axis to place plot into (must also specify figure)

        Returns
        -------
        matplotlib figure and matplotlib axis
            The matplotlib figure and axis the data was plotted to
        """
        from .plotting import plot_metric_history

        return plot_metric_history(self, metric, x_axis=x_axis, fig=fig, ax=ax)

    def __repr__(self) -> str:
        dims = (
            (
                f"vars={self.reports[0].x.shape[1]}, objs=[{self.reports[0].obj_directions}], "
                f"cons={self.reports[0]._get_constraint_direction_str()}"
            )
            if self.reports
            else "empty"
        )
        return f"History(problem='{self.problem}', reports={len(self.reports)}, {dims})"

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.reports)


class Experiment(BaseModel):
    """
    Represents on "experiment" performed on a multibojective genetic algorithm. It may contain several evaluations of the
    algorithm on different problems or repeated iterations on the same problem.

    Note: when loading files from disk, the `file_version` field will indicate the version of the file
    that was loaded. When the `Experiment` is saved, the resulting file will contain the file version
    used to save the data.
    """

    runs: List[History]
    name: str
    author: str = ""
    software: str = ""
    software_version: str = ""
    comment: str = ""
    creation_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    file_version: str = "1.1.0"

    def __eq__(self, other):
        if not isinstance(other, Experiment):
            return False
        return (
            self.name == other.name
            and self.author == other.author
            and self.software == other.software
            and self.software_version == other.software_version
            and self.comment == other.comment
            and self.creation_time == other.creation_time
            and self.runs == other.runs
        )

    def __repr__(self) -> str:
        metadata = [
            f"name='{self.name}'",
            f"created='{self.creation_time.strftime('%Y-%m-%d')}'",
        ]
        if self.author:
            metadata.append(f"author='{self.author}'")
        if self.software:
            version = f" {self.software_version}" if self.software_version else ""
            metadata.append(f"software='{self.software}{version}'")
        metadata.append(f"runs={len(self.runs)}")
        return f"Experiment({', '.join(metadata)})"

    def __str__(self):
        return self.__repr__()

    @classmethod
    def from_random(
        cls,
        n_histories: int,
        n_populations: int,
        n_objectives: int,
        n_decision_vars: int,
        n_constraints: int,
        pop_size: int,
        generate_names: bool = False,
        generate_obj_constraint_settings: bool = False,
    ) -> "Experiment":
        """
        Generate a randomized instance of the Experiment class.

        Parameters
        ----------
        n_histories : int
            The number of histories to generate.
        n_populations : int
            The number of populations in each history.
        n_objectives : int
            The number of objectives for each individual in each population.
        n_decision_vars : int
            The number of decision variables for each individual in each population.
        n_constraints : int
            The number of inequality constraints for each individual in each population.
        pop_size : int
            The number of individuals in each population.
        generate_names : bool, optional
            Whether to include names for the decision variables, objectives, and constraints, by default False.
        generate_obj_constraint_settings : bool, optional
            Randomize the objective and constraint settings, default to minimization problem and g >= 0 constraint

        Returns
        -------
        Experiment
            An instance of the Experiment class with a list of randomly generated History instances and a random name.
        """
        # Generate random histories
        runs = [
            History.from_random(
                n_populations,
                n_objectives,
                n_decision_vars,
                n_constraints,
                pop_size,
                generate_names=generate_names,
                generate_obj_constraint_settings=generate_obj_constraint_settings,
            )
            for _ in range(n_histories)
        ]

        # Randomly generate an name for the experiment
        name = f"Experiment_{random.randint(1000, 9999)}"

        # Generate random values or placeholders for other attributes
        author = f"Author_{random.randint(1, 100)}"
        software = f"Software_{random.randint(1, 10)}"
        software_version = f"{random.randint(1, 5)}.{random.randint(0, 9)}"
        comment = "Randomly generated experiment"

        return cls(
            runs=runs,
            name=name,
            author=author,
            software=software,
            software_version=software_version,
            comment=comment,
        )

    def save(self, fname):
        """
        Saves the experiment data into an HDF5 file at the specified filename.

        Parameters
        ----------
        fname : str
            Filename to save to
        """
        with h5py.File(fname, mode="w") as f:
            # Save metadata as attributes
            f.attrs["name"] = self.name
            f.attrs["author"] = self.author
            f.attrs["software"] = self.software
            f.attrs["software_version"] = self.software_version
            f.attrs["comment"] = self.comment
            f.attrs["creation_time"] = self.creation_time.isoformat()
            f.attrs["file_version"] = "1.1.0"
            f.attrs["file_format"] = "ParetoBench Multi-Objective Optimization Data"

            # Calculate the necessary zero padding based on the number of runs
            max_len = len(str(len(self.runs) - 1))

            # Save each run into its own group
            for idx, run in enumerate(self.runs):
                run._to_h5py_group(f.create_group(f"run_{idx:0{max_len}d}"))

    @classmethod
    def load(cls, fname):
        """
        Creates a new Experiment object from an HDF5 file on disk.

        Parameters
        ----------
        fname : str
            Filename to load the data from

        Returns
        -------
        Experiment
            The loaded experiment object

        Examples
        --------
        >>> exp = Experiment.load('state_of_the_art_algorithm_benchmarking_data.h5')
        >>> print(exp.name)
        NewAlg
        >>> print(len(exp.runs))
        64
        """
        # Load the data
        with h5py.File(fname, mode="r") as f:
            # Load each of the runs keeping track of the order of the indices
            file_version = f.attrs["file_version"]
            idx_runs = []
            for idx_str, run_grp in f.items():
                m = re.match(r"run_(\d+)", idx_str)
                if m:
                    idx_runs.append((int(m.group(1)), History._from_h5py_group(run_grp, file_version=file_version)))
            runs = [x[1] for x in sorted(idx_runs, key=lambda x: x[0])]

            # Convert the creation_time back to a timezone-aware datetime object
            creation_time = datetime.fromisoformat(f.attrs["creation_time"]).astimezone(timezone.utc)

            # Return as an experiment object
            return cls(
                runs=runs,
                name=f.attrs["name"],
                author=f.attrs["author"],
                software=f.attrs["software"],
                software_version=f.attrs["software_version"],
                comment=f.attrs["comment"],
                creation_time=creation_time,
                file_version=file_version,
            )
