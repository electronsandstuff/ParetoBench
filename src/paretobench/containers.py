from datetime import datetime, timezone
from functools import reduce
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from typing import List, Dict, Union, Any, Optional
import h5py
import numpy as np
import random
import re
import string


class Population(BaseModel):
    '''
    Stores the individuals in a population for one reporting interval in a genetic algorithm. Conventional names are used for
    the decision variables (x), the objectives (f), and inequality constraints (g). The first dimension of each array is the
    batch dimension. The number of evaluations of the objective functions performed to reach this state is also recorded.
    '''
    x: np.ndarray
    f: np.ndarray
    g: np.ndarray
    feval: int
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Optional lists of names for decision variables, objectives, and constraints
    names_x: Optional[List[str]] = None
    names_f: Optional[List[str]] = None
    names_g: Optional[List[str]] = None
    
    @model_validator(mode='before')
    @classmethod
    def set_default_arrays(cls, values):
        # Determine the batch size from the first non-None array
        batch_size = next((arr.shape[0] for arr in [values.get('x'), values.get('f'), values.get('g')] if arr is not None), None)
        if batch_size is None:
            raise ValueError('Must specify one of x, f, or g')

        # Set empty arrays for unspecified fields
        if values.get('x') is None:
            values['x'] = np.empty((batch_size, 0), dtype=np.float64)
        if values.get('f') is None:
            values['f'] = np.empty((batch_size, 0), dtype=np.float64)
        if values.get('g') is None:
            values['g'] = np.empty((batch_size, 0), dtype=np.float64)

        return values

    @model_validator(mode='after')
    def validate_batch_dimensions(self):
        # Validate batch dimensions
        x_size, f_size, g_size = self.x.shape[0], self.f.shape[0], self.g.shape[0]
        if len(set([x_size, f_size, g_size])) != 1:
            raise ValueError(f'Batch dimensions do not match (len(x)={x_size}, len(f)={f_size}, len(g)={g_size})')
        return self

    @model_validator(mode='after')
    def validate_names(self):
        if self.names_x and len(self.names_x) != self.x.shape[1]:
            raise ValueError("Length of names_x must match the number of decision variables in x.")
        if self.names_f and len(self.names_f) != self.f.shape[1]:
            raise ValueError("Length of names_f must match the number of objectives in f.")
        if self.names_g and len(self.names_g) != self.g.shape[1]:
            raise ValueError("Length of names_g must match the number of constraints in g.")
        return self
    
    @field_validator('x', 'f', 'g')
    @classmethod
    def validate_numpy_arrays(cls, value: np.ndarray, field: Any) -> np.ndarray:
        expected_dtype = np.float64
        expected_ndim = 2

        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected a NumPy array for field '{field.name}', got {type(value)}")
        if value.dtype != expected_dtype:
            raise TypeError(f"Expected array of type {expected_dtype} for field '{field.name}', got {value.dtype}")
        if value.ndim != expected_ndim:
            raise ValueError(f"Expected array with {expected_ndim} dimensions for field '{field.name}', got {value.ndim}")
        
        return value

    @field_validator('feval')
    @classmethod
    def validate_feval(cls, v):
        if v < 0:
            raise ValueError("feval must be a non-negative integer")
        return v
    
    def __eq__(self, other):
        if not isinstance(other, Population):
            return False
        return (np.array_equal(self.x, other.x) and
                np.array_equal(self.f, other.f) and
                np.array_equal(self.g, other.g) and
                self.feval == other.feval and
                self.names_x == other.names_x and
                self.names_f == other.names_f and
                self.names_g == other.names_g)

    def __add__(self, other: 'Population') -> 'Population':
        if not isinstance(other, Population):
            raise TypeError("Operands must be instances of Population")
        
        # Check that the names are consistent
        if self.names_x != other.names_x:
            raise ValueError("names_x are inconsistent between populations")
        if self.names_f != other.names_f:
            raise ValueError("names_f are inconsistent between populations")
        if self.names_g != other.names_g:
            raise ValueError("names_g are inconsistent between populations")
        
        # Concatenate the arrays along the batch dimension (axis=0)
        new_x = np.concatenate((self.x, other.x), axis=0)
        new_f = np.concatenate((self.f, other.f), axis=0)
        new_g = np.concatenate((self.g, other.g), axis=0)
        
        # Unique the arrays
        _, indices = np.unique(np.concatenate([new_x, new_f, new_g], axis=1), return_index=True, axis=0)
        new_x = new_x[indices, :]
        new_f = new_f[indices, :]
        new_g = new_g[indices, :]
        
        # Set feval to the maximum of the two feval values
        new_feval = max(self.feval, other.feval)
        
        # Return a new Population instance
        return Population(
            x=new_x,
            f=new_f,
            g=new_g,
            feval=new_feval,
            names_x=self.names_x,
            names_f=self.names_f,
            names_g=self.names_g
        )

    def __getitem__(self, idx: Union[slice, np.ndarray, List[int]]) -> 'Population':
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
            x= self.x[idx],
            f=self.f[idx],
            g=self.g[idx],
            feval=self.feval,
            names_x=self.names_x,
            names_f=self.names_f,
            names_g=self.names_g
        )

    def get_nondominated_indices(self):
        # Compare the objectives
        dom = np.bitwise_and(
            (self.f[:, None, :] <= self.f[None, :, :]).all(axis=-1), 
            (self.f[:, None, :] <  self.f[None, :, :]).any(axis=-1)
        )
        
        # If one individual is feasible and the other isn't, set domination
        feas = self.g >= 0.0
        ind = np.bitwise_and(feas.all(axis=1)[:, None], ~feas.all(axis=1)[None, :])
        dom[ind] = True
        ind = np.bitwise_and(~feas.all(axis=1)[:, None], feas.all(axis=1)[None, :])
        dom[ind] = False

        # If both are infeasible, then the individual with the least constraint violation wins
        constraint_violation = -np.sum(np.minimum(self.g, 0), axis=1)
        comp = constraint_violation[:, None] < constraint_violation[None, :]
        ind = ~np.bitwise_or(feas.all(axis=1)[:, None], feas.all(axis=1)[None, :])
        dom[ind] = comp[ind]

        # Return the nondominated individuals
        nondominated = (np.sum(dom, axis=0) == 0)
        return nondominated
    
    def get_nondominated_set(self):
        return self[self.get_nondominated_indices()]
    
    @classmethod
    def from_random(cls, n_objectives: int, n_decision_vars: int, n_constraints: int, pop_size: int,
                    feval: int = 0, generate_names: bool = False) -> 'Population':
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
        feval : int, optional
            The number of evaluations of the objective functions performed to reach this state, by default 0.
        generate_names : bool, optional
            Whether to include names for the decision variables, objectives, and constraints, by default False.

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
        >>> print(random_population.feval)
        0
        """
        x = np.random.rand(pop_size, n_decision_vars)
        f = np.random.rand(pop_size, n_objectives)
        g = np.random.rand(pop_size, n_constraints) if n_constraints > 0 else np.empty((pop_size, 0))

        # Optionally generate names if include_names is True
        names_x = [f"x{i+1}" for i in range(n_decision_vars)] if generate_names else None
        names_f = [f"f{i+1}" for i in range(n_objectives)] if generate_names else None
        names_g = [f"g{i+1}" for i in range(n_constraints)] if generate_names else None

        return cls(x=x, f=f, g=g, feval=feval, names_x=names_x, names_f=names_f, names_g=names_g)
    
    def __len__(self):
        return self.x.shape[0]


class History(BaseModel):
    """
    Represents the "history" of an optimization algorithm solving a multi-objective optimization problem.
    Populations are recorded at some reporting interval.
    
    Assumptions:
     - All reports must have a consistent number of objectives, decision variables, and constraints.
    """
    reports: List[Population]
    problem: str
    metadata: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_consistent_populations(self):
        n_decision_vars = [x.x.shape[1] for x in self.reports]
        n_objectives = [x.f.shape[1] for x in self.reports]
        n_constraints = [x.g.shape[1] for x in self.reports]

        if n_decision_vars and len(set(n_decision_vars)) != 1:
            raise ValueError(f'Inconsistent number of decision variables in reports: {n_decision_vars}')
        if n_objectives and len(set(n_objectives)) != 1:
            raise ValueError(f'Inconsistent number of objectives in reports: {n_objectives}')
        if n_constraints and len(set(n_constraints)) != 1:
            raise ValueError(f'Inconsistent number of constraints in reports: {n_constraints}')
        
        # Validate consistency of names across populations
        names_x = [tuple(x.names_x) if x.names_x is not None else None for x in self.reports]
        names_f = [tuple(x.names_f) if x.names_f is not None else None for x in self.reports]
        names_g = [tuple(x.names_g) if x.names_g is not None else None for x in self.reports]

        # If names are provided, check consistency
        if names_x and len(set(names_x)) != 1:
            raise ValueError(f'Inconsistent names for decision variables in reports: {names_x}')
        if names_f and len(set(names_f)) != 1:
            raise ValueError(f'Inconsistent names for objectives in reports: {names_f}')
        if names_g and len(set(names_g)) != 1:
            raise ValueError(f'Inconsistent names for constraints in reports: {names_g}')
        
        return self
    
    def __eq__(self, other):
        if not isinstance(other, History):
            return False
        return (self.reports == other.reports and
                self.problem == other.problem and
                self.metadata == other.metadata)

    @classmethod
    def from_random(cls, n_populations: int, n_objectives: int, n_decision_vars: int, n_constraints: int, 
                    pop_size: int, generate_names: bool = False) -> 'History':
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
            ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)),
            random.uniform(1.0, 3.0),
            "Randomly generated metadata",
            f"{random.randint(2020, 2024)}-0{random.randint(1, 9)}-{random.randint(10, 28)}"
        ]
        metadata = dict(zip(metadata_keys, metadata_values))

        # Generate populations with increasing feval values
        reports = [
            Population.from_random(n_objectives, n_decision_vars, n_constraints, pop_size, feval=(i+1) * pop_size, 
                                   generate_names=generate_names) for i in range(n_populations)
        ]

        return cls(reports=reports, problem=problem, metadata=metadata)
    
    def _to_h5py_group(self, g: h5py.Group):
        # Save the metadata
        g.attrs['problem'] = self.problem
        g_md = g.create_group("metadata")
        for k, v in self.metadata.items():
            g_md.attrs[k] = v
        
        # Save data from each population into one bigger dataset to reduce API calls to HDF5 file reader
        g.attrs['pop_sizes'] = [len(r) for r in self.reports]
        g.attrs['fevals'] = [r.feval for r in self.reports]
        g['x'] = np.concatenate([r.x for r in self.reports], axis=0)
        g['f'] = np.concatenate([r.f for r in self.reports], axis=0)
        g['g'] = np.concatenate([r.g for r in self.reports], axis=0)
        
        # Save names
        if self.reports and self.reports[0].names_x is not None:
            g['x'].attrs['names'] = self.reports[0].names_x
        if self.reports and self.reports[0].names_f is not None:
            g['f'].attrs['names'] = self.reports[0].names_f
        if self.reports and self.reports[0].names_g is not None:
            g['g'].attrs['names'] = self.reports[0].names_g

    @classmethod
    def _from_h5py_group(cls, grp: h5py.Group):
        # Get the decision vars, objectives, and constraints
        x = grp['x'][()]
        f = grp['f'][()]
        g = grp['g'][()]
        
        # Get the names
        names_x = grp['x'].attrs.get('names', None)
        names_f = grp['f'].attrs.get('names', None)
        names_g = grp['g'].attrs.get('names', None)

        # Create the population objects
        start_idx = 0
        reports = []
        for pop_size, feval in zip(grp.attrs['pop_sizes'], grp.attrs['fevals']):
            reports.append(Population(
                x = x[start_idx:start_idx+pop_size],
                f = f[start_idx:start_idx+pop_size],
                g = g[start_idx:start_idx+pop_size],
                feval=feval,
                names_x=names_x,
                names_f=names_f,
                names_g=names_g,
            ))
            start_idx += pop_size

        # Return as a history object
        return cls(
            problem=grp.attrs['problem'],
            reports=reports,
            metadata={k: v for k, v in grp['metadata'].attrs.items()},
        )

    def to_nondominated(self):
        """
        Returns a history object with the same number of population objects, but the individuals
        in each generation are the nondominated solutions seen up to this point.

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
            n.feval = o.feval

        return History(reports=new_reports, problem=self.problem, metadata=self.metadata.copy())

    
class Experiment(BaseModel):
    runs: List[History]
    identifier: str
    author: str = ''
    software: str = ''
    software_version: str = ''
    comment: str = ''
    creation_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    file_version: str = '1.0.0'

    def __eq__(self, other):
        if not isinstance(other, Experiment):
            return False
        return (self.identifier == other.identifier and
                self.author == other.author and
                self.software == other.software and
                self.software_version == other.software_version and
                self.comment == other.comment and
                self.creation_time == other.creation_time and
                self.runs == other.runs)

    def __repr__(self) -> str:
        return f"Experiment(identifier='{self.identifier}', n_runs={len(self.runs)})"

    @classmethod
    def from_random(cls, n_histories: int, n_populations: int, n_objectives: int, n_decision_vars: int, n_constraints: int,
                    pop_size: int, generate_names: bool = False) -> 'Experiment':
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

        Returns
        -------
        Experiment
            An instance of the Experiment class with a list of randomly generated History instances and a random identifier.
        """
        # Generate random histories
        runs = [
            History.from_random(n_populations, n_objectives, n_decision_vars, n_constraints, pop_size,
                                generate_names=generate_names) for _ in range(n_histories)
        ]
        
        # Randomly generate an identifier for the experiment
        identifier = f"Experiment_{random.randint(1000, 9999)}"

        # Generate random values or placeholders for other attributes
        author = f"Author_{random.randint(1, 100)}"
        software = f"Software_{random.randint(1, 10)}"
        software_version = f"{random.randint(1, 5)}.{random.randint(0, 9)}"
        comment = "Randomly generated experiment"

        return cls(runs=runs, identifier=identifier, author=author, software=software,
                   software_version=software_version, comment=comment)
    
    def save(self, fname):
        with h5py.File(fname, mode='w') as f:
            # Save metadata as attributes
            f.attrs['identifier'] = self.identifier
            f.attrs['author'] = self.author
            f.attrs['software'] = self.software
            f.attrs['software_version'] = self.software_version
            f.attrs['comment'] = self.comment
            f.attrs['creation_time'] = self.creation_time.isoformat()
            f.attrs['file_version'] = self.file_version
            f.attrs['file_format'] = 'ParetoBench Multi-Objective Optimization Data'
    
            # Calculate the necessary zero padding based on the number of runs
            max_len = len(str(len(self.runs) - 1))
        
            # Save each run into its own group
            for idx, run in enumerate(self.runs):
                run._to_h5py_group(f.create_group(f"run_{idx:0{max_len}d}"))
              
    @classmethod
    def load(cls, fname):
        # Load the data
        with h5py.File(fname, mode='r') as f:
            # Load each of the runs keeping track of the order of the indices
            idx_runs = []
            for idx_str, run_grp in f.items():
                m = re.match(r'run_(\d+)', idx_str)
                if m:
                    idx_runs.append((int(m.group(1)), History._from_h5py_group(run_grp)))
            runs = [x[1] for x in sorted(idx_runs, key=lambda x: x[0])]
            
            # Extract other attributes
            identifier = f.attrs['identifier']
            author = f.attrs['author']
            software = f.attrs['software']
            software_version = f.attrs['software_version']
            comment = f.attrs['comment']

            # Convert the creation_time back to a timezone-aware datetime object
            creation_time = datetime.fromisoformat(f.attrs['creation_time']).astimezone(timezone.utc)

            # Return as an experiment object
            return cls(
                runs=runs,
                identifier=identifier,
                author=author,
                software=software,
                software_version=software_version,
                comment=comment,
                creation_time=creation_time
            )
