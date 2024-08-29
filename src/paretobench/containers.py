from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Union
import h5py
import re
import random
import string


@dataclass
class Population:
    '''
    Stores the individuals in a population for one reporting interval in a genetic algorithm. Conventional names are used for
    the decision variables (x), the objectives (f), and inequality constraints (g). The first dimension of each array is the
    batch dimension. The number of evaluations of the objective functions performed to reach this state is also recored.
    '''
    x: np.ndarray
    f: np.ndarray
    g: np.ndarray
    feval: int
    
    def __eq__(self, other):
        if not isinstance(other, Population):
            return False
        return (np.array_equal(self.x, other.x) and
                np.array_equal(self.f, other.f) and
                np.array_equal(self.g, other.g) and
                self.feval == other.feval)

    @classmethod
    def from_random(cls, n_objectives: int, n_decision_vars: int, n_constraints: int, pop_size: int,
                    feval: int = 0) -> 'Population':
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

        Returns
        -------
        Population
            An instance of the Population class with random values for decision variables (`x`), objectives (`f`), 
            and inequality constraints (`g`).

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
        return cls(x=x, f=f, g=g, feval=feval)
    
    def __len__(self):
        return self.x.shape[0]


@dataclass
class History:
    """
    Reprsents the "history" of an optimizatoin algorithm solving a multiobjective optimization problem. Populations are recorded
    at some reporting interval.
    
    Assumptions:
     - All reports must have consistent number of objectives, decision variables, and constraints.
    """
    reports: List[Population]
    problem: str
    metadata: Dict[str, Union[str, int, float, bool]]

    def __eq__(self, other):
        if not isinstance(other, History):
            return False
        return (self.reports == other.reports and
                self.problem == other.problem and
                self.metadata == other.metadata)

    @classmethod
    def from_random(cls, n_populations: int, n_objectives: int, n_decision_vars: int, n_constraints: int, 
                    pop_size: int) -> 'History':
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
            Population.from_random(n_objectives, n_decision_vars, n_constraints, pop_size, feval=(i+1) * pop_size)
            for i in range(n_populations)
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

    @classmethod
    def _from_h5py_group(cls, grp: h5py.Group):
        start_idx = 0
        reports = []
        x = grp['x'][()]
        f = grp['f'][()]
        g = grp['g'][()]

        for pop_size, feval in zip(grp.attrs['pop_sizes'], grp.attrs['fevals']):
            reports.append(Population(
                x = x[start_idx:start_idx+pop_size],
                f = f[start_idx:start_idx+pop_size],
                g = g[start_idx:start_idx+pop_size],
                feval=feval,
            ))

        # Return as a history object
        return cls(
            problem=grp.attrs['problem'],
            reports=reports,
            metadata={k: v for k, v in grp['metadata'].attrs.items()},
        )

    
@dataclass
class Experiment:
    runs: List[History]
    identifier: str

    def __eq__(self, other):
        if not isinstance(other, Experiment):
            return False
        return (self.runs == other.runs and
                self.identifier == other.identifier)

    def __repr__(self) -> str:
        return f"Experiment(identifier='{self.identifier}', n_runs={len(self.runs)})"

    @classmethod
    def from_random(cls, n_histories: int, n_populations: int, n_objectives: int, n_decision_vars: int, n_constraints: int,
                    pop_size: int) -> 'Experiment':
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

        Returns
        -------
        Experiment
            An instance of the Experiment class with a list of randomly generated History instances and a random identifier.
        """
        # Generate random histories
        runs = [
            History.from_random(n_populations, n_objectives, n_decision_vars, n_constraints, pop_size)
            for _ in range(n_histories)
        ]
        
        # Randomly generate an identifier for the experiment
        identifier = f"Experiment_{random.randint(1000, 9999)}"

        return cls(runs=runs, identifier=identifier)
    
    def save(self, fname, version=1):
        with h5py.File(fname, mode='w') as f:
            # Save metadata as attributes
            f.attrs['identifier'] = self.identifier
            
            # Save each run into its own group
            if version == 1:
                for idx, run in enumerate(self.runs):
                    run._to_h5py_group(f.create_group("run_{:d}".format(idx)))
            elif version == 2:
                for idx, run in enumerate(self.runs):
                    run._to_h5py_group(f.create_group("run_{:d}".format(idx)))
            else:
                raise ValueError(f'unrecognized version: {version}')
                
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
            
            # Return as an experiment object
            return cls(
                identifier=f.attrs['identifier'],
                runs=runs,
            )
