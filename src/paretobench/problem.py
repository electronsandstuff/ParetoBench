import numpy as np
from pydantic import BaseModel
from dataclasses import dataclass

from .exceptions import DeserializationError
from .factory import create_problem
from .simple_serialize import dumps, loads


@dataclass
class Result:
    """
    This class represents the batched output from running one of the problems. The first index is the batched index and common
    literature names are used for the objectives (f) and inequality constraints (g).
    """
    f: np.ndarray
    g: np.ndarray


class Problem(BaseModel):
    """
    The overarching class all problems inherit from. Children must implement the following methods and properties.
     * `m`: property, the number of objectives
     * `n`: property, the number of decision variables
     * `n_constraints`: property, the number of constraints
     * `var_upper_bounds`: property, the array of upper bounds for decision variables
     * `var_lower_bounds`: property, the array of lower bounds for decision variables
     * `_call`: method, accepts `x` the decision variables (first dimension is batch), return `Result` object
    """    
    def __call__(self, x: np.ndarray) -> any:
        """
        Returns the values of the objective functions and constraints at the decision variables x.

        Args:
            x (np.ndarray): The decision variables. An mxn array representing the m variables and n individuals.

        Returns:
            any: A tuple containing first the objective function values and second the values of the constraints
        """
        ret = self._call(x)
        return ret if isinstance(ret, tuple) else (ret, np.empty((x.shape[1], 0)))
    
    def _call(self, x: np.ndarray) -> any:
        """
        This method is implemented by the child classes of problem and can return either the objectives only or both the objectives and constraints.
        """
        raise NotImplementedError()

    @property
    def n_vars(self):
        """
        Returns the number of decision variables expected by this problem. Passed through to property `n`.
        """
        return self.n

    @property
    def n_objs(self):
        """
        Returns the number of objective functions used in this problem. Passed through to property `m`.
        """
        return self.m
    
    @property
    def n_constraints(self):
        """
        Returns the number of constraints in the problem
        """
        return 0
    
    @property
    def var_lower_bound(self):
        """
        Returns the rectangular lower boundaries of the decision variables.
        """
        raise NotImplementedError()
    
    @property
    def var_upper_bound(self):
        """
        Returns the rectangular upper boundaries of the decision variables 
        """
        raise NotImplementedError()
    
    @property
    def var_bounds(self):
        """
        Returns the rectangular boundaries of the decision variables (2d numpy array
        where first row is lower bound of each variable and second row are the upper bounds)
        """
        return np.vstack((self.var_lower_bound, self.var_upper_bound))
    
    def get_reference(self):
        """
        Returns an APA formatted reference to where the problem was defined.
        """
        raise NotImplementedError()

    def to_line_fmt(self):
        """Serializes the problem object and returns it in a single line human readable format with the problem name and all of
        the data required to recreate it.

        Returns
        -------
        str
            The serialized problem object.
        """
        # Grab problem name and parameters
        name = type(self).__name__
        params = self.model_dump()
        
        # Save with parameters or just give name if no parameters
        if params:
            return f"{   name } ({ dumps(params) })"
        return name

    @classmethod
    def from_line_fmt(cls, s: str):
        """Create a problem object from the "single line" format. When run from the abstract class `Problem` this expects a
        string of the format `NAME (PARAMETERS)` or `NAME` and will create a problem object of the right class name with the
        specified parameters. If called from a child class, it expects the argument to only contain the paraemeters and creates
        the class based on that.

        Parameters
        ----------
        s : str
            The single line describing the problem object

        Returns
        -------
        Problem
            The instantiated problem
            
        Raises
        ------
        DeserializationError
            The string couldn't be parsed into the format NAME (PARAMETERS)
        """
        # Run from the abstract class
        if cls == Problem:
            # Find the section of the string corresponding to serialized parameters
            serialization_beg = s.find('(')
            serialization_end = s.find(')')
            
            # No parameters were passed
            if (serialization_beg == -1) and (serialization_end == -1):
                name = s.strip()
                kwargs = {}
            elif (serialization_beg != -1) and (serialization_end != -1):
                name = s[:serialization_beg].strip()
                kwargs = loads(s[serialization_beg+1:serialization_end])
            else:
                raise DeserializationError('could not interpret line "s"')
            
            # Create the problem and return
            return create_problem(name, **kwargs)
        
        # We are called from a child class; load parameters and create
        else:
            return cls(**loads(s))
            

class ProblemWithPF:
    """
    Mixin class for problems with a defined Pareto front where you can request a certain number of points from it.
    """
    def get_pareto_front(self, n=1000):
        """
        Returns at lesat n points along the Pareto front.
        """
        raise NotImplementedError()


class ProblemWithFixedPF:
    """
    Mixin class for problems that have a limited number of points along the Pareto front (ie you can't request a number of them)
    """
    def get_pareto_front(self):
        """
        Returns all of the points on the Pareto front.
        """
        raise NotImplementedError()
