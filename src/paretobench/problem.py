import numpy as np
from pydantic import BaseModel

from .simple_serialize import dumps


class Problem(BaseModel):
    """
    The overarching class all problems inherit from
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
        return ret if isinstance(ret, tuple) else (ret, np.empty((0, x.shape[1])))
    
    def _call(self, x: np.ndarray) -> any:
        """
        This method is implemented by the child classes of problem and can return either the objectives only or both the objectives and constraints.
        """
        raise NotImplementedError()

    @property
    def n_decision_vars(self):
        """
        Returns the number of decision variables expected by this problem.
        """
        raise NotImplementedError()

    @property
    def n_objectives(self):
        """
        Returns the number of objective functions used in this problem
        """
        raise NotImplementedError()
    
    @property
    def n_constraints(self):
        """
        Returns the number of constraints in the problem
        """
        return 0
    
    @property
    def decision_var_bounds(self):
        """
        Returns the rectangular boundaries of the decision variables (2d numpy array
        where first row is lower bound of each variable and second row are the upper bounds)
        """
        raise NotImplementedError()
    
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
