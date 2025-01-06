import numpy as np
from pydantic import BaseModel

from .exceptions import DeserializationError, InputError
from .factory import create_problem
from .simple_serialize import dumps, loads


class Problem(BaseModel):
    """
    The overarching class all problems inherit from. Children must implement the following methods and properties.
     * `m`: property, the number of objectives
     * `n`: property, the number of decision variables
     * `n_constraints`: property, the number of constraints
     * `var_upper_bounds`: property, the array of upper bounds for decision variables
     * `var_lower_bounds`: property, the array of lower bounds for decision variables
     * `_call`: method, accepts `x` the decision variables (first dimension is batch), return `Population` object
    """

    def __call__(self, x: np.ndarray, check_bounds=True):
        """
        Returns the values of the objective functions and constraints at the decision variables `x`.
        The input can be either batched or a single value.

        Note: When subclassing `Problem`, the function must be implemented by defining `_call`, not `__call__`.

        Parameters
        ----------
        x : np.ndarray
            The decision variables. When batched, the first dimension is the batch dimension.
        check_bounds : bool, optional
            Whether or not to check that `x` is within the boundaries of the problem. Defaults to True.

        Returns
        -------
        Population
            A population object containing the objectives and constraints.
        """
        # If a single input was provided
        if len(x.shape) == 1:
            if x.shape[0] != self.n_vars:
                msg = (
                    f"Input does not match number of decision variables (n_vars={self.n_vars}, x.shape[0]={x.shape[0]})"
                )
                raise InputError(msg)
            if check_bounds and ((x > self.var_upper_bounds).all() or (x < self.var_lower_bounds).all()):
                raise InputError("Input lies outside of problem bounds.")
            pop = self._call(x[None, :])

        # If batched input is used
        elif len(x.shape) == 2:
            if x.shape[1] != self.n_vars:
                msg = (
                    f"Input does not match number of decision variables (n_vars={self.n_vars}, x.shape[1]={x.shape[1]})"
                )
                raise InputError(msg)
            if check_bounds and ((x > self.var_upper_bounds).all() or (x < self.var_lower_bounds).all()):
                raise InputError("Input lies outside of problem bounds.")
            pop = self._call(x)

        # If user provided something not usable
        else:
            raise ValueError(f"Incompatible shape of input array x: {x.shape}")

        # Set the decision variables
        pop.x = x
        return pop

    def _call(self, x: np.ndarray):
        """
        This method is implemented by the child classes of `Problem` and should operate on a batched array of inputs.
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
    def var_lower_bounds(self):
        """
        Returns the rectangular lower boundaries of the decision variables.
        """
        raise NotImplementedError()

    @property
    def var_upper_bounds(self):
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
        return np.vstack((self.var_lower_bounds, self.var_upper_bounds))

    @property
    def reference(self):
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
            serialization_beg = s.find("(")
            serialization_end = s.find(")")

            # No parameters were passed
            if (serialization_beg == -1) and (serialization_end == -1):
                name = s.strip()
                kwargs = {}
            elif (serialization_beg != -1) and (serialization_end != -1):
                name = s[:serialization_beg].strip()
                kwargs = loads(s[serialization_beg + 1 : serialization_end])
            else:
                raise DeserializationError('could not interpret line "s"')

            # Create the problem and return
            return create_problem(name, **kwargs)

        # We are called from a child class; load parameters and create
        else:
            return cls(**loads(s))

    def __repr__(self):
        return self.to_line_fmt()

    def __str__(self):
        return self.__repr__()


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
