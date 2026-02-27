from pymoo.core.problem import Problem as PymooProblem
from paretobench import Problem


class PymooProblemWrapper(PymooProblem):
    def __init__(self, problem: Problem):
        """
        Initialize the pymoo wrapper with a given problem object.

        Parameters
        ----------
        problem : Problem
            A problem object that follows the Problem class interface.
        """
        self.prob = problem

        super().__init__(
            n_var=self.prob.n_vars,
            n_obj=self.prob.n_objs,
            n_ieq_constr=self.prob.n_constraints,
            xl=self.prob.var_lower_bounds,
            xu=self.prob.var_upper_bounds,
            vtype=float,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate the problem for pymoo using vectorization.

        Parameters
        ----------
        X : np.ndarray
            2D array of decision variables, where each row is a solution
        out : dict
            Dictionary to store the output

        Returns
        -------
        None
        """
        # Evaluate the problem
        pop = self.prob(X)

        # Set the objectives
        out["F"] = pop.f

        # Set the constraints (if any)
        if self.prob.n_constraints > 0:
            out["G"] = pop.g
