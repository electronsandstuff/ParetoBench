import numpy as np
import paretobench as pb


class ProblemExample(pb.Problem, pb.ProblemWithFixedPF):
    """
    Problem with specific Pareto front for `test_inverse_generational_distance`
    """
    def get_pareto_front(self):
        return np.array([
            [0, 1],
            [0.5, 0.5],
            [1, 0],
        ])


def test_inverse_generational_distance():
    """
    Make sure IGD calculation works on analytical cases
    """
    # Create the metric
    igd = pb.InverseGenerationalDistance()
    
    # Get the IGD of a test population and compare with analytical value
    test_pop = pb.Population(f=np.array([[0.0, 0.0]]))
    val = igd(test_pop, ProblemExample())
    actual1 = np.mean([1, np.sqrt(0.5**2 + 0.5**2), 1])
    assert val == actual1

    # Another point
    test_pop = pb.Population(f=np.array([[0.0, 1.0]]))
    val = igd(test_pop, ProblemExample())
    actual2 = np.mean([0, np.sqrt(0.5**2 + 0.5**2), np.sqrt(1**2 + 1**2)])
    assert val == actual2
    
    # Do multiple points
    test_pop = pb.Population(f=np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    val = igd(test_pop, ProblemExample())
    assert val == np.mean([0, 0, np.sqrt(0.5**2 + 0.5**2)])
