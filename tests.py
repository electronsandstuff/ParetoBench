import unittest
import numpy as np
import random
import string

import paretobench as pb
from paretobench.simple_serialize import split_unquoted, dumps, loads


class TestProblemsBase(unittest.TestCase):
    def assertNoNaNs(self, value, msg=None):
        standardMsg = "Provided array has NaNs"
        try:
            if not np.isfinite(value).all():
                self.fail(self._formatMessage(msg, standardMsg))
        except:
            pass
    
    def test_evaluate(self, n_eval = 64):
        """
        Try creating each registered problem w/ default parameters and then call it.
        """
        for name in pb.get_problem_names():
            with self.subTest(name=name):
                p = pb.create_problem(name)
                bnd = p.decision_var_bounds
                x = np.random.random((bnd.shape[1], n_eval))*(bnd[1, :] - bnd[0, :])[:, None] + bnd[0, :][:, None]
                f, g = p(x)

                self.assertIsInstance(f, np.ndarray)
                self.assertIsInstance(g, np.ndarray)
                self.assertNoNaNs(f)
                self.assertNoNaNs(g)

    def test_get_params(self):
        """
        Checks all parameters like numbe rof decision variables, objectives, and constraints are set w/ right type and that when you 
        call the problem, those values are consistant with what comes out.
        """
        for name in pb.get_problem_names():
            with self.subTest(name=name):
                p = pb.create_problem(name)
                
                # Check the properties themselves for the right type
                self.assertIsInstance(p.n_decision_vars, int)
                self.assertIsInstance(p.n_objectives, int)
                self.assertIsInstance(p.n_constraints, int)
                self.assertIsInstance(p.decision_var_bounds, np.ndarray)
                
                # Check that if you actually call the values, you get the right sized objects (everything is consistent)
                bnd = p.decision_var_bounds
                x = np.random.random((bnd.shape[1], 1))*(bnd[1, :] - bnd[0, :])[:, None] + bnd[0, :][:, None]
                f, g = p(x)
                self.assertEqual(p.n_decision_vars, x.shape[0])
                self.assertEqual(p.n_objectives, f.shape[0])
                self.assertEqual(p.n_constraints, g.shape[0])

    def test_pareto_front(self, npoints=1000):
        """
        Try getting a pareto front for each of the registered test problems.
        """
        for name in pb.get_problem_names():
            with self.subTest(name=name):
                p = pb.create_problem(name)
            
                if not isinstance(p, (pb.ProblemWithPF, pb.ProblemWithFixedPF)):
                    continue
                if isinstance(p, pb.ProblemWithPF):  # If we can choose number of points, check at lesat that many are returned
                    f = p.get_pareto_front(npoints)
                    self.assertGreaterEqual(f.shape[1], npoints)
                else:  # If it's the fixed PF case
                    f = p.get_pareto_front()

                # Make sure the right size array is returned and it doesn't give bad values
                self.assertEqual(p.n_objectives, f.shape[0])
                self.assertNoNaNs(f)

    def test_refs(self):
        for name in pb.get_problem_names():
            with self.subTest(name=name):
                p = pb.create_problem(name)
                self.assertEqual(type(p.get_reference()), str)


def generate_random_dict(n_vals=32):
    """Generates a randomized dict of strings, ints, floats, and bools for testing serialization functions.

    Parameters
    ----------
    n_vals : int, optional
        Number of elements in the dict, by default 32
    """
    d = {}
    for _ in range(n_vals):  
        # Random key name
        k = ''.join(random.choice(string.ascii_letters) for i in range(10))
        
        # Random value
        idx = random.randrange(0, 4)  # The datatype
        if idx == 0:  # String
            v = ''.join(random.choice(string.ascii_letters) for i in range(10))
        elif idx == 1:  # Float
            v = random.random()
        elif idx == 2:  # Int
            v = random.randint(0, 999999)
        elif idx == 3:  # Bool
            v = bool(random.randint(0, 1))
            
        # Set the key
        d[k] = v
    return d


class TestSerializer(unittest.TestCase):
    def test_split_unquoted(self):
        # Basic test
        test_val = split_unquoted(r'a="fdas", b=fwqej, c="jlsfd"')
        true_val = ['a="fdas"',' b=fwqej',' c="jlsfd"']
        self.assertEqual(true_val, test_val)
        
        # Test ending with comma
        test_val = split_unquoted(r'a="fdas", b=fwqej,')
        true_val = ['a="fdas"',' b=fwqej']
        self.assertEqual(true_val, test_val)
        
        # Comma in string object
        test_val = split_unquoted(r'a="fd,a,s", b=fwqej, c="jlsf,d"')
        true_val = ['a="fd,a,s"',' b=fwqej',' c="jlsf,d"']
        self.assertEqual(true_val, test_val)
        
        # Test escaped quote
        test_val = split_unquoted('a="fdas", b=fwqej, c="jlsf\\",d"')
        true_val = ['a="fdas"',' b=fwqej',' c="jlsf\\",d"']
        self.assertEqual(true_val, test_val)
        
        # Check unterminated string
        with self.assertRaises(ValueError):
            split_unquoted('a="fdas", b=fwqej, c="jlsf\\",d')
        
        # Check escaped char outside of string
        with self.assertRaises(ValueError):
            split_unquoted('a="fdas", b=\\fwqef')
    
    def test_serialize_deserialize(self):
        # Run the test multiple times
        for _ in range(32):
            # Get a random dict
            d_true = generate_random_dict()
            
            # Serialize then deserialize the object
            d_test = loads(dumps(d_true))
            
            # Confirm objects are the same
            self.assertEqual(d_true, d_test)
            

# TODO problem family specific test varying parameters (ie changing n and k for WFGx)

if __name__ == "__main__":
    unittest.main()
    