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


def randstr(n):
    """Generates a random string of length n

    Parameters
    ----------
    n : int
        length of the random string
    """
    return ''.join(random.choice(string.ascii_letters) for _ in range(n))


def randlenstr(a=1, b=16):
    """Generates a random string of random length between a and b

    Parameters
    ----------
    a : int
        lower limit of string length
    b : int
        upper limit of string length
    """
    return randstr(random.randint(a, b))


def generate_random_dict(n_vals=32, int_lb=0, int_ub=999999):
    """Generates a randomized dict of strings, ints, floats, and bools for testing serialization functions. If n_vals is greater
    than 4 you are guaranteed to get at least one of each data-type.

    Parameters
    ----------
    n_vals : int, optional
        Number of elements in the dict, by default 32
    int_lb : int, optional
        Lower bound of the random ints
    int_ub : int, optional
        Upper bound of the random ints
    """
    d = {}
    for idx in range(n_vals):  
        # Random key name
        k = randlenstr()
        
        # Random value
        if idx%4 == 0:  # String
            v = randlenstr()
        elif idx%4 == 1:  # Float
            v = random.random()
        elif idx%4 == 2:  # Int
            v = random.randint(int_lb, int_ub)
        elif idx%4 == 3:  # Bool
            v = bool(random.randint(0, 1))
            
        # Set the key
        d[k] = v
    return d


class TestSerializer(unittest.TestCase):
    def test_split_unquoted(self):
        """Tests for the function `split_unquoted`
        """
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
        with self.assertRaises(pb.DeserializationError):
            split_unquoted('a="fdas", b=fwqej, c="jlsf\\",d')
        
        # Check escaped char outside of string
        with self.assertRaises(pb.DeserializationError):
            split_unquoted('a="fdas", b=\\fwqef')
    
    def test_serialize_deserialize(self):
        """Tries to serialize and then deserialize a series of random dicts
        """
        # Get a random dict, pass through serializer, then compare
        for _ in range(32):
            d_true = generate_random_dict()
            d_test = loads(dumps(d_true))
            self.assertEqual(d_true, d_test)
    
    def test_deserialize_empty(self):
        """Make sure deserializing empty strings gives an empty dict
        """
        # Dict mapping lines to the expected dicts
        lines_and_true_vals = {
            '': {},
            '   ': {},
        }
        
        for line, true_val in lines_and_true_vals.items():
            with self.subTest(name=line):
                # Create from line and compare against expected value
                self.assertEqual(true_val, loads(line))  
        
    def test_deserialize_whitespace(self):
        """Confirm whitespace is ignored around the objects
        """
        # Dict mapping lines to the expected dicts
        lines_and_true_vals = {
            'asdf=1,jkpl=1.0': {"asdf": 1, "jkpl": 1.0},
            '   asdf=1,jkpl=1.0': {"asdf": 1, "jkpl": 1.0},
            'asdf   =1,jkpl=1.0': {"asdf": 1, "jkpl": 1.0},
            'asdf=   1,jkpl=1.0': {"asdf": 1, "jkpl": 1.0},
            'asdf=1   ,jkpl=1.0': {"asdf": 1, "jkpl": 1.0},
            'asdf=1,   jkpl=1.0': {"asdf": 1, "jkpl": 1.0},
            'asdf=1,jkpl   =1.0': {"asdf": 1, "jkpl": 1.0},
            'asdf=1,jkpl=   1.0': {"asdf": 1, "jkpl": 1.0},
            'asdf=1,jkpl=1.0   ': {"asdf": 1, "jkpl": 1.0},
            '   asdf   =   1   ,   jkpl   =   1.0   ': {"asdf": 1, "jkpl": 1.0},
            'asdf=1,jkpl="hello"': {"asdf": 1, "jkpl": "hello"},
            'asdf=1,jkpl=   "hello"': {"asdf": 1, "jkpl": "hello"},
            'asdf=1,jkpl="hello"   ': {"asdf": 1, "jkpl": "hello"},
            'asdf=1,jkpl=   "hello"   ': {"asdf": 1, "jkpl": "hello"},
        }
        
        for line, true_val in lines_and_true_vals.items():
            with self.subTest(name=line):
                # Create from line and compare against expected value
                self.assertEqual(true_val, loads(line))  
    
    def test_serialize_deserialize_bad_val_chars(self):
        """Test that serialization works with "problem" characters in string value
        """
        # Get a random dict with extra "problem characters", pass through serializer, then compare
        for bad_char in '.=",\\':
            for _ in range(32):
                d_true = generate_random_dict()
                d_true['my_val'] = randlenstr() + bad_char + randlenstr()
                d_test = loads(dumps(d_true))
                self.assertEqual(d_true, d_test)
            
    def test_serialize_deserialize_bad_key_chars(self):
        """Test that serialization gives us the right error when there are bad characters in the key
        """
        # Try to serialize dict with bad characters in key
        for bad_char in '=,"':
            d_true = generate_random_dict()
            d_true[randlenstr() + bad_char + randlenstr()] = 0
            with self.assertRaises(pb.SerializationError):
                dumps(d_true)
    
    def test_serialize_deserialize_bad_datatype(self):
        """Test that serialization gives us the right error when there are values with an unserializable datatype in them
        """
        # Try to serialize dict with bad characters in key
        for bad_val in [[1, 2, 3], (1, 2, 3), {'a': 1}]:
            d_true = generate_random_dict()
            d_true['my_key'] = bad_val
            with self.assertRaises(pb.SerializationError):
                dumps(d_true)
    
    def test_serialize_problem(self):
        """For each problem, try to serialize it, deserialize it, and compare with original. Randomize the objects a little to
        make it more difficult.
        """
        for name in pb.get_problem_names():
            with self.subTest(name=name):
                # Create test problem and try to randomize a parameter
                p_true = pb.create_problem(name)
                if p_true.model_dump():  # Some problems don't have parameters to randomize
                    rand_key = list(p_true.model_dump().keys())[0]  # Get a key
                    param_type = type(p_true.model_dump()[rand_key])  # Randomize based on the parameter type
                    if param_type == int:
                        kwargs = {rand_key: random.randint(0, 10)}
                    elif param_type == float:
                        kwargs = {rand_key: random.random()}
                    elif param_type == str:
                        kwargs = {rand_key: randlenstr()}
                    elif param_type == bool:
                        kwargs = {rand_key: bool(random.randint(0, 1))}
                    else:
                        raise ValueError(f'Couldn\'t randomize object of type "{param_type}"')
                    
                    # Generate the new object with the randomized parameter
                    p_true = pb.create_problem(name, **kwargs)

                # Convert to line format, generate the object, and then make sure it loads correctly
                line_fmt = p_true.to_line_fmt()
                p_test = pb.Problem.from_line_fmt(line_fmt)
                self.assertEqual(p_true.model_dump(), p_test.model_dump())
              
    def test_deserialize_problems_manual(self):
        """Manually specify some cases of problems to test
        """
        
        # Dict mapping lines and the expected problem
        lines_and_probs = {
            "ZDT1": pb.ZDT1(),
            "ZDT1 ()": pb.ZDT1(),
            "ZDT1 (   )": pb.ZDT1(),
            "ZDT1()": pb.ZDT1(),
            "   ZDT1 (  n = 4 )   ": pb.ZDT1(n=4),
        }
        
        for line, prob in lines_and_probs.items():
            with self.subTest(name=line):
                # Create from line and compare against expected value
                p_test = pb.Problem.from_line_fmt(line)
                self.assertEqual(prob.model_dump(), p_test.model_dump())  

    def test_deserialize_problem_errors(self):
        """Test expected issues in problem deserialization
        """
        
        # Dict mapping lines and the expected problem
        lines = [
            "ZDT1 (",
            "ZDT1 )",
        ]
        
        # Confirm each of them causes an error
        for line in lines:
            with self.subTest(name=line):
                with self.assertRaises(pb.DeserializationError):
                    pb.Problem.from_line_fmt(line)
                
    def test_parenthesis_no_params(self):
        """Makes sure objects without parameters get printed without an extra set of parenthesis.
        """
        self.assertNotIn("(", pb.SRN().to_line_fmt())
        self.assertNotIn(")", pb.SRN().to_line_fmt())
        self.assertNotIn(" ", pb.SRN().to_line_fmt())

    def test_problem_names_are_safe(self):
        """Checks all registered problem names and parameters for bad characters
        """
        for name in pb.get_problem_names():
            with self.subTest(name=name):
                # Check name for bad chars
                self.assertTrue(name.replace("_", "").isalnum())
                
                # Check the parameters for bad characters or types
                prob = pb.create_problem(name)
                for param, val in prob.model_dump().items():
                    self.assertTrue(param.replace("_", "").isalnum())
                    self.assertTrue(type(val) in [int, float, bool, str])

    def test_from_line_fmt_child_class(self):
        """Tests running the method `from_line_fmt` in a child class.
        """
        tests = [
            (pb.ZDT1, 'n=3', pb.ZDT1(n=3)),
            (pb.WFG1, 'n=10, k=5', pb.WFG1(n=10, k=5)),
        ]
        
        for cls, line, actual in tests:
            with self.subTest(name=line):
                # Create from line and compare against expected value
                p_test = cls.from_line_fmt(line)
                self.assertEqual(actual.model_dump(), p_test.model_dump())  


if __name__ == "__main__":
    unittest.main()
    