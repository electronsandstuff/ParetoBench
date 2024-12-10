import pytest
import random
import string

import paretobench as pb
from paretobench.simple_serialize import split_unquoted, dumps, loads


def randstr(n):
    """Generates a random string of length n

    Parameters
    ----------
    n : int
        length of the random string
    """
    return "".join(random.choice(string.ascii_letters) for _ in range(n))


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
        if idx % 4 == 0:  # String
            v = randlenstr()
        elif idx % 4 == 1:  # Float
            v = random.random()
        elif idx % 4 == 2:  # Int
            v = random.randint(int_lb, int_ub)
        elif idx % 4 == 3:  # Bool
            v = bool(random.randint(0, 1))

        # Set the key
        d[k] = v
    return d


def test_split_unquoted():
    """Tests for the function `split_unquoted`"""
    # Basic test
    test_val = split_unquoted(r'a="fdas", b=fwqej, c="jlsfd"')
    true_val = ['a="fdas"', " b=fwqej", ' c="jlsfd"']
    assert true_val == test_val

    # Test ending with comma
    test_val = split_unquoted(r'a="fdas", b=fwqej,')
    true_val = ['a="fdas"', " b=fwqej"]
    assert true_val == test_val

    # Comma in string object
    test_val = split_unquoted(r'a="fd,a,s", b=fwqej, c="jlsf,d"')
    true_val = ['a="fd,a,s"', " b=fwqej", ' c="jlsf,d"']
    assert true_val == test_val

    # Test escaped quote
    test_val = split_unquoted('a="fdas", b=fwqej, c="jlsf\\",d"')
    true_val = ['a="fdas"', " b=fwqej", ' c="jlsf\\",d"']
    assert true_val == test_val

    # Check unterminated string
    with pytest.raises(pb.DeserializationError):
        split_unquoted('a="fdas", b=fwqej, c="jlsf\\",d')

    # Check escaped char outside of string
    with pytest.raises(pb.DeserializationError):
        split_unquoted('a="fdas", b=\\fwqef')


def test_serialize_deserialize():
    """Tries to serialize and then deserialize a series of random dicts"""
    # Get a random dict, pass through serializer, then compare
    for _ in range(32):
        d_true = generate_random_dict()
        d_test = loads(dumps(d_true))
        assert d_true == d_test


@pytest.mark.parametrize("line", ["", "   "])
def test_deserialize_empty(line):
    """Make sure deserializing empty strings gives an empty dict"""
    # Create from line and compare against expected value
    assert loads(line) == {}


@pytest.mark.parametrize(
    "line,true_val",
    [
        ("asdf=1,jkpl=1.0", {"asdf": 1, "jkpl": 1.0}),
        ("   asdf=1,jkpl=1.0", {"asdf": 1, "jkpl": 1.0}),
        ("asdf   =1,jkpl=1.0", {"asdf": 1, "jkpl": 1.0}),
        ("asdf=   1,jkpl=1.0", {"asdf": 1, "jkpl": 1.0}),
        ("asdf=1   ,jkpl=1.0", {"asdf": 1, "jkpl": 1.0}),
        ("asdf=1,   jkpl=1.0", {"asdf": 1, "jkpl": 1.0}),
        ("asdf=1,jkpl   =1.0", {"asdf": 1, "jkpl": 1.0}),
        ("asdf=1,jkpl=   1.0", {"asdf": 1, "jkpl": 1.0}),
        ("asdf=1,jkpl=1.0   ", {"asdf": 1, "jkpl": 1.0}),
        ("   asdf   =   1   ,   jkpl   =   1.0   ", {"asdf": 1, "jkpl": 1.0}),
        ('asdf=1,jkpl="hello"', {"asdf": 1, "jkpl": "hello"}),
        ('asdf=1,jkpl=   "hello"', {"asdf": 1, "jkpl": "hello"}),
        ('asdf=1,jkpl="hello"   ', {"asdf": 1, "jkpl": "hello"}),
        ('asdf=1,jkpl=   "hello"   ', {"asdf": 1, "jkpl": "hello"}),
    ],
)
def test_deserialize_whitespace(line, true_val):
    """Confirm whitespace is ignored around the objects"""
    # Create from line and compare against expected value
    assert true_val == loads(line)


@pytest.mark.parametrize("bad_char", '.=",\\')
def test_serialize_deserialize_bad_val_chars(bad_char):
    """Test that serialization works with "problem" characters in string value"""
    # Get a random dict with extra "problem characters", pass through serializer, then compare
    for _ in range(32):
        d_true = generate_random_dict()
        d_true["my_val"] = randlenstr() + bad_char + randlenstr()
        d_test = loads(dumps(d_true))
        assert d_true == d_test


@pytest.mark.parametrize("bad_char", '=,"')
def test_serialize_deserialize_bad_key_chars(bad_char):
    """Test that serialization gives us the right error when there are bad characters in the key"""
    # Try to serialize dict with bad characters in key
    d_true = generate_random_dict()
    d_true[randlenstr() + bad_char + randlenstr()] = 0
    with pytest.raises(pb.SerializationError):
        dumps(d_true)


@pytest.mark.parametrize("bad_val", [[1, 2, 3], (1, 2, 3), {"a": 1}])
def test_serialize_deserialize_bad_datatype(bad_val):
    """Test that serialization gives us the right error when there are values with an unserializable datatype in them"""
    # Try to serialize dict with bad value
    d_true = generate_random_dict()
    d_true["my_key"] = bad_val
    with pytest.raises(pb.SerializationError):
        dumps(d_true)


@pytest.mark.parametrize("name", pb.get_problem_names())
def test_serialize_problem(name):
    """For each problem, try to serialize it, deserialize it, and compare with original. Randomize the objects a little to
    make it more difficult.
    """
    # Create test problem and try to randomize a parameter
    p_true = pb.create_problem(name)
    if p_true.model_dump():  # Some problems don't have parameters to randomize
        rand_key = list(p_true.model_dump().keys())[0]  # Get a key
        param = p_true.model_dump()[rand_key]
        if isinstance(param, int):
            kwargs = {rand_key: random.randint(0, 10)}
        elif isinstance(param, float):
            kwargs = {rand_key: random.random()}
        elif isinstance(param, str):
            kwargs = {rand_key: randlenstr()}
        elif isinstance(param, bool):
            kwargs = {rand_key: bool(random.randint(0, 1))}
        else:
            raise ValueError(f'Couldn\'t randomize object of type "{type(param)}"')

        # Generate the new object with the randomized parameter
        p_true = pb.create_problem(name, **kwargs)

    # Convert to line format, generate the object, and then make sure it loads correctly
    line_fmt = p_true.to_line_fmt()
    p_test = pb.Problem.from_line_fmt(line_fmt)
    assert p_true.model_dump() == p_test.model_dump()


@pytest.mark.parametrize(
    "line,prob",
    [
        ("ZDT1", pb.ZDT1()),
        ("ZDT1 ()", pb.ZDT1()),
        ("ZDT1 (   )", pb.ZDT1()),
        ("ZDT1()", pb.ZDT1()),
        ("   ZDT1 (  n = 4 )   ", pb.ZDT1(n=4)),
    ],
)
def test_deserialize_problems_manual(line, prob):
    """Manually specify some cases of problems to test"""
    # Create from line and compare against expected value
    p_test = pb.Problem.from_line_fmt(line)
    assert prob.model_dump() == p_test.model_dump()


@pytest.mark.parametrize("line", ["ZDT1 (", "ZDT1 )"])
def test_deserialize_problem_errors(line):
    """Test expected issues in problem deserialization"""
    with pytest.raises(pb.DeserializationError):
        pb.Problem.from_line_fmt(line)


def test_parenthesis_no_params():
    """Makes sure objects without parameters get printed without an extra set of parenthesis."""
    assert "(" not in pb.SRN().to_line_fmt()
    assert ")" not in pb.SRN().to_line_fmt()
    assert " " not in pb.SRN().to_line_fmt()


@pytest.mark.parametrize("name", pb.get_problem_names())
def test_problem_names_are_safe(name):
    """Checks all registered problem names and parameters for bad characters"""
    # Check name for bad chars
    assert name.replace("_", "").isalnum()

    # Check the parameters for bad characters or types
    prob = pb.create_problem(name)
    for param, val in prob.model_dump().items():
        assert param.replace("_", "").isalnum()
        assert type(val) in [int, float, bool, str]


@pytest.mark.parametrize(
    "cls,line,actual",
    [
        (pb.ZDT1, "n=3", pb.ZDT1(n=3)),
        (pb.WFG1, "n=10, k=5", pb.WFG1(n=10, k=5)),
    ],
)
def test_from_line_fmt_child_class(cls, line, actual):
    """Tests running the method `from_line_fmt` in a child class."""
    # Create from line and compare against expected value
    p_test = cls.from_line_fmt(line)
    assert actual.model_dump() == p_test.model_dump()


@pytest.mark.parametrize("val", ["1E+04", "1e+04", "1E-04", "1e-04"])
def test_buggy_values(val):
    """
    Confirm loading of certain values that have raised exceptions in the past and should not.
    """
    loads(f"x={val}")
