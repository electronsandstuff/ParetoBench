from .exceptions import SerializationError, DeserializationError


def dumps(data: dict):
    """Converts a dict of basic python types (float, int, string, bool) to a single line string. This method is used to help
    instantiate problem objects with parameters from single line strings (such as those in config files).

    Parameters
    ----------
    data : dict
        The data to serialize

    Returns
    -------
    str
        The single line containing the data

    Raises
    ------
    SerializationError
        Object could not be converted
    """
    # Confirm object is appropriate for the function
    for k, v in data.items():
        # Check for illegal characters in the keys
        if not k.replace("_", "").isalnum():
            raise SerializationError(f'Keys must only contain alphanumeric characters. Found key named "{k}"')

        # Check for illegal datatypes
        if type(v) not in [int, float, bool, str]:
            raise SerializationError(f'Cannot serialize object of type {type(v)} at key "{k}"')

    # Perform the conversion
    items = []
    for k, v in data.items():
        if isinstance(v, str):
            # Escape problem characters before writing
            v_safe = v.replace("\\", "\\\\")
            v_safe = v_safe.replace('"', r"\"")
            items.append(f'{k}="{v_safe}"')
        else:
            items.append(f"{k}={v}")
    return ", ".join(sorted(items))


def split_unquoted(s: str, split_char=",", quote_char='"', escape_char="\\"):
    """Breaks string into pieces by the character `split_char` as long as it isn't in a quoted section. Quotes may be escaped
    using a user provided escape character.

    Parameters
    ----------
    s : str
        The string to split up
    split_char : str, optional
        The character to split on, by default ','
    quote_char : str, optional
        The character recognized as a quote, by default '"'
    escape_char : str, optional
        The character recognized as an escape character, by default '\\'

    Raises
    ------
    DeserializationError
        Detected a badly formed string. For instance unclosed quotes or escape character in a weird spot.
    """
    # Note: tried to replace with csv.reader, but it only supports quotes around the entire field separated by commas
    # Go through char by char while keeping track of "depth" into quotes. Store chunks of str as we go
    last_escape_char = -2  # The index to the last escape character seen
    in_quotes = False  # Are we currently in quotes?
    chunk_start = 0  # The start of the current chunk
    chunks = []  # The list of chunks we have seen
    for idx, c in enumerate(s):
        # We saw a non-escaped quote marks
        if c == quote_char and (last_escape_char != idx - 1):
            in_quotes = not in_quotes

        # We saw the splitter character outside of quotes
        elif not in_quotes and c == split_char:
            chunks.append(s[chunk_start:idx])
            chunk_start = idx + 1

        # We saw the escape character
        elif c == escape_char:
            last_escape_char = idx
            if not in_quotes:
                raise DeserializationError("Detected escaped character outside of a string. Possible corrupted data.")

    # If we didn't end on a split_char, add remaining values
    if chunk_start < len(s):
        chunks.append(s[chunk_start:])

    # Check that we ended with a quote
    if in_quotes:
        raise DeserializationError("Unterminated quotations marks. Possible corrupted data.")

    return chunks


def can_convert_to_int(x):
    """
    Whether or not the "raw value" string used in the deserialization function `loads(...)` can be converted to an int.

    Parameters
    ----------
    x : str
        The string under test

    Returns
    -------
    bool
        Whether or not the string can be converted
    """
    try:
        int(x)
        return True
    except ValueError:
        return False


def can_convert_to_float(x):
    """
    Whether or not the "raw value" string used in the deserialization function `loads(...)` can be converted to a float.

    Parameters
    ----------
    x : str
        The string under test

    Returns
    -------
    bool
        Whether or not the string can be converted
    """
    try:
        float(x)
        return True
    except ValueError:
        return False


def can_convert_to_bool(x):
    """
    Whether or not the "raw value" string used in the deserialization function `loads(...)` can be converted to a bool.

    Parameters
    ----------
    x : str
        The string under test

    Returns
    -------
    bool
        Whether or not the string can be converted
    """
    return x in ["True", "False"]


def is_quoted_str(x):
    """
    Is the string x surrounded by quotes. Used to determine whether or not the string represents a string-type value in the
    serialization format.

    Parameters
    ----------
    x : str
        The string to be parsed

    Returns
    -------
    bool
        Whether or not x should be treated as a string
    """
    return x[0] == '"'


def loads(s: str):
    """Converts a string in the "single line" serialization format back to a dict of basic python objects.

    Parameters
    ----------
    s : str
        The string containing data in "single line" format

    Returns
    -------
    dict
        The dict containing data loaded from the string

    Raises
    ------
    DeserializationError
        Deserializing the data failed
    """
    # Clean all whitespace around the string
    s = s.strip()

    # Break string into sections
    ret = {}
    for chunk in split_unquoted(s):
        # Make sure we can break out the key and value
        if "=" not in chunk:
            raise DeserializationError(f'Bad key value pair: "{chunk}"')

        # Break into key and value
        idx = chunk.index("=")  # Must use first instance in case '=' in a string object
        k = chunk[:idx].strip()
        v_raw = chunk[idx + 1 :].strip()

        # Handle strings
        if is_quoted_str(v_raw):
            # Replace escaped characters and get rid of surrounding quotes
            v_raw = v_raw.replace("\\\\", "\\")
            v_raw = v_raw.replace('\\"', '"')
            v_parsed = v_raw[1:-1]
        elif can_convert_to_bool(v_raw):
            v_parsed = {"True": True, "False": False}[v_raw]
        elif can_convert_to_int(v_raw):
            v_parsed = int(v_raw)
        elif can_convert_to_float(v_raw):
            v_parsed = float(v_raw)
        else:
            raise DeserializationError(
                f'Could not determine data type for the value associated with key "{k}": "{v_raw}"'
            )

        # Finally set the dict entry
        ret[k] = v_parsed
    return ret
