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
    ValueError
        Object could not be converted
    """
    # Confirm object is appropriate for the function
    for k, v in data.items():
        # Check for illegal characters in the keys
        for c in [',', '=']:
            if c in k:
                raise ValueError(f'Keys cannot have "{c}" character. Found key named "{k}"')
        
        # Check for illegal datatypes
        if type(v) not in [int, float, bool, str]:
            raise ValueError(f'Cannot serialize object of type {type(v)} at key "{k}"')
    
    # Perform the conversion
    items = []
    for k, v in data.items():
        if isinstance(v, str):
            # Escape problem characters before writing
            v_safe = v.replace('\\', '\\\\')
            v_safe = v_safe.replace('"', r'\"')
            items.append('{k}="{v_safe}"')
        else:
            items.append('{k}={v}')
    return ", ".join(items)


def split_unquoted(s: str, split_char=','):
    """Breaks string into pieces by the character `split_char` as long as it isn't in a quoted section.

    Parameters
    ----------
    s : str
        The string to split up
    split_char : str, optional
        The characeter to split on, by default ','
    """
    # Go through char by char while keeping track of "depth" into quotes. Store chunks of str as we go
    last_escape_char = -2
    depth = 0
    chunk_start = 0
    chunks = []
    for idx, c in enumerate(s):
        if c == '"' and last_escape_char != idx - 1:
            depth = 1 - depth
        elif c == split_char and depth == 0:
            chunks.append(s[chunk_start:idx])
            chunk_start = idx+1
        elif c == '\\' and depth == 1:
            last_escape_char = idx
        elif c == '\\' and depth == 0:
            raise ValueError('Detected escaped character outside of a string. Possible corrupted data.')
        
    # If we didn't end on a split_char, add remaining values
    if chunk_start < len(s):
        chunks.append(s[chunk_start:])

    # Check that we ended with a quote
    if depth != 0:
        raise ValueError("Unterminated quotations marks. Possible corrupted data.")
    
    return chunks
