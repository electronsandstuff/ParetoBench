class SerializationError(Exception):
    """
    Exception raised when something goes wrong during serialization.
    """
    pass


class DeserializationError(Exception):
    """
    Exception raised when something goes wrong during deserialization.
    """
    pass


class UnknownProblemError(Exception):
    """
    Used when a problem name passed by the user is not registered to the package.
    """
    pass
