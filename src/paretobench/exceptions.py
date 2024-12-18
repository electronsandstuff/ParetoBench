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
    Exception raised when a problem name passed by the user is not registered to the package.
    """

    pass


class InputError(Exception):
    """
    The input provided to a problem is wrong somehow.
    """

    pass


class EmptyPopulationError(Exception):
    """
    The supplied Population object was empty.
    """

    pass


class NoDecisionVarsError(Exception):
    """
    No decision variables were detected in the population.
    """

    pass


class NoObjectivesError(Exception):
    """
    No objectives were detected in the population.
    """

    pass


class NoConstraintsError(Exception):
    """
    No contraints were detected in the population.
    """

    pass
