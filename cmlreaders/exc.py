
class CMLReaderException(Exception):
    """ Generic exception for all CML Reader exceptions """


class UnsupportedOutputFormat(CMLReaderException):
    """ Raised when a save method is not supported for the given data type """


class UnsupportedRepresentation(CMLReaderException):
    """ Raised when a data loading method is not supported for the given data type """


class MissingParameter(CMLReaderException):
    """ Raised when a required parameter is missing """


class ReferencingNotPossibleError(CMLReaderException):
    """ Raised when the requested EEG referencing scheme is not possible """


class UnmetOptionalDependencyError(CMLReaderException):
    """ Raised when an optional dependency is used without having first been installed """


class UnsupportedExperimentError(CMLReaderException):
    """ Raised when data for a particular experiment type is not supported """
