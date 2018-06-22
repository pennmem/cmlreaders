class CMLReaderException(Exception):
    """ Generic exception for all CML Reader exceptions """


class UnsupportedOutputFormat(CMLReaderException):
    """ Raised when a save method is not supported for the given data type """


class UnsupportedRepresentation(CMLReaderException):
    """Raised when a data loading method is not supported for the given data
    type.

    """


class UnknownProtocolError(CMLReaderException):
    """Raised when an unknown protocol (e.g., "r2") is found."""


class MissingParameter(CMLReaderException):
    """ Raised when a required parameter is missing """


class RereferencingNotPossibleError(CMLReaderException):
    """ Raised when the requested EEG referencing scheme is not possible """


class IncompatibleParametersError(CMLReaderException):
    """Raised mutually exclusive parameters are given to a function."""


class UnmetOptionalDependencyError(CMLReaderException):
    """ Raised when an optional dependency is used without having first been installed """


class UnsupportedExperimentError(CMLReaderException):
    """ Raised when data for a particular experiment type is not supported """


class ImproperlyDefinedReader(CMLReaderException):
    """" Raised when a CML reader has not been defined correctly """
