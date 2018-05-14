
class CMLReaderException(Exception):
    """ Generic exception for all CML Reader exceptions """


class UnsupportedOutputFormat(CMLReaderException):
    """ Raised when a save method is not supported for the given data type """


class UnsupportedRepresentation(CMLReaderException):
    """ Raised when a data loading method is not supported for the given data type """
