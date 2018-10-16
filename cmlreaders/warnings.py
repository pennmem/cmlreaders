class MultiplePathsFoundWarning(RuntimeWarning):
    """Warning when :class:`PathFinder` finds multiple files for a given type
    and returns only the first one found.

    """


class MissingChannelsWarning(RuntimeWarning):
    """Warning used when there is a discrepancy between requested EEG channels
    and what are available.

    """

class MissingCoordinatesWarning(UserWarning):
    """ Warning when coordinates could not be loaded
    """