import pandas as pd

from .decorator import accessor


@accessor("events")
class EventsAccessor(object):
    """Shortcuts for selecting events of various types.

    Examples
    --------

    .. code-block:: python

        >>> reader = CMLReader("R1111M", "FR1", 0)
        >>> df = reader.load("events")
        >>> word_events = df.events.words
        >>> stim_events = df.events.stim

    """
    def __init__(self, obj):
        self._obj = obj

    @property
    def words(self) -> pd.DataFrame:
        """Select all WORD onset events."""
        return self._obj[self._obj["type"] == "WORD"]

    @property
    def stim(self) -> pd.DataFrame:
        """Select all stim events."""
        return self._obj[self._obj["type"] == "STIM_ON"]
