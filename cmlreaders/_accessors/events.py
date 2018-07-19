import pandas as pd

from .decorator import accessor


@accessor("events")
class EventsAccessor(object):
    """Shortcuts for selecting events of various types."""
    def __init__(self, obj):
        self._obj = obj

    @property
    def words(self) -> pd.DataFrame:
        """Return all WORD onset events."""
        return self._obj[self._obj["type"] == "WORD"]

    @property
    def stim(self) -> pd.DataFrame:
        """Return all stim events."""
        return self._obj[self._obj["type"] == "STIM_ON"]
