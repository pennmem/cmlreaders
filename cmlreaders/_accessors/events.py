import pandas as pd
from .decorator import accessor


@accessor("events")
class EventsAccessor(object):
    """Shortcuts for selecting events of various types.

    Examples
    --------

    .. code-block:: python

        >>> from cmlreaders import CMLReader
        >>> reader = CMLReader("R1111M", "FR1", 0)
        >>> df = reader.load("events")
        >>> word_events = df.events.words
        >>> stim_events = df.events.stim
        >>> recalled_words = df.events.words_recalled
        >>> forgotten_words = df.events.words_not_recalled

    """
    def __init__(self, obj):
        self._obj = obj

    @property
    def words(self) -> pd.DataFrame:
        """Select all WORD onset events."""
        return self._obj[self._obj["type"] == "WORD"]

    def _words_recalled_or_not(self, recalled: bool) -> pd.DataFrame:
        recalled = int(recalled)
        mask = (self._obj["type"] == "WORD") & (self._obj["recalled"] == recalled)
        return self._obj[mask]

    @property
    def words_recalled(self) -> pd.DataFrame:
        """Select all recalled word events."""
        return self._words_recalled_or_not(True)

    @property
    def words_not_recalled(self) -> pd.DataFrame:
        """Select word events where the word was not recalled."""
        return self._words_recalled_or_not(False)

    @property
    def stim(self) -> pd.DataFrame:
        """Select all stim events."""
        return self._obj[self._obj["type"] == "STIM_ON"]

    @property
    def stim_params(self) -> pd.DataFrame:
        """ Expand the stim_params field in a friendly manner"""
        sp = [x.stim_params for _, x in self._obj.iterrows()]
        if not all(isinstance(x, list) for x in sp):
            return self._obj.stim_params
        if not all(len(x) < 2 for x in sp):
            return self._obj.stim_params
        for x in sp:
            if len(x) == 0:
                x.append({})
        sp = [x[0] for x in sp]
        df = pd.DataFrame.from_records(sp)
        return df
