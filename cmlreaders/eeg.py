import pandas as pd
import ptsa.data.TimeSeriesX as TimeSeries

from .exc import UnsupportedOutputFormat, ReferencingNotPossibleError
from .readers import BaseCMLReader


class EEGReader(BaseCMLReader):
    """Reads EEG data.

    Returns a :class:`TimeSeries`.

    Keyword arguments
    -----------------
    events : pd.DataFrame
        Events to use to determine epochs to load.

    Examples
    --------
    All examples start by defining a reader::

        >>> from cmlreaders import CMLReader
        >>> reader = CMLReader('R1111M', experiment='FR1', session=0)

    Loading data based on word events and including data 100 ms before and
    after::

        >>> events = reader.load('events')
        >>> words = events[events.type == 'WORD']
        >>> eeg = reader.load_eeg(events=words, pre=100, post=100)

    Loading a subset of EEG based on brain region::

        >>> contacts = reader.load('contacts')
        >>> eeg = reader.load_eeg(contacts=contacts[contacts.region == 'MTL'])

    Loading from explicitly specified epochs::

        >>> epochs = [(100, 200), (300, 400)]
        >>> eeg = reader.load_eeg(epochs=epochs)

    Loading an entire session::

        >>> eeg = reader.load_eeg()

    """
    default_representation = "timeseries"

    def as_dataframe(self):
        raise UnsupportedOutputFormat

    def as_recarray(self):
        raise UnsupportedOutputFormat

    def as_dict(self):
        raise UnsupportedOutputFormat

    def as_timeseries(self):
        pass

    def rereference(self, data: TimeSeries, scheme: pd.DataFrame) -> TimeSeries:
        """Attempt to rereference the EEG data using the specified scheme.

        Parameters
        ----------
        data
            Input timeseries data.
        scheme
            Bipolar pairs to use.

        Returns
        -------
        reref
            Rereferenced timeseries.

        Raises
        ------
        RereferencingNotPossibleError
            When rereferincing is not possible.

        """
        raise NotImplementedError
