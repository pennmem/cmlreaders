from abc import abstractmethod, ABC
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import ptsa.data.TimeSeriesX as TimeSeries

from .exc import UnsupportedOutputFormat, ReferencingNotPossibleError
from .path_finder import PathFinder
from .readers import BaseCMLReader

__all__ = ['EEGReader']


class BaseEEGReader(ABC):
    """Base class for actually reading EEG data. Subclasses will be used by
    :class:`EEGReader` to actually read the format-specific EEG data.

    Parameters
    ----------
    filename
        Base name for EEG file(s) including absolute path
    sample_rate
        Sample rate in 1/s
    dtype
        numpy dtype to use for reading data
    epochs
        Epochs to include.
    channels
        A list of channel indices (1-based) to include when reading. When not
        given, read all available channels.

    """
    def __init__(self, filename: str, sample_rate: Union[int, float],
                 dtype: Type[np.dtype], epochs: List[Tuple[int, int]],
                 channels: Optional[List[int]] = None,):
        self.filename = filename
        self.sample_rate = sample_rate
        self.dtype = dtype

        self.epochs = epochs
        self.channels = channels

    @abstractmethod
    def read(self) -> TimeSeries:
        """Read the data."""


class SplitEEGReader(BaseEEGReader):
    """Read so-called split EEG data (that is, raw binary data stored as one
    channel per file).

    """
    @staticmethod
    def _read_epoch(mmap: np.memmap, epoch: Tuple[int, int]) -> np.array:
        return np.array(mmap[epoch[0]:epoch[1]])

    def read(self) -> TimeSeries:
        if self.channels is None:
            files = sorted(Path(self.filename).parent.glob('*'))
        else:
            raise NotImplementedError("FIXME: we can only read all channels now")

        memmaps = [np.memmap(f, dtype=self.dtype, mode='r') for f in files]
        data = [
            [self._read_epoch(mmap, epoch) for mmap in memmaps]
            for epoch in self.epochs
        ]
        return TimeSeries.create(data, samplerate=self.sample_rate)


class EDFReader(BaseEEGReader):
    def read(self):
        raise NotImplementedError


class RamulatorHDF5Reader(BaseEEGReader):
    """Reads Ramulator HDF5 EEG files."""
    def read(self):
        raise NotImplementedError


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

    @staticmethod
    def _get_reader_class(basename: str) -> Type[BaseEEGReader]:
        """Return the class to use for loading EEG data."""
        if basename.endswith('.h5'):
            return RamulatorHDF5Reader
        else:
            return SplitEEGReader

    def as_timeseries(self, events: Optional[pd.DataFrame] = None,
                      pre: int = 0, post: int = 0,
                      epochs: Optional[List[Tuple[int, int]]] = None,
                      contacts: Optional[pd.DataFrame] = None,
                      scheme: Optional[pd.DataFrame] = None) -> TimeSeries:
        """Read the timeseries."""
        finder = PathFinder(subject=self.subject,
                            experiment=self.experiment,
                            session=self.session)

        meta_path = Path(finder.find('sources'))
        with meta_path.open() as metafile:
            meta = list(json.load(metafile).values())[0]

        basename = meta['name']
        sample_rate = meta['sample_rate']
        dtype = meta['data_format']
        n_samples = meta['n_samples']
        reader_class = self._get_reader_class(basename)

        if events is not None:
            epochs = self._events_to_epochs(events, pre, post)
        elif epochs is None:
            epochs = [(0, int(1000 * n_samples / sample_rate))]

        reader = reader_class(filename=meta_path.joinpath(basename),
                              sample_rate=sample_rate,
                              dtype=dtype,
                              epochs=epochs)  # FIXME: channels
        ts = reader.read()

        if scheme is not None:
            return self.rereference(ts, scheme)
        else:
            return ts

    def _events_to_epochs(self, events: pd.DataFrame, pre: int, post: int) -> List[Tuple[int, int]]:
        """Convert events to epochs."""

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


if __name__ == "__main__":
    finder = PathFinder(subject='R1111M', experiment='FR1', session=0, rootdir='/Users/depalati/mnt/rhino')

    meta_path = Path(finder.find('sources'))
    with meta_path.open() as metafile:
        meta = list(json.load(metafile).values())[0]

    basename = meta['name']
    sample_rate = meta['sample_rate']
    dtype = meta['data_format']
    n_samples = meta['n_samples']

    reader = SplitEEGReader(basename, sample_rate, dtype, [(0, n_samples)])
    print(reader.read())
