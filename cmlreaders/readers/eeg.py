from abc import abstractmethod, ABC
import copy
import functools
import json
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union

import h5py
import numpy as np
import pandas as pd

from cmlreaders.base_reader import BaseCMLReader
from cmlreaders.exc import (
    RereferencingNotPossibleError, UnsupportedOutputFormat
)
from cmlreaders.path_finder import PathFinder
from cmlreaders.timeseries import TimeSeries

__all__ = ['EEGReader']


def events_to_epochs(events: pd.DataFrame, rel_start: int = 0,
                     rel_stop: int = 100,
                     sample_rate: Union[int, float] = 1000) -> List[Tuple[int, int]]:
    """Convert events to epochs.

    Parameters
    ----------
    events
        Events to read.
    rel_start
        Start time relative to events in ms (default: 0).
    rel_stop
        Stop time relative to events in ms (default: 100).
    sample_rate
        Sample rate in Hz (default: 1000).

    Returns
    -------
    epochs
        A list of tuples giving absolute start and stop times.

    """
    offsets = events.eegoffset.values * sample_rate / 1000.
    epochs = [(int(offset + rel_start), int(offset + rel_stop)) for offset in offsets]
    return epochs


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

    Notes
    -----
    The :meth:`read` method must be implemented by subclasses to return a 3-D
    array with dimensions (epochs x channels x time).

    """
    def __init__(self, filename: str, sample_rate: Union[int, float],
                 dtype: Type[np.dtype], epochs: List[Tuple[int, int]],
                 channels: Optional[List[int]] = None,):
        self.filename = filename
        self.sample_rate = sample_rate
        self.dtype = dtype

        self.epochs = epochs
        self.channels = channels

        # in cases where we can't rereference, this will get changed to False
        self.rereferencing_possible = True

    @abstractmethod
    def read(self) -> np.ndarray:
        """Read the data."""


class SplitEEGReader(BaseEEGReader):
    """Read so-called split EEG data (that is, raw binary data stored as one
    channel per file).

    """
    @staticmethod
    def _read_epoch(mmap: np.memmap, epoch: Tuple[int, int]) -> np.array:
        return np.array(mmap[epoch[0]:epoch[1]])

    def read(self) -> np.ndarray:
        if self.channels is None:
            files = sorted(Path(self.filename).parent.glob('*'))
        else:
            raise NotImplementedError("FIXME: we can only read all channels now")

        memmaps = [np.memmap(f, dtype=self.dtype, mode='r') for f in files]
        data = np.array([
            [self._read_epoch(mmap, epoch) for mmap in memmaps]
            for epoch in self.epochs
        ])
        return data


class EDFReader(BaseEEGReader):
    def read(self) -> np.ndarray:
        raise NotImplementedError


class RamulatorHDF5Reader(BaseEEGReader):
    """Reads Ramulator HDF5 EEG files."""
    def read(self) -> np.ndarray:
        if self.channels is not None:
            raise NotImplementedError("FIXME: we can only read all channels now")

        with h5py.File(self.filename, 'r') as hfile:
            try:
                self.rereferencing_possible = bool(hfile['monopolar_possible'][0])
            except KeyError:
                # Older versions of Ramulator recorded monopolar channels only
                # and did not include a flag indicating this.
                pass

            ts = hfile['/timeseries']

            if 'orient' in ts.attrs.keys() and ts.attrs['orient'] == b'row':
                data = np.array([ts[epoch[0]:epoch[1], :].T for epoch in self.epochs])
            else:
                data = np.array([ts[:, epoch[0]:epoch[1]] for epoch in self.epochs])

            return data


class EEGReader(BaseCMLReader):
    """Reads EEG data.

    Returns a :class:`TimeSeries`.

    Examples
    --------
    All examples start by defining a reader::

        >>> from cmlreaders import CMLReader
        >>> reader = CMLReader('R1111M', experiment='FR1', session=0)

    Loading a subset of EEG based on brain region::

        >>> contacts = reader.load('contacts')
        >>> eeg = reader.load_eeg(contacts=contacts[contacts.region == 'MTL'])

    Loading from explicitly specified epochs::

        >>> epochs = [(100, 200), (300, 400)]
        >>> eeg = reader.load_eeg(epochs=epochs)

    Loading an entire session::

        >>> eeg = reader.load_eeg()

    """
    data_types = ["eeg"]
    default_representation = "timeseries"

    # metainfo loaded from sources.json
    sources_info = {}

    def load(self, **kwargs):
        """Overrides the generic load method so as to accept keyword arguments
        to pass along to :meth:`as_timeseries`.

        """
        finder = PathFinder(subject=self.subject,
                            experiment=self.experiment,
                            session=self.session,
                            rootdir=self.rootdir)

        path = Path(finder.find('sources'))
        with path.open() as metafile:
            self.sources_info = list(json.load(metafile).values())[0]
            self.sources_info['path'] = path

        if 'events' in kwargs:
            # convert events to epochs
            events = kwargs.pop('events')
            epochs = events_to_epochs(events,
                                      kwargs.pop('rel_start'),
                                      kwargs.pop('rel_stop'),
                                      self.sources_info['sample_rate'])
            kwargs['epochs'] = epochs

        return self.as_timeseries(**kwargs)

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

    @staticmethod
    def _ms_to_samples(ms: int, rate: Union[int, float]) -> int:
        """Convert milliseconds to samples given a sample rate in Hz."""
        return int(rate * ms / 1000.)

    def as_timeseries(self, epochs: Optional[List[Tuple[float, float]]] = None,
                      contacts: Optional[pd.DataFrame] = None,
                      scheme: Optional[pd.DataFrame] = None) -> TimeSeries:
        """Read the timeseries.

        Keyword arguments
        -----------------
        epochs
            When given, specify which epochs to read in ms.
        contacts
            Contacts to include when reading EEG data.
        scheme
            When given, attempt to rereference the data.

        Returns
        -------
        A time series with shape (channels, epochs, time). By default, this
        returns data as it was physically recorded (e.g., if recorded with a
        common reference, each channel will be a contact's reading referenced to
        the common reference, a.k.a. "monopolar channels").

        Raises
        ------
        ValueError
            When provided epochs are not all the same length
        RereferencingNotPossibleError
            When rereferincing is not possible.

        """
        basename = self.sources_info['name']
        sample_rate = self.sources_info['sample_rate']
        dtype = self.sources_info['data_format']
        eeg_filename = self.sources_info['path'].parent.joinpath('noreref', basename)
        reader_class = self._get_reader_class(basename)

        orig_epochs = None

        if epochs is None:
            epochs = [(0, -1)]
        else:
            to_samples = functools.partial(self._ms_to_samples, rate=sample_rate)
            orig_epochs = copy.deepcopy(epochs)
            epochs = [(to_samples(e[0]), to_samples(e[1])) for e in epochs]

        if contacts is not None:
            raise NotImplementedError("filtering contacts is not yet implemented")

        tlens = np.array([e[-1] - e[0] for e in epochs])
        if not np.all(tlens == tlens[0]):
            raise ValueError("Epoch lengths are not all the same!")

        reader = reader_class(filename=eeg_filename,
                              sample_rate=sample_rate,
                              dtype=dtype,
                              epochs=epochs)  # TODO: channels
        data = reader.read()

        if scheme is not None:
            if not reader.rereferencing_possible:
                raise RereferencingNotPossibleError
            data = self.rereference(data, scheme)

        # TODO: channels, tstart
        ts = TimeSeries(data, sample_rate, epochs=epochs)
        return ts

    def rereference(self, data: np.ndarray, scheme: pd.DataFrame) -> np.ndarray:
        """Attempt to rereference the EEG data using the specified scheme.

        Parameters
        ----------
        data
            Input timeseries data shaped as (epochs, channels, time).
        scheme
            Bipolar pairs to use. This should be at a minimum a
            :class:`pd.DataFrame` with columns ``contact_1`` and ``contact_2``.

        Returns
        -------
        reref
            Rereferenced timeseries.

        Notes
        -----
        This method is meant to be used when loading data and so returns a raw
        Numpy array. If used externally, a :class:`TimeSeries` will need to be
        constructed manually.

        """
        c1, c2 = scheme.contact_1 - 1, scheme.contact_2 - 1
        reref = np.array(
            [data[i, c1, :] - data[i, c2, :] for i in range(data.shape[0])]
        )
        return reref
