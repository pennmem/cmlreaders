from abc import abstractmethod, ABC
from collections import OrderedDict
import itertools
import json
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union
import warnings

with warnings.catch_warnings():  # noqa
    # Some versions of h5py produce a FutureWarning from a numpy import; we can
    # safely ignore it.
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

import numpy as np
import pandas as pd

from cmlreaders.base_reader import BaseCMLReader
from cmlreaders.exc import (
    RereferencingNotPossibleError, UnsupportedOutputFormat
)
from cmlreaders.path_finder import PathFinder
from cmlreaders.timeseries import TimeSeries


__all__ = [
    "EEGReader",
    "milliseconds_to_events",
    "milliseconds_to_samples",
]


def milliseconds_to_samples(millis: Union[int, float],
                            sample_rate: Union[int, float]) -> int:
    """Covert times in milliseconds to number of samples.

    Parameters
    ----------
    millis
        Time in ms.
    sample_rate
        Sample rate in samples per second.

    Returns
    -------
    Number of samples.

    """
    return int(sample_rate * millis / 1000.)


def samples_to_milliseconds(samples: int,
                            sample_rate: Union[int, float]) -> Union[int, float]:
    """Convert samples to milliseconds.

    Parameters
    ----------
    samples
        Number of samples.
    sample_rate
        Sample rate in samples per second.

    Returns
    -------
    Samples converted to milliseconds.

    """
    return 1000 * samples / sample_rate


def milliseconds_to_events(onsets: List[Union[int, float]],
                           sample_rate: Union[int, float]) -> pd.DataFrame:
    """Take times and produce a minimal events :class:`pd.DataFrame` to load
    EEG data with.

    Parameters
    ----------
    onsets
        Onset times in ms.
    sample_rate
        Sample rate in samples per second.

    Returns
    -------
    events
        A :class:`pd.DataFrame` with ``eegoffset`` as the only column.

    """
    samples = [milliseconds_to_samples(onset, sample_rate) for onset in onsets]
    return pd.DataFrame({"eegoffset": samples})


def events_to_epochs(events: pd.DataFrame, rel_start: int, rel_stop: int,
                     sample_rate: Union[int, float],
                     basenames: Optional[List[str]] = None
                     ) -> List[Tuple[int, int, int]]:
    """Convert events to epochs.

    Parameters
    ----------
    events
        Events to read.
    rel_start
        Start time relative to events in ms.
    rel_stop
        Stop time relative to events in ms.
    sample_rate
        Sample rate in Hz.

    Returns
    -------
    epochs
        A list of tuples giving absolute start and stop times in number of
        samples.

    """
    rel_start = milliseconds_to_samples(rel_start, sample_rate)
    rel_stop = milliseconds_to_samples(rel_stop, sample_rate)
    offsets = events.eegoffset.values
    if basenames is not None:
        eegfiles = events.eegfile.values
        epochs = [(offset + rel_start, offset + rel_stop,
                   basenames.index(eegfile))
                  for (offset, eegfile) in zip(offsets, eegfiles)]
    else:
        epochs = [(offset + rel_start, offset + rel_stop, 0)
                  for offset in offsets]
    return epochs


class BaseEEGReader(ABC):
    """Base class for actually reading EEG data. Subclasses will be used by
    :class:`EEGReader` to actually read the format-specific EEG data.

    Parameters
    ----------
    filename
        Base name for EEG file(s) including absolute path
    dtype
        numpy dtype to use for reading data
    epochs
        Epochs to include. Epochs are defined with start and stop sample
        counts.

    Notes
    -----
    The :meth:`read` method must be implemented by subclasses to return a tuple
    containing a 3-D array with dimensions (epochs x channels x time) and a
    list of contact numbers.

    """
    def __init__(self, filename: str, dtype: Type[np.dtype],
                 epochs: List[Tuple[int, Union[int, None]]]):
        self.filename = filename
        self.dtype = dtype

        self.epochs = epochs

        # in cases where we can't rereference, this will get changed to False
        self.rereferencing_possible = True

    @abstractmethod
    def read(self) -> Tuple[np.ndarray, List[int]]:
        """Read the data."""

    def rereference(self, data: np.ndarray, contacts: List[int],
                    scheme: pd.DataFrame) -> np.ndarray:
        """Attempt to rereference the EEG data using the specified scheme.

        Parameters
        ----------
        data
            Input timeseries data shaped as (epochs, channels, time).
        contacts
            List of contact numbers (1-based) that index the data.
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
        contact_to_index = {
            c: i
            for i, c in enumerate(contacts)
        }

        c1 = [contact_to_index[c] for c in scheme.contact_1]
        c2 = [contact_to_index[c] for c in scheme.contact_2]

        reref = np.array(
            [data[i, c1, :] - data[i, c2, :] for i in range(data.shape[0])]
        )
        return reref


class NumpyEEGReader(BaseEEGReader):
    """Read EEG data stored in Numpy's .npy format.

    Notes
    -----
    This reader is currently only used to do some testing so lacks some features
    such as being able to determine what contact numbers it's actually using.
    Instead, it will just give contacts as a sequential list of ints.

    """
    def read(self) -> Tuple[np.ndarray, List[int]]:
        raw = np.load(self.filename)
        data = np.array([raw[:, e[0]:(e[1] if e[1] > 0 else None)]
                         for e in self.epochs])
        contacts = [i + 1 for i in range(data.shape[1])]
        return data, contacts


class SplitEEGReader(BaseEEGReader):
    """Read so-called split EEG data (that is, raw binary data stored as one
    channel per file).

    """
    @staticmethod
    def _read_epoch(mmap: np.memmap, epoch: Tuple[int, ...]) -> np.array:
        return np.array(mmap[epoch[0]:epoch[1]])

    def read(self) -> Tuple[np.ndarray, List[int]]:
        basename = Path(self.filename).name
        files = sorted(Path(self.filename).parent.glob(basename + '.*'))
        contacts = [int(f.name.split(".")[-1]) for f in files]

        memmaps = [np.memmap(f, dtype=self.dtype, mode='r') for f in files]

        data = np.array([
            [self._read_epoch(mmap, epoch) for mmap in memmaps]
            for epoch in self.epochs
        ])

        return data, contacts


class EDFReader(BaseEEGReader):
    def read(self) -> Tuple[np.ndarray, List[int]]:
        raise NotImplementedError


class RamulatorHDF5Reader(BaseEEGReader):
    """Reads Ramulator HDF5 EEG files."""
    def read(self) -> Tuple[np.ndarray, List[int]]:
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

            contacts = hfile["ports"][:]

            return data, contacts

    def rereference(self, data: np.ndarray, contacts: List[int],
                    scheme: pd.DataFrame):
        if self.rereferencing_possible:
            return BaseEEGReader.rereference(self, data, contacts, scheme)
        else:
            with h5py.File(self.filename, 'r') as hfile:
                bpinfo = hfile['bipolar_info']
                all_nums = [
                    (int(a), int(b)) for (a, b) in list(
                        zip(bpinfo['ch0_label'][()], bpinfo['ch1_label'][()])
                    )
                ]
            scheme_nums = list(zip(scheme['contact_1'], scheme['contact_2']))
            is_valid_channel = [channel in all_nums for channel in scheme_nums]

            if not all(is_valid_channel):
                raise RereferencingNotPossibleError(
                    'The following channels are missing: %s' % (
                        ', '.join(label) for (label, valid) in
                        zip(scheme.label, is_valid_channel)
                        if not valid)
                )

            channel_to_index = {c: i for (i, c) in enumerate(contacts)}
            channel_inds = [channel_to_index[c] for c in scheme.contact_1]

            return data[:, channel_inds, :]


class EEGReader(BaseCMLReader):
    """Reads EEG data.

    Returns a :class:`TimeSeries`.

    Examples
    --------
    All examples start by defining a reader::

        >>> from cmlreaders import CMLReader
        >>> reader = CMLReader("R1111M", experiment="FR1", session=0)

    Loading a subset of EEG based on brain region (this automatically
    re-references)::

        >>> pairs = reader.load("pairs")
        >>> filtered = pairs[pairs["avg.region"] == "middletemporal"]
        >>> eeg = reader.load_eeg(scheme=pairs)

    Loading from explicitly specified epochs (units are samples)::

        >>> epochs = [(100, 200), (300, 400)]
        >>> eeg = reader.load_eeg(epochs=epochs)

    Loading from specified epochs, when there are multiple files for the
    session (units are relative to the start of each file):

        >>> epochs = [(100,200,0), (100,200,1)]
        >>> eeg = reader.load_eeg(epochs=epochs)

    Loading EEG from -100 ms to +100 ms relative to a set of events:
        >>> events = reader.load("events")
        >>> eeg = reader.load_eeg(events, rel_start=-100, rel_stop=100)

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
            self.sources_info = json.load(metafile,
                                          object_pairs_hook=OrderedDict)
            self.sources_info['path'] = path

        if 'events' in kwargs:
            # convert events to epochs
            events = kwargs.pop('events')
            source_info_list = list(self.sources_info.values())[:-1]
            sample_rate = source_info_list[0]['sample_rate']
            basenames = [info['name'] for info in source_info_list]
            epochs = events_to_epochs(events,
                                      kwargs.pop('rel_start'),
                                      kwargs.pop('rel_stop'),
                                      sample_rate,
                                      basenames)
            kwargs['epochs'] = epochs

        if "epochs" not in kwargs:
            kwargs["epochs"] = [(0, -1)]

        kwargs["epochs"] = [e if len(e) == 3
                            else e + (0,)
                            for e in kwargs['epochs']]

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
        if basename.endswith(".h5"):
            return RamulatorHDF5Reader
        elif basename.endswith(".npy"):
            return NumpyEEGReader
        else:
            return SplitEEGReader

    def as_timeseries(self,
                      epochs: Optional[List[Tuple[int, Union[int, None], int]]] = None,
                      scheme: Optional[pd.DataFrame] = None) -> TimeSeries:
        """Read the timeseries.

        Keyword arguments
        -----------------
        epochs
            When given, specify which epochs to read in ms.
        scheme
            When given, attempt to rereference and/or filter the data.

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
        if epochs is None:
            epochs = [(0, None, 0)]

        ts = []

        for fileno, epoch_lst in itertools.groupby(epochs, key=lambda x: x[-1]):
            sources_info = list(self.sources_info.values())[fileno]
            basename = sources_info['name']
            sample_rate = sources_info['sample_rate']
            dtype = sources_info['data_format']
            eeg_filename = self.sources_info['path'].parent.joinpath('noreref', basename)
            reader_class = self._get_reader_class(basename)

            tlens = np.array([e[1] - e[0] for e in epochs])
            if not np.all(tlens == tlens[0]):
                raise ValueError("Epoch lengths are not all the same!")

            reader = reader_class(filename=eeg_filename,
                                  dtype=dtype,
                                  epochs=epochs)
            data, contacts = reader.read()

            if scheme is not None:
                data = reader.rereference(data, contacts, scheme)
                channels = scheme.label.tolist()
            else:
                channels = ["CH{}".format(n + 1) for n in range(data.shape[1])]

            # TODO: tstart
            ts.append(
                TimeSeries(data, sample_rate, epochs=epochs, channels=channels)
            )
        return TimeSeries.concatenate(ts)
