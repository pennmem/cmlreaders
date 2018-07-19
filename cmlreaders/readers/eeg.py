from abc import abstractmethod, ABC
import json
import os
from pathlib import Path
from typing import List, Tuple, Type, Union
import warnings

with warnings.catch_warnings():  # noqa
    # Some versions of h5py produce a FutureWarning from a numpy import; we can
    # safely ignore it.
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

import numpy as np
import pandas as pd

from cmlreaders import constants, convert
from cmlreaders.base_reader import BaseCMLReader
from cmlreaders.eeg_container import EEGContainer
from cmlreaders.exc import (
    RereferencingNotPossibleError, UnsupportedOutputFormat,
    IncompatibleParametersError,
)
from cmlreaders.path_finder import PathFinder
from cmlreaders.readers.readers import EventReader
from cmlreaders.util import get_protocol, get_root_dir


class EEGMetaReader(BaseCMLReader):
    """Reads the ``sources.json`` or ``params.txt`` files which describes
    metainfo about EEG data.

    Returns a :class:`dict`.

    """
    data_types = ["sources"]
    default_representation = "dict"

    def _read_sources_json(self) -> dict:
        """Read from a sources.json file."""
        with open(self._file_path, 'r') as metafile:
            sources_info = list(json.load(metafile).values())[0]
            sources_info['path'] = self._file_path
            return sources_info

    def _read_params_txt(self) -> dict:
        """Read from a params.txt file and coerces to a similar format as
        sources.json.

        """
        df = pd.read_table(self._file_path, sep=' ', header=None, index_col=0).T

        sources_info = {
            "sample_rate": float(df["samplerate"].iloc[0]),
            "data_format": df["dataformat"].str.replace("'", "").iloc[0],
            "n_samples": None,
            "path": self._file_path,
        }

        return sources_info

    def as_dict(self) -> dict:
        if self.protocol in ["r1", "ltp"]:
            return self._read_sources_json()
        else:
            return self._read_params_txt()


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
    scheme
        Scheme data to use for rereferencing/channel filtering. This should be
        loaded/manipulated from ``pairs.json`` data.

    Notes
    -----
    The :meth:`read` method must be implemented by subclasses to return a tuple
    containing a 3-D array with dimensions (epochs x channels x time) and a
    list of contact numbers.

    """
    def __init__(self, filename: str, dtype: Type[np.dtype],
                 epochs: List[Tuple[int, Union[int, None]]],
                 scheme: Union[pd.DataFrame, None]):
        self.filename = filename
        self.dtype = dtype
        self.epochs = epochs
        self.scheme = scheme

        self._unique_contacts = np.union1d(
            self.scheme["contact_1"],
            self.scheme["contact_2"]
        ) if self.scheme is not None else None

        # in cases where we can't rereference, this will get changed to False
        self.rereferencing_possible = True

    def include_contact(self, contact_num: int):
        """Filter to determine if we need to include a contact number when
        reading data.

        """
        if self._unique_contacts is not None:
            return contact_num in self._unique_contacts
        else:
            return True

    @abstractmethod
    def read(self) -> Tuple[np.ndarray, List[int]]:
        """Read the data."""

    def rereference(self, data: np.ndarray, contacts: List[int]) -> np.ndarray:
        """Attempt to rereference the EEG data using the specified scheme.

        Parameters
        ----------
        data
            Input timeseries data shaped as (epochs, channels, time).
        contacts
            List of contact numbers (1-based) that index the data.

        Returns
        -------
        reref
            Rereferenced timeseries.

        Notes
        -----
        This method is meant to be used when loading data and so returns a raw
        Numpy array. If used externally, a :class:`EEGContainer` will need to be
        constructed manually.

        """
        contact_to_index = {
            c: i
            for i, c in enumerate(contacts)
        }

        c1 = [contact_to_index[c] for c in self.scheme["contact_1"]
              if c in contact_to_index]
        c2 = [contact_to_index[c] for c in self.scheme["contact_2"]
              if c in contact_to_index]

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

        contacts = []
        memmaps = []

        for f in files:
            contact_num = int(f.name.split(".")[-1])
            if not self.include_contact(contact_num):
                continue
            contacts.append(contact_num)
            memmaps.append(np.memmap(f, dtype=self.dtype, mode='r'))

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

            # FIXME: only select channels we care about
            if 'orient' in ts.attrs.keys() and ts.attrs['orient'] == b'row':
                data = np.array([ts[epoch[0]:epoch[1], :].T for epoch in self.epochs])
            else:
                data = np.array([ts[:, epoch[0]:epoch[1]] for epoch in self.epochs])

            contacts = hfile["ports"][:]

            return data, contacts

    def rereference(self, data: np.ndarray, contacts: List[int]) -> np.ndarray:
        """Overrides the default rereferencing to first check validity of the
        passed scheme or if rereferencing is even possible in the first place.

        """
        if self.rereferencing_possible:
            return BaseEEGReader.rereference(self, data, contacts)

        with h5py.File(self.filename, 'r') as hfile:
            bpinfo = hfile['bipolar_info']
            all_nums = [
                (int(a), int(b)) for (a, b) in list(
                    zip(bpinfo['ch0_label'][:], bpinfo['ch1_label'][:])
                )
            ]

        scheme_nums = list(zip(self.scheme["contact_1"],
                               self.scheme["contact_2"]))
        is_valid_channel = [channel in all_nums for channel in scheme_nums]

        if not all(is_valid_channel):
            raise RereferencingNotPossibleError(
                'The following channels are missing: %s' % (
                    ', '.join(
                        label for (label, valid) in
                        zip(self.scheme["label"], is_valid_channel)
                        if not valid
                    )
                )
            )

        # allow a subset of channels
        channel_inds = [chan in scheme_nums for chan in all_nums]
        return data[:, channel_inds, :]


class EEGReader(BaseCMLReader):
    """Reads EEG data.

    Returns a :class:`EEGContainer`.

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

    Loading EEG from -100 ms to +100 ms relative to a set of events::

        >>> events = reader.load("events")
        >>> eeg = reader.load_eeg(events, rel_start=-100, rel_stop=100)

    Loading an entire session::

        >>> eeg = reader.load_eeg()

    """
    data_types = ["eeg"]
    default_representation = "timeseries"

    # referencing scheme
    scheme = None  # type: pd.DataFrame

    def _eegfile_absolute(self, events: pd.DataFrame) -> pd.DataFrame:
        """Convert possibly relative paths to EEG files in events to absolute
        paths.

        """
        # Only reformat if we have relative path names. Data stored in
        # /protocols usually uses relative paths, whereas the older, Matlab-
        # based event processing uses absolute paths.
        def to_absolute(row):
            if row["eegfile"].startswith("/"):
                return row["eegfile"]

            subject = row["subject"]

            return "/" + constants.rhino_paths["processed_eeg"][0].format(
                protocol=get_protocol(subject),
                subject=subject,
                experiment=row["experiment"],
                session=row["session"],
                basename=row["eegfile"]
            )

        events.loc[:, "eegfile"] = events.apply(to_absolute, axis=1)
        return events

    def load(self, **kwargs):
        """Overrides the generic load method so as to accept keyword arguments
        to pass along to :meth:`as_timeseries`.

        """
        if "events" in kwargs:
            events = kwargs["events"]
        else:
            if self.session is None:
                raise ValueError(
                    "A session must be specified to load an entire session of "
                    "EEG data!"
                )

            finder = PathFinder(subject=self.subject,
                                experiment=self.experiment,
                                session=self.session,
                                rootdir=self.rootdir)

            events_file = finder.find("task_events")
            all_events = EventReader.fromfile(events_file, self.subject,
                                              self.experiment, self.session)

            # Select only a single event with a valid eegfile just to get the
            # filename
            valid = all_events[
                (all_events["eegfile"].notnull()) &
                (all_events["eegfile"].str.len() > 0)
            ]
            events = pd.DataFrame(valid.iloc[0]).T.reset_index(drop=True)

            # Set relative start and stop times if necessary. If they were
            # already specified, these will allow us to subset the session.
            if "rel_start" not in kwargs:
                kwargs["rel_start"] = 0
            if "rel_stop" not in kwargs:
                kwargs["rel_stop"] = -1

        if not len(events):
            raise ValueError("No events found! Hint: did filtering events "
                             "result in at least one?")
        elif len(events["subject"].unique()) > 1:
            raise ValueError("Loading multiple sessions of EEG data requires "
                             "using events from only a single subject.")

        if "rel_start" not in kwargs or "rel_stop" not in kwargs:
            raise IncompatibleParametersError(
                "rel_start and rel_stop must be given with events"
            )

        # info = EEGMetaReader.fromfile(path, subject=self.subject)
        # sample_rate = info["sample_rate"]
        # dtype = info["data_format"]

        self.scheme = kwargs.get("scheme", None)

        events = self._eegfile_absolute(events.copy())
        return self.as_timeseries(events, kwargs["rel_start"], kwargs["rel_stop"])

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

    def as_timeseries(self, events: pd.DataFrame,
                      rel_start: Union[float, int],
                      rel_stop: Union[float, int]) -> EEGContainer:
        """Read the timeseries.

        Parameters
        ----------
        events
            Events to read EEG data from
        rel_start
            Relative start times in ms
        rel_stop
            Relative stop times in ms

        Returns
        -------
        A time series with shape (channels, epochs, time). By default, this
        returns data as it was physically recorded (e.g., if recorded with a
        common reference, each channel will be a contact's reading referenced to
        the common reference, a.k.a. "monopolar channels").

        Raises
        ------
        RereferencingNotPossibleError
            When rereferencing is not possible.

        """
        eegs = []

        for filename in events["eegfile"].unique():
            # select subset of events for this basename
            ev = events[events["eegfile"] == filename]

            # determine experiment, session, dtype, and sample rate
            experiment = ev["experiment"].unique()[0]
            session = ev["session"].unique()[0]
            finder = PathFinder(subject=self.subject,
                                experiment=experiment,
                                session=session,
                                rootdir=self.rootdir)
            sources = EEGMetaReader.fromfile(finder.find("sources"),
                                             subject=self.subject)
            sample_rate = sources["sample_rate"]
            dtype = sources["data_format"]

            # convert events to epochs
            if rel_stop < 0:
                # We're trying to load to the end of the session. Only allow
                # this in cases where we're trying to load a whole session.
                if len(events) > 1:
                    raise ValueError("rel_stop must not be negative")
                epochs = [(0, None)]
            else:
                epochs = convert.events_to_epochs(ev,
                                                  rel_start, rel_stop,
                                                  sample_rate)

            root = get_root_dir(self.rootdir)
            eeg_filename = os.path.join(root, filename.lstrip("/"))
            reader_class = self._get_reader_class(filename)
            reader = reader_class(filename=eeg_filename,
                                  dtype=dtype,
                                  epochs=epochs,
                                  scheme=self.scheme)
            data, contacts = reader.read()

            if self.scheme is not None:
                data = reader.rereference(data, contacts)
                channels = self.scheme.label.tolist()
            else:
                channels = ["CH{}".format(n + 1) for n in range(data.shape[1])]

            eegs.append(
                EEGContainer(
                    data,
                    sample_rate,
                    epochs=epochs,
                    events=ev,
                    channels=channels,
                )
            )

        eegs = EEGContainer.concatenate(eegs)
        eegs.attrs["rereferencing_possible"] = reader.rereferencing_possible
        return eegs
