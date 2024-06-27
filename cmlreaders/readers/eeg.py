from abc import abstractmethod, ABC
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

from cmlreaders import constants, convert, exc
from cmlreaders.base_reader import BaseCMLReader
from cmlreaders.eeg_container import EEGContainer
from cmlreaders.path_finder import PathFinder
from cmlreaders.readers.readers import EventReader
from cmlreaders.util import get_protocol, get_root_dir
from cmlreaders.warnings import MissingChannelsWarning


class EEGMetaReader(BaseCMLReader):
    """Reads the ``sources.json`` or ``params.txt`` files which describes
    metainfo about EEG data.

    EEGMetaReader uses the following logic to combine entries in
    ``sources.json``:

    - If all recordings in ``sources.json`` have the same value for a field,
      then the dictionary returned by EEGMetaReader has that value for the
      field
    - Otherwise, that field should be populated by a list of the values
      present in ``sources.json``

    """

    data_types = ["sources"]
    default_representation = "dict"

    def _read_sources_json(self) -> dict:
        """Read from a sources.json file."""
        df = pd.read_json(self.file_path, orient="index")
        sources_info = {}
        for k in df:
            if any(df[k].apply(lambda x: isinstance(x, dict))):
                continue
            v = df[k].unique()
            sources_info[k] = v[0] if len(v) == 1 else v
        sources_info["path"] = self.file_path
        return sources_info

    def _read_params_txt(self) -> dict:
        """Read from a params.txt file and coerces to a similar format as
        sources.json.

        """
        df = pd.read_csv(self.file_path, sep=" ", header=None, index_col=0).T

        sources_info = {
            "sample_rate": float(df["samplerate"].iloc[0]),
            "data_format": df["dataformat"].str.replace("'", "").iloc[0],
            "n_samples": None,
            "path": self.file_path,
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
        loaded/manipulated from ``pairs.json`` data. (Currently available for
        iEEG only.)
    clean
        If True, load re-referenced, filtered, and ICA/LCF-cleaned version of
        data (currently available for scalp EEG only). If false, load raw data.

    Notes
    -----
    The :meth:`read` method must be implemented by subclasses to return a tuple
    containing a 3-D array with dimensions (epochs x channels x time) and a
    list of contact numbers.

    """

    def __init__(
        self,
        filename: str,
        dtype: Type[np.dtype],
        epochs: List[Tuple[int, Union[int, None]]],
        scheme: Union[pd.DataFrame, None],
        clean: Union[bool, None],
    ):
        self.filename = filename
        self.dtype = dtype
        self.epochs = epochs
        self.scheme = scheme
        self.clean = False if clean is None else clean

        try:
            if self.scheme_type == "contacts":
                self._unique_contacts = self.scheme.contact.unique()
            elif self.scheme_type == "pairs":
                self._unique_contacts = np.union1d(
                    self.scheme["contact_1"], self.scheme["contact_2"]
                )
            else:
                self._unique_contacts = None
        except KeyError:
            self._unique_contacts = None

        # in cases where we can't rereference, this will get changed to False
        self.rereferencing_possible = True

    @property
    def scheme_type(self) -> Union[str, None]:
        """Returns "contacts" when the input scheme is in the form of monopolar
        contacts and "pairs" when bipolar.

        Returns
        -------
        The scheme type or ``None`` is no scheme was specified.

        Raises
        ------
        KeyError
            When the passed scheme doesn't include any of the following keys:
            ``contact_1``, ``contact_2``, ``contact``

        """
        if self.scheme is None:
            return None

        if "contact_1" in self.scheme and "contact_2" in self.scheme:
            return "pairs"
        elif "contact" in self.scheme:
            return "contacts"
        else:
            raise KeyError(
                "The passed scheme appears to be neither contacts " "nor pairs"
            )

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

    def rereference(
        self, data: np.ndarray, contacts: List[int]
    ) -> Tuple[np.ndarray, List[str]]:
        """Rereference and/or select a subset of raw channels.

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
        labels
            List of channel labels used (included in case some don't get used).

        Notes
        -----
        This method is meant to be used when loading data and so returns a raw
        Numpy array. If used externally, a :class:`EEGContainer` will need to
        be constructed manually.

        """
        if self.scheme_type == "pairs":
            contact_1_to_index_df = pd.DataFrame({'contact_1': contacts}).reset_index()
            contact_2_to_index_df = pd.DataFrame({'contact_2': contacts}).reset_index()
            pairs = self.scheme.reset_index(names='pairs_index')
            pairs_to_index_df = pairs.merge(contact_1_to_index_df
                                            ).merge(contact_2_to_index_df,
                                                    on='contact_2', suffixes=('_1', '_2')
                                                    ).sort_values('pairs_index')
            c1 = pairs_to_index_df["index_1"].tolist()
            c2 = pairs_to_index_df["index_2"].tolist()

            reref = np.array(
                [data[i, c1, :] - data[i, c2, :] for i in range(data.shape[0])]
            )
            return reref, pairs_to_index_df["label"].tolist()
        else:
            contact_to_index = {c: i for i, c in enumerate(contacts)}
            channels = [contact_to_index[c] for c in self.scheme["contact"]]
            subset = np.array([data[i, channels, :] for i in range(data.shape[0])])
            return subset, self.scheme["label"].tolist()


class NumpyEEGReader(BaseEEGReader):
    """Read EEG data stored in Numpy's .npy format.

    Notes
    -----
    This reader is currently only used to do some testing so lacks some
    features such as being able to determine what contact numbers it's actually
    using. Instead, it will just give contacts as a sequential list of ints.

    """

    def read(self) -> Tuple[np.ndarray, List[int]]:
        raw = np.load(self.filename)
        data = np.array(
            [raw[:, e[0] : (e[1] if e[1] > 0 else None)] for e in self.epochs]
        )
        contacts = [i + 1 for i in range(data.shape[1])]
        return data, contacts


class SplitEEGReader(BaseEEGReader):
    """Read so-called split EEG data (that is, raw binary data stored as one
    channel per file).

    """

    def _get_files(self, glob_pattern: str) -> List[Path]:
        files = sorted(Path(self.filename).parent.glob(glob_pattern + ".*"))
        return files

    def read(self) -> Tuple[np.ndarray, List[int]]:
        pattern = Path(self.filename).name
        files = self._get_files(pattern)

        # Some experiments have errors in the EEG splitting which results in
        # mismatches between the names of the actual EEG files and what the
        # events say they should be.
        if len(files) == 0:
            names = Path(self.filename).name.split("_")
            pattern = "*".join(names)
            files = self._get_files(pattern)

            if len(files) == 0:
                raise ValueError(
                    "split EEG filenames don't seem to match what" " are in the events"
                )

        contacts = []
        memmaps = []

        for f in files:
            contact_num = int(f.name.split(".")[-1])
            if not self.include_contact(contact_num):
                continue
            contacts.append(contact_num)
            memmaps.append(np.memmap(f, dtype=self.dtype, mode="r"))
        # check epochs exist in eeg
        # for i, epoch in enumerate(self.epochs):
        #    for mmap in memmaps:
        #        assert epoch[1] < len(mmap), f"Must have epoch end
        #       {epoch[1]} < length {len(mmap)}"
        #        assert len(mmap[epoch[0]:epoch[1]]) == epoch[1] - epoch[0],
        #           f"epoch difference not equal to length of mmap {len(mmap)}"
        try:
            data = np.array(
                [[mmap[epoch[0] : epoch[1]] for mmap in memmaps] for epoch in self.epochs]
            )
        except ValueError:  # requesting event-epoched data beyond final sample of EEG
            raise exc.MissingDataError(
                "Requesting EEG beyond the boundaries of the EEG recording. If loading " +
                "un-epoched EEG, try loading event-epoched EEG. If loading event-epoched " +
                "EEG, either modify rel_start/rel_stop parameters or try dropping " +
                "the first or last event(s)."
            )

        return data, contacts


class EDFReader(BaseEEGReader):
    def read(self) -> Tuple[np.ndarray, List[int]]:
        raise NotImplementedError


class RamulatorHDF5Reader(BaseEEGReader):
    """Reads Ramulator HDF5 EEG files."""

    def read(self) -> Tuple[np.ndarray, List[int]]:
        with h5py.File(self.filename, "r") as hfile:
            try:
                self.rereferencing_possible = bool(hfile["monopolar_possible"][0])
            except KeyError:
                # Older versions of Ramulator recorded monopolar channels only
                # and did not include a flag indicating this.
                pass
                # Equivalent to:
                # self.rereferencing_possible = True
                # Because set as true in constructor

            ts = hfile["/timeseries"]


            if "orient" in ts.attrs.keys() and ts.attrs["orient"] == b"row":
                data = np.array(
                    [ts[epoch[0] : epoch[1], :].T for epoch in self.epochs]
                )
            else:
                data = np.array(
                    [ts[:, epoch[0] : epoch[1]] for epoch in self.epochs]
                )

            if self.rereferencing_possible or self.scheme_type == "contacts":
                contacts = hfile["ports"]#[idxs]
            else:
                bpinfo = hfile["bipolar_info"]
                bpinfo_df = pd.DataFrame({'ch0_label': bpinfo["ch0_label"][:].astype(int), 
                                          'ch1_label': bpinfo["ch1_label"][:].astype(int),
                                          'contact_name': bpinfo["contact_name"][:]}
                                          ).reset_index(names='bp_index')
                
                # hack to pass test suite
                if self.scheme_type != "pairs":
                    bpinfo_df['ch0_label_ordered'] = np.min(bpinfo_df[['ch0_label', 'ch1_label']], axis=1)
                    bpinfo_df['ch1_label_ordered'] = np.max(bpinfo_df[['ch0_label', 'ch1_label']], axis=1)
                    bpinfo_df.drop_duplicates(subset=['ch0_label_ordered', 'ch1_label_ordered'], inplace=True)
                    contacts = bpinfo_df['contact_name'].tolist()
                    # use bpinfo to select inds in eeg data
                    channel_inds = bpinfo_df['bp_index'].astype(int).tolist()
                else:
                    pairs = self.scheme.reset_index(names='pairs_index')
                    pairs_bpinfo_all_df = pairs.merge(bpinfo_df, right_on=['ch0_label', 'ch1_label'],
                                                      left_on=['contact_1', 'contact_2'], how='left',
                                                      indicator=True).sort_values('pairs_index')
                    # only use pairs that are in the scheme and the actual recording
                    pairs_bpinfo_df = pairs_bpinfo_all_df.query('_merge == "both"')
                    contacts = pairs_bpinfo_df['label'].tolist()
                    # use bpinfo to select inds in eeg data
                    channel_inds = pairs_bpinfo_df['bp_index'].astype(int).tolist()

                    if not len(contacts) > 0:
                        raise exc.RereferencingNotPossibleError(
                            "No channels specified in scheme are present in EEG recording"
                        )

                    pairs_only_df = pairs_bpinfo_df.query('_merge == "left"')
                    if len(pairs_only_df) > 0:
                        # Some channels included in the scheme are not present in the
                        # actual recording
                        msg = "The following channels are missing: {:s}".format(
                            ", ".join(pairs_only_df["label"])
                        )
                        warnings.warn(msg, MissingChannelsWarning)

            return data[:, channel_inds, :], contacts

    def rereference(
        self, data: np.ndarray, contacts: List[int]
    ) -> Tuple[np.ndarray, List[str]]:
        """Overrides the default rereferencing to first check validity of the
        passed scheme or if rereferencing is even possible in the first place.

        """
        if self.rereferencing_possible or self.scheme_type == "contacts":
            return BaseEEGReader.rereference(self, data, contacts)

        if not len(contacts) > 0:
            raise exc.RereferencingNotPossibleError(
                "No channels specified in scheme are present in EEG recording"
            )

        # otherwise, data is already rereferenced and contacts IS labels
        labels = contacts
        return data, labels


class ScalpEEGReader(BaseEEGReader):
    def read(self):
        import mne

        # To read cleaned/preprocessed data (.fif)
        if self.clean:
            clean_eegfile = os.path.splitext(self.filename)[0] + "_clean_raw.fif"
            eeg = mne.io.read_raw_fif(clean_eegfile, preload=True)
        # To read BioSemi data (.bdf)
        elif self.dtype == ".bdf":
            eeg = mne.io.read_raw_bdf(
                self.filename,
                eog=["EXG1", "EXG2", "EXG3", "EXG4"],
                misc=["EXG5", "EXG6", "EXG7", "EXG8"],
                stim_channel="Status",
                preload=True,
            )
        # To read EGI data (.raw/.mff)
        else:
            eeg = mne.io.read_raw_egi(self.filename, preload=True)
            eeg.rename_channels({"E129": "Cz"})
            eeg.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-129"))
            eeg.set_channel_types(
                {"E8": "eog", "E25": "eog", "E126": "eog", "E127": "eog", "Cz": "misc"}
            )

        # If no event information is provided, return continuous data as a
        # single epoch
        if self.epochs is None:
            data = np.expand_dims(eeg._data, axis=0)
        # If event information is provided, return epoched data
        else:
            # Remove any events that run beyond the beginning or end of the EEG
            # recording
            truncated_events_pre = 0
            truncated_events_post = 0
            while (
                self.epochs["epochs"][0, 0] + eeg.info["sfreq"] * self.epochs["tmin"]
                < 0
            ):
                self.epochs["epochs"] = self.epochs["epochs"][1:]
                truncated_events_pre += 1
            while (
                self.epochs["epochs"][-1, 0] + eeg.info["sfreq"] * self.epochs["tmax"]
                >= eeg.n_times
            ):
                self.epochs["epochs"] = self.epochs["epochs"][:-1]
                truncated_events_post += 1
            # Cut continuous data into epochs
            eeg = mne.Epochs(
                eeg,
                self.epochs["epochs"],
                tmin=self.epochs["tmin"],
                tmax=self.epochs["tmax"],
                baseline=None,
                preload=True,
            )
            data = eeg._data
            # Add information about how many events needed to be truncated
            eeg.info["temp"] = {
                "truncated_events_pre": truncated_events_pre,
                "truncated_events_post": truncated_events_post,
            }

        return data, eeg.info


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

    Loading multiple sessions from the same subject::

        >>> events = CMLReader.load_events(["R1111M"], ["FR1"])
        >>> words = events[events["type"] == "WORD"]
        >>> reader = CMLReader("R1111M")
        >>> eeg = reader.load_eeg(events=words, rel_start=-100, rel_stop=100)

    """

    data_types = ["eeg"]
    default_representation = "timeseries"

    # referencing scheme
    clean = None  # type: bool
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

            subject = self.subject if self.subject is not None else row["subject"]
            experiment = (
                self.experiment if self.experiment is not None else row["experiment"]
            )
            session = self.session if self.session is not None else row["session"]

            return "/" + constants.rhino_paths["processed_eeg"][0].format(
                protocol=get_protocol(subject),
                subject=subject,
                experiment=experiment,
                session=session,
                basename=row["eegfile"],
            )

        events.loc[:, "eegfile"] = events.apply(to_absolute, axis=1)
        return events

    def load(self, **kwargs):
        """Overrides the generic load method so as to accept keyword arguments
        to pass along to :meth:`as_timeseries`.

        """
        if "events" in kwargs:
            events = kwargs["events"]  # type: pd.DataFrame
            # drop any invalid eegoffset events
            if np.any(events["eegoffset"] < 0):
                warnings.warn("Some events have eegoffset < 0 and will be dropped.")
            events = events[events["eegoffset"] >= 0]
        else:
            if self.session is None:
                raise ValueError(
                    "A session must be specified to load an entire session of "
                    "EEG data!"
                )

            # Because of reasons, PS4 experiments may or may not end with a 5.
            if self.experiment.startswith("PS4") and self.experiment.endswith("5"):
                experiment = self.experiment[:-1]
            else:
                experiment = self.experiment

            finder = PathFinder(
                subject=self.subject,
                experiment=experiment,
                session=self.session,
                rootdir=self.rootdir,
            )

            events_file = finder.find("task_events")
            all_events = EventReader.fromfile(
                events_file, self.subject, self.experiment, self.session
            )

            # Select only a single event with a valid eegfile just to get the
            # filename
            valid = all_events[
                (all_events["eegfile"].notnull())
                & (all_events["eegfile"].str.len() > 0)
            ]
            events = pd.DataFrame(valid.iloc[0]).T.reset_index(drop=True)

            # Set relative start and stop times if necessary. If they were
            # already specified, these will allow us to subset the session.
            if "rel_start" not in kwargs:
                kwargs["rel_start"] = 0
            if "rel_stop" not in kwargs:
                kwargs["rel_stop"] = -1

        if not len(events):
            raise ValueError(
                "No events found! Hint: did filtering events " "result in at least one?"
            )
        elif len(events["subject"].unique()) > 1:
            raise ValueError(
                "Loading multiple sessions of EEG data requires "
                "using events from only a single subject."
            )

        if "rel_start" not in kwargs or "rel_stop" not in kwargs:
            raise exc.IncompatibleParametersError(
                "rel_start and rel_stop must be given with events"
            )

        # info = EEGMetaReader.fromfile(path, subject=self.subject)
        # sample_rate = info["sample_rate"]
        # dtype = info["data_format"]

        self.clean = kwargs.get("clean", None)
        self.scheme = kwargs.get("scheme", None)

        events = self._eegfile_absolute(events.copy())
        return self.as_timeseries(events, kwargs["rel_start"], kwargs["rel_stop"])

    def as_dataframe(self):
        raise exc.UnsupportedOutputFormat

    def as_recarray(self):
        raise exc.UnsupportedOutputFormat

    def as_dict(self):
        raise exc.UnsupportedOutputFormat

    @staticmethod
    def _get_reader_class(basename: str) -> Type[BaseEEGReader]:
        """Return the class to use for loading EEG data."""
        if basename.endswith(".h5"):
            return RamulatorHDF5Reader
        elif basename.endswith((".bdf", ".mff", ".raw")):
            return ScalpEEGReader
        elif basename.endswith(".npy"):
            return NumpyEEGReader
        else:
            return SplitEEGReader

    def as_timeseries(
        self,
        events: pd.DataFrame,
        rel_start: Union[float, int],
        rel_stop: Union[float, int],
    ) -> EEGContainer:
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
        common reference, each channel will be a contact's reading referenced
        to the common reference, a.k.a. "monopolar channels").

        Raises
        ------
        RereferencingNotPossibleError
            When rereferencing is not possible.

        """
        eegs = []

        # sanity check on the offsets

        if rel_start != 0 and rel_stop != -1 and rel_start > rel_stop:
            raise ValueError("rel_start must precede rel_stop")

        for filename in events["eegfile"].unique():
            # select subset of events for this basename
            ev = events[events["eegfile"] == filename]

            # determine experiment, session, dtype, and sample rate
            experiment = ev["experiment"].unique()[0]
            session = ev["session"].unique()[0]
            basename = os.path.basename(filename)
            finder = PathFinder(
                subject=self.subject,
                experiment=experiment,
                session=session,
                eeg_basename=basename,
                rootdir=self.rootdir,
            )
            sources = EEGMetaReader.fromfile(
                finder.find("sources"), subject=self.subject
            )
            sample_rate = sources["sample_rate"]
            dtype = sources["data_format"]
            is_scalp = dtype in (".bdf", ".raw", ".mff")

            # Convert events to epochs (onset & offset times)
            if rel_start == 0 and rel_stop == -1 and len(events) == 1:
                epochs = [(0, None)]
            else:
                # drop events beyond bounds of EEG samples
                n_samples = sources["n_samples"]
                if not isinstance(n_samples, np.ndarray):         # pass if multiple EEG files
                    rstart = convert.milliseconds_to_samples(rel_start, sample_rate)
                    rstop = convert.milliseconds_to_samples(rel_stop, sample_rate)
                    # only events within boundaries
                    evs = ev[(ev['eegoffset'] - rstart >= 0) &
                             (ev['eegoffset'] + rstop <= n_samples)]
                    if len(evs) == 0:                         # raise error if all events dropped
                        raise exc.MissingDataError(
                            f"All {len(ev)} event epochs beyond boundaries of EEG recording."
                        )
                    elif len(evs) < len(ev):
                        drop_idx = ev.index.difference(evs.index).to_numpy()   # event idx to drop
                        warnings.warn(
                            f"Dropping {len(ev) - len(evs)} event(s) at index(es) {drop_idx} " +
                            "with epochs beyond the boundaries of the EEG recording."
                        )
                        ev = evs             # need to store in "ev" variable

                epochs = convert.events_to_epochs(ev, rel_start, rel_stop, sample_rate)

            # Scalp EEG reader requires onsets, rel_start (in sec), and rel_
            # stop (in sec) to cut data into epochs
            if is_scalp:
                on_off_epochs = epochs  # The onset & offset times will still
                # be passed to the EEGContainer later
                if rel_start == 0 and rel_stop == -1 and len(events) == 1:
                    epochs = None
                else:
                    epochs = np.zeros((len(ev), 3), dtype=int)
                    epochs[:, 0] = ev["eegoffset"]
                    epochs = dict(
                        epochs=epochs, tmin=rel_start / 1000.0, tmax=rel_stop / 1000.0
                    )

            root = get_root_dir(self.rootdir)
            eeg_filename = os.path.join(root, filename.lstrip("/"))
            reader_class = self._get_reader_class(filename)
            reader = reader_class(
                filename=eeg_filename,
                dtype=dtype,
                epochs=epochs,
                scheme=self.scheme,
                clean=self.clean,
            )
            # if scalp EEG, info is an MNE Info object; if iEEG, info is a list
            # of contacts
            data, info = reader.read()

            attrs = {}
            if is_scalp:
                # Pass MNE info and events as extra attributes, to be able to
                # fully reconstruct MNE Raw/Epochs objects
                attrs["mne_info"] = info
                channels = info["ch_names"]
                if epochs is not None:
                    #                     Crop out any events/epoch times that ran beyond the
                    #                     bounds of the EEG recording
                    te_pre = (
                        info["temp"]["truncated_events_pre"]
                        if info["temp"]["truncated_events_pre"] > 0
                        else None
                    )
                    te_post = (
                        -info["temp"]["truncated_events_post"]
                        if info["temp"]["truncated_events_post"] > 0
                        else None
                    )
                    on_off_epochs = on_off_epochs[te_pre:te_post]
                    ev = ev[te_pre:te_post]
                # Pass the onset & offset time epoch list to EEGContainer, NOT
                # the MNE-formatted epoch list
                epochs = on_off_epochs
            elif self.scheme is not None:
                data, channels = reader.rereference(data, info)
            else:
                channels = ["CH{}".format(n + 1) for n in range(data.shape[1])]

            eegs.append(
                EEGContainer(
                    data,
                    sample_rate,
                    epochs=epochs,
                    events=ev,
                    channels=channels,
                    tstart=rel_start,
                    attrs=attrs,
                )
            )

        eegs = EEGContainer.concatenate(eegs)
        eegs.attrs["rereferencing_possible"] = reader.rereferencing_possible
        return eegs
