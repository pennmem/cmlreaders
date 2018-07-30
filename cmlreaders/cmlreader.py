import functools
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from . import readers
from .data_index import get_data_index
from .exc import IncompatibleParametersError, UnsupportedProtocolError
from .util import get_protocol, get_root_dir


__all__ = ['CMLReader']


class CMLReader(object):
    """ Generic reader for all CML-specific files

    Notes
    -----
    At import, all the readers from :mod:`cmlreaders.readers` will register the
    data types that should correspond to that reader by updating the
    reader_names dictionary. reader_names is a dict whose keys are one of
    the data types understood by :class:`cmlreaders.PathFinder` and defined in
    :mod:`cmlreaders.constants`. Values are the name of the reader class
    that should be used for loading/reading the data type. When an instance of
    :class:`cmlreaders.cmlreader.CMLReader` is instantiated, a new dictionary is
    created that maps the data types to the actual reader class, rather than
    just the class name. In essence, :class:`cmlreaders.cmlreader.CMLReader` is
    a factory that routes the requests for loading a particular data type to
    the reader defined to handle that data.

    """
    reader_names = {}

    def __init__(self, subject: Optional[str] = None,
                 experiment: Optional[str] = None,
                 session: Optional[int] = None,
                 localization: Optional[int] = None,
                 montage: Optional[int] = None,
                 rootdir: Optional[str] = None):

        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.rootdir = get_root_dir(rootdir)

        self._localization = localization
        self._montage = montage

        self.readers = {k: getattr(readers, v) for k, v in self.reader_names.items()}
        self.reader_protocols = {k: getattr(readers, v).protocols for k, v in self.reader_names.items()}

    def _load_index(self) -> pd.DataFrame:
        """Loads the data index. Used internally to determine montage and
        localization nubmers.

        """
        if self.subject is None:
            return pd.DataFrame({})

        index = get_data_index(self.protocol, rootdir=self.rootdir)

        # Some subjects don't explicitly specify localization/montage
        # numbers in the index, so they appear as NaNs.
        try:
            index["montage"].replace({np.nan: "0"}, inplace=True)
            index["localization"].replace({np.nan: "0"}, inplace=True)
        except KeyError:
            # We're using a protocol that doesn't include localization data
            # (e.g., ltp)
            pass

        return index

    def _determine_localization_or_montage(self, which: str) -> Optional[int]:
        """Inner workings of localization/montage properties.

        Returns
        -------
        Montage or localization number if all are the same, otherwise None.

        """
        if which not in ["localization", "montage"]:
            raise ValueError

        index = self._load_index()
        df = index[index["subject"] == self.subject]

        if which not in df:
            setattr(self, "_" + which, None)
            return None

        if self.experiment is not None:
            df = df[df.experiment == self.experiment]

        if self.session is not None:
            df = df[df.session == self.session]

        if len(df[which].unique()) != 1:
            setattr(self, "_" + which, None)
            return None
        else:
            value = int(df[which].unique()[0])
            setattr(self, "_" + which, value)
            return value

    @property
    def protocol(self):
        return get_protocol(self.subject)

    @property
    def localization(self) -> Union[int, None]:
        """Determine the localization number."""
        if self.subject is None:
            return None

        if self._localization is not None:
            return self._localization
        return self._determine_localization_or_montage("localization")

    @property
    def montage(self) -> Union[int, None]:
        """Determine the montage number."""
        if self.subject is None:
            return None

        if self._montage is not None:
            return self._montage
        return self._determine_localization_or_montage("montage")

    @property
    def path_finder(self):
        """Return a path finder using the proper kwargs."""
        from .path_finder import PathFinder
        return PathFinder(self.subject, self.experiment, self.session,
                          self.localization, self.montage, self.rootdir)

    @functools.lru_cache()
    def _construct_reader(self, data_type, subject, experiment, session,
                          localization, montage, rootdir):
        return self.readers[data_type](data_type,
                                       subject=subject,
                                       experiment=experiment,
                                       session=session,
                                       localization=localization,
                                       montage=montage,
                                       rootdir=rootdir)

    def get_reader(self, data_type):
        """Return an instance of the reader class for the given data type.

        Notes
        -----
        Reader instances get cached via :func:`functools.lru_cache`.

        """
        return self._construct_reader(data_type,
                                      self.subject,
                                      self.experiment,
                                      self.session,
                                      self.localization,
                                      self.montage,
                                      self.rootdir)

    def load(self, data_type: str, **kwargs):
        """Load requested data into memory.

        Parameters
        ----------
        data_type
            Type of data to load (see :attr:`readers` for available options)

        Notes
        -----
        Keyword arguments that are accepted depend on the type of data being
        loaded. See :meth:`load_eeg` for details.

        """
        if data_type not in self.readers:
            raise NotImplementedError("There is no reader to support the "
                                      "requested file type")

        # By default we want task + math events when requesting events so
        # coerce to "all_events" unless we're looking at experiments that don't
        # include these.
        if data_type == "events":
            if self.experiment.startswith("PS") or self.experiment.startswith("TH"):
                data_type = "task_events"
            else:
                data_type = "all_events"

        cls = self._construct_reader(data_type,
                                     self.subject,
                                     self.experiment,
                                     self.session,
                                     self.localization,
                                     self.montage,
                                     self.rootdir)

        if self.protocol not in cls.protocols:
            raise UnsupportedProtocolError(
                "Data type {} is not supported under protocol {}".format(
                    data_type, self.protocol
                )
            )

        return cls.load(**kwargs)

    def load_eeg(self, events: Optional[pd.DataFrame] = None,
                 rel_start: int = None, rel_stop: int = None,
                 scheme: Optional[pd.DataFrame] = None):
        """Load EEG data.

        Keyword arguments
        -----------------
        events
            Events to load EEG epochs from. Incompatible with passing
            ``epochs``.
        rel_start
            Start time in ms relative to passed event onsets. This parameter is
            required when passing events and not used otherwise.
        rel_stop
            Stop time in ms relative to passed event onsets. This  parameter is
            required when passing events and not used otherwise.
        scheme
            When specified, a bipolar scheme to rereference the data with and/or
            filter by channel. Rereferencing is only possible if the data were
            recorded in monopolar (a.k.a. common reference) mode.

        Returns
        -------
        EEGContainer

        Raises
        ------
        RereferencingNotPossibleError
            When passing ``scheme`` and the data do not support rereferencing.
        IncompatibleParametersError
            When both ``events`` and ``epochs`` are specified or ``events`` are
            used without passing ``rel_start`` and/or ``rel_stop``.

        """
        kwargs = {
            'scheme': scheme,
        }

        if events is not None:
            if rel_start is None or rel_stop is None:
                raise IncompatibleParametersError(
                    "rel_start and rel_stop are required keyword arguments"
                    " when passing events"
                )

            kwargs.update({
                'events': events,
                'rel_start': rel_start,
                'rel_stop': rel_stop,
            })

        return self.load('eeg', **kwargs)

    @classmethod
    def load_events(cls, subjects: Optional[Union[str, List[str]]] = None,
                    experiments: Optional[Union[str, List[str]]] = None,
                    rootdir: Optional[str] = None) -> pd.DataFrame:
        """Load events from multiple sessions.

        Parameters
        ----------
        subjects
            Subject or list of subjects.
        experiments
            Experiment or list of experiments to include.
        rootdir
            Path to root data directory.

        """
        if subjects is None and experiments is None:
            raise ValueError(
                "Please specify at least one subject or experiment."
            )

        rootdir = get_root_dir(rootdir)
        df = get_data_index("all", rootdir=rootdir)

        if isinstance(subjects, str):
            subjects = [subjects]
        elif subjects is None:
            subjects = df["subject"].unique()

        if isinstance(experiments, str):
            experiments = [experiments]
        elif experiments is None:
            experiments = df["experiment"].unique()

        events = []

        for subject in subjects:
            for experiment in experiments:
                mask = (df["subject"] == subject) & (df["experiment"] == experiment)
                sessions = df[mask]["session"].unique()

                for session in sessions:
                    reader = CMLReader(subject, experiment, session,
                                       rootdir=rootdir)
                    events.append(reader.load("events"))

        return pd.concat(events, sort=True)
