from typing import List, Tuple, Optional

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

    def __init__(self, subject: str,
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

        self.protocol = get_protocol(self.subject)

        self.readers = {k: getattr(readers, v) for k, v in self.reader_names.items()}
        self.reader_protocols = {k: getattr(readers, v).protocols for k, v in self.reader_names.items()}

    def _load_index(self) -> pd.DataFrame:
        """Loads the data index. Used internally to determine montage and
        localization nubmers.

        """
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
            return None

        if self.experiment is not None:
            df = df[df.experiment == self.experiment]

        if self.session is not None:
            df = df[df.session == self.session]

        if len(df[which].unique()) != 1:
            return None
        else:
            return int(df[which].unique()[0])

    @property
    def localization(self) -> int:
        """Determine the localization number."""
        if self._localization is not None:
            return self._localization
        return self._determine_localization_or_montage("localization")

    @property
    def montage(self) -> int:
        """Determine the montage number."""
        if self._montage is not None:
            return self._montage
        return self._determine_localization_or_montage("montage")

    @property
    def path_finder(self):
        """Return a path finder using the proper kwargs."""
        from .path_finder import PathFinder
        return PathFinder(self.subject, self.experiment, self.session,
                          self.localization, self.montage, self.rootdir)

    def get_reader(self, data_type, file_path=None):
        """ Return an instance of the reader class for the given data type """

        return self.readers[data_type](data_type,
                                       subject=self.subject,
                                       experiment=self.experiment,
                                       session=self.session,
                                       localization=self.localization,
                                       montage=self.montage,
                                       file_path=file_path,
                                       rootdir=self.rootdir)

    def load(self, data_type: str, file_path: str = None, **kwargs):
        """Load requested data into memory.

        Parameters
        ----------
        data_type
            Type of data to load (see :attr:`readers` for available options)
        file_path
            Absolute path to load if given. This overrides the default search
            paths.

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
            if self.experiment.startswith("PS2"):
                data_type = "task_events"
            elif self.experiment.startswith("PS4"):
                data_type = "task_events"
            else:
                data_type = "all_events"

        cls = self.readers[data_type]
        if self.protocol not in cls.protocols:
            raise UnsupportedProtocolError(
                "Data type {} is not supported under protocol {}".format(
                    data_type, self.protocol
                )
            )

        return cls(data_type,
                   subject=self.subject,
                   experiment=self.experiment,
                   session=self.session,
                   localization=self.localization,
                   montage=self.montage,
                   file_path=file_path,
                   rootdir=self.rootdir).load(**kwargs)

    def load_eeg(self, events: Optional[pd.DataFrame] = None,
                 rel_start: int = None, rel_stop: int = None,
                 epochs: Optional[List[Tuple[int, ...]]] = None,
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
        epochs
            A list of tuples to specify epochs to retrieve data from. These can
            be in one of two forms:

            - (start_index, stop_index)
            - (start_index, stop_index, file_number) when the EEG
              for the session is is split over multiple recordings.

            Incompatible with passing ``events``.
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
        if events is not None and epochs is not None:
            raise IncompatibleParametersError("events and epochs are mutually exclusive")

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
        elif epochs is not None:
            kwargs.update({
                'epochs': epochs,
            })

        return self.load('eeg', **kwargs)
