from pathlib import Path
from typing import Optional, Union
from typing import Set  # noqa
import weakref

from . import cache
from .constants import PROTOCOLS
from .path_finder import PathFinder
from .exc import UnsupportedOutputFormat, ImproperlyDefinedReader
from .cmlreader import CMLReader
from .util import get_protocol, get_root_dir


class _MetaReader(type):
    """ Metaclass for all CML readers

        The responsibility of the metaclass is to register the reader with
        the generic CMLReader to avoid needing to do this as a manual step.

    Notes
    -----
    When a child class uses :class:`cmlreaders.base_reader._MetaReader` as a
    meta class, the reader_names dictionary stored as a class variable in
    :class:`cmlreaders.cmlreader.CMLReader` is updated based on the data_types
    class variable in the child class.

    Any child class that starts with ``Base`` will not be added to the mapping
    of data types to reader classes. This allows for combining logic where
    appropriate for similar reader types.

    """
    def __new__(cls, name, bases, d):
        if not name.startswith("Base") and not name.startswith("Dummy"):
            if "data_types" not in d:
                raise ImproperlyDefinedReader(
                    "All CML readers must define a list called 'data_types' "
                    "whose elements are the data types that should use the "
                    "reader")
            CMLReader.reader_names.update({x: name for x in d['data_types']})
        return type.__new__(cls, name, bases, d)


class BaseCMLReader(object, metaclass=_MetaReader):
    """Base class for CML data readers

    All CML readers should inherit from this base class in order to have the
    reader be registered with the generic :class:`cmlreaders.CMLReader` class.
    This happens through the metaclass of BaseCMLReader. To ensure the
    registration happens correctly, new readers must define a list called
    `data_types` as a class variable containing all of the data types that
    should use the reader.

    By default, readers are assumed to work with all protocol types. If only a
    subset of protocols support a data type, then they should be specified using
    the ``protocols`` variable.

    Results from calling :meth:`load` may be cached. This is configured by
    setting the ``caching`` variable which defaults to ``None`` (caching
    disabled for this reader type). Available caching types are:

    * ``"memory"`` for in-memory caching

    """
    data_types = []
    default_representation = "dataframe"
    protocols = PROTOCOLS
    caching = None

    # weakref references to all instantiated readers
    _instances = set()  # type: Set[weakref.ref]

    # We set this default value here for easier mocking in tests.
    _file_path = None

    def __init__(self, data_type: str,
                 subject: Optional[str] = None,
                 experiment: Optional[str] = None,
                 session: Optional[int] = None,
                 localization: Optional[int] = 0,
                 montage: Optional[int] = 0,
                 file_path: Optional[str] = None,
                 rootdir: Optional[str] = None):

        # This is for mocking in tests, do not remove!
        if self._file_path is None:
            self._file_path = file_path

        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.localization = localization
        self.montage = montage
        self.data_type = data_type
        self.rootdir = get_root_dir(rootdir)

        # in-memory cached result (if enabled)
        self._result = None

        BaseCMLReader._instances.add(weakref.ref(self))

    @property
    def protocol(self):
        return get_protocol(self.subject)

    @property
    def file_path(self):
        """When no file path is given, look it up using PathFinder unless we're
        loading EEG data. EEG data is treated differently because of the way
        it is stored on rhino: sometimes it is split into one file per channel
        and other times it is a single HDF5 or EDF/BDF file.

        """
        if self._file_path is None and self.data_type != 'eeg':
            finder = PathFinder(subject=self.subject, experiment=self.experiment,
                                session=self.session, localization=self.localization,
                                montage=self.montage, rootdir=self.rootdir)
            self._file_path = finder.find(self.data_type)
        return self._file_path

    @classmethod
    def fromfile(cls, path: Union[str, Path],
                 subject: Optional[str] = None,
                 experiment: Optional[str] = None,
                 session: Optional[int] = None,
                 data_type: Optional[str] = None):
        """Directly load data from a file using the default representation.
        This is equivalent to creating a reader with the ``file_path`` keyword
        argument given but without the need to then call ``load`` or specify
        other, unnecessary arguments.

        Parameters
        ----------
        path
            Path to the file to load.
        subject
            Subject code to use; required when we need to determine the protocol
        experiment
        session

        """
        if subject is None:
            subject = ""

        if experiment is None:
            experiment = "experiment"

        # Try to guess data_type if it wasn't given (likely only required if
        # testing things).
        if data_type is None:
            try:
                data_type = Path(path).name.split(".")[0]
            except:  # noqa
                data_type = "dtype"

        reader = cls(data_type, subject, experiment, session, file_path=path)
        return reader.load()

    def _load_no_cache(self):
        """Load stored data without using a cache."""
        method_name = "_".join(["as", self.default_representation])
        return getattr(self, method_name)()

    def _load_from_memory(self):
        """Load from disk or in-memory cache."""
        if self._result is None:
            self._result = self._load_no_cache()

        return self._result

    def load(self):
        """Load data into memory in the default representation."""
        if self.caching is None or not cache.enabled:
            return self._load_no_cache()
        elif self.caching == "memory":
            return self._load_from_memory()
        else:
            raise ValueError("Invalid caching type: " + self.caching)

    def _clear_memory_cache(self):
        """Clear in-memory cached result."""
        self._result = None

    def clear_cache(self):
        """Clear cache if configured."""
        {
            "memory": self._clear_memory_cache,
            None: lambda: None,
        }[self.caching]()

    @classmethod
    def clear_all_caches(cls):
        """Clear caches for every instantiated reader."""
        instances = cls._instances.copy()
        for ref in instances:
            obj = ref()
            if obj is not None:
                obj.clear_cache()
            else:
                cls._instances.remove(ref)

    def as_dataframe(self):
        """ Return data as dataframe """
        raise NotImplementedError

    def as_recarray(self):
        """Return data as a numpy recarray. By default, this calls
        :meth:`as_dataframe` and converts to a recarray with
        :meth:`pd.DataFrame.to_records`.

        """
        return self.as_dataframe().to_records()

    def as_dict(self):
        """ Return data as a list of dictionaries """
        return self.as_dataframe().to_dict(orient='records')

    def to_json(self, file_name):
        self.as_dataframe().to_json(file_name)

    def to_csv(self, file_name, **kwargs):
        """ Save data to csv file """
        self.as_dataframe().to_csv(file_name, index=False, **kwargs)

    def to_hdf(self, file_name):
        raise UnsupportedOutputFormat
