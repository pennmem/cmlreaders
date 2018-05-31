from typing import Optional

from .path_finder import PathFinder
from .exc import UnsupportedOutputFormat, ImproperlyDefinedReader
from .cmlreader import CMLReader


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

    """
    def __new__(cls, name, bases, d):
        if name is not "BaseCMLReader":
            if 'data_types' not in d:
                raise ImproperlyDefinedReader(
                    "All CML readers must define a list called 'data_types' "
                    "whose elements are the data types that should use the "
                    "reader")
            CMLReader.reader_names.update({x: name for x in d['data_types']})
        return type.__new__(cls, name, bases, d)


class BaseCMLReader(object, metaclass=_MetaReader):
    """ Base class for CML data readers

    Notes
    -----
    All CML readers should inherit from this base class in order to have the
    reader be registered with the generic :class:`cmlreaders.CMLReader` class.
    This happens through the metaclass of BaseCMLReader. To ensure the
    registration happens correctly, new readers must define a list called
    `data_types` as a class variable containing all of the data types that
    should use the reader.

    """
    data_types = []
    default_representation = "dataframe"

    def __init__(self, data_type: str, subject: Optional[str] = None,
                 experiment: Optional[str] = None,
                 session: Optional[int] = None,
                 localization: Optional[int] = 0, montage: Optional[int] = 0,
                 file_path: Optional[str] = None, rootdir: Optional[str] = "/"):

        self._file_path = file_path

        # When no file path is given, look it up using PathFinder unless we're
        # loading EEG data
        if file_path is None and data_type != 'eeg':
            finder = PathFinder(subject=subject, experiment=experiment,
                                session=session, localization=localization,
                                montage=montage, rootdir=rootdir)
            self._file_path = finder.find(data_type)

        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.localization = localization
        self.montage = montage
        self.data_type = data_type
        self.rootdir = rootdir

    def load(self):
        """ Load data into memory """
        method_name = "_".join(["as", self.default_representation])
        return getattr(self, method_name)()

    def as_dataframe(self):
        """ Return data as dataframe """
        raise NotImplementedError

    def as_recarray(self):
        """ Return data as a `np.rec.array` """
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
