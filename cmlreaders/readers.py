import pandas as pd

from .path_finder import PathFinder
from .exc import UnsupportedOutputFormat, MissingParameter
from abc import abstractmethod, ABC


__all__ = ['BaseCMLReader', 'TextReader', 'CSVReader']


class BaseCMLReader(ABC):
    """ Abstract base class that defines the interface for all CML data readers """

    @abstractmethod
    def as_dataframe(self):
        raise NotImplementedError

    @abstractmethod
    def as_recarray(self):
        raise NotImplementedError

    @abstractmethod
    def as_dict(self):
        raise NotImplementedError

    def to_json(self, file_name, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def to_csv(self, file_name, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def to_hdf(self, file_name):
        raise NotImplementedError


class TextReader(BaseCMLReader):
    """ Generic reader class for reading RAM text files """
    headers = {
        'voxel_coordinates': ['label', 'vox_x', 'vox_y', 'vox_z', 'type',
                              'min_contact_num', 'max_contact_num'],
        'jacksheet': ['channel_label'],
        'classifier_excluded_leads': ['channel_label'],
        'good_leads': ['channel_num'],
        'leads': ['channel_num'],
        'area': ['lead_label', 'surface_area'],
    }

    def __init__(self, data_type, subject, localization, file_path=None,
                 rootdir="/", **kwargs):

        self._file_path = file_path
        # When no file path is given, look it up using PathFinder
        if file_path is None:
            finder = PathFinder(subject=subject, localization=localization,
                                rootdir=rootdir)
            self._file_path = finder.find(data_type)
        self._headers = self.headers[data_type]

    def as_dataframe(self):
        df = pd.read_csv(self._file_path, names=self._headers)
        return df

    def as_recarray(self):
        records = self.as_dataframe().to_records()
        return records

    def as_dict(self):
        df = self.as_dataframe()
        return df.to_dict(orient='records')

    def to_csv(self, file_path, **kwargs):
        self.as_dataframe().to_csv(file_path, index=False, **kwargs)

    def to_json(self, file_path, **kwargs):
        self.as_dataframe().to_json(file_path)

    def to_hdf(self, file_path):
        raise UnsupportedOutputFormat


class CSVReader(BaseCMLReader):
    """ Generic reader class for loading csv files with headers """
    def __init__(self, data_type, subject, localization, experiment=None,
                 file_path=None, rootdir="/", **kwargs):

        if (data_type == 'target_selection_table') and experiment is None:
            raise MissingParameter("Experiment required with target_selection_"
                                   "table data type")

        self._file_path = file_path
        # When no file path is given, look it up using PathFinder
        if file_path is None:
            finder = PathFinder(subject=subject, localization=localization,
                                experiment=experiment,
                                rootdir=rootdir)
            self._file_path = finder.find(data_type)

    def as_dataframe(self):
        df = pd.read_csv(self._file_path)
        return df

    def as_recarray(self):
        records = self.as_dataframe().to_records()
        return records

    def as_dict(self):
        df = self.as_dataframe()
        return df.to_dict(orient='records')

    def to_csv(self, file_path, **kwargs):
        self.as_dataframe().to_csv(file_path, index=False, **kwargs)

    def to_json(self, file_path, **kwargs):
        self.as_dataframe().to_json(file_path, **kwargs)

    def to_hdf(self, file_path):
        raise UnsupportedOutputFormat


