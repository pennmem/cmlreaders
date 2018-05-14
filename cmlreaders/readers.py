import pandas as pd

from .path_finder import PathFinder
from abc import abstractmethod, ABC


__all__ = ['BaseCMLReader', 'TextReader', 'CSVReader']


class BaseCMLReader(ABC):

    @abstractmethod
    def as_dataframe(self):
        raise NotImplementedError

    @abstractmethod
    def as_recarray(self):
        raise NotImplementedError

    @abstractmethod
    def to_json(self, file_name, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def to_csv(self, file_name, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def to_hdf(self, file_name):
        raise NotImplementedError


class TextReader(BaseCMLReader):
    headers = {
        'voxel_coordinates': ['label', 'vox_x', 'vox_y', 'vox_z', 'type',
                              'min_contact_num', 'max_contact_num'],
        'jacksheet': ['channel_label'],
        'classifier_excluded_leads': ['channel_label'],
        'good_leads': ['channel_num'],
        'leads': ['channel_num'],
        'area': ['lead_label', 'surface_area'],
    }

    def __init__(self, data_type, subject, localization, rootdir="/",
                 **kwargs):
        """ Create a TextReader for loading text-based RAM data """
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

    def to_csv(self, file_path, **kwargs):
        self.as_dataframe().to_csv(file_path, index=False, **kwargs)

    def to_json(self, file_path, **kwargs):
        self.as_dataframe().to_json(file_path, index=False, **kwargs)

    def to_hdf(self, file_path):
        raise NotImplementedError


class CSVReader(BaseCMLReader):
    def __init__(self, data_type, subject, localization, rootdir="/"):
        """ Generic reader class for loading csv files with headers """
        finder = PathFinder(subject=subject, localization=localization,
                            rootdir=rootdir)
        self._file_path = finder.find(data_type)

    def as_dataframe(self):
        df = pd.read_csv(self._file_path)
        return df

    def as_recarray(self):
        records = self.as_dataframe().to_records()
        return records

    def to_csv(self, file_path, **kwargs):
        self.as_dataframe().to_csv(file_path, index=False, **kwargs)

    def to_json(self, file_path, **kwargs):
        self.as_dataframe().to_json(file_path, index=False, **kwargs)

    def to_hdf(self, file_path):
        raise NotImplementedError


