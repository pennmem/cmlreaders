import pandas as pd

from .path_finder import PathFinder
from abc import ABCMeta, abstractmethod, ABC


__all__ = ['BaseCMLReader', 'TextReader']


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
    }

    def __init__(self, data_type, subject=None, localization=None, rootdir="/",
                 **kwargs):
        """ Create a TextReader for loading text-based RAM data """

        if (data_type is None) or (subject is None) or (localization is None):
            pass

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
