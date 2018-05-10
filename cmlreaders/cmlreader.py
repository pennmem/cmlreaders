from typing import Optional
from .readers import *


__all__ = ['CMLReader']


class CMLReader(BaseCMLReader):
    readers = {
            'voxel_coordinates': TextReader
        }

    def __init__(self, subject: Optional[str] =None,
                 experiment: Optional[str] = None,
                 session: Optional[str] = None,
                 localization: Optional[str] = None,
                 montage: Optional[str] = None,
                 rootdir: Optional[str] = "/"):
        """ Instatiates a general reader for CML-specific data """

        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.localization = localization
        self.montage = montage
        self.rootdir = rootdir

        # reader is unintialized initially, but are populated
        # when the user requests that a particular file be loaded
        self._reader = None

    def load(self, data_type):
        if data_type not in self.readers:
            raise NotImplementedError("There is no reader to support the requested file type")

        self._reader = self.readers[data_type](data_type,
                                               subject=self.subject,
                                               experiment=self.experiment,
                                               session=self.session,
                                               localization=self.localization,
                                               montage=self.montage,
                                               rootdir=self.rootdir)

    def as_dataframe(self):
        return self._reader.as_dataframe()

    def as_recarray(self):
        return self._reader.as_recarray()

    def to_json(self, file_name, **kwargs):
        return self._reader.to_json(file_name, **kwargs)

    def to_csv(self, file_name, **kwargs):
        return self._reader.to_csv(file_name, **kwargs)

    def to_hdf(self, file_name):
        return self._reader.to_hdf(file_name)
