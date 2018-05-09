from .readers import *


__all__ = ['CMLReader']


class CMLReader(BaseCMLReader):
    readers = {
            'voxel_coordinates': TextReader
        }

    def __init__(self, subject=None, experiment=None, session=None,
                 localization=None, montage=None, rootdir="/"):
        """ Instatiates a general reader for CML-specific data """

        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.localization = localization
        self.montage = montage
        self.rootdir = rootdir

        # File type and reader are unintialized initially, but are populated
        # when the user requests that a particular file be loaded
        self.file_type = None
        self._reader = None

    def load(self, file_type):
        self.file_type = file_type
        self._reader = self.readers[self.file_type](self.file_type,
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
