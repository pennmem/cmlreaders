from typing import Optional
from .readers import *


__all__ = ['CMLReader']


class CMLReader(BaseCMLReader):
    """ Generic reader for all CML-specific files """
    readers = {
        'voxel_coordinates': TextReader,
        'jacksheet': TextReader,
        'good_leads': TextReader,
        'leads': TextReader,
        'classifier_excluded_leads': TextReader,
        'prior_stim_results': CSVReader,
        'electrode_coordinates': CSVReader,
        'target_selection_table': CSVReader
    }

    def __init__(self, subject: Optional[str] =None,
                 experiment: Optional[str] = None,
                 session: Optional[str] = None,
                 localization: Optional[str] = None,
                 montage: Optional[str] = None,
                 rootdir: Optional[str] = "/"):

        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.localization = localization
        self.montage = montage
        self.rootdir = rootdir

        # reader is unintialized initially, but are populated
        # when the user requests that a particular file be loaded
        self._reader = None

    def load(self, data_type, file_path=None):
        if data_type not in self.readers:
            raise NotImplementedError("There is no reader to support the "
                                      "requested file type")

        self._reader = self.readers[data_type](data_type,
                                               subject=self.subject,
                                               experiment=self.experiment,
                                               session=self.session,
                                               localization=self.localization,
                                               montage=self.montage,
                                               file_path=file_path,
                                               rootdir=self.rootdir)

    def as_dataframe(self):
        """ Return data as `pd.DataFrame` """
        return self._reader.as_dataframe()

    def as_recarray(self):
        """ Return data as `np.rec.array` """
        return self._reader.as_recarray()

    def as_dict(self):
        """ Return data as dict """
        return self._reader.as_dict()

    def to_json(self, file_name, **kwargs):
        """ Save data to JSON formatted file """
        return self._reader.to_json(file_name, **kwargs)

    def to_csv(self, file_name, **kwargs):
        """ Save data to CSV file """
        return self._reader.to_csv(file_name, **kwargs)

    def to_hdf(self, file_name):
        """ Save data to HDF5 file """
        return self._reader.to_hdf(file_name)
