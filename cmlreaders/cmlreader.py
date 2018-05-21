from typing import Optional
from .readers import TextReader, CSVReader, RamulatorEventLogReader


__all__ = ['CMLReader']


class CMLReader(object):
    """ Generic reader for all CML-specific files """
    readers = {
        'voxel_coordinates': TextReader,
        'jacksheet': TextReader,
        'good_leads': TextReader,
        'leads': TextReader,
        'classifier_excluded_leads': TextReader,
        'prior_stim_results': CSVReader,
        'electrode_coordinates': CSVReader,
        'target_selection_table': CSVReader,
        'experiment_log': RamulatorEventLogReader
    }

    def __init__(self, subject: Optional[str] =None,
                 experiment: Optional[str] = None,
                 session: Optional[int] = None,
                 localization: Optional[int] = 0,
                 montage: Optional[int] = 0,
                 rootdir: Optional[str] = "/"):

        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.localization = localization
        self.montage = montage
        self.rootdir = rootdir

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

    def load(self, data_type, file_path=None):
        """ Load requested data into memory """
        if data_type not in self.readers:
            raise NotImplementedError("There is no reader to support the "
                                      "requested file type")

        return self.readers[data_type](data_type,
                                       subject=self.subject,
                                       experiment=self.experiment,
                                       session=self.session,
                                       localization=self.localization,
                                       montage=self.montage,
                                       file_path=file_path,
                                       rootdir=self.rootdir).load()

