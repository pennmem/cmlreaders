import json
import pandas as pd
from pandas.io.json import json_normalize

from .path_finder import PathFinder
from .exc import UnsupportedOutputFormat, MissingParameter


__all__ = ['TextReader', 'CSVReader', 'RamulatorEventLogReader']


class BaseCMLReader(object):
    """ Base class for CML data readers """

    default_representation = "dataframe"

    def __init__(self, data_type, subject=None, experiment=None, session=None,
                 localization=None, montage=None, file_path=None, rootdir="/"):
        self._file_path = file_path
        # When no file path is given, look it up using PathFinder
        if file_path is None:
            finder = PathFinder(subject=subject, experiment=experiment,
                                session=session, localization=localization,
                                montage=montage, rootdir=rootdir)
            self._file_path = finder.find(data_type)

    def load(self, dtype=default_representation):
        """ Load data into memory """
        method_name = "_".join(["as", dtype])
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

    def to_json(self, file_name, **kwargs):
        self.as_dataframe().to_json(file_name)

    def to_csv(self, file_name, **kwargs):
        """ Save data to csv file """
        self.as_dataframe().to_csv(file_name, index=False, **kwargs)

    def to_hdf(self, file_name):
        raise UnsupportedOutputFormat


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
        super(TextReader, self).__init__(data_type, subject=subject,
                                         localization=localization,
                                         file_path=file_path,
                                         rootdir=rootdir)
        self._headers = self.headers[data_type]

    def as_dataframe(self):
        df = pd.read_csv(self._file_path, names=self._headers)
        return df

    def to_hdf(self, file_path):
        raise UnsupportedOutputFormat


class CSVReader(BaseCMLReader):
    """ Generic reader class for loading csv files with headers """
    def __init__(self, data_type, subject, localization, experiment=None,
                 file_path=None, rootdir="/", **kwargs):

        if (data_type == 'target_selection_table') and experiment is None:
            raise MissingParameter("Experiment required with target_selection_"
                                   "table data type")
        super(CSVReader, self).__init__(data_type, subject=subject,
                                        localization=localization,
                                        experiment=experiment,
                                        file_path=file_path, rootdir=rootdir)

    def as_dataframe(self):
        df = pd.read_csv(self._file_path)
        return df


class RamulatorEventLogReader(BaseCMLReader):
    """ Reader for Ramulator event log """

    def __init__(self, data_type, subject, experiment, session, file_path=None,
                 rootdir="/", **kwargs):
        super(RamulatorEventLogReader, self).__init__(data_type, subject=subject,
                                                      experiment=experiment,
                                                      session=session,
                                                      file_path=file_path,
                                                      rootdir=rootdir)

    def as_dataframe(self):
        with open(self._file_path, 'r') as efile:
            raw = json.loads(efile.read())['events']

        exclude = ['to_id', 'from_id', 'event_id', 'command_id']
        df = json_normalize(raw)
        return df.drop(exclude, axis=1)


class BasicJSONReader(BaseCMLReader):
    """ Generic reader class for loading simple JSON files """
    pass
