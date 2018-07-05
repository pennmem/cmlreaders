import json
import pandas as pd
from pandas.io.json import json_normalize

from cmlreaders.base_reader import BaseCMLReader
from cmlreaders.exc import (
    MissingParameter, UnmetOptionalDependencyError, UnsupportedRepresentation,
)


__all__ = ['TextReader', 'CSVReader', 'RamulatorEventLogReader',
           'BasicJSONReader', 'EventReader',
           'ClassifierContainerReader', 'EEGMetaReader']


# TODO: separate into a base class so that we can use this for ltp
class TextReader(BaseCMLReader):
    """ Generic reader class for reading RAM text files """
    data_types = ['voxel_coordinates', 'jacksheet', 'classifier_excluded_leads',
                  'good_leads', 'leads', 'area']
    protocols = ["r1"]

    headers = {
        'voxel_coordinates': ['label', 'vox_x', 'vox_y', 'vox_z', 'type',
                              'min_contact_num', 'max_contact_num'],
        'jacksheet': ["number", "label"],
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
        if self.data_type == "jacksheet":
            sep = " "
        else:
            sep = ","  # read_csv's default value
        df = pd.read_csv(self._file_path, sep=sep, names=self._headers)
        return df


# TODO: separate into a base class so that we can use this for ltp
class CSVReader(BaseCMLReader):
    """ Generic reader class for loading csv files with headers """
    data_types = [
        "electrode_coordinates",
        "prior_stim_results",
        "target_selection_table",
    ]
    protocols = ["r1"]

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
    """Reader for Ramulator event log"""
    data_types = ["experiment_log"]
    protocols = ["r1"]

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

    def as_dict(self):
        with open(self._file_path, 'r') as efile:
            raw_dict = json.load(efile)
        return raw_dict


class BasicJSONReader(BaseCMLReader):
    """Generic reader class for loading simple JSON files.

    Returns a :class:`pd.DataFrame`.

    """
    data_types = []

    def as_dataframe(self):
        return pd.read_json(self._file_path)


class EEGMetaReader(BaseCMLReader):
    """Reads the ``sources.json`` file which describes metainfo about EEG data.

    Returns a :class:`dict`.

    """
    data_types = ["sources"]
    default_representation = "dict"

    def as_dict(self):
        with open(self._file_path, 'r') as metafile:
            sources_info = list(json.load(metafile).values())[0]
            sources_info['path'] = self._file_path
        return sources_info


class EventReader(BasicJSONReader):
    """Reader for all experiment events.

    Returns a :class:`pd.DataFrame`.

    """

    data_types = ['all_events', 'math_events', 'task_events', 'events', 'ps4_events']

    def as_dataframe(self):
        df = super().as_dataframe()
        first = ['eegoffset']
        df = df[first + [col for col in df.columns if col not in first]]
        return df


class ClassifierContainerReader(BaseCMLReader):
    """ Reader class for loading a serialized classifier classifier

    Notes
    -----
    By default, a :class:`classiflib.container.ClassifierContainer` class is
    returned.

    """
    data_types = ["used_classifier", "baseline_classifier"]
    protocols = ["r1"]
    default_representation = "pyobject"

    def __init__(self, data_type, subject, experiment, session, localization,
                 file_path=None, rootdir="/", **kwargs):
        super(ClassifierContainerReader, self).__init__(data_type,
                                                        subject=subject,
                                                        experiment=experiment,
                                                        session=session,
                                                        localization=localization,
                                                        file_path=file_path,
                                                        rootdir=rootdir)
        try:
            from classiflib.container import ClassifierContainer
        except ImportError:
            raise UnmetOptionalDependencyError("Install classiflib to use this reader")

        self.pyclass_mapping = {
            'classifier': ClassifierContainer
        }

    def as_pyobject(self):
        summary_obj = self.pyclass_mapping['classifier']
        return summary_obj.load(self._file_path)

    def as_dataframe(self):
        raise UnsupportedRepresentation("Unable to represent classifier as a dataframe")

    def to_binary(self, file_name, **kwargs):
        """Saves classifier to a serialized format as determined by  the file
        extension. By default, if the file already exists, it will note be
        overwritten.

        Notes
        -----
        See :meth:`classiflib.container.ClassifierContainer.save()` for more
        details on supported output formats.

        """
        self.as_pyobject().save(file_name, **kwargs)
