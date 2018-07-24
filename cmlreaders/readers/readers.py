import json
import pandas as pd
from pandas.io.json import json_normalize
import scipy.io as sio
import warnings
import h5py

from cmlreaders.base_reader import BaseCMLReader
from cmlreaders.exc import (
    MissingParameter, UnmetOptionalDependencyError, UnsupportedRepresentation,
)

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
        df = pd.read_csv(self.file_path, sep=sep, names=self._headers)
        return df


class BaseCSVReader(BaseCMLReader):
    """Base class for reading CSV files."""
    def as_dataframe(self):
        df = pd.read_csv(self.file_path)
        return df


class RAMCSVReader(BaseCSVReader):
    """CSV reader type for RAM data."""
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

        super().__init__(data_type, subject=subject,
                         localization=localization,
                         experiment=experiment,
                         file_path=file_path, rootdir=rootdir)


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
        with open(self.file_path, 'r') as efile:
            raw = json.loads(efile.read())['events']

        exclude = ['to_id', 'from_id', 'event_id', 'command_id']
        df = json_normalize(raw)
        return df.drop(exclude, axis=1)

    def as_dict(self):
        with open(self.file_path, 'r') as efile:
            raw_dict = json.load(efile)
        return raw_dict


class BaseJSONReader(BaseCMLReader):
    """Generic reader class for loading simple JSON files.

    Returns a :class:`pd.DataFrame`.

    """
    data_types = []

    def as_dataframe(self):
        return pd.read_json(self.file_path)


class EventReader(BaseCMLReader):
    """Reader for all experiment events.

    Returns a :class:`pd.DataFrame`.

    """
    data_types = [
        "all_events", "events", "math_events", "ps4_events", "task_events",
    ]

    def _read_json_events(self) -> pd.DataFrame:
        return pd.read_json(self.file_path)

    def _read_matlab_events(self) -> pd.DataFrame:
        df = pd.DataFrame(sio.loadmat(self.file_path, squeeze_me=True)["events"])

        if self.session is not None:
            df = df[df["session"] == self.session]

        return df

    def as_dataframe(self):
        if self.file_path.endswith(".json"):
            df = self._read_json_events()
        else:
            df = self._read_matlab_events()
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
        return summary_obj.load(self.file_path)

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


class BrainObjectReader(BaseCMLReader):
    """Reader class for loading a SuperEEG brain object """

    data_types = ['brain_object']
    default_representation = "pyobject"

    def as_pyobject(self):
        try:
            import supereeg
        except ImportError:
            warnings.warn("Could not import supereeg -- data will be returned as dict")
            supereeg = None
        try:
            import deepdish
        except ImportError:
            warnings.warn("Could not import deepdish -- data will be returned as HDF5 file")
            deepdish = None

        if deepdish is None:
            return h5py.File(self.file_path)
        else:
            data = deepdish.io.load(self.file_path)
            if supereeg is None:
                return supereeg.Brain(**data)
            else:
                return data
