import json
import warnings

import pandas as pd
from pandas.errors import ParserWarning
from pandas import json_normalize
import scipy.io as sio

from cmlreaders.base_reader import BaseCMLReader
from cmlreaders.exc import (
    MissingParameter,
    UnmetOptionalDependencyError,
    UnsupportedRepresentation,
)


class TextReader(BaseCMLReader):
    """Generic reader class for reading RAM text files"""

    data_types = [
        "voxel_coordinates",
        "jacksheet",
        "classifier_excluded_leads",
        "good_leads",
        "leads",
        "area",
    ]
    protocols = ["r1"]

    headers = {
        "voxel_coordinates": [
            "label",
            "vox_x",
            "vox_y",
            "vox_z",
            "type",
            "min_contact_num",
            "max_contact_num",
        ],
        "jacksheet": ["number", "label"],
        "classifier_excluded_leads": ["channel_label"],
        "good_leads": ["channel_num"],
        "leads": ["channel_num"],
        "area": ["lead_label", "surface_area"],
    }

    def __init__(self, data_type: str, subject: str, **kwargs):
        super(TextReader, self).__init__(data_type, subject=subject, **kwargs)
        self._headers = self.headers[data_type]

    def as_dataframe(self):
        if self.data_type == "jacksheet":
            sep = r"\s+"  # Split on any whitespace
        else:
            sep = ","  # read_csv's default value

        # When sep is None, we get a warning that the Python parser is slower,
        # but for jacksheet files, it automatically DTRT and the file is small
        # enough for speed to not be an issue.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ParserWarning)
            df = pd.read_csv(self.file_path, sep=sep, names=self._headers)

        return df


class MNICoordinatesReader(TextReader):
    data_types = ["mni_coordinates"]

    protocols = ["r1"]

    headers = {
        "mni_coordinates": [
            "label",
            "mni.x",
            "mni.y",
            "mni.z",
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
        ],
    }  # Ignoring these last 5 fields at the moment

    def as_dataframe(self):
        df = super(MNICoordinatesReader, self).as_dataframe()
        return df[["label", "mni.x", "mni.y", "mni.z"]]


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

    def __init__(
        self,
        data_type,
        subject,
        localization,
        experiment=None,
        file_path=None,
        rootdir="/",
        **kwargs
    ):
        if (data_type == "target_selection_table") and experiment is None:
            raise MissingParameter(
                "Experiment required with target_selection_" "table data type"
            )

        super().__init__(
            data_type,
            subject=subject,
            localization=localization,
            experiment=experiment,
            file_path=file_path,
            rootdir=rootdir,
        )


class RamulatorEventLogReader(BaseCMLReader):
    """Reader for Ramulator event log"""

    data_types = ["experiment_log"]
    protocols = ["r1"]

    def __init__(
        self,
        data_type,
        subject,
        experiment,
        session,
        file_path=None,
        rootdir="/",
        **kwargs
    ):
        super(RamulatorEventLogReader, self).__init__(
            data_type,
            subject=subject,
            experiment=experiment,
            session=session,
            file_path=file_path,
            rootdir=rootdir,
        )

    def as_dataframe(self):
        with open(self.file_path, "r") as efile:
            raw = json.loads(efile.read())["events"]

        exclude = ["to_id", "from_id", "event_id", "command_id"]
        df = json_normalize(raw)
        return df.drop(exclude, axis=1)

    def as_dict(self):
        with open(self.file_path, "r") as efile:
            raw_dict = json.load(efile)
        return raw_dict


class BaseJSONReader(BaseCMLReader):
    """Generic reader class for loading simple JSON files.

    Returns a :class:`pd.DataFrame`.

    """

    data_types = []

    def as_dataframe(self):
        return pd.read_json(self.file_path)


class SessionJSONLogReader(BaseCMLReader):
    """Reads the ``session.json`` file produced by UnityEPL"""

    data_types = ["session_json"]

    def as_dataframe(self):
        df = pd.read_json(self._file_path, lines=True)
        contents_dict = [{col: df.iloc[i][col] for col in df.columns} for i in df.index]
        df_normalized = pd.io.json.json_normalize(contents_dict)
        fixed_columns = {
            col: col.replace(" ", "_").lower() for col in df_normalized.columns
        }
        return df_normalized.rename(columns=fixed_columns)


class EventReader(BaseCMLReader):
    """Reader for all experiment events.

    Returns a :class:`pd.DataFrame`.

    """

    data_types = [
        "all_events",
        "events",
        "math_events",
        "ps4_events",
        "task_events",
    ]
    caching = "memory"

    def _read_json_events(self) -> pd.DataFrame:
        return pd.read_json(self.file_path)

    def _read_matlab_events(self) -> pd.DataFrame:
        df = pd.DataFrame(sio.loadmat(self.file_path, squeeze_me=True)["events"])

        if self.session is not None:
            df = df[df["session"] == self.session]

        # ensure we have an experiment column
        if "experiment" not in df:
            df.loc[:, "experiment"] = self.experiment

        return df

    def as_dataframe(self):
        if self.file_path.endswith(".json"):
            df = self._read_json_events()
        else:
            df = self._read_matlab_events()

        if df.empty:
            raise ValueError(
                "Events DataFrame is empty. Events JSON or MATLAB \
                file likely empty, and experiment session likely not run or uploaded properly."
            )

        first = ["eegoffset"]
        df = df[first + [col for col in df.columns if col not in first]]
        
        # ensure session field matches data index
        if df['session'].unique()[0] != self.session:
            # have to split up to appease Travis CI
            wm1 = f'Changing events session field from {df["session"].unique()[0]} '
            wm2 = f'to {self.session} to match data index.'
            wm = wm1 + wm2
            warnings.warn(wm)
            df['session'] = self.session

        return df


class ClassifierContainerReader(BaseCMLReader):
    """Reader class for loading a serialized classifier classifier

    Notes
    -----
    By default, a :class:`classiflib.container.ClassifierContainer` class is
    returned.

    """

    data_types = ["used_classifier", "baseline_classifier"]
    protocols = ["r1"]
    default_representation = "pyobject"

    def __init__(
        self,
        data_type,
        subject,
        experiment,
        session,
        localization,
        file_path=None,
        rootdir="/",
        **kwargs
    ):
        super(ClassifierContainerReader, self).__init__(
            data_type,
            subject=subject,
            experiment=experiment,
            session=session,
            localization=localization,
            file_path=file_path,
            rootdir=rootdir,
        )
        try:
            from classiflib.container import ClassifierContainer
        except ImportError:
            raise UnmetOptionalDependencyError("Install classiflib to use this reader")

        self.pyclass_mapping = {"classifier": ClassifierContainer}

    def as_pyobject(self):
        summary_obj = self.pyclass_mapping["classifier"]
        return summary_obj.load(self.file_path)

    def as_dataframe(self):
        raise UnsupportedRepresentation("Unable to represent classifier as a dataframe")
