import json
import pandas as pd
from pandas.io.json import json_normalize
from typing import List, Optional

from .path_finder import PathFinder
from .exc import (
    UnsupportedOutputFormat, MissingParameter,
    UnmetOptionalDependencyError, UnsupportedRepresentation,
    UnsupportedExperimentError
)


__all__ = ['TextReader', 'CSVReader', 'RamulatorEventLogReader',
           'BasicJSONReader', 'EventReader', 'MontageReader',
           'LocalizationReader', 'ElectrodeCategoriesReader',
           'BaseReportDataReader', 'ReportSummaryDataReader']


class BaseCMLReader(object):
    """ Base class for CML data readers """

    default_representation = "dataframe"

    def __init__(self, data_type: str, subject: Optional[str] = None,
                 experiment: Optional[str] = None,
                 session: Optional[int] = None,
                 localization: Optional[int] = 0, montage: Optional[int] = 0,
                 file_path: Optional[str] = None, rootdir: Optional[str] = "/"):

        self._file_path = file_path

        # When no file path is given, look it up using PathFinder
        if file_path is None:
            finder = PathFinder(subject=subject, experiment=experiment,
                                session=session, localization=localization,
                                montage=montage, rootdir=rootdir)
            self._file_path = finder.find(data_type)

        self.subject = subject
        self.data_type = data_type

    def load(self):
        """ Load data into memory """
        method_name = "_".join(["as", self.default_representation])
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

    def to_json(self, file_name):
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

    def as_dict(self):
        with open(self._file_path, 'r') as efile:
            raw_dict = json.load(efile)
        return raw_dict


class BasicJSONReader(BaseCMLReader):
    """Generic reader class for loading simple JSON files.

    Returns a :class:`pd.DataFrame`.

    """
    def as_dataframe(self):
        return pd.read_json(self._file_path)


class EventReader(BasicJSONReader):
    """Reader for all experiment events.

    Returns a :class:`pd.DataFrame`.

    """
    def as_dataframe(self):
        df = super().as_dataframe()
        first = ['eegoffset']
        df.columns = first + [col for col in df.columns if col not in first]
        return df


class MontageReader(BaseCMLReader):
    """Reads montage files (contacts.json, pairs.json).

    Returns a :class:`pd.DataFrame`.

    """
    @staticmethod
    def _flatten_row(data: dict, labels: List[str], i: int) -> pd.DataFrame:
        entry = data[labels[i]].copy()
        atlases = entry.pop('atlases')
        atlas_row = json_normalize(atlases)
        atlas_row.index = [i]
        row = pd.concat([pd.DataFrame(entry, index=[i]), atlas_row], axis=1)
        return row

    def as_dataframe(self):
        from concurrent.futures import ProcessPoolExecutor as Pool
        from functools import partial
        from multiprocessing import cpu_count

        with open(self._file_path) as f:
            data = json.load(f)[self.subject][self.data_type]

        labels = [l for l in data]

        flatten = partial(self._flatten_row, data, labels)
        with Pool(min(4, cpu_count())) as pool:
            rows = pool.map(flatten, range(len(labels)))
        df = pd.concat(rows)

        # Drop useless atlas.id tags
        df = df[[col for col in df.columns if not col.endswith('.id')]]

        # rename poorly named columns
        if self.data_type == 'contacts':
            renames = {'channel': 'contact'}
        else:
            renames = {'channel_1': 'contact_1', 'channel_2': 'contact_2'}
        renames.update({'code': 'label'})
        df = df.rename(renames, axis=1)

        # ensure that contact and label appear first
        names = df.columns
        if self.data_type == 'contacts':
            first = ['contact']
        else:
            first = ['contact_1', 'contact_2']
        first += ['label']
        df = df[first + [name for name in names if name not in first]]

        # sort by contact
        key = 'contact' if self.data_type == 'contacts' else 'contact_1'
        df = df.sort_values(by=key).reset_index(drop=True)

        return df


class LocalizationReader(BaseCMLReader):
    """Reads data stored in localization.json.

    Returns a :class:`pd.DataFrame`.

    """
    def as_dataframe(self):
        import itertools

        with open(self._file_path) as f:
            data = json.load(f)

        leads = list(data['leads'].values())

        for lead in leads:
            contacts = lead["contacts"]
            if isinstance(contacts, dict):
                contacts = contacts.values()
            for c in contacts:
                c.update({"type": lead["type"]})
            pairs = lead["pairs"]
            if isinstance(pairs, dict):
                pairs = pairs.values()
            for p in pairs:
                p['names'] = tuple(p['names'])
                p.update({"type": lead["type"]})

        flat_contact_data = list(itertools.chain(*[x["contacts"] for x in leads]))
        flat_pairs_data = list(itertools.chain(*[x["pairs"] for x in leads]))
        all_data = []
        all_data.append(pd.io.json.json_normalize(flat_contact_data).set_index('name'))
        all_data.append(pd.io.json.json_normalize(flat_pairs_data).set_index('names'))
        combined_df = pd.concat(all_data, keys=['contacts', 'pairs'])
        return combined_df


class ElectrodeCategoriesReader(BaseCMLReader):
    """Reads electrode_categories.txt and handles the many inconsistencies in
    those files.

    Returns a ``dict``.

    """
    default_representation = 'dict'

    def _read_categories(self) -> dict:
        """Returns a dictionary mapping categories to electrode from the
        electrode_categories.txt file

        Returns
        -------
        groups: dict,
            dictionary mapping relevant field values (bad channel, SOZ, etc.) to
            the corresponding channels

        Notes
        -----
        This function is only required because there's so much inconsistency in
        where and how the data corresponding to bad electrodes are stored.

        """
        # Used to indicate relevant strings in the text files
        relevant = {
            'seizure onset zone', 'seizure onset zones', 'seizure onset',
            'interictal', 'interictal spiking', 'interictal spikes',
            'ictal onset', 'ictal onset:', 'interictal spiking:',
            'brain lesions', 'brain lesions:', 'octal onset zone',
            'bad electrodes', 'bad electrodes:', 'broken leads', 'broken leads:'
        }

        with open(self._file_path, 'r') as f:
            ch_info = f.read().split('\n')

        # This will be used to initalize a before after kind of check to sort
        # the groups
        count = 0
        groups = {}  # Save the groups here

        for index, line in enumerate(ch_info[2:]):
            # We skip to two because all files start with line one being subject
            # followed by another line of '', if the user wishes to access the
            # information feel free to modify below. Blank spaces used to
            # seperate, if we encountered one count increases
            if line == '':
                count += 1
                continue  # Ensures '' isn't appended to dict[group_name]

            # Ignore a line containing only '-' (sometimes found at the end of
            # files)
            if line == '-':
                continue

            # Check if the line is relevant if so add a blank list to the dict
            if line.lower() in relevant:
                count = 0
                group_name = line.lower()
                groups[group_name] = []
                # Sanity check to ensure that they had a '' after the relevant
                if ch_info[2:][index + 1] != '':  # Skipping two for same reason
                    count += 1
                continue

            # Triggered when inside of a group e.g. they're channel names
            if (count == 1) and (line != ''):  # indicates start of group values
                groups[group_name].append(line)

        return groups

    def _get_categories_dict(self) -> dict:
        """Return electrode categories from relevant textfile; ensures that the
        fields are consistent regardless of the actual field the RA entered into
        the textfile

        Returns
        -------
        e_cat_reader: dict
            dictionary mapping relevant field values (bad channel, SOZ, etc.)
            to the corresponding channels

        """
        import numpy as np

        convert = {
            'seizure onset zone': 'soz',
            'seizure onset zones': 'soz',
            'seizure onset': 'soz',

            # Interictal activity
            'interictal': 'interictal',
            'interictal spiking': 'interictal',
            'interictal spikes': 'interictal',
            'ictal onset': 'interictal',
            'ictal onset:': 'interictal',
            'interictal spiking:': 'interictal',
            'octal onset zone': 'interictal',

            # Lesioned Tissue
            'brain lesions': 'brain_lesion',
            'brain lesions:': 'brain_lesion',

            # Bad channels
            'bad electrodes': 'bad_channel',
            'bad electrodes:': 'bad_channel',
            'broken leads': 'bad_channel',
            'broken leads:': 'bad_channel'
        }

        e_cat_reader = self._read_categories()
        if e_cat_reader is not None:
            e_cat_reader = {convert[v]: np.array([s.upper() for s in e_cat_reader[v]])
                            for k, v in enumerate(e_cat_reader)}

        return e_cat_reader

    def as_dict(self):
        categories = {
            key: sorted(value.tolist())
            for key, value in self._get_categories_dict().items()
        }

        # make sure we have all the keys
        for key in ['soz', 'interictal', 'brain_lesion', 'bad_channel']:
            if key not in categories:
                categories[key] = []

        return categories


class BaseReportDataReader(BaseCMLReader):
    """
        Reader class for classifier summary data produced in reporting pipeline

    Notes
    -----
    By default, a python class is returned. For report data read with this class
    a python object is the only supported return type. The returned class will
    be `ramutils.reports.summary.Classifiersummary`

    """

    default_representation = 'pyobject'

    def __init__(self, data_type, subject, experiment, session, localization,
                 file_path=None, rootdir="/", **kwargs):
        super(BaseReportDataReader, self).__init__(data_type, subject=subject,
                                                   experiment=experiment,
                                                   session=session,
                                                   localization=localization,
                                                   file_path=file_path,
                                                   rootdir=rootdir)
        self.data_type = data_type

        try:
            from ramutils.reports.summary import ClassifierSummary
        except ImportError:
            raise UnmetOptionalDependencyError("Install ramutils to use this reader")

        self.pyclass_mapping = {
            'classifier_summary': ClassifierSummary,
        }

    def as_pyobject(self):
        """ Return data as a python object specific to this data type """
        if self.data_type in self.pyclass_mapping:
            return self.pyclass_mapping[self.data_type].from_hdf(self._file_path)

    def as_dataframe(self):
        raise UnsupportedRepresentation("Unable to represent this data as a dataframe")

    def to_hdf(self, file_name):
        pyobj = self.as_pyobject()
        pyobj.to_hdf(file_name)


class ReportSummaryDataReader(BaseReportDataReader):
    """
        Reader class for session and math summary data produced in the reporting
        pipeline

    Notes
    -----
    By default, a python class is returned based on the type of data. It could
    be one of

    - `ramutils.reports.summary.MathSummary`
    - `ramutils.reports.summary.FRStimSessionSummary`

    """

    default_representation = "pyobject"

    def __init__(self, data_type, subject, experiment, session, localization,
                 file_path=None, rootdir="/", **kwargs):
        super(BaseReportDataReader, self).__init__(data_type, subject=subject,
                                                   experiment=experiment,
                                                   session=session,
                                                   localization=localization,
                                                   file_path=file_path,
                                                   rootdir=rootdir)
        self.data_type = data_type
        self.subject = subject
        self.experiment = experiment
        self.session = session

        try:
            from ramutils.reports.summary import FRStimSessionSummary, \
                MathSummary
            from ramutils.utils import is_stim_experiment
        except ImportError:
            raise UnmetOptionalDependencyError("Install ramutils to use this reader")

        self.pyclass_mapping = {
            'math_summary': MathSummary,
            'fr_stim_summary': FRStimSessionSummary,
        }

        self.is_stim_experiment = is_stim_experiment

    def as_pyobject(self):
        if self.data_type == 'math_summary':
            return super(ReportSummaryDataReader, self).as_pyobject()

        stim_experiment = self.is_stim_experiment(self.experiment)

        # TODO: Loading record-only data is a bit more complicated since it is
        # not tied to a particular session
        if not stim_experiment:
            raise UnsupportedExperimentError("Only stim report data is currently supported. The readers in ramutils can still be used")

        summary_obj = self.pyclass_mapping['fr_stim_summary']

        return summary_obj.from_hdf(self._file_path)

    def as_dataframe(self):
        pyobj = self.as_pyobject()
        return pyobj.to_dataframe()


class ClassifierContainerReader(BaseCMLReader):
    """ Reader class for loading a serialized classifier classifier

    Notes
    -----
    By default, a `classiflib.container.ClassifierContainer` class is returned

    """

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

    def to_binary(self, file_name, **kwargs):
        """
            Saves classifier to a serialized format as determined by  the file
            extension. By default, if the file already exists, it will note be
            overwritten.

        Notes
        -----
        See :method:`classiflib.container.ClassifierContainer.save()` for more
        details on supported output formats
        """
        self.as_pyobject().save(file_name, **kwargs)
