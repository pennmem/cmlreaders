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


class ElectrodeCategoryReader(BaseCMLReader):
    """Reads electrode_categories.txt and handles the many inconsistencies in
    those files.

    Returns a :class:`pd.DataFrame`.

    """
    def electrode_categories_reader(self, subject):
        """Returns a dictionary mapping categories to electrode from the electrode_categories.txt file

        Parameters
        ----------
        subject: str
            subject ID for subjects in the RAM project. E.g. 'R1111M'

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
        import os.path

        # Used to indicate relevant strings in the text files
        relevant = {
            'seizure onset zone', 'seizure onset zones', 'seizure onset',
            'interictal', 'interictal spiking', 'interictal spikes',
            'ictal onset', 'ictal onset:', 'interictal spiking:',
            'brain lesions', 'brain lesions:', 'octal onset zone',
            'bad electrodes', 'bad electrodes:', 'broken leads', 'broken leads:'
        }

        # Open the file, which may exist in multiple places
        filename = '/data/eeg/{}/docs/electrode_categories.txt'.format(subject)

        if not os.path.exists(filename):
            print('{} does not exist'.format(filename))
            filename = '/scratch/pwanda/electrode_categories/{}_electrode_categories.txt'.format(subject)
        if not os.path.exists(filename):  # Check spot one, if not go to paul's scratch directory
            # Because literally it's ACTUALLY SAVED ON PAUL'S SCRATCH DIRECTORY
            filename = '/scratch/pwanda/electrode_categories/electrode_categories_{}.txt'.format(subject)

        try:
            with open(filename, 'r') as f:
                ch_info = f.read().split('\n')

        except IOError:
            print('File {} not found'.format(filename))
            return

        # This will be used to initalize a before after kind of check to sort
        # the groups
        count = 0
        groups = {}  # Save the groups here

        for index, current in enumerate(ch_info[2:]):
            # We skip to two because all files start with line one being subject
            # followed by another line of '', if the user wishes to access the
            # information feel free to modify below. Blank spaces used to
            # seperate, if we encountered one count increases
            if (current == ''):
                count += 1
                continue  # Ensures '' isn't appended to dict[group_name]

            # Check if the line is relevant if so add a blank list to the dict
            if current.lower() in relevant:
                count = 0
                group_name = current.lower()
                groups[group_name] = []
                # Sanity check to ensure that they had a '' after the relevant
                if ch_info[2:][index + 1] != '':  # Skipping two for same reason
                    count += 1
                continue

            # Triggered when inside of a group e.g. they're channel names
            if (count == 1) and (current != ''):  # indicates start of group values
                groups[group_name].append(current)

        return groups

    def get_elec_cat(self, subject):
        """Return electrode categories from relevant textfile; ensures that the
        fields are consistent regardless of the actual field the RA entered into
        the textfile

        Parameters
        ----------
        subject: str
            subject ID for subjects in the RAM project. E.g. 'R1111M'

        Returns
        -------
        e_cat_reader: dict
            dictionary mapping relevant field values (bad channel, SOZ, etc.)
            to the corresponding channels

        """
        import numpy as np

        convert = {
            'seizure onset zone': 'SOZ',
            'seizure onset zones': 'SOZ',
            'seizure onset': 'SOZ',

            # Interictal activity
            'interictal': 'IS',
            'interictal spiking': 'IS',
            'interictal spikes': 'IS',
            'ictal onset': 'IS',
            'ictal onset:': 'IS',
            'interictal spiking:': 'IS',
            'octal onset zone': 'IS',

            # Lesioned Tissue
            'brain lesions': 'brain lesion',
            'brain lesions:': 'brain lesion',

            # Bad channels
            'bad electrodes': 'bad ch',
            'bad electrodes:': 'bad ch',
            'broken leads': 'bad ch',
            'broken leads:': 'bad ch'
        }

        e_cat_reader = self.electrode_categories_reader(subject)
        if e_cat_reader is not None:
            e_cat_reader = {convert[v]: np.array([s.upper() for s in e_cat_reader[v]])
                            for k, v in enumerate(e_cat_reader)}

        return e_cat_reader

    def as_dataframe(self):
        pass
