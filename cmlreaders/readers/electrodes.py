from functools import partial
import json
import os.path
from typing import List

import pandas as pd
from pandas.io.json import json_normalize

from cmlreaders.base_reader import BaseCMLReader


class MontageReader(BaseCMLReader):
    """Reads montage files (contacts.json, pairs.json).

    Returns a :class:`pd.DataFrame`.

    """

    data_types = ['pairs', 'contacts']

    @staticmethod
    def _flatten_row(data: dict, labels: List[str], i: int) -> pd.DataFrame:
        entry = data[labels[i]].copy()
        atlases = entry.pop('atlases')
        atlas_row = json_normalize(atlases)
        atlas_row.index = [i]
        row = pd.concat([pd.DataFrame(entry, index=[i]), atlas_row], axis=1)
        return row

    def as_dataframe(self):
        with open(self._file_path) as f:
            raw = json.load(f)

            # we're using fromfile, so we need to infer subject/data_type
            if not len(self.data_type):
                self.subject = [key for key in raw.keys() if key != "version"][0]
                self.data_type = (
                    "contacts" if "contacts" in os.path.basename(self._file_path)
                    else "pairs"
                )

            data = raw[self.subject][self.data_type]

        labels = [l for l in data]

        flatten = partial(self._flatten_row, data, labels)
        rows = [flatten(i) for i in range(len(labels))]
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

    data_types = ['localization']

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

    data_types = ["electrode_categories"]
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
