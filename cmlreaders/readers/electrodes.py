import json
import os.path
import scipy.io as sio
import numpy as np

import pandas as pd
from pandas.io.json import json_normalize

from cmlreaders import exc
from cmlreaders.base_reader import BaseCMLReader
from cmlreaders.readers.readers import MNICoordinatesReader
import cmlreaders.warnings
import warnings


class MontageReader(BaseCMLReader):
    """Reads montage files (contacts.json, pairs.json). When loading via
    :meth:`CMLReader.load`, pass ``read_categories=True`` to additionally load
    electrode category information.

    Returns a :class:`pd.DataFrame`.

    """
    data_types = ["pairs", "contacts", "matlab_contacts", "matlab_pairs"]
    caching = "memory"

    # MATLAB struct names
    struct_names = ['bpTalStruct', 'talStruct', 'virtualTalStruct',
                    'subjTalEvents', 'events']

    read_categories = False

    def load(self, read_categories: bool = False):
        self.read_categories = read_categories
        return super().load()

    def _insert_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inserts electrode category information into the loaded DataFrame.

        Parameters
        ----------
        df
            Montage DataFrame.

        Returns
        -------
        DataFrame updated with a column named ``category`` containing a comma
        separated list of category names.

        """
        try:
            category_reader = ElectrodeCategoriesReader(
                data_type="electrode_categories",
                subject=self.subject,
                experiment=self.experiment,
                session=self.session,
                localization=self.localization,
                montage=self.montage,
                rootdir=self.rootdir,
            )
            categories = category_reader.load()
        except IOError:
            raise exc.MissingDataError(
                "Cannot find electrode category information for " +
                "{}, {} session {},".format(
                    self.subject, self.experiment, self.session) +
                "localization {}, montage {}".format(
                    self.localization, self.montage)
            )

        column = [None] * len(df)

        for i, label in enumerate(df["label"]):
            if self.data_type == "contacts":
                cat = [key for key in categories if label in categories[key]]
            else:
                l1, l2 = label.split("-")
                cat = [key for key in categories
                       if l1 in categories[key] or l2 in categories[key]]

            column[i] = cat

        df["category"] = column

        return df

    def _insert_mni_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        mni_reader = MNICoordinatesReader(
            data_type='mni_coordinates',
            subject=self.subject,
            experiment=self.experiment,
            session=self.session,
            localization=self.localization,
            montage=self.montage,
            rootdir=self.rootdir,
        )
        mni_coords = mni_reader.as_dataframe()
        # Normalize labels to upper case
        mni_coords['label'] = [lbl.upper() for lbl in mni_coords['label']]
        if 'pairs' in self.data_type:
            mni_coords = mni_coords.set_index('label')
            # Normalize labels to upper case
            pair_labels = [p.upper().split('-') for p in df.label]
            mni_coords = [mni_coords.loc[p, ['mni.x', 'mni.y', 'mni.z']].mean()
                          for p in pair_labels]
            mni_coords = pd.DataFrame(data=mni_coords,
                                      index=df.label).reset_index()

        df = df.merge(mni_coords, how='left', on='label')

        return df

    def _as_dataframe_json(self):
        """Load montage data from newer JSON formats."""
        with open(self.file_path) as f:
            raw = json.load(f)

            # we're using fromfile, so we need to infer subject/data_type
            if not len(self.data_type):
                self.data_type = (
                    "contacts" if "contacts" in os.path.basename(
                        self.file_path) else "pairs"
                )

            subject_key = [key for key in raw.keys() if key != "version"][0]
            pairs = raw[subject_key][self.data_type]

        records = []
        for pair, data in pairs.items():
            atlases = data.pop("atlases", {})

            for atlas_label, atlas_data in atlases.items():
                data.update({
                    "{}.{}".format(atlas_label, key): value
                    for key, value in atlas_data.items()
                    if not key.endswith("id")  # this just duplicates the key
                })

            records.append(data)

        df = pd.DataFrame(records)

        return df

    def _as_dataframe_matlab(self):
        """Load montage data from older MATLAB format."""
        data_dict = sio.loadmat(self.file_path, squeeze_me=True)
        for name in self.struct_names:
            if name in data_dict:
                self.struct_name = name
                break
        else:
            raise ValueError("Montage info has unknown name ")
        arr = data_dict[self.struct_name]
        flat_cols = [c for c in arr.dtype.names
                     if not isinstance(arr[c][0], np.ndarray) or
                     arr[c][0].dtype.names is None]
        nested_cols = [c for c in arr.dtype.names if c not in flat_cols]

        # I'll implement arbitrary nesting as soon as you show me an example
        # with depth > 1

        df = pd.DataFrame(arr[flat_cols])
        for col in nested_cols:
            subcols = arr[col][0].dtype.names
            for c in subcols:
                datacol = [x[c] if x.shape else x[c].item() for x in
                           arr[col]]
                idxs = range(len(datacol))
                new_datacol, new_idxs = list(
                    zip(*[(x, i) for (x, i) in zip(datacol, idxs)
                          if (((not isinstance(x, np.ndarray))
                              or x.size > 0) and x)])
                )
                new_df = pd.DataFrame(np.array(new_datacol),
                                      index=new_idxs,
                                      columns=['{}.{}'.format(col, c)])
                df = df.merge(new_df, how='outer',
                              left_index=True, right_index=True)

        if hasattr(df.channel.iloc[0], "__len__"):
            channels = np.array([[ch[0], ch[1]] for ch in df.pop("channel")])
            df["channel_1"] = channels[:, 0]
            df["channel_2"] = channels[:, 1]

        return df

    def as_dataframe(self):
        """Read montage data. This will explicitly use the MATLAB reader logic
        if the filename ends with ".mat" and otherwise uses the newer JSON
        format.

        """
        if self.file_path.endswith(".mat"):
            df = self._as_dataframe_matlab()
        else:
            df = self._as_dataframe_json()

        # rename poorly named columns
        if "contacts" in self.data_type:
            renames = {"channel": "contact"}
        else:
            renames = {"channel_1": "contact_1", "channel_2": "contact_2"}

        renames.update({
            "code": "label",
            "tagName": "label",
            "eType": "type",
        })
        df = df.rename(renames, axis=1)

        # ensure that contact and label appear first
        names = df.columns
        if "contacts" in self.data_type:
            first = ["contact", "label"]
        else:
            first = ["contact_1", "contact_2", "label"]
        df = df[first + [name for name in names if name not in first]]

        # sort by contact
        key = "contact" if "contacts" in self.data_type else "contact_1"
        df = df.sort_values(by=key).reset_index(drop=True)

        # try to insert categories; silently fail if it doesn't work
        if self.read_categories:
            try:
                df = self._insert_categories(df)
            except Exception:  # noqa
                pass

        # Insert MNI coordinates if they're missing; emit a warning if it
        # doesn't work
        if not any('mni' in c for c in df.columns) and self.subject:
            try:
                df = self._insert_mni_coordinates(df)
            except Exception:  # noqa
                warnings.warn(cmlreaders.warnings.MissingCoordinatesWarning(
                    "Could not load MNI coordinates"))

        return df


class LocalizationReader(BaseCMLReader):
    """Reads data stored in localization.json.

    Returns a :class:`pd.DataFrame`.

    """
    data_types = ["localization"]
    protocols = ["r1"]
    caching = "memory"

    def as_dataframe(self):
        import itertools

        with open(self.file_path) as f:
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

        flat_contact_data = list(itertools.chain(*[x["contacts"]
                                                   for x in leads]))
        flat_pairs_data = list(itertools.chain(*[x["pairs"]
                                                 for x in leads]))
        all_data = [
            json_normalize(flat_contact_data).set_index('name'),
            json_normalize(flat_pairs_data).set_index('names')
        ]
        combined_df = pd.concat(all_data, keys=['contacts', 'pairs'],
                                sort=True)
        return combined_df


class ElectrodeCategoriesReader(BaseCMLReader):
    """Reads electrode_categories.txt and handles the many inconsistencies in
    those files.

    Returns a ``dict``.

    """
    data_types = ["electrode_categories"]
    protocols = ["r1"]
    default_representation = "dict"
    caching = "memory"

    def _read_categories(self) -> dict:
        """Returns a dictionary mapping categories to electrode from the
        electrode_categories.txt file

        Returns
        -------
        groups: dict,
            dictionary mapping relevant field values (bad channel, SOZ, etc.)
            to the corresponding channels

        Notes
        -----
        This function is only required because there's so much inconsistency in
        where and how the data corresponding to bad electrodes are stored.

        """
        # Used to indicate relevant strings in the text files
        relevant = {
            'seizure onset zone', 'seizure onset zones', 'seizure onset',
            'interictal spiking', 'interictal spikes', 'interictal spikes:',
            'ictal onset', 'ictal onset:',
            'interictal spiking:', 'interictal:',
            'brain lesions', 'brain lesions:', 'octal onset zone',
            'bad electrodes', 'bad electrodes:', 'broken leads',
            'broken leads:'
        }

        with open(self.file_path, 'r') as f:
            ch_info = f.read().split('\n')

        # This will be used to initialize a before after kind of check to sort
        # the groups
        count = 0
        groups = {}  # Save the groups here

        for index, line in enumerate(ch_info[2:]):
            # We skip to 2 because all files start with line 1 being subject
            # followed by another line of '', if the user wishes to access the
            # information feel free to modify below. Blank spaces used to
            # separate, if we encountered one count increases
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
                if ch_info[2:][index + 1] != '':  # Skipping 2 for same reason
                    count += 1
                continue

            # Triggered when inside of a group e.g. they're channel names
            if (count == 1) and (line != ''):  # start of group values
                groups[group_name].append(line)

        return groups

    def _get_categories_dict(self) -> dict:
        """Return electrode categories from relevant textfile; ensures that the
        fields are consistent regardless of the actual field the RA entered
        into the textfile

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
            'ictal onset': 'soz',
            'ictal onset:': 'soz',
            'octal onset zone': 'soz',

            # Interictal activity
            'interictal': 'interictal',
            'interictal spiking': 'interictal',
            'interictal spikes': 'interictal',
            'interictal spiking:': 'interictal',
            'interictal spikes:': 'interictal',

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
            e_cat_reader = {convert[v]:
                            np.array([s.upper() for s in e_cat_reader[v]])
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
