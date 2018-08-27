import pandas as pd
import pytest

from cmlreaders.readers.electrodes import MontageReader, LocalizationReader, \
    ElectrodeCategoriesReader
from cmlreaders.test.test_readers import datafile


class TestMontageReader:
    @pytest.mark.parametrize("kind", ["contacts", "pairs"])
    @pytest.mark.parametrize("subject,localization,montage", [
        ("R1405E", 0, 0),
        ("R1006P", 0, 0),
        ("R1006P", 0, 1),
    ])
    def test_load_json(self, kind, subject, localization, montage):
        if montage == 0:
            path = datafile("{}_{}.json".format(subject, kind))
        else:
            path = datafile("{}_{}_{}.json".format(subject, montage, kind))

        reader = MontageReader(kind,
                               subject=subject,
                               localization=localization,
                               montage=montage,
                               file_path=path)
        df = reader.load()

        if kind == 'contacts':
            assert 'contact' in df.columns
        else:
            assert 'contact_1' in df.columns
            assert 'contact_2' in df.columns

        if montage == 0:
            filename = datafile("{}_{}.csv".format(subject, kind))
        else:
            filename = datafile("{}_{}_{}.csv".format(subject, montage, kind))

        reference = pd.read_csv(filename, index_col=0)
        assert all(reference == df)

    @pytest.mark.parametrize("kind,subject,montage", [
        ("matlab_contacts", "R1001P", 0),
        ("matlab_pairs", "R1001P", 0),
        ("contacts", "R1001P", 0),
        ("pairs", "R1001P", 0),
    ])
    def test_load_matlab(self, kind, subject, montage):
        reader = MontageReader(kind, subject=subject, localization=0,
                               montage=montage)
        path = datafile("{}_{}.mat".format(subject, "pairs" if "pairs" in kind else "contacts"))

        df = reader.fromfile(path)

        if "contacts" in kind:
            assert "contact" in df.columns
        else:
            assert "contact_1" in df.columns
            assert "contact_2" in df.columns


class TestLocalizationReader:
    def test_load(self):
        path = datafile('localization.json')
        reader = LocalizationReader('localization', subject='R1405E', file_path=path)
        df = reader.load()
        assert isinstance(df, pd.DataFrame)


@pytest.mark.rhino
class TestElectrodeCategoriesReader:
    @pytest.mark.parametrize("subject,lens", [
        ("R1111M", {'soz': 9, 'interictal': 15, 'brain_lesion': 5, 'bad_channel': 6}),
        ("R1052E", {'soz': 2, 'interictal': 14, 'brain_lesion': 0, 'bad_channel': 0})
    ])
    def test_load(self, subject, lens, rhino_root):
        reader = ElectrodeCategoriesReader('electrode_categories',
                                           subject=subject,
                                           rootdir=rhino_root)
        categories = reader.load()
        for key, len_ in lens.items():
            assert len(categories[key]) == len_
