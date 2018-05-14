import pytest
from cmlreaders.readers import TextReader, CSVReader


class TestTextReader:

    @pytest.mark.rhino
    @pytest.mark.parametrize("file_type", [
        "voxel_coordinates", "leads", "classifier_excluded_leads", "good_leads",
        "jacksheet", "area"])
    @pytest.mark.parametrize("subject,localization", [
        ('R1389J', '0'),
    ])
    def test_as_dataframe(self, file_type, subject, localization, rhino_root):
        reader = TextReader(file_type, subject, localization,
                            rootdir=rhino_root)
        df = reader.as_dataframe()
        assert df is not None

    @pytest.mark.rhino
    @pytest.mark.parametrize("file_type", [
        "voxel_coordinates", "leads", "classifier_excluded_leads", "good_leads",
        "jacksheet", "area"])
    @pytest.mark.parametrize("subject,localization", [
        ('R1389J', '0'),
    ])
    def test_as_recarray(self, file_type, subject, localization, rhino_root):
        reader = TextReader(file_type, subject, localization,
                            rootdir=rhino_root)
        df = reader.as_recarray()
        assert df is not None


class TestCSVReader:

    @pytest.mark.rhino
    @pytest.mark.parametrize("data_type", [
        'electrode_coordinates.csv', 'prior_stim_results.csv'
    ])
    @pytest.mark.parametrize("subject,localization", [
        ('R1409D', '0'),
        ('R1409D', '1')
    ])
    def test_as_dataframe(self, data_type, subject, localization, rhino_root):
        reader = CSVReader(data_type, subject, localization, rootdir=rhino_root)
        df = reader.as_dataframe()
        assert df is not None

