import pytest
from cmlreaders.readers import TextReader


class TestTextReader:

    @pytest.mark.rhino
    @pytest.mark.parametrize("file_type", ["voxel_coordinates"])
    @pytest.mark.parametrize("subject,localization", [
        ('R1405E', 0),
        ('R1405E', 1)
    ])
    def test_as_dataframe(self, file_type, subject, localization, rhino_root):
        reader = TextReader(file_type, subject, localization,
                            rootdir=rhino_root)
        df = reader.as_dataframe()
        assert len(df) > 0

    @pytest.mark.rhino
    @pytest.mark.parametrize("file_type", ["voxel_coordinates"])
    @pytest.mark.parametrize("subject,localization", [
        ('R1405E', 0),
        ('R1405E', 1)
    ])
    def test_as_recarray(self, file_type, subject, localization, rhino_root):
        reader = TextReader(file_type, subject, localization,
                            rootdir=rhino_root)
        df = reader.as_recarray()
        assert len(df) > 0
