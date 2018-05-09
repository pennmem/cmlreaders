import pytest
from cmlreaders import CMLReader


class TestCMLReader:
    @pytest.mark.rhino
    @pytest.mark.parametrize("file_type", [
        'voxel_coordinates',
    ])
    def test_load(self, file_type, rhino_root):
        reader = CMLReader(subject="R1405E", localization=1, rootdir=rhino_root)
        reader.load(file_type)
