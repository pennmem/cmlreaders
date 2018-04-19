import pytest
from cml_data_readers.path_finder import PathFinder
from cml_data_readers import constants


class TestPathFinder:
    @classmethod
    def setup_class(cls):
        cls.finder = PathFinder('R1389J', rootdir='/Volumes/RHINO/', experiment='catFR5', session=1,
                                localization=0, montage=0)

    @pytest.mark.parametrize("file_type", constants.localization_files)
    def test_find_localization_files(self, file_type):
        file_path = self.finder.find_file(file_type)
        assert file_path is not None
        return

    @pytest.mark.parametrize("file_type", constants.montage_files)
    def test_find_montage_files(self, file_type):
        file_path = self.finder.find_file(file_type)
        assert file_path is not None
        return

    @pytest.mark.parametrize("file_type", constants.session_files)
    def test_find_session_files(self, file_type):
        if file_type in constants.used_classifier_files:
            with pytest.raises(RuntimeWarning):
                file_path = self.finder.find_file(file_type)
                assert file_path is not None
        else:
            file_path = self.finder.find_file(file_type)
            assert file_path is not None
        return

# Test case for multiple timestamped directories: R1354E PS4_FR5
# Test case for original classifier
# No timestamped directories case