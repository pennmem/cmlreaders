import pytest
from cmlreaders.path_finder import PathFinder, InvalidFileTypeRequest
from cmlreaders import constants


# Test case for multiple timestamped directories: R1354E PS4_FR5
# Test case for original classifier
# No timestamped directories case

@pytest.fixture()
def all_files_subject(rhino_root):
    finder = PathFinder('R1389J', rootdir=rhino_root, experiment='catFR5',
                        session=1, localization=0, montage=0)
    return finder


@pytest.mark.rhino
@pytest.mark.parametrize("file_type", list(constants.rhino_paths.keys()))
def test_find_file(file_type, all_files_subject):
    if file_type == 'target_selection_table':
        return # does not exist for stim sessions

    if file_type in constants.used_classifier_files:
        with pytest.raises(RuntimeWarning):
            file_path = all_files_subject.find_file(file_type)
            assert file_path is not None
    else:
        file_path = all_files_subject.find_file(file_type)
        assert file_path is not None
    return


@pytest.mark.rhino
def test_invalid_file_request(all_files_subject):
    with pytest.raises(InvalidFileTypeRequest):
        all_files_subject.find_file('fake_file_type')
