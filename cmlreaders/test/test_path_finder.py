import os
import pytest
from cmlreaders.path_finder import PathFinder, InvalidDataTypeRequest
from cmlreaders import constants


# Test case for multiple timestamped directories: R1354E PS4_FR5
# Test case for original classifier
# No timestamped directories case

@pytest.fixture()
def all_files_subject(rhino_root):
    finder = PathFinder('R1389J', rootdir=rhino_root, experiment='catFR5',
                        session='1', localization='0', montage='0')
    return finder


@pytest.fixture
def ramulator_files_finder(rhino_root):
    finder = PathFinder('R1354E', rootdir=rhino_root, experiment='PS4_FR5',
                        session=0)
    return finder


@pytest.mark.rhino
@pytest.mark.parametrize("file_type", list(constants.rhino_paths.keys()))
def test_find_file(file_type, all_files_subject):
    if file_type in ['target_selection_table', 'ps4_events']:
        return  # does not exist for stim sessions

    file_path = all_files_subject.find(file_type)
    assert file_path is not None
    return


@pytest.mark.rhino
def test_invalid_file_request(all_files_subject):
    with pytest.raises(InvalidDataTypeRequest):
        all_files_subject.find('fake_file_type')


@pytest.mark.ramulator
@pytest.mark.rhino
def test_get_ramulator_files(ramulator_files_finder):
    path = ramulator_files_finder.find('experiment_config')
    assert path.endswith(os.path.join('20171027_144048', 'experiment_config.json'))
