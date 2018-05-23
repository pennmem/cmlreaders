import os
import pytest
from cmlreaders.path_finder import PathFinder, InvalidDataTypeRequest
from cmlreaders import constants


# Test case for multiple timestamped directories: R1354E PS4_FR5
# Test case for original classifier
# No timestamped directories case

@pytest.fixture()
def current_files_subject(rhino_root):
    finder = PathFinder('R1389J', rootdir=rhino_root, experiment='catFR5',
                        session=1, localization=0, montage=0)
    return finder


@pytest.fixture()
def non_zero_localization_subject(rhino_root):
    finder = PathFinder("R1405E", rootdir=rhino_root, experiment='FR1',
                        session=1, localization=1, montage=1)
    return finder


@pytest.fixture()
def legacy_files_subject(rhino_root):
    finder = PathFinder('R1111M', rootdir=rhino_root, localization=0)
    return finder


@pytest.fixture
def ramulator_files_finder(rhino_root):
    finder = PathFinder('R1354E', rootdir=rhino_root, experiment='PS4_FR5',
                        session=0, localization=0)
    return finder


@pytest.mark.rhino
@pytest.mark.parametrize("file_type", list(constants.rhino_paths.keys()))
def test_find_file(file_type, current_files_subject, legacy_files_subject):
    if file_type in ['target_selection_table', 'ps4_events']:
        return  # does not exist for stim sessions

    if file_type in ['matlab_bipolar_talstruct', 'matlab_monopolar_talstruct']:
        myfinder = legacy_files_subject
    else:
        myfinder = current_files_subject

    file_path = myfinder.find(file_type)
    assert file_path is not None


@pytest.mark.rhino
def test_invalid_file_request(current_files_subject):
    with pytest.raises(InvalidDataTypeRequest):
        current_files_subject.find('fake_file_type')


@pytest.mark.rhino
def test_nonzero_localization_lookup(non_zero_localization_subject):
    path = non_zero_localization_subject.find("pairs")
    assert path is not None


@pytest.mark.ramulator
@pytest.mark.rhino
def test_get_ramulator_files(ramulator_files_finder):
    path = ramulator_files_finder.find('experiment_config')
    assert path.endswith(os.path.join('20171027_144048',
                                      'experiment_config.json'))

    folder_path = ramulator_files_finder.find("ramulator_session_folder")
    assert folder_path.endswith('20171027_144048')
