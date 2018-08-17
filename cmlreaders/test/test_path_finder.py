import os
import pytest
from cmlreaders.path_finder import PathFinder, InvalidDataTypeRequest
from cmlreaders import constants


# Test case for multiple timestamped directories: R1354E PS4_FR5
# Test case for original classifier
# No timestamped directories case

@pytest.fixture()
def current_files_subject_finder(rhino_root):
    finder = PathFinder('R1389J', rootdir=rhino_root, experiment='catFR5',
                        session=1, localization=0, montage=0)
    return finder


@pytest.fixture()
def legacy_files_subject_finder(rhino_root):
    finder = PathFinder('R1111M', rootdir=rhino_root, localization=0)
    return finder


@pytest.fixture
def ramulator_files_finder(rhino_root):
    finder = PathFinder('R1354E', rootdir=rhino_root, experiment='PS4_FR5',
                        session=0, localization=0)
    return finder


@pytest.mark.rhino
@pytest.mark.parametrize("file_type", list(constants.rhino_paths.keys()))
def test_find_file(file_type, current_files_subject_finder,
                   legacy_files_subject_finder):
    if file_type in ['target_selection_table', 'ps4_events']:
        return  # does not exist for stim sessions
    elif file_type == "processed_eeg":
        return  # special case that isn't actually used by PathFinder

    if file_type in ['matlab_bipolar_talstruct', 'matlab_monopolar_talstruct',
                     'matlab_pairs', 'matlab_contacts']:
        myfinder = legacy_files_subject_finder
    else:
        myfinder = current_files_subject_finder

    file_path = myfinder.find(file_type)
    assert file_path is not None


@pytest.mark.rhino
def test_invalid_file_request(current_files_subject_finder):
    with pytest.raises(InvalidDataTypeRequest):
        current_files_subject_finder.find('fake_file_type')


@pytest.mark.rhino
@pytest.mark.parametrize("subject,localization,montage", [
    ('R1006P', 0, 0),  # standard case
    ('R1006P', 0, 1),  # re-montage without localization change
    ('R1024E', 0, 0),
    ('R1024E', 1, 1),  # localization change resulting in montage change
    ('R1024E', 1, 2),  # montage change after a re-implant
])
def test_montage_file_lookup(subject, localization, montage, rhino_root):
    finder = PathFinder(subject, localization=localization, montage=montage,
                        rootdir=rhino_root)
    path = finder.find('pairs')
    assert path is not None


@pytest.mark.ramulator
@pytest.mark.rhino
def test_get_ramulator_files(ramulator_files_finder):
    path = ramulator_files_finder.find('experiment_config')
    assert path.endswith(os.path.join('20171027_164013',
                                      'experiment_config.json'))

    folder_path = ramulator_files_finder.find("ramulator_session_folder")
    assert folder_path.endswith('20171027_164013')


@pytest.mark.rhino
@pytest.mark.parametrize('use_basename', [True, False])
def test_session_params(rhino_root, use_basename):
    subject = 'TJ012'
    eeg_basename = 'TJ012_20Apr10_1329'
    if use_basename:
        finder = PathFinder(subject=subject, eeg_basename=eeg_basename,
                            rootdir=rhino_root)
    else:
        finder = PathFinder(subject=subject, rootdir=rhino_root)
    path = finder.find('sources')
    assert (eeg_basename in path) == use_basename
