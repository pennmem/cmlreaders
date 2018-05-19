import pytest
import functools
from pkg_resources import resource_filename
from cmlreaders import CMLReader

datafile = functools.partial(resource_filename, 'cmlreaders.test.data')


class TestCMLReader:
    @pytest.mark.rhino
    @pytest.mark.parametrize("file_type", [
        'voxel_coordinates', 'classifier_excluded_leads', 'jacksheet',
        'good_leads', 'leads', 'electrode_coordinates', 'prior_stim_results',
        'target_selection_table'
    ])
    def test_load_from_rhino(self, file_type, rhino_root):
        reader = CMLReader(subject="R1405E", localization='0', experiment='FR1',
                           rootdir=rhino_root)
        reader.load(file_type)

    @pytest.mark.parametrize("file_type", [
        'voxel_coordinates.txt', 'classifier_excluded_leads.txt',
        'jacksheet.txt', 'good_leads.txt', 'leads.txt',
        'electrode_coordinates.csv', 'prior_stim_results.csv',
        'target_selection_table.csv'
    ])
    def test_load(self, file_type):
        reader = CMLReader(subject="R1405E", localization='0', experiment="FR1")
        reader.load(data_type=file_type[:-4], file_path=datafile(file_type))

    @pytest.mark.parametrize("file_type", [
        'voxel_coordinates', 'classifier_excluded_leads', 'jacksheet',
        'good_leads', 'leads', 'electrode_coordinates', 'prior_stim_results',
        'target_selection_table'
    ])
    def test_load_proxy(self, file_type):
        reader = CMLReader(subject='R1405E', localization='0', experiment='FR1',
                           session='0', montage='0')
        reader_obj = reader.load_proxy(file_type, file_path=datafile(file_type))
        assert type(reader_obj) == reader.readers[file_type]
