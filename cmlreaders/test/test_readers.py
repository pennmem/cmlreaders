import os
import pytest
import pandas as pd
import numpy as np
import functools

from cmlreaders.readers import (
    BasicJSONReader, TextReader, CSVReader, EEGMetaReader,
    ElectrodeCategoriesReader, EventReader, LocalizationReader, MontageReader,
    RamulatorEventLogReader, ReportSummaryDataReader, BaseReportDataReader,
    ClassifierContainerReader
)
from cmlreaders.exc import UnsupportedRepresentation, UnsupportedExperimentError
from pkg_resources import resource_filename
from ramutils.reports.summary import ClassifierSummary, FRStimSessionSummary,\
    MathSummary
from classiflib.container import ClassifierContainer

datafile = functools.partial(resource_filename, 'cmlreaders.test.data')


class TestTextReader:

    @pytest.mark.parametrize("method", ['dataframe', 'recarray', 'dict'])
    @pytest.mark.parametrize("data_type", [
        "voxel_coordinates", "leads", "classifier_excluded_leads", "good_leads",
        "jacksheet", "area"])
    @pytest.mark.parametrize("subject,localization", [
        ('R1389J', '0'),
    ])
    def test_as_methods(self, method, data_type, subject, localization):
        file_path = datafile(data_type + ".txt")
        reader = TextReader(data_type, subject, localization,
                            file_path=file_path)
        expected_types = {
            'dataframe': pd.DataFrame,
            'recarray': np.recarray,
            'dict': list
        }
        method_name = "as_{}".format(method)
        callable_method = getattr(reader, method_name)
        data = callable_method()
        assert data is not None
        assert type(data) == expected_types[method]

    @pytest.mark.parametrize("method", ['json', 'csv'])
    @pytest.mark.parametrize("data_type", [
        "voxel_coordinates", "leads", "classifier_excluded_leads", "good_leads",
        "jacksheet", "area"])
    @pytest.mark.parametrize("subject,localization", [
        ('R1389J', '0'),
    ])
    def test_to_methods(self, method, data_type, subject, localization,
                        rhino_root):
        file_path = datafile(data_type + ".txt")
        reader = TextReader(data_type, subject, localization,
                            file_path=file_path, rootdir=rhino_root)

        # Save as specified format
        method_name = "to_{}".format(method)
        callable_method = getattr(reader, method_name)
        exp_output = datafile("output/" + data_type + "." + method)
        callable_method(exp_output)
        assert os.path.exists(exp_output)

        # Check that data can be reloaded
        re_reader = TextReader(data_type, subject, localization,
                               file_path=exp_output)
        reread_data = re_reader.as_dataframe()
        assert reread_data is not None
        os.remove(exp_output)


class TestCSVReader:

    @pytest.mark.parametrize("method", ["dataframe", "recarray", "dict"])
    @pytest.mark.parametrize("data_type", [
        'electrode_coordinates', 'prior_stim_results', 'target_selection_table'
    ])
    @pytest.mark.parametrize("subject,localization", [
        ('R1409D', '0'),
    ])
    def test_as_methods(self, method, data_type, subject, localization):
        file_path = datafile(data_type + ".csv")
        reader = CSVReader(data_type, subject, localization, experiment="FR1",
                           file_path=file_path)
        expected_types = {
            'dataframe': pd.DataFrame,
            'recarray': np.recarray,
            'dict': list
        }
        method_name = "as_{}".format(method)
        callable_method = getattr(reader, method_name)
        data = callable_method()
        assert data is not None
        assert type(data) == expected_types[method]

    @pytest.mark.parametrize("method", ['json', 'csv'])
    @pytest.mark.parametrize("data_type", [
        'electrode_coordinates', 'prior_stim_results', 'target_selection_table'
    ])
    @pytest.mark.parametrize("subject,localization", [
        ('R1409D', '0'),
    ])
    def test_to_methods(self, method, data_type, subject, localization,
                        rhino_root):
        # Load the test data
        file_path = datafile(data_type + ".csv")
        reader = CSVReader(data_type, subject, localization, experiment="FR1",
                           file_path=file_path, rootdir=rhino_root)

        # Save as specified format
        method_name = "to_{}".format(method)
        callable_method = getattr(reader, method_name)
        exp_output = datafile("output/" + data_type + "." + method)
        callable_method(exp_output)
        assert os.path.exists(exp_output)

        # Check that data can be reloaded
        re_reader = CSVReader(data_type, subject, localization,
                              experiment="FR1", file_path=exp_output)
        reread_data = re_reader.as_dataframe()
        assert reread_data is not None
        os.remove(exp_output)


class TestRamulatorEventLogReader:
    @pytest.mark.parametrize("method", ["dataframe", "recarray", "dict"])
    @pytest.mark.parametrize("data_type", ['event_log'])
    @pytest.mark.parametrize("subject,experiment,session", [
        ('R1409D', 'catFR1', '1'),
    ])
    def test_as_methods(self, method, data_type, subject, experiment, session,
                        rhino_root):
        file_path = datafile(data_type + ".json")
        reader = RamulatorEventLogReader(data_type, subject=subject,
                                         experiment=experiment, session=session,
                                         file_path=file_path,
                                         rootdir=rhino_root)
        expected_types = {
            'dataframe': pd.DataFrame,
            'recarray': np.recarray,
            'dict': dict
        }
        method_name = "as_{}".format(method)
        callable_method = getattr(reader, method_name)
        data = callable_method()
        assert data is not None
        assert type(data) == expected_types[method]

    @pytest.mark.parametrize("method", ['json', 'csv'])
    @pytest.mark.parametrize("data_type", ['event_log'])
    @pytest.mark.parametrize("subject,experiment,session", [
        ('R1409D', 'catFR1', '1'),
    ])
    def test_to_methods(self, method, data_type, subject, experiment, session,
                        rhino_root):
        # Load the test data
        file_path = datafile(data_type + ".json")
        reader = RamulatorEventLogReader(data_type, subject=subject,
                                         experiment=experiment, session=session,
                                         file_path=file_path,
                                         rootdir=rhino_root)
        # Save as specified format
        method_name = "to_{}".format(method)
        callable_method = getattr(reader, method_name)
        exp_output = datafile("output/" + data_type + "." + method)
        callable_method(exp_output)
        assert os.path.exists(exp_output)
        os.remove(exp_output)

        # Note: We are not testing that the data can be reloaded with the
        # reader because the format has materially changed from the original
        # source. This does not happen for all readers, which is why we can
        # test reloading for some


class TestBasicJSONReader:
    def test_load(self):
        path = datafile('index.json')
        reader = BasicJSONReader('index.json', file_path=path)
        df = reader.load()
        assert isinstance(df, pd.DataFrame)


class TestEEGMetaReader:
    def test_load(self):
        path = datafile("sources.json")
        reader = EEGMetaReader("sources.json", file_path=path)
        sources = reader.load()

        assert isinstance(sources, dict)
        assert sources["data_format"] == "int16"
        assert sources["n_samples"] == 1641165
        assert sources["sample_rate"] == 1000


class TestEventReader:
    def test_load(self):
        path = datafile('all_events.json')
        reader = EventReader('all_events', file_path=path)
        df = reader.load()
        assert df.columns[0] == 'eegoffset'


class TestMontageReader:
    @pytest.mark.parametrize('kind', ['contacts', 'pairs'])
    def test_load(self, kind):
        path = datafile(kind + '.json')
        reader = MontageReader(kind, subject='R1405E', file_path=path)
        df = reader.load()

        if kind == 'contacts':
            assert 'contact' in df.columns
        else:
            assert 'contact_1' in df.columns
            assert 'contact_2' in df.columns

    @pytest.mark.rhino
    @pytest.mark.parametrize("subject,expected_pairs", [
        ("R1111M", 141),
        ("R1405E", 158),
    ])
    def test_load_pairs(self, subject, expected_pairs, rhino_root):
        reader = MontageReader("pairs", subject=subject, experiment="FR1",
                               session=0, rootdir=rhino_root)
        df = reader.load()
        assert len(df) == expected_pairs


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


class TestBaseReportDataReader:
    @pytest.mark.parametrize("method", ['pyobject', 'dataframe', 'dict', 'recarray'])
    @pytest.mark.parametrize("data_type", ['classifier_summary'])
    @pytest.mark.parametrize("subject,experiment,session", [
        ('R1409D', 'catFR1', '1'),
    ])
    def test_as_methods(self, method, data_type, subject, experiment, session):
        file_path = datafile(data_type + ".h5")
        reader = BaseReportDataReader(data_type, subject=subject,
                                      experiment=experiment, session=session,
                                      localization=0, file_path=file_path)

        pyobj_expected_types = {
            'classifier_summary': ClassifierSummary,
        }

        method_name = "as_{}".format(method)
        callable_method = getattr(reader, method_name)

        if method != "pyobject":
            with pytest.raises(UnsupportedRepresentation):
                callable_method()
            return

        data = callable_method()
        assert data is not None
        assert type(data) == pyobj_expected_types[data_type]

    @pytest.mark.parametrize("method", ['hdf'])
    @pytest.mark.parametrize("data_type", ["classifier_summary"])
    def test_to_methods(self, method, data_type):
        # Load the test data
        file_path = datafile(data_type + ".h5")
        reader = BaseReportDataReader(data_type, subject='R1409D',
                                      experiment='catFR1', session=1,
                                      localization=0, file_path=file_path)
        # Save as specified format
        method_name = "to_{}".format(method)
        callable_method = getattr(reader, method_name)
        exp_output = datafile("output/" + data_type + ".h5")
        callable_method(exp_output)
        assert os.path.exists(exp_output)
        os.remove(exp_output)


class TestReportSummaryReader:
    @pytest.mark.parametrize("method", ['pyobject', 'dataframe', 'recarray', 'dict'])
    @pytest.mark.parametrize("data_type", ['math_summary', "session_summary"])
    def test_as_methods(self, method, data_type):
        file_path = datafile(data_type + ".h5")
        reader = ReportSummaryDataReader(data_type, subject='R1409D',
                                         experiment='catFR5', session=1,
                                         localization=0, file_path=file_path)

        pyobj_expected_types = {
            'math_summary': MathSummary,
            'session_summary': FRStimSessionSummary
        }

        expected_types = {
            'dataframe': pd.DataFrame,
            'recarray': np.recarray,
            'dict': list,

        }

        method_name = "as_{}".format(method)
        callable_method = getattr(reader, method_name)
        data = callable_method()
        assert data is not None

        if method == "pyobject":
            assert type(data) == pyobj_expected_types[data_type]
        else:
            assert type(data) == expected_types[method]

    def test_load_nonstim_session(self):
        file_path = datafile('session_summary' + ".h5")
        reader = ReportSummaryDataReader('session_summary', subject='R1409D',
                                         experiment='catFR1', session=1,
                                         localization=0, file_path=file_path)
        with pytest.raises(UnsupportedExperimentError):
            reader.as_pyobject()

    @pytest.mark.xfail
    @pytest.mark.parametrize("method", ['csv', 'json', 'hdf'])
    @pytest.mark.parametrize("data_type", ["math_summary", "session_summary"])
    def test_to_methods(self, method, data_type):
        # Load the test data
        file_path = datafile(data_type + ".h5")
        reader = ReportSummaryDataReader(data_type, subject='R1409D',
                                         experiment='catFR5', session=1,
                                         localization=0, file_path=file_path)
        # Save as specified format
        method_name = "to_{}".format(method)
        callable_method = getattr(reader, method_name)

        extension = "." + method
        if method == "hdf":
            extension = ".h5"

        exp_output = datafile("output/" + data_type + extension)
        callable_method(exp_output)

        assert os.path.exists(exp_output)
        os.remove(exp_output)


class TestClassifierContainerReader:
    @pytest.mark.parametrize("method", ['pyobject'])
    @pytest.mark.parametrize("data_type", [
        'baseline_classifier', 'used_classifier'])
    def test_as_methods(self, method, data_type):
        file_path = datafile(data_type + ".zip")
        reader = ClassifierContainerReader(data_type, subject='R1389J',
                                           experiment='catFR5', session=1,
                                           localization=0, file_path=file_path)

        pyobj_expected_types = {
            'baseline_classifier': ClassifierContainer,
            'used_classifier': ClassifierContainer,
        }

        method_name = "as_{}".format(method)
        callable_method = getattr(reader, method_name)
        data = callable_method()
        assert data is not None
        assert type(data) == pyobj_expected_types[data_type]

    @pytest.mark.parametrize("method", ['binary'])
    @pytest.mark.parametrize("data_type", [
        'baseline_classifier', 'used_classifier'])
    def test_to_methods(self, method, data_type):
        # Load the test data
        file_path = datafile(data_type + ".zip")
        reader = ClassifierContainerReader(data_type, subject='R1389J',
                                           experiment='catFR5', session=1,
                                           localization=0, file_path=file_path)
        # Save as specified format
        method_name = "to_{}".format(method)
        callable_method = getattr(reader, method_name)
        exp_output = datafile("output/" + data_type + ".zip")
        callable_method(exp_output, overwrite=True)
        assert os.path.exists(exp_output)
        os.remove(exp_output)
