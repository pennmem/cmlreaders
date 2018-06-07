import json
from pathlib import Path
from pkg_resources import resource_filename
import pytest

import numpy as np
from numpy.testing import assert_equal
import pandas as pd

from cmlreaders import CMLReader, PathFinder
from cmlreaders import exc
from cmlreaders.readers.eeg import (
    events_to_epochs, milliseconds_to_events, milliseconds_to_samples,
    NumpyEEGReader, RamulatorHDF5Reader, samples_to_milliseconds,
    SplitEEGReader,
)


@pytest.fixture
def events():
    cml_reader = CMLReader()
    path = resource_filename('cmlreaders.test.data', 'all_events.json')
    reader = cml_reader.get_reader('events', file_path=path)
    return reader.as_dataframe()


@pytest.mark.parametrize("millis,rate,samples", [
    (1000, 1000, 1000),
    (1000, 500, 500),
    (1000, 250, 250),
])
def test_milliseconds_to_samples(millis, rate, samples):
    assert milliseconds_to_samples(millis, rate) == samples


@pytest.mark.parametrize("samples,rate,millis", [
    (1000, 1000, 1000),
    (1000, 500, 2000),
    (1000, 250, 4000),
])
def test_samples_to_milliseconds(samples, rate, millis):
    assert samples_to_milliseconds(samples, rate) == millis


@pytest.mark.parametrize("onsets,rate,expected", [
    ([0], 1000, [0]),
    ([10], 100, [1])
])
def test_milliseconds_to_events(onsets, rate, expected):
    df = milliseconds_to_events(onsets, rate)
    assert_equal(expected, df.eegoffset.values)


@pytest.mark.parametrize('rel_start', [-100, 0])
@pytest.mark.parametrize('rel_stop', [100, 500])
def test_events_to_epochs(events, rel_start, rel_stop):
    words = events[events.type == 'WORD']

    epochs = events_to_epochs(words, rel_start, rel_stop, 1000)
    assert len(epochs) == 156
    for epoch in epochs:
        assert epoch[1] - epoch[0] == rel_stop - rel_start


def test_events_to_epochs_simple():
    offsets = [100, 200]  # in samples
    events = pd.DataFrame({"eegoffset": offsets})
    rate = 1000  # in samples / s
    rel_start = -10  # in ms
    rel_stop = 10  # in ms

    epochs = events_to_epochs(events, rel_start, rel_stop, rate)

    assert epochs[0][0] == 90
    assert epochs[0][1] == 110
    assert epochs[1][0] == 190
    assert epochs[1][1] == 210


class TestFileReaders:
    def get_finder(self, subject, experiment, session, rootdir):
        return PathFinder(subject, experiment, session, rootdir=rootdir)

    def get_meta(self, subject, experiment, session, rootdir):
        finder = self.get_finder(subject, experiment, session, rootdir)
        meta_path = Path(finder.find('sources'))
        with meta_path.open() as metafile:
            meta = list(json.load(metafile).values())[0]

        basename = meta['name']
        sample_rate = meta['sample_rate']
        dtype = meta['data_format']
        filename = str(meta_path.parent.joinpath('noreref', basename))

        return basename, sample_rate, dtype, filename

    @pytest.mark.only
    def test_npy_reader(self):
        filename = resource_filename("cmlreaders.test.data", "eeg.npy")
        reader = NumpyEEGReader(filename, np.int16, [(0, -1)])
        ts = reader.read()
        assert ts.shape == (1, 32, 1000)

        orig = np.load(filename)
        assert_equal(orig, ts[0])

    @pytest.mark.rhino
    def test_split_eeg_reader(self, rhino_root):
        basename, sample_rate, dtype, filename = self.get_meta('R1111M', 'FR1', 0, rhino_root)

        events = pd.DataFrame({"eegoffset": list(range(0, 500, 100))})
        rel_start, rel_stop = 0, 200
        epochs = events_to_epochs(events, rel_start, rel_stop, sample_rate)

        eeg_reader = SplitEEGReader(filename, dtype, epochs)
        ts = eeg_reader.read()

        assert ts.shape == (len(epochs), 100, 100)

    @pytest.mark.rhino
    def test_ramulator_hdf5_reader(self, rhino_root):
        basename, sample_rate, dtype, filename = self.get_meta('R1386T', 'FR1', 0, rhino_root)

        events = pd.DataFrame({"eegoffset": list(range(0, 500, 100))})
        rel_start, rel_stop = 0, 200
        epochs = events_to_epochs(events, rel_start, rel_stop, sample_rate)

        eeg_reader = RamulatorHDF5Reader(filename, dtype, epochs)
        ts = eeg_reader.read()

        num_expected_channels = 214
        time_steps = 200
        assert ts.shape == (len(epochs), num_expected_channels, time_steps)


@pytest.mark.rhino
class TestEEGReader:
    @pytest.mark.parametrize('subject', ['R1298E', 'R1387E'])
    def test_eeg_reader(self, subject, rhino_root):
        """Note: R1387E uses Ramulator's HDF5 format, R1298E uses split EEG."""
        reader = CMLReader(subject=subject, experiment='FR1', session=0,
                           rootdir=rhino_root)
        eeg = reader.load_eeg(epochs=[(0, 100), (100, 200)])
        assert len(eeg.time) == 100
        assert len(eeg.epochs) == 2

    def test_eeg_reader_with_events(self, rhino_root):
        reader = CMLReader(subject='R1387E', experiment='FR1', session=0,
                           rootdir=rhino_root)
        events = reader.load('events')
        word_events = events[events.type == 'WORD'].iloc[:10]
        eeg = reader.load_eeg(events=word_events, rel_start=-75, rel_stop=75)
        assert eeg.shape == (10, 121, 150)

        ErrorType = exc.IncompatibleParametersError

        with pytest.raises(ErrorType):
            reader.load_eeg(events=word_events, rel_start=0)

        with pytest.raises(ErrorType):
            reader.load_eeg(events=word_events, rel_stop=0)

        with pytest.raises(ErrorType):
            reader.load_eeg(events=word_events)

    @pytest.mark.parametrize("subject,reref_possible", [
        ('R1387E', False),
        ('R1111M', True),
    ])
    def test_rereference(self, subject, reref_possible, rhino_root):
        reader = CMLReader(subject=subject, experiment='FR1', session=0,
                           rootdir=rhino_root)
        rate = reader.load("sources")["sample_rate"]

        events = pd.DataFrame({"eegoffset": [0]})
        rel_start, rel_stop = 0, 100
        epochs = events_to_epochs(events, rel_start, rel_stop, rate)

        expected_samples = int(rate * rel_stop / 1000)
        scheme = reader.load('pairs')

        if not reref_possible:
            with pytest.raises(exc.RereferencingNotPossibleError):
                reader.load_eeg(epochs=epochs, scheme=scheme)

        else:
            data = reader.load_eeg(epochs=epochs)
            assert data.shape == (1, 100, expected_samples)
            data = reader.load_eeg(epochs=epochs, scheme=scheme)
            assert data.shape == (1, 141, expected_samples)
