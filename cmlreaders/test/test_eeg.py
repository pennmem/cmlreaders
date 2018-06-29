from functools import partial
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
    BaseEEGReader, NumpyEEGReader, RamulatorHDF5Reader,
    samples_to_milliseconds, SplitEEGReader,
)
from cmlreaders.readers import MontageReader
from cmlreaders.test.utils import patched_cmlreader


@pytest.fixture
def events():
    with patched_cmlreader():
        cml_reader = CMLReader("R1389J")
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


class TestBaseEEGReader:
    @pytest.mark.parametrize("use_scheme", [True, False])
    def test_include_contact(self, use_scheme):
        class DummyReader(BaseEEGReader):
            def read(self):
                return

        scheme = pd.DataFrame({
            "contact_1": list(range(1, 10)),
            "contact_2": list(range(2, 11)),
        }) if use_scheme else None

        reader = DummyReader("", np.int16, [(0, None)], scheme=scheme)

        if use_scheme:
            assert len(reader._unique_contacts) == 10

        for i in range(1, 20):
            if i <= 10 or not use_scheme:
                assert reader.include_contact(i)
            else:
                assert not reader.include_contact(i)


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

    def test_npy_reader(self):
        filename = resource_filename("cmlreaders.test.data", "eeg.npy")
        reader = NumpyEEGReader(filename, np.int16, [(0, -1)], None)
        ts, contacts = reader.read()
        assert ts.shape == (1, 32, 1000)

        orig = np.load(filename)
        assert_equal(orig, ts[0])

    @pytest.mark.rhino
    def test_split_eeg_reader(self, rhino_root):
        basename, sample_rate, dtype, filename = self.get_meta('R1111M', 'FR1', 0, rhino_root)

        events = pd.DataFrame({"eegoffset": list(range(0, 500, 100))})
        rel_start, rel_stop = 0, 200
        epochs = events_to_epochs(events, rel_start, rel_stop, sample_rate)

        eeg_reader = SplitEEGReader(filename, dtype, epochs, None)
        ts, contacts = eeg_reader.read()

        assert ts.shape == (len(epochs), 100, 100)
        assert len(contacts) == 100

    @pytest.mark.rhino
    def test_split_eeg_reader_missing_contacts(self, rhino_root):
        basename, sample_rate, dtype, filename = self.get_meta('R1006P', 'FR2', 0, rhino_root)

        events = pd.DataFrame({"eegoffset": list(range(0, 500, 100))})
        rel_start, rel_stop = 0, 200
        epochs = events_to_epochs(events, rel_start, rel_stop, sample_rate)

        eeg_reader = SplitEEGReader(filename, dtype, epochs, None)
        ts, contacts = eeg_reader.read()

        assert ts.shape == (len(epochs), 123, 102)
        assert 1 not in contacts
        assert 98 not in contacts
        assert 99 not in contacts
        assert 100 not in contacts
        assert len(contacts) == 123

    @pytest.mark.rhino
    @pytest.mark.parametrize("subject,experiment,session,num_channels,sequential", [
        ("R1363T", "FR1", 0, 178, True),  # all contacts sequential
        ("R1392N", "PAL1", 0, 112, False),  # missing some contacts in jacksheet
    ])
    def test_ramulator_hdf5_reader_rhino(self, subject, experiment, session,
                                         num_channels, sequential, rhino_root):
        basename, sample_rate, dtype, filename = self.get_meta(
            subject, experiment, session, rhino_root)

        events = pd.DataFrame({"eegoffset": list(range(0, 500, 100))})
        rel_start, rel_stop = 0, 200
        epochs = events_to_epochs(events, rel_start, rel_stop, sample_rate)

        eeg_reader = RamulatorHDF5Reader(filename, dtype, epochs, None)
        ts, contacts = eeg_reader.read()

        time_steps = 200
        assert ts.shape == (len(epochs), num_channels, time_steps)

        df = pd.DataFrame({"contacts": contacts})
        assert sequential == all(df.index + 1 == contacts)

    def test_ramulator_hdf5_reader(self):
        filename = resource_filename('cmlreaders.test.data', 'eeg.h5')
        reader = RamulatorHDF5Reader(filename, np.int16, [(0, None)], None)
        ts, channels = reader.read()

        time_steps = 3000
        assert ts.shape == (1, len(channels), time_steps)

    def test_ramulator_hdf5_rereference(self):
        pairs_file = resource_filename("cmlreaders.test.data",
                                       "R1405E_pairs_loc1_mon1.json")
        pairs = MontageReader("pairs", subject="R1405E",
                              file_path=pairs_file,).load()

        filename = resource_filename("cmlreaders.test.data", "eeg.h5")

        make_reader = partial(RamulatorHDF5Reader, filename, np.int16, [(0, None)])
        reader = make_reader(pairs)
        ts, contacts = reader.read()

        new_ts = reader.rereference(ts, contacts)
        assert (new_ts == ts).all()

        pairs = pairs[:10]
        reader = make_reader(pairs)
        ts, contacts = reader.read()
        new_ts = reader.rereference(ts, contacts)
        assert (new_ts == ts[:, :10, :]).all()

        pairs['contact_1'][0] = pairs.iloc[0].contact_2
        reader = make_reader(pairs)
        ts, contacts = reader.read()
        with pytest.raises(exc.RereferencingNotPossibleError):
            reader.rereference(ts, contacts)


@pytest.mark.rhino
class TestEEGReader:
    @pytest.mark.parametrize("subject,index,channel", [
        ("R1298E", 87, "CH88"),  # Split EEG
        ("R1387E", 13, "CH14"),  # Ramulator HDF5
    ])
    def test_eeg_reader(self, subject, index, channel, rhino_root):
        reader = CMLReader(subject=subject, experiment='FR1', session=0,
                           rootdir=rhino_root)
        eeg = reader.load_eeg(epochs=[(0, 100), (100, 200)])
        assert len(eeg.time) == 100
        assert len(eeg.epochs) == 2
        assert eeg.channels[index] == channel

    @pytest.mark.parametrize("subject", ["R1161E"])
    def test_read_whole_session(self, subject, rhino_root):
        reader = CMLReader(subject=subject, experiment="FR1", session=0,
                           rootdir=rhino_root)
        reader.load_eeg()

    @pytest.mark.parametrize('subject', ['R1161E', 'R1387E'])
    def test_eeg_reader_with_events(self, subject, rhino_root):
        """Note: R1161E is split over two separate sets of files"""

        reader = CMLReader(subject=subject, experiment='FR1', session=0,
                           rootdir=rhino_root)
        events = reader.load('events')
        word_events = events[events.type == 'WORD']
        eeg = reader.load_eeg(events=word_events, rel_start=-75, rel_stop=75)
        assert eeg.shape[0] == len(word_events)
        assert eeg.shape[-1] == 150

        ErrorType = exc.IncompatibleParametersError

        with pytest.raises(ErrorType):
            reader.load_eeg(events=word_events, rel_start=0)

        with pytest.raises(ErrorType):
            reader.load_eeg(events=word_events, rel_stop=0)

        with pytest.raises(ErrorType):
            reader.load_eeg(events=word_events)

    @pytest.mark.parametrize("subject,reref_possible,index,channel", [
        ("R1384J", False, 43, "LS12-LS1"),
        ("R1111M", True, 43, "LPOG23-LPOG31"),
    ])
    def test_rereference(self, subject, reref_possible, index, channel,
                         rhino_root):
        reader = CMLReader(subject=subject, experiment='FR1', session=0,
                           rootdir=rhino_root)
        rate = reader.load("sources")["sample_rate"]

        events = pd.DataFrame({"eegoffset": [0]})
        rel_start, rel_stop = 0, 100
        epochs = events_to_epochs(events, rel_start, rel_stop, rate)

        expected_samples = int(rate * rel_stop / 1000)
        scheme = reader.load('pairs')
        print(scheme.label)

        if reref_possible:
            data = reader.load_eeg(epochs=epochs)
            assert data.shape == (1, 100, expected_samples)
            data = reader.load_eeg(epochs=epochs, scheme=scheme)
            assert data.shape == (1, 141, expected_samples)
            assert data.channels[index] == channel
        else:
            data_noreref = reader.load_eeg(epochs=epochs)
            data_reref = reader.load_eeg(epochs=epochs, scheme=scheme)
            assert_equal(data_noreref.data, data_reref.data)
            assert data_reref.channels[index] == channel

    @pytest.mark.rhino
    @pytest.mark.parametrize("subject,region_key,region_name,expected_channels,tlen", [
        ("R1384J", "ind.region", "insula", 10, 200),  # Ramulator bipolar
        ("R1288P", "ind.region", "lateralorbitofrontal", 5, 200),  # Ramulator monopolar (but will load from split...)
        ("R1111M", "ind.region", "middletemporal", 18, 100),  # "split" EEG
    ])
    def test_filter_channels(self, subject, region_key, region_name,
                             expected_channels, tlen, rhino_root):
        """Test that we can actually filter channels. This happens via
        rereference, so it's really just a special case check of that.

        """
        reader = CMLReader(subject, "FR1", 0, rootdir=rhino_root)
        pairs = reader.load("pairs")
        scheme = pairs[pairs[region_key] == region_name]
        all_events = reader.load("events")
        events = all_events[all_events["type"] == "WORD"]

        eeg = reader.load_eeg(events, rel_start=-100, rel_stop=100,
                              scheme=scheme)

        assert eeg.shape[0] == len(events)
        assert eeg.shape[1] == expected_channels
        assert eeg.shape[2] == tlen
        assert eeg.attrs["rereferencing_possible"] is (
            True if subject != "R1384J" else False
        )
