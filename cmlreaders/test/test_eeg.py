from functools import partial
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from pkg_resources import resource_filename
import pytest

import h5py
import numpy as np
from numpy.testing import assert_equal
import pandas as pd

from cmlreaders import CMLReader, PathFinder
from cmlreaders import convert, exc
from cmlreaders.readers.eeg import (
    BaseEEGReader, EEGMetaReader, EEGReader, NumpyEEGReader,
    RamulatorHDF5Reader, SplitEEGReader,
)
from cmlreaders.readers.electrodes import MontageReader
from cmlreaders.readers.readers import EventReader
from cmlreaders.test.utils import patched_cmlreader
from cmlreaders.warnings import MissingChannelsWarning


@pytest.fixture
def events():
    with patched_cmlreader():
        cml_reader = CMLReader("R1389J")
        path = resource_filename('cmlreaders.test.data', 'all_events.json')
        reader = cml_reader.get_reader('events', file_path=path)
        return reader.as_dataframe()


class TestEEGMetaReader:
    @pytest.mark.parametrize("subject,filename,data_format,n_samples,sample_rate", [
        ("R1389J", "sources.json", "int16", 1641165, 1000),
        ("TJ001", "TJ001_pyFR_params.txt", "int16", None, 400.0),
    ])
    def test_load(self, subject, filename, data_format, n_samples, sample_rate):
        path = resource_filename("cmlreaders.test.data", filename)
        sources = EEGMetaReader.fromfile(path, subject=subject)

        assert isinstance(sources, dict)
        assert sources["data_format"] == data_format
        assert sources["sample_rate"] == sample_rate

        if n_samples is not None:
            assert sources["n_samples"] == n_samples


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
        epochs = convert.events_to_epochs(events, rel_start, rel_stop, sample_rate)

        eeg_reader = SplitEEGReader(filename, dtype, epochs, None)
        ts, contacts = eeg_reader.read()

        assert ts.shape == (len(epochs), 100, 100)
        assert len(contacts) == 100

    @pytest.mark.rhino
    def test_split_eeg_reader_missing_contacts(self, rhino_root):
        basename, sample_rate, dtype, filename = self.get_meta('R1006P', 'FR2', 0, rhino_root)

        events = pd.DataFrame({"eegoffset": list(range(0, 500, 100))})
        rel_start, rel_stop = 0, 200
        epochs = convert.events_to_epochs(events, rel_start, rel_stop, sample_rate)

        eeg_reader = SplitEEGReader(filename, dtype, epochs, None)
        ts, contacts = eeg_reader.read()

        assert ts.shape == (len(epochs), 123, 102)
        assert 1 not in contacts
        assert 98 not in contacts
        assert 99 not in contacts
        assert 100 not in contacts
        assert len(contacts) == 123

    @pytest.mark.rhino
    @pytest.mark.parametrize("subject,experiment,session", [
        ("R1345D", "FR1", 0,),
        ("R1363T", "FR1", 0,),
        ("R1392N", "PAL1", 0),
    ])
    def test_ramulator_hdf5_reader_rhino(self, subject, experiment, session,
                                         rhino_root):
        basename, sample_rate, dtype, filename = self.get_meta(
            subject, experiment, session, rhino_root)

        events = pd.DataFrame({"eegoffset": list(range(0, 500, 100))})
        rel_start, rel_stop = 0, 200
        epochs = convert.events_to_epochs(events, rel_start, rel_stop, sample_rate)

        eeg_reader = RamulatorHDF5Reader(filename, dtype, epochs, None)
        ts, contacts = eeg_reader.read()

        num_channels = len(CMLReader(subject, experiment, session, rootdir=rhino_root).load('pairs'))

        time_steps = 200
        assert ts.shape == (len(epochs), num_channels, time_steps)

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

        new_ts, _ = reader.rereference(ts, contacts)
        assert (new_ts == ts).all()

        pairs = pairs[:10]
        reader = make_reader(pairs)
        ts, contacts = reader.read()
        new_ts, _ = reader.rereference(ts, contacts)
        assert (new_ts == ts[:, :10, :]).all()

        pairs['contact_1'][0] = pairs.iloc[0].contact_2
        reader = make_reader(pairs)
        ts, contacts = reader.read()
        with pytest.raises(exc.RereferencingNotPossibleError):
            reader.rereference(ts, contacts)


@pytest.mark.rhino
class TestEEGReader:
    # FIXME: add LTP, pyFR cases
    @pytest.mark.parametrize("subject,index,channel", [
        ("R1298E", 87, "CH88"),  # Split EEG
        ("R1387E", 13, "CH14"),  # Ramulator HDF5
    ])
    def test_eeg_reader(self, subject, index, channel, rhino_root):
        reader = CMLReader(subject=subject, experiment='FR1', session=0,
                           rootdir=rhino_root)
        events = reader.load("events")
        events = events[events["type"] == "WORD"].iloc[:2]
        eeg = reader.load_eeg(events=events, rel_start=0, rel_stop=100)
        assert len(eeg.time) == 100
        assert eeg.data.shape[0] == 2
        assert eeg.channels[index] == channel

    @pytest.mark.parametrize("subject", ["R1161E"])
    def test_read_whole_session(self, subject, rhino_root):
        reader = CMLReader(subject=subject, experiment="FR1", session=0,
                           rootdir=rhino_root)

        eeg = reader.load_eeg()
        assert eeg.shape == (1, 70, 3304786)

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
        events = reader.load("events")
        events = events[events.type == "WORD"].iloc[:1]

        rel_start, rel_stop = 0, 100

        expected_samples = int(rate * rel_stop / 1000)
        scheme = reader.load('pairs')

        load_eeg = partial(reader.load_eeg, events=events, rel_start=rel_start,
                           rel_stop=rel_stop)

        if reref_possible:
            data = load_eeg()
            assert data.shape == (1, 100, expected_samples)
            data = load_eeg(scheme=scheme)
            assert data.shape == (1, 141, expected_samples)
            assert data.channels[index] == channel
        else:
            data_noreref = load_eeg()
            data_reref = load_eeg(scheme=scheme)
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

    @pytest.mark.parametrize("subject,events_filename,expected_basenames", [
        ("TJ001", "TJ001_events.mat", [
            "/data1/eeg/TJ001/eeg.reref/TJ001_14Jan09_1057",
            "/data1/eeg/TJ001/eeg.reref/TJ001_15Jan09_1134",
            "/data1/eeg/TJ001/eeg.reref/TJ001_16Jan09_1107",
        ]),
        ("R1389J", "task_events.json", [
            "/protocols/r1/subjects/R1389J/experiments/catFR1/sessions/0/ephys/current_processed/noreref/R1389J_catFR1_0_20Feb18_1720.h5",
        ]),
    ])
    def test_eeg_absolute(self, subject, events_filename, expected_basenames):
        path = resource_filename("cmlreaders.test.data", events_filename)
        events = EventReader.fromfile(path)
        reader = EEGReader("eeg", subject)
        new_events = reader._eegfile_absolute(events)

        for eegfile in new_events[new_events["eegfile"].notnull()]["eegfile"].unique():
            assert eegfile in expected_basenames


class TestRereference:
    def setup_method(self):
        size = 1000
        t = np.arange(size)

        self.data = np.empty((3, size), dtype=np.int16)
        self.data[0] = 1000 * np.sin(t)
        self.data[1] = 1000 * np.cos(t)
        self.data[2] = np.array([1000] * size)

        self.anodes = [0, 0, 1]
        self.cathodes = [1, 2, 2]

        self.contact_labels = ["sine", "cosine", "constant"]
        self.pair_labels = ["sine-cosine", "sine-constant", "cosine-constant"]
        self.contact_nums = [i for i in range(1, len(self.contact_labels) + 1)]

        self.rootdir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.rootdir.name, ignore_errors=True)

    def events(self, filename: str) -> pd.DataFrame:
        return pd.DataFrame({
            "eegoffset": [0],
            "eegfile": [filename],
            "subject": ["R1111M"],
            "experiment": ["doesn't matter"],
            "session": [10],
        })

    @property
    def sources_path(self) -> str:
        return str(self.rootdir.joinpath("sources.json"))

    @property
    def reref_data(self) -> np.ndarray:
        return self.data[self.anodes] - self.data[self.cathodes]

    def make_sources(self, name: str) -> dict:
        return {
            name: {
                "data_format": "int16",
                "n_samples": self.data.shape[1],
                "name": name,
                "sample_rate": 1000,
                "source_file": "na",
                "start_time_ms": 1490441398781,
                "start_time_str": "25Mar17_1129",
                "path": "./sources.json",
            }
        }

    def prepare_dirs(self, name: str) -> Path:
        """Prepare directories for writing EEG data. Returns path to write EEG
        data files to.

        """
        sources = self.make_sources(name)

        with self.rootdir.joinpath("sources.json").open("w") as outfile:
            json.dump(sources, outfile, indent=2)

        eeg_dir = self.rootdir.joinpath("noreref")
        eeg_dir.mkdir()
        return eeg_dir

    def to_split_eeg(self) -> Path:
        """Save files as raw binary "split" EEG files."""
        prefix = "split"
        eeg_dir = self.prepare_dirs(prefix)

        for channel in range(self.data.shape[0]):
            filepath = eeg_dir.joinpath(prefix + ".{:03d}".format(channel + 1))
            with filepath.open("w") as eegfile:
                self.data[channel].tofile(eegfile.name)

        return eeg_dir.joinpath(prefix)

    def to_ramulator_hdf5(self, rerefable: bool) -> Path:
        """Save files in the Ramulator HDF5 format."""
        name = "eeg_timeseries.h5"
        eeg_dir = self.prepare_dirs(name)
        eeg_path = eeg_dir.joinpath(name)

        with h5py.File(eeg_dir.joinpath(name), "w") as hfile:
            if not rerefable:
                # these names are *all* incorrect, but that's the format we
                # have, so we have no choice but to go with it...
                bpinfo = {
                    "contact_name": [
                        b"sine-cosine",
                        b"sine-constant",
                        b"cosine-constant",
                    ],
                    "ch0_label": [
                        "{:03d}".format(e + 1).encode() for e in self.anodes
                    ],
                    "ch1_label": [
                        "{:03d}".format(e + 1).encode() for e in self.cathodes
                    ]
                }

                bpgroup = hfile.create_group("bipolar_info")
                for key, value in bpinfo.items():
                    bpgroup[key] = value

                hfile.create_dataset("monopolar_possible", dtype=np.int, data=[0])
                data = self.reref_data.transpose()
            else:
                hfile.create_dataset("monopolar_possible", dtype=np.int, data=[1])
                data = self.data.transpose()

            hfile["ports"] = [i + 1 for i in range(3)]
            ts = hfile.create_dataset("timeseries", data=data)
            ts.attrs["orient"] = b"row"

        return eeg_path

    @pytest.mark.parametrize("reader_class,rerefable", [
        (SplitEEGReader, True),
        (RamulatorHDF5Reader, True),
        (RamulatorHDF5Reader, False),
    ])
    def test_rereference(self, reader_class, rerefable):
        """Test rereferencing without rhino by using known, fake data."""
        if reader_class == SplitEEGReader:
            eeg_path = self.to_split_eeg()

        if reader_class == RamulatorHDF5Reader:
            eeg_path = self.to_ramulator_hdf5(rerefable)

        scheme = pd.DataFrame({
            "contact_1": [1 + a for a in self.anodes],
            "contact_2": [1 + c for c in self.cathodes],
            "label": self.pair_labels,
        })

        with patch.object(PathFinder, "find", return_value=self.sources_path):
            reader = EEGReader("eeg", subject="R1111M")
            eeg = reader.load(events=self.events(str(eeg_path)),
                              rel_start=0, rel_stop=self.data.shape[-1],
                              scheme=scheme)
            assert_equal(eeg.data[0], self.reref_data)


class TestLoadEEG:
    def test_load_with_empty_events(self):
        sources_file = resource_filename("cmlreaders.test.data", "sources.json")
        with patch.object(PathFinder, "find", return_value=sources_file):
            reader = EEGReader("eeg")

            with pytest.raises(ValueError):
                data = np.random.random((10, 10, 10))
                with patch.object(RamulatorHDF5Reader, "read", return_value=[data, None]):
                    reader.load()

    @pytest.mark.rhino
    @pytest.mark.parametrize("subjects,experiments", [
        (["R1111M"], ["FR1"]),
        (["R1111M"], ["FR1", "catFR1"]),
        (["R1111M", "R1286J"], ["FR1"]),
    ])
    def test_load_multisession(self, subjects, experiments, rhino_root):
        events = CMLReader.load_events(subjects, experiments, rootdir=rhino_root)

        good_sample = False

        while not good_sample:
            events = events.copy()[events["type"] == "WORD"].sample(20)
            good_sample = (
                all([s in events.subject.values for s in subjects]) and
                all([e in events.experiment.values for e in experiments])
            )

        reader = CMLReader(events["subject"].unique()[0], rootdir=rhino_root)

        load = lambda: reader.load_eeg(events, rel_start=0, rel_stop=10)  # noqa

        if len(subjects) > 1:
            with pytest.raises(ValueError):
                load()
            return

        eeg = load()

        assert len(eeg.epochs) == len(events)
        assert len(eeg.events) == len(events)

        for subject in subjects:
            assert subject in set(events["subject"])

        for experiment in experiments:
            assert experiment in set(events["experiment"])

    @pytest.mark.rhino
    @pytest.mark.parametrize("subject,experiment,session,eeg_channels,pairs_channels", [
        ("R1387E", "catFR5", 0, 120, 125),
    ])
    def test_channel_discrepancies(self, subject, experiment, session,
                                   eeg_channels, pairs_channels, rhino_root):
        """Test loading of known subjects with differences between channels in
        pairs.json and channels actually recorded.

        """
        reader = CMLReader(subject, experiment, session, rootdir=rhino_root)
        pairs = reader.load("pairs")
        events = reader.load("events")

        with pytest.warns(MissingChannelsWarning):
            eeg = reader.load_eeg(events.sample(n=1), rel_start=0, rel_stop=10,
                                  scheme=pairs)

        assert len(eeg.channels) == eeg_channels
        assert len(pairs) == pairs_channels
