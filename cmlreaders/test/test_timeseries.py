import numpy as np
from numpy.testing import assert_equal
import pandas as pd
from pkg_resources import resource_filename
import pytest

from ptsa.data.timeseries import TimeSeries as PtsaTimeSeries

from cmlreaders.timeseries import TimeSeries


class TestTimeSeries:

    @pytest.mark.parametrize("tstart", [0, 1])
    def test_make_time_array_tstart(self, tstart):
        data = np.random.random((1, 32, 2))
        rate = 1000

        ts = TimeSeries(data, rate, tstart=tstart)
        assert_equal([tstart, tstart + 1], ts.time)

    @pytest.mark.parametrize("samplerate", [50, 1000, 5000])
    def test_make_time_array_samplerate(self, samplerate):
        data = np.random.random((1, 1, 100))
        ts = TimeSeries(data, samplerate=samplerate)
        assert len(ts.time) == ts.data.shape[-1]
        assert_equal(ts.time[1] - ts.time[0], 1000. / samplerate)

    @pytest.mark.parametrize("data", [
        np.random.random((1, 32, 100)),
        np.random.random((32, 100)),
        np.random.random((100,))
    ])
    def test_create_defaults(self, data):
        if len(data.shape) < 2:
            with pytest.raises(ValueError):
                TimeSeries(data, 1000)
            return

        ts = TimeSeries(data, 1000)

        assert ts.shape == ts.data.shape

        for epoch in ts.epochs:
            assert epoch == (-1, -1)

        assert_equal(ts.channels, [i + 1 for i in range(ts.data.shape[1])])

    @pytest.mark.parametrize("data", [
        np.random.random((1, 32, 100)),
        np.random.random((32, 100)),
    ])
    def test_create_epochs(self, data):
        good_epochs = [(0, 100)]
        bad_epochs = [(0, 100), (100, 200)]

        ts = TimeSeries(data, 1000, epochs=good_epochs)
        assert ts.epochs == good_epochs

        with pytest.raises(ValueError):
            TimeSeries(data, 1000, epochs=bad_epochs)

    def test_create_channels(self):
        data = np.random.random((11, 32, 100))
        contacts = [i + 1 for i in range(data.shape[1])]

        ts = TimeSeries(data, 1000, channels=contacts)
        assert ts.channels == contacts

        with pytest.raises(ValueError):
            TimeSeries(data, 1000, channels=[1, 2])

    def test_resample(self):
        x = np.linspace(0, 4 * np.pi, 400)
        data = np.sin([x, x])
        ts = TimeSeries(data, 100)
        new_ts = ts.resample(200)
        assert len(new_ts.time) == 2 * len(ts.time)
        assert (new_ts.time[1] - new_ts.time[0]) * 2 == (ts.time[1] - ts.time[0])

    @pytest.mark.parametrize("dim", ["events", "time"])
    def test_concatenate(self, dim):
        n_channels = 32
        n_samples = 100
        rate = 1000

        data = [
            np.random.random((1, n_channels, n_samples)),
            np.random.random((1, n_channels, n_samples)),
        ]

        def get_tstart(i):
            if dim == "time":
                return (i * n_samples) / rate * 1000
            else:
                return 0

        series = [
            TimeSeries(d, rate, tstart=get_tstart(i), attrs={'test': 'me'})
            for i, d in enumerate(data)
        ]

        ts = TimeSeries.concatenate(series, dim=dim)

        if dim == "events":
            assert ts.shape == (2, n_channels, n_samples)
            assert_equal(ts.data, np.concatenate(data, axis=0))
        elif dim == "time":
            assert ts.shape == (1, n_channels, n_samples * 2)
            assert_equal(ts.data, np.concatenate(data, axis=2))

    @pytest.mark.only
    @pytest.mark.parametrize("which", ["events", "epochs"])
    def test_to_ptsa(self, which):
        data = np.random.random((10, 32, 100))
        rate = 1000

        if which == "epochs":
            epochs = [(i, i + 100) for i in range(data.shape[0])]
            ts = TimeSeries(data, rate, epochs=epochs)
        else:
            filename = resource_filename("cmlreaders.test.data", "R1111M_FR1_0_events.json")
            events = pd.read_json(filename).iloc[:data.shape[0]]
            ts = TimeSeries(data, rate, events=events)

        tsx = ts.to_ptsa()

        assert isinstance(tsx, PtsaTimeSeries)
        assert tsx["samplerate"] == ts.samplerate
        assert_equal(tsx.data, data)

        offsets = tsx["event"].data["eegoffset"]

        if which == "epochs":
            assert_equal(offsets, ts.start_offsets)
        else:
            assert_equal(offsets, events["eegoffset"])

        assert_equal(tsx["time"].data, ts.time)

    def test_to_mne(self):
        events = int(np.random.uniform(1, 10))
        channels = int(np.random.uniform(1, 128))
        samples = int(np.random.uniform(10, 100))

        data = np.random.random((events, channels, samples))
        rate = 1000
        ts = TimeSeries(data, rate)
        ea = ts.to_mne()

        assert len(ea.times) == samples
        assert len(ea.events) == events
        assert len(ea.info['chs']) == channels
        assert_equal(ea.get_data().shape, data.shape)
