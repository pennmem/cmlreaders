import numpy as np
from numpy.testing import assert_equal
import pytest

from ptsa.data.TimeSeriesX import TimeSeriesX

from cmlreaders.timeseries import TimeSeries


class TestTimeSeries:
    def test_make_time_array(self):
        data = np.random.random((1, 32, 2))
        rate = 1000

        ts = TimeSeries(data, rate)
        assert_equal([0, 1], ts.time)

        ts = TimeSeries(data, rate, tstart=1)
        assert_equal([1, 2], ts.time)

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

        for i, ch in enumerate(ts.channels):
            assert ch == 'CH{}'.format(i)

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
        channels = ["channel{}".format(ch) for ch in range(32)]
        data = np.random.random((11, len(channels), 100))

        ts = TimeSeries(data, 1000, channels=channels)
        assert ts.channels == channels

        with pytest.raises(ValueError):
            TimeSeries(data, 1000, channels=['a', 'b'])

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

    def test_to_ptsa(self):
        data = np.random.random((10, 32, 100))
        rate = 1000
        ts = TimeSeries(data, rate)
        tsx = ts.to_ptsa()

        assert isinstance(tsx, TimeSeriesX)
        assert tsx['samplerate'] == ts.samplerate
        assert_equal(tsx['start_offset'].data, ts.start_offsets)
        assert_equal(tsx['time'].data, ts.time)

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
