import numpy as np
from numpy.testing import assert_equal
import pytest

from ptsa.data.TimeSeriesX import TimeSeriesX

from cmlreaders.timeseries import TimeSeries


class TestTimeSeries:
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

    def test_to_ptsa(self):
        data = np.random.random((10, 32, 100))
        rate = 1000
        ts = TimeSeries(data, rate)
        tsx = ts.to_ptsa()

        assert isinstance(tsx, TimeSeriesX)
        assert tsx['samplerate'] == ts.samplerate
        assert_equal(tsx['start_offset'].data, ts.start_offsets)
        assert_equal(tsx['time'].data, ts.time)
