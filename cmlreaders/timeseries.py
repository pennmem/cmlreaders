import numpy as np
from typing import List, Optional, Tuple, Union

from ptsa.data.TimeSeriesX import TimeSeriesX


class TimeSeries(object):
    """A simple wrapper around a ndarray to represent EEG time series data.

    Parameters
    ----------
    data
        Numpy array shaped as (epochs, channels, time) or (channels, time).
    samplerate
        Sample rate in Hz.
    epochs
        Optional list of tuples defining epochs in ms.
    channels
        Optional list of channel labels.
    tstart
        Start time for each epoch in ms (default: 0).

    Raises
    ------
    ValueError
        When data is not 2- or 3-D; when epochs is given and doesn't match the
        first data dimension

    """
    def __init__(self, data: np.ndarray, samplerate: Union[int, float],
                 epochs: Optional[List[Tuple[int, int]]] = None,
                 channels: Optional[List[str]] = None,
                 tstart: Union[int, float] = 0):
        if len(data.shape) == 2:
            data = np.array([data])
        if len(data.shape) != 3:
            raise ValueError("Data must be 2- or 3-dimensional")

        self.data = data
        self.time = np.arange(tstart, data.shape[-1] * samplerate, samplerate)
        self.samplerate = samplerate

        if epochs is not None:
            if len(epochs) != self.data.shape[0]:
                raise ValueError("epochs must be the same length as the first data dimension")
            self.epochs = epochs
        else:
            self.epochs = [(-1, -1) for _ in range(self.data.shape[0])]

        if channels is not None:
            if len(channels) != self.data.shape[1]:
                raise ValueError("channels must be the same length as the second data dimension")
            self.channels = channels
        else:
            self.channels = ["CH{}".format(i) for i in range(self.data.shape[1])]

    @property
    def start_offsets(self) -> np.ndarray:
        """Returns the start offsets in ms for each epoch."""
        return np.array([e[0] for e in self.epochs])

    def resample(self, rate: Union[int, float]) -> "TimeSeries":
        """Resample the time series."""
        raise NotImplementedError

    def filter(self, filter) -> "TimeSeries":
        """Apply a filter to the data and return a new :class:`TimeSeries`."""
        raise NotImplementedError

    def to_ptsa(self) -> TimeSeriesX:
        """Convert to a PTSA :class:`TimeSeriesX` object."""
        return TimeSeriesX.create(
            self.data,
            samplerate=self.samplerate,
            dims=('start_offset', 'channel', 'time'),
            coords={
                'start_offset': self.start_offsets,
                'channel': self.channels,
                'time': self.time
            }
        )
