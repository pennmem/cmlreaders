import numpy as np
import scipy
from typing import Any, Dict, List, Optional, Tuple, Union


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
    contacts
        Optional list of contact numbers (these are the same as the jackbox
        numbers). When not given, this will default to a list ranged like
        ``[1, data.shape[1]]``.
    tstart
        Start time for each epoch in ms (default: 0).
    attrs
        Arbitrary additional attributes to store.

    Raises
    ------
    ValueError
        When data is not 2- or 3-D; when epochs is given and doesn't match the
        first data dimension

    """
    def __init__(self, data: np.ndarray, samplerate: Union[int, float],
                 epochs: Optional[List[Tuple[int, int]]] = None,
                 contacts: Optional[List[int]] = None,
                 tstart: Union[int, float] = 0,
                 attrs: Optional[Dict[str, Any]] = None):
        if len(data.shape) == 2:
            data = np.array([data])
        if len(data.shape) != 3:
            raise ValueError("Data must be 2- or 3-dimensional")

        self.data = data
        self.samplerate = samplerate
        self.time = self._make_time_array(tstart)  #: time in milliseconds

        if epochs is not None:
            if len(epochs) != self.data.shape[0]:
                raise ValueError("epochs must be the same length as the first data dimension")
            self.epochs = epochs
        else:
            self.epochs = [(-1, -1) for _ in range(self.data.shape[0])]

        if contacts is not None:
            if len(contacts) != self.data.shape[1]:
                raise ValueError("contacts must be the same length as the second data dimension")
            self.contacts = contacts
        else:
            self.contacts = np.linspace(1, self.data.shape[1],
                                        self.data.shape[1],
                                        dtype=np.int).tolist()

        self.attrs = attrs if attrs is not None else {}

    def _make_time_array(self, tstart):
        rate = 1000. / self.samplerate
        n_samples = self.data.shape[-1]
        return np.arange(tstart, n_samples * rate + tstart, rate)

    @classmethod
    def concatenate(cls, series: List["TimeSeries"], dim="events") -> "TimeSeries":
        """Concatenate several :class:`TimeSeries` objects.

        Parameters
        ----------
        series
            The time series to concatenate.
        dim
            The dimension to concatenate on. Allowed options are: "events",
            "time". Default: "events".

        Returns
        -------
        combined
            The concatenated time series.

        Raises
        ------
        ValueError
            When trying to concatenate along the wrong dimension.

        Notes
        -----
        This attempts to combine attributes using a :class:`ChainMap`. This is
        likely not the right solution, so don't rely on keeping attributes.

        """
        if dim not in ["events", "time"]:
            raise ValueError("Invalid dimension to concatenate on: " + dim)

        samplerate = series[0].samplerate
        if not all([s.samplerate == samplerate for s in series]):
            raise ValueError("Sample rates must be the same for all series")

        def check_samples():
            if not all([s.shape[-1] == series[0].shape[-1] for s in series]):
                raise ValueError("Number of samples must match to concatenate"
                                 " events")

        def check_times():
            if not all([s.time == series[0].time for s in series]):
                raise ValueError("Times must be the same for all series")

        def check_channels():
            if not all([np.all(s.contacts == series[0].contacts)
                        for s in series]):
                raise ValueError("Channels must be the same for all series")

        def check_starts():
            if len(series) == 1:
                return

            step = series[0].samplerate / 1000.
            last = series[0].time[-1]
            for s in series[1:]:
                if last + step != s.time[0]:
                    raise ValueError("Start times are not properly aligned for concatenation")
                last += step

        attrs = {
            key: [s.attrs.get(key, None) for s in series]
            for key in series[0].attrs.keys()
        }

        if dim == "events":
            check_samples()
            check_channels()

            data = np.concatenate([s.data for s in series], axis=0)
            epochs = list(np.concatenate([s.epochs for s in series]))

            return TimeSeries(data, samplerate, epochs,
                              contacts=series[0].contacts,
                              tstart=series[0].time[0],
                              attrs=attrs)

        elif dim == "time":
            check_channels()
            check_starts()

            data = np.concatenate([s.data for s in series], axis=2)
            return TimeSeries(data, samplerate, series[0].epochs,
                              contacts=series[0].contacts,
                              tstart=series[0].time[0],
                              attrs=attrs)

    @property
    def shape(self):
        """Get the shape of the data."""
        return self.data.shape

    @property
    def start_offsets(self) -> np.ndarray:
        """Returns the start offsets in samples for each epoch."""
        return np.array([e[0] for e in self.epochs])

    def resample(self, rate: Union[int, float]) -> "TimeSeries":
        """Resample the time series.

        Parameters
        ----------
        rate: Union[int,float]
            new sampling rate, in Hz
        """
        new_len = int(len(self.time) * rate / self.samplerate)
        new_data, _ = scipy.signal.resample(self.data, new_len,
                                            t=self.time, axis=-1)
        return TimeSeries(new_data, rate, epochs=self.epochs,
                          contacts=self.contacts, tstart=self.time[0],
                          attrs=self.attrs)

    def filter(self, filter) -> "TimeSeries":
        """Apply a filter to the data and return a new :class:`TimeSeries`."""
        raise NotImplementedError

    def to_ptsa(self) -> "TimeSeriesX":
        """Convert to a PTSA :class:`TimeSeriesX` object."""
        from ptsa.data.TimeSeriesX import TimeSeriesX

        return TimeSeriesX.create(
            self.data,
            samplerate=self.samplerate,
            dims=('start_offset', 'channel', 'time'),
            coords={
                'start_offset': self.start_offsets,
                'channel': self.contacts,
                'time': self.time
            }
        )

    def to_mne(self) -> "mne.EpochsArray":
        """Convert data to MNE's ``EpochsArray`` format."""
        import mne

        info = mne.create_info([str(c) for c in self.contacts], self.samplerate,
                               ch_types='eeg')

        events = np.empty([self.data.shape[0], 3], dtype=int)
        events[:, 0] = list(range(self.data.shape[0]))
        # FIXME: are these ok?
        events[:, 1] = 0
        events[:, 2] = 1

        epochs = mne.EpochsArray(self.data, info, events, verbose=False)
        return epochs
