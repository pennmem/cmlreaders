from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy


class EEGContainer(object):
    """A simple wrapper around a ndarray to represent EEG time series data.

    Parameters
    ----------
    data
        Numpy array shaped as (epochs, channels, time) or (channels, time).
    samplerate
        Sample rate in Hz.
    epochs
        Optional list of tuples defining epochs in ms.
    events
        Events corresponding to epochs (optional).
    channels
        A list of labels for each channel (optional).
    tstart
        Start time for each epoch in ms (default: 0).
    attrs
        Arbitrary additional attributes to store.

    Attributes
    ----------
    data
        Numpy array shaped as (number of events, channels, time).
    samplerate
        Sample rate in Hz.
    events
        DataFrame of events.
    epochs
        Epoch start and stop times.
    time
        Array of time points in milliseconds.

    Raises
    ------
    ValueError
        When data is not 2- or 3-D; when epochs is given and doesn't match the
        first data dimension

    """

    def __init__(self, data: np.ndarray, samplerate: Union[int, float],
                 epochs: Optional[List[Tuple[int, ...]]] = None,
                 events: Optional[pd.DataFrame] = None,
                 channels: Optional[List[str]] = None,
                 tstart: Union[int, float] = 0,
                 attrs: Optional[Dict[str, Any]] = None):
        if len(data.shape) == 2:
            data = np.array([data])
        if len(data.shape) != 3:
            raise ValueError("Data must be 2- or 3-dimensional")

        self.data = data
        self.samplerate = samplerate
        self.time = self._make_time_array(tstart)  # time in milliseconds
        self.events = events

        if epochs is not None:
            if len(epochs) != self.data.shape[0]:
                raise ValueError("epochs must be the same length as the first "
                                 "data dimension")
            self.epochs = epochs
        else:
            self.epochs = [(-1, -1) for _ in range(self.data.shape[0])]

        if channels is not None:
            if len(channels) != self.data.shape[1]:
                raise ValueError(
                    "len(channels) (%d) "
                    "must be the same length as the second data dimension (%d)"
                    % (len(channels), self.data.shape[1]))
            self.channels = channels
        else:
            self.channels = np.linspace(1, self.data.shape[1],
                                        self.data.shape[1],
                                        dtype=np.int).tolist()

        self.attrs = attrs if attrs is not None else {}

    def _make_time_array(self, tstart):
        rate = 1000. / self.samplerate
        n_samples = self.data.shape[-1]
        return np.arange(tstart, n_samples * rate + tstart, rate)

    @classmethod
    def concatenate(cls, containers: List["EEGContainer"], dim="events") -> \
            "EEGContainer":
        """Concatenate several :class:`EEGContainer` objects.

        Parameters
        ----------
        containers
            The containers to concatenate.
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

        samplerate = containers[0].samplerate
        if not all([s.samplerate == samplerate for s in containers]):
            raise ValueError("Sample rates must be the same for all series")

        def check_samples():
            if not all([s.shape[-1] == containers[0].shape[-1]
                        for s in containers]):
                raise ValueError("Number of samples must match to concatenate"
                                 " events")

        def check_times():
            if not all([s.time == containers[0].time for s in containers]):
                raise ValueError("Times must be the same for all series")

        def check_channels():
            if not all([np.all(s.channels == containers[0].channels)
                        for s in containers]):
                raise ValueError("Channels must be the same for all series")

        def check_starts():
            if len(containers) == 1:
                return

            step = containers[0].samplerate / 1000.
            last = containers[0].time[-1]
            for s in containers[1:]:
                if last + step != s.time[0]:
                    raise ValueError("Start times are not properly aligned for"
                                     " concatenation")
                last += step

        attrs = {
            key: [s.attrs.get(key, None) for s in containers]
            for key in containers[0].attrs.keys()
        }

        if all([s.events is None for s in containers]):
            all_events = None
        else:
            all_events = pd.concat([s.events for s in containers], sort=True)

        if dim == "events":
            check_samples()
            check_channels()

            data = np.concatenate([s.data for s in containers], axis=0)
            epochs = list(np.concatenate([s.epochs for s in containers]))

            return EEGContainer(data, samplerate,
                                epochs=epochs,
                                events=all_events,
                                channels=containers[0].channels,
                                tstart=containers[0].time[0],
                                attrs=attrs)

        elif dim == "time":
            check_channels()
            check_starts()

            data = np.concatenate([s.data for s in containers], axis=2)
            return EEGContainer(data, samplerate,
                                epochs=containers[0].epochs,
                                events=all_events,
                                channels=containers[0].channels,
                                tstart=containers[0].time[0],
                                attrs=attrs)

    @property
    def shape(self):
        """Get the shape of the data."""
        return self.data.shape

    @property
    def start_offsets(self) -> np.ndarray:
        """Returns the start offsets in samples for each epoch."""
        return np.array([e[0] for e in self.epochs])

    def resample(self, rate: Union[int, float]) -> "EEGContainer":
        """Resample the time series.

        Parameters
        ----------
        rate: Union[int,float]
            new sampling rate, in Hz
        """
        new_len = int(len(self.time) * rate / self.samplerate)
        new_data, _ = scipy.signal.resample(self.data, new_len,
                                            t=self.time, axis=-1)
        return EEGContainer(new_data, rate, epochs=self.epochs,
                            channels=self.channels, tstart=self.time[0],
                            attrs=self.attrs)

    def filter(self, filter) -> "EEGContainer":
        """
        Apply a filter to the data and return a new :class:`EEGContainer`.
        """
        raise NotImplementedError

    def to_ptsa(self, recarray=False) -> "TimeSeries":  # noqa: F821
        """Convert to a PTSA :class:`TimeSeriesX` object.

        Parameters
        ----------
        recarray
            If True, events get stored as a recarray, to preserve backwards compatibility
            If False, events get stored as xarray coordinates.

        Notes
        -----
        Events are first converted from a :class:`pd.DataFrame` to a Numpy
        recarray and are available as the ``event`` coordinate.

        """
        from ptsa.data.timeseries import TimeSeries
        dims = ("event", "channel", "time")

        if self.events is not None:
            if recarray:
                events = self.events.to_records()
            else:
                for col in self.events.columns:
                    if isinstance(self.events[col].iloc[0], list):
                        self.events[col] = self.events[col].apply(tuple)
                events = pd.MultiIndex.from_frame(self.events)
        else:
            columns = ["eegoffset", "epoch_end"]
            if len(self.epochs[0]) > 2:
                columns = [columns[i] if i < 2 else "column_{}".format(i)
                           for i in range(len(self.epochs[0]))]
            events = pd.MultiIndex.from_frame(pd.DataFrame(self.epochs,
                                                           columns=columns))
            if recarray:
                events = events.to_records(index=False)

        coords = {
            "event": events,
            "channel": self.channels,
            "time": self.time,
        }

        return TimeSeries.create(
            self.data,
            samplerate=self.samplerate,
            dims=dims,
            coords=coords,
        )

    def to_mne(self) -> \
            Union["mne.EpochsArray", "mne.io.RawArray"]:  # noqa: F821
        """
        Convert data to MNE's RawArray or EpochsArray format. If loading
        continuous data, a RawArray will be returned. If loading epoched data,
        an EpochsArray will be returned. Events are accessible made accessible
        through the 'events' field of the RawArray/EpochsArray's info.

        If the data was originally loaded into the EEGContainer using MNE, the
        returned MNE object will retain the original info structure created by
        mne.io.read_raw_*. Otherwise, a default info object will be attached to
        the RawArray/EpochsArray.

        :return: A RawArray or EpochsArray constructed from the EEGContainer's
                data.
        """
        import mne

        # Create a default info object if one has not been provided
        if "mne_info" not in self.attrs:
            info = mne.create_info([str(c) for c in self.channels],
                                   self.samplerate, ch_types='eeg')
        # Otherwise, use the info from the session's first recording
        else:
            info = self.attrs["mne_info"][0]

        # Return RawArray if loading continuous data (must be from from a
        # single recording)
        if len(self.epochs) == 1 and \
                np.all(self.epochs[0] == np.array([0, None])):
            eeg = mne.io.RawArray(self.data[0], info, first_samp=0)
        # Return EpochsArray if loading epoched data
        else:
            eeg = mne.EpochsArray(self.data, info, tmin=self.time[0] / 1000.)

        # Attach events to MNE object as record array
        if self.events is not None:
            events = self.events.to_records()
        else:
            columns = ["eegoffset", "epoch_end"]
            if len(self.epochs[0]) > 2:
                columns = [columns[i] if i < 2 else "column_{}".format(i)
                           for i in range(len(self.epochs[0]))]
            events = pd.DataFrame(self.epochs, columns=columns).to_records(
                index=False)
        eeg.info['temp'] = {"events":events}

        return eeg
