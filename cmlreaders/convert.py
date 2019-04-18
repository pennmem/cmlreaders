from typing import List, Optional, Tuple, Union
import pandas as pd


def milliseconds_to_samples(millis: Union[int, float],
                            sample_rate: Union[int, float]) -> int:
    """Covert times in milliseconds to number of samples.

    Parameters
    ----------
    millis
        Time in ms.
    sample_rate
        Sample rate in samples per second.

    Returns
    -------
    Number of samples.

    """
    return int(sample_rate * millis / 1000.)


def samples_to_milliseconds(samples: int, sample_rate: Union[int, float]) -> \
        Union[int, float]:
    """Convert samples to milliseconds.

    Parameters
    ----------
    samples
        Number of samples.
    sample_rate
        Sample rate in samples per second.

    Returns
    -------
    Samples converted to milliseconds.

    """
    return 1000 * samples / sample_rate


def milliseconds_to_events(onsets: List[Union[int, float]],
                           sample_rate: Union[int, float]) -> pd.DataFrame:
    """Take times and produce a minimal events :class:`pd.DataFrame` to load
    EEG data with.

    Parameters
    ----------
    onsets
        Onset times in ms.
    sample_rate
        Sample rate in samples per second.

    Returns
    -------
    events
        A :class:`pd.DataFrame` with ``eegoffset`` as the only column.

    """
    samples = [milliseconds_to_samples(onset, sample_rate) for onset in onsets]
    return pd.DataFrame({"eegoffset": samples})


def events_to_epochs(events: pd.DataFrame, rel_start: int, rel_stop: int,
                     sample_rate: Union[int, float],
                     basenames: Optional[List[str]] = None
                     ) -> List[Tuple[int, int, int]]:
    """Convert events to epochs.

    Parameters
    ----------
    events
        Events to read.
    rel_start
        Start time relative to events in ms.
    rel_stop
        Stop time relative to events in ms.
    sample_rate
        Sample rate in Hz.
    basenames
        EEG file basenames.

    Returns
    -------
    epochs
        A list of tuples giving absolute start and stop times in number of
        samples.

    """
    rel_start = milliseconds_to_samples(rel_start, sample_rate)
    rel_stop = milliseconds_to_samples(rel_stop, sample_rate)
    offsets = events.eegoffset.values
    if basenames is not None:
        eegfiles = events.eegfile.values
        epochs = [(offset + rel_start, offset + rel_stop,
                   basenames.index(eegfile))
                  for (offset, eegfile) in zip(offsets, eegfiles)]
    else:
        epochs = [(offset + rel_start, offset + rel_stop, 0)
                  for offset in offsets]
    return epochs
