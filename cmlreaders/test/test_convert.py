from numpy.testing import assert_equal
import pandas as pd
from pkg_resources import resource_filename
import pytest

from cmlreaders.cmlreader import CMLReader
from cmlreaders.convert import (
    events_to_epochs,
    milliseconds_to_events,
    milliseconds_to_samples,
    samples_to_milliseconds,
)
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
