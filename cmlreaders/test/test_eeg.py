from pkg_resources import resource_filename
import pytest

from cmlreaders import CMLReader
from cmlreaders.readers.eeg import events_to_epochs


@pytest.fixture
def events():
    cml_reader = CMLReader()
    path = resource_filename('cmlreaders.test.data', 'all_events.json')
    reader = cml_reader.get_reader('events', file_path=path)
    return reader.as_dataframe()


@pytest.mark.parametrize('rel_start', [-100, 0])
@pytest.mark.parametrize('rel_stop', [100, 500])
def test_events_to_epochs(events, rel_start, rel_stop):
    words = events[events.type == 'WORD']

    epochs = events_to_epochs(words, rel_start=rel_start, rel_stop=rel_stop)
    assert len(epochs) == 156
    for epoch in epochs:
        assert epoch[1] - epoch[0] == rel_stop - rel_start
