from pkg_resources import resource_filename
import pytest

import pandas as pd

from cmlreaders.ramulator import events_to_dataframe


@pytest.fixture
def event_log_path():
    return resource_filename('cmlreaders.test.data', 'event_log.json')


@pytest.mark.ramulator
class TestRamulator:
    def test_events_to_dataframe(self, event_log_path):
        df = events_to_dataframe(event_log_path)
        assert isinstance(df, pd.DataFrame)
