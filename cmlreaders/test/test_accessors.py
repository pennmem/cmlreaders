import random

import pandas as pd
import pytest


@pytest.fixture
def events():
    types = (
        ["WORD"] * random.randint(1, 100) +
        ["STIM_ON"] * random.randint(1, 100)
    )
    random.shuffle(types)

    df = pd.DataFrame({
        "type": types
    })
    return df


class TestEventsAccessors:
    def test_words(self, events):
        expected = events[events.type == "WORD"]
        assert all(expected == events.events.words)

    def test_stim(self, events):
        expected = events[events.type == "STIM_ON"]
        assert all(expected == events.events.stim)
