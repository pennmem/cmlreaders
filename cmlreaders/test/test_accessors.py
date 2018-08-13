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

    recalled = [
        random.choice([0, 1, -999])
        for _ in range(len(types))
    ]

    df = pd.DataFrame({
        "type": types,
        "recalled": recalled,
    })
    return df


class TestEventsAccessors:
    def test_words(self, events):
        expected = events[events.type == "WORD"]
        assert all(expected == events.events.words)

    def test_stim(self, events):
        expected = events[events.type == "STIM_ON"]
        assert all(expected == events.events.stim)

    def test_recalled_words(self, events):
        expected_recalled = events[(events.type == "WORD") & (events.recalled == 1)]
        expected_not_recalled = events[(events.type == "WORD") & (events.recalled == 0)]
        assert all(expected_recalled == events.events.words_recalled)
        assert all(expected_not_recalled == events.events.words_not_recalled)
