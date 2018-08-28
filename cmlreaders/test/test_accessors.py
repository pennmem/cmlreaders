import random
import cmlreaders
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


@pytest.fixture
def stim_param_events(events):
    params = {"field_1": 0,
              "field_2": "hello"}
    param_list = [[params]] * 70
    empty_params_list = [[]] * (len(events)-70)
    full_list = param_list+ empty_params_list
    random.shuffle(full_list)
    events = events.copy()
    events['stim_params'] = full_list
    return events


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

    def test_stim_params(self,stim_param_events):
        stim_params = stim_param_events.events.stim_params
        assert all(stim_params.columns == ['field_1', 'field_2'])
        assert (~stim_params.field_1.isna()).sum() == 70


@pytest.mark.rhino
@pytest.mark.parametrize("subject, experiment, session",
                         [
                             ('R1002P', 'FR2', 0),
                             ('R1226D', 'catFR3', 0),
                             ('R1154D', 'PS2.1', 0),
                             ('R1293P', 'PS4_FR', 1),
                             ('R1384J', 'PS5_catFR', 0),
                             # Should multi-site stim raise a warning?
                             # ('R1409D', 'FR6', 0),
                             ('R1436J', 'LocationSearch', 5)
                         ]
                         )
def test_stim_params_rhino(rhino_root, subject, experiment, session):
    reader = cmlreaders.CMLReader(subject, experiment, session, rootdir=rhino_root)
    events = reader.load('task_events')
    stim_params = events.events.stim_params
    assert len(stim_params.columns) > 1
