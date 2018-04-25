"""Utilities for reading raw data stored by Ramulator."""

import json
from pandas.io.json import json_normalize


def events_to_dataframe(path):
    """Read a Ramulator event log and return as a flattened ``DataFrame``.

    Parameters
    ----------
    path : str
        Path to ``event_log.json``

    Returns
    -------
    pandas DataFrame with uninteresting columns removed

    """
    with open(path, 'r') as efile:
        raw = json.loads(efile.read())['events']

    exclude = ['to_id', 'from_id', 'event_id', 'command_id']
    df = json_normalize(raw)
    return df.drop(exclude, axis=1)
