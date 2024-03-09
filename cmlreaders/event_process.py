import pandas as pd
import os
import json
import warnings

from .path_finder import PathFinder
from . import constants


def correct_retrieval_offsets(events, reader):
    """Correct eegoffset and mstime values for retrieval events with unityEPL-FR bug.

    Parameters
    ----------
    events
        The :class: `pd.DataFrame` loaded from reader.load('events').
    reader
        The :class: `cmlreaders.CMLReader` used to load the events dataframe.

    Returns
    -------
    events
        A :class: `pd.DataFrame` with corrected eegoffset and mstime fields,
        as necessary.

    """
    rhino_root = '/'
    # event types that require offset correction
    retrieval_events = ['REC_START', 'REC_WORD', 'REC_WORD_VV', 'REC_END']
    # load in csv with offset correction sessions
    offset_corrections = pd.read_csv(os.path.join(rhino_root, constants.offset_corrections[0]))
    oc = offset_corrections[(offset_corrections['subject'] == reader.subject) &
                            (offset_corrections['experiment'] == reader.experiment) &
                            (offset_corrections['session'] == reader.session)]

    # no correction necessary (TICL experiments corrections problematic)
    if len(oc) == 0 or reader.experiment == 'TICL_FR' or reader.experiment == 'TICL_catFR':
        return events
    else:
        ms = int(oc.offset_ms.iloc[0])                            # 1000 or 500 ms correction
        if ms == 1000:     # don't correct REC_END for 1000 ms sessions
            retrieval_events = retrieval_events[:-1]
        pf = PathFinder(subject=reader.subject, experiment=reader.experiment,
                        session=reader.session, localization=reader.localization,
                        montage=reader.montage)
        path = pf.find('sources')
        with open(path, 'r') as f:
            sources = json.load(f)
        sr = sources[list(sources.keys())[0]]['sample_rate']      # samplerate
        samples = int((ms / 1000) * sr)                           # samples for correction

        # apply offset correction to retrieval events (also adjust mstimes)
        events['eegoffset'] = [row.eegoffset + samples if row.type in retrieval_events else
                               row.eegoffset for _, row in events.iterrows()]
        events['mstime'] = [row.mstime + ms if row.type in retrieval_events else
                            row.mstime for _, row in events.iterrows()]
        warnings.warn(f'Applying {ms} ms offset correction to retrieval events.')

        return events


def sort_eegfiles(events):
    """Sort events by mstime for sessions with multipl eegfile values.

    Parameters
    ----------
    events:
        The :class: `pd.DataFrame` loaded from reader.load('events').

    Returns
    -------
    events:
        A :class: `pd.DataFrame` with rows sorted by mstime, as necessary.

    """
    # find number of eegfiles, not including empty rows
    eegfiles = [x for x in events['eegfile'].unique() if x != '']
    if len(eegfiles) > 1:
        events = events.sort_values(['mstime', 'eegoffset'], ignore_index=True)
        warnings.warn("Multiple eegfile values, sorting events by mstime.")

    return events
