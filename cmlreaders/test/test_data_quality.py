import os
import pytest
import traceback
import pandas as pd

from cmlreaders import get_data_index, CMLReader


@pytest.mark.rhino
def test_eeg_loading(rhino_root, output_dest):
    # Loop over very session of data and check that eeg can be read
    data_index = get_data_index(kind='r1', rootdir=rhino_root)
    total_sessions = len(data_index) + 1
    problematic_sessions = []
    for index, row in data_index.iterrows():
        print("Working on session {}/{}".format(index + 1, total_sessions))
        subject = row['subject']
        experiment = row['experiment']
        session = row['session']
        localization = row['localization']
        montage = row['montage']

        reader = CMLReader(subject=subject, experiment=experiment,
                           session=session, localization=localization,
                           montage=montage, rootdir=rhino_root)
        try:
            eeg = reader.load_eeg(epochs=[(0, 1), (1, 2)])
        except Exception:
            # Accumulate problematic sessions
            problematic_sessions.append((subject, experiment, session,
                                         localization, montage,
                                         traceback.format_exc()))

    cols = ['subject', 'experiment', 'session', 'localization', 'montage',
            'traceback']
    problem_session_df = pd.DataFrame.from_records(problematic_sessions,
                                                   columns=cols)
    outfile = os.path.join(output_dest, "eeg_load_failures.csv")
    problem_session_df.to_csv(outfile, index=False)


@pytest.mark.rhino
def test_channel_counts(rhino_root):
    # Loop over every session and check that the number of recorded channels
    # matches the number of pairs in pairs.json (hardware bipolar recordings)
    # or the number of contacts in contacts.json (monopolar recordings)
    data_index = get_data_index(kind='r1', rootdir=rhino_root)
    return


if __name__ == "__main__":
    #test_eeg_loading("/Volumes/RHINO/")
    test_channel_counts("/Volumes/RHINO/")
