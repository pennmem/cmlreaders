from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Dict, Optional
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio

from .constants import PROTOCOLS, rhino_paths
from .path_finder import PathFinder
from .util import get_root_dir


def read_index(path: str) -> Dict:
    """Reads the data index, removing the initial stages of nesting."""
    path = Path(path)
    kind = os.path.splitext(path.name)[0]
    data = json.loads(path.read_text())
    return data["protocols"][kind]["subjects"]


def _index_dict_to_dataframe(data: Dict) -> pd.DataFrame:
    subjects = data.keys()
    entries = []

    for subject in subjects:
        experiments = data[subject]["experiments"]
        for experiment in experiments:
            sessions = experiments[experiment]["sessions"]
            for session in sessions:
                entry = sessions[session]
                entry["subject"] = subject
                entry["experiment"] = experiment
                entry["session"] = int(session)
                entries.append(entry)

    df = pd.DataFrame(entries)
    return df


def generate_pyfr_index(outdir: str, rootdir: str):
    """Generates an index file for pyFR data. This needs to be run once (and
    hopefully only once!) as a user with correct permissions and writes a file
    to ``/data/events/pyFR/index.csv``.

    Parameters
    ----------
    outdir
        Absolute path to the directory to write the index file to.
    rootdir
        Data root directory.

    """
    path = Path(get_root_dir(rootdir)).joinpath(*rhino_paths["pyfr_root"])
    event_files = list(path.glob("*_events.mat"))

    subjects = []
    sessions = []
    montages = []

    for filename in event_files:
        subject = filename.name.split("_events")[0]

        if "_" in subject:
            subject, montage = subject.split("_")
            montage = int(montage)
        else:
            montage = 0

        events = sio.loadmat(str(filename), squeeze_me=True)["events"]

        try:
            unique_sessions = np.unique(events["session"]).tolist()
        except ValueError:
            warnings.warn("No session field found for {};".format(subject) +
                          " assuming single session", UserWarning)
            unique_sessions = [0]

        subjects += [subject] * len(unique_sessions)
        sessions += unique_sessions
        montages += [montage] * len(unique_sessions)

    experiments = ["pyFR"] * len(sessions)

    df = pd.DataFrame({
        "subject": subjects,
        "experiment": experiments,
        "session": sessions,
        "localization": [0] * len(sessions),
        "montage": montages,
    })

    df.to_json(Path(outdir).joinpath("pyFR.json"))


@lru_cache()
def get_data_index(kind: str = "all",
                   rootdir: Optional[str] = None) -> pd.DataFrame:
    """Get an index to all available data.

    Parameters
    ----------
    kind
        Which kind of data index to return (default: "all"). Choices are:
        ``"r1"``, ``"ltp"``, ``"pyfr"``, ``"all"``. Using ``"all"`` will read
        all available indices.
    rootdir
        Root search path (default: ``"/"``).

    Returns
    -------
    index
        A flattened :class:`pd.DataFrame` version of the data index.

    """
    kinds = PROTOCOLS + ("all",)

    if kind not in kinds:
        raise ValueError("Unknown data index: " + kind)

    finder = PathFinder(rootdir=get_root_dir(rootdir))
    data = []

    if kind == "ltp" or kind == "all":
        data.append(read_index(finder.find("ltp_index")))
    if kind == "r1" or kind == "all":
        data.append(read_index(finder.find("r1_index")))
    if kind == "pyfr":  # or kind == "all":
        raise NotImplementedError
        # data.append(pd.read_json(finder.find("pyfr_index")))

    df = pd.concat([_index_dict_to_dataframe(d) for d in data])

    if kind not in ["ltp"]:
        # make sure localization and montage are integers
        for key in ["localization", "montage"]:
            df.loc[df[key].isnull(), key] = 0
            df[key] = df[key].astype(int)

    return df


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--generate-pyfr-index", action="store_true",
                        help="generate a pyFR.json index file")
    parser.add_argument("--outdir", default=".", type=str,
                        help="output directory (default: .)")
    parser.add_argument("--rootdir", default=None, help="data root directory")

    args = parser.parse_args()

    if args.generate_pyfr_index:
        generate_pyfr_index(args.outdir, args.rootdir)
    else:
        parser.print_usage()
