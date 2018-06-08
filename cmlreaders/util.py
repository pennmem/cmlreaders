import os
from pathlib import Path
from typing import Optional, Union


def get_root_dir(path: Union[str, Path] = None) -> str:
    """Used to set a default root directory. The root directory is resolved in
    the following order:

    1. Using the ``path`` argument to this function.
    2. Using the CML_ROOT environment variable.
    3. Using "/".

    """
    if path is not None:
        return os.path.expanduser(path)

    return os.path.expanduser(os.environ.get("CML_ROOT", "/"))


def is_rerefable(subject: str, experiment: str, session: int,
                 localization: int = 0, montage: int = 0,
                 rootdir: Optional[str] = None) -> bool:
    """Checks if a subject's EEG data can be arbitrarily rereferenced.

    Parameters
    ----------
    subject
        Subject ID.
    experiment
        Experiment.
    session
        Session number.
    localization
        Localization number (default: 0).
    montage
        Montage number (default: 0).
    rootdir
        Root data directory.

    Returns
    -------
    Whether or not the EEG data can be rereferenced.

    """
    from cmlreaders import CMLReader

    reader = CMLReader(subject, experiment, session, localization, montage,
                       rootdir=rootdir)
    sources = reader.load("sources")

    if sources["source_file"] == "eeg_timeseries.h5":
        path = Path(sources["path"]).parent.joinpath("noreref")
        if len(list(path.glob("*.h5"))) == 1:
            # only one HDF5 is present which indicates we recorded in hardware
            # bipolar mode
            return False

    return True
