import os
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from cmlreaders import constants
from cmlreaders.exc import UnknownProtocolError


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


def get_protocol(subject: str) -> str:
    """Get the protocol name from the subject code.

    This returns the ``<protocol> `` in ``/protocols/<protocol>``. For
    example, it returns ``"r1"`` for RAM subjects.

    """
    if subject.startswith("R1"):
        return "r1"
    elif subject.startswith("LTP") or subject.startswith("PLTP"):
        return "ltp"
    elif subject[:2] in constants.PYFR_SUBJECT_CODE_PREFIXES:
        return "pyfr"
    else:
        raise UnknownProtocolError(
            "Can't determine protocol for subject id " + subject
        )


class DefaultTuple(tuple):
    """A tuple that will return a default value if an entry is None.

    Parameters
    ----------
    iterable
        Iterable to initialize the tuple with.
    default
        Default value to return when an item is None (default: 0).

    """
    __default = 0

    def __new__(cls, iterable: Iterable, default: Any = 0):
        self = super().__new__(cls, iterable)
        self.__default = default
        return self

    def __getitem__(self, item):
        value = super().__getitem__(item)
        if value is None:
            return self.__default
        else:
            return value
