import os
from pathlib import Path
from typing import Union


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
