from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union
from unittest.mock import patch

from pkg_resources import resource_filename

from cmlreaders import CMLReader, PathFinder
from cmlreaders.data_index import _index_dict_to_dataframe, read_index


@contextmanager
def patched_cmlreader(file_path: Optional[Union[str, Path]] = None):
    """Patches CMLReader to load the data index locally.

    Parameters
    ----------
    file_path

    """
    raw = read_index(resource_filename("cmlreaders.test.data", "r1.json"))

    with patch.object(CMLReader, "_load_index",
                      return_value=_index_dict_to_dataframe(raw)):
        with patch.object(PathFinder, "find", return_value=str(file_path)):
            yield
