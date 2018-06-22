from contextlib import contextmanager
from unittest.mock import patch

from pkg_resources import resource_filename

from cmlreaders import CMLReader
from cmlreaders.data_index import _index_dict_to_dataframe, read_index


@contextmanager
def patched_cmlreader():
    """Patches CMLReader to load the data index locally."""
    raw = read_index(resource_filename("cmlreaders.test.data", "r1.json"))

    with patch.object(CMLReader, "_load_index",
                      side_effect=setattr(CMLReader, "_index", _index_dict_to_dataframe(raw))):
        yield
