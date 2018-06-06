from collections import namedtuple

from .data_index import get_data_index  # noqa

# n.b. the order below matters so as to avoid circular imports
from .path_finder import PathFinder  # noqa
from .readers import *  # noqa
from .cmlreader import CMLReader  # noqa

__version__ = '0.4.0'
version_info = namedtuple("VersionInfo", "major,minor,patch")(*__version__.split('.'))
