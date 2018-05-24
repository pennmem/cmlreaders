from collections import namedtuple
from .path_finder import PathFinder  # noqa
from .readers import *  # noqa
from .cmlreader import CMLReader  # noqa

__version__ = '0.2.0'
version_info = namedtuple("VersionInfo", "major,minor,patch")(*__version__.split('.'))
