from collections import namedtuple
from .path_finder import *

__version__ = '0.1.0'
version_info = namedtuple("VersionInfo", "major,minor,patch")(*__version__.split('.'))
