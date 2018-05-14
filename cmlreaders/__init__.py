from collections import namedtuple
from .path_finder import *  # noqa
from .cmlreader import *
from .readers import *

__version__ = '0.2.0'
version_info = namedtuple("VersionInfo", "major,minor,patch")(*__version__.split('.'))
