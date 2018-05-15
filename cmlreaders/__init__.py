from collections import namedtuple
from .path_finder import *  # noqa
from .cmlreader import * # noqa
from .readers import * # noqa

__version__ = '0.2.0'
version_info = namedtuple("VersionInfo", "major,minor,patch")(*__version__.split('.'))
