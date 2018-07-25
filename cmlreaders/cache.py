import functools
from pathlib import Path
from typing import Set, Type


_cached_readers = set()  # type: Set[CachedReader]


class CachedReader(object):
    """Wraps a reader class so that its load method can be cached."""
    # Location to store cached results
    # FIXME: put in the proper place for app-specific data
    # TODO: on-disk or in-memory cache options
    _cachedir = Path.home().joinpath(".cmlreaders", "cache")

    _result = None

    def __init__(self, reader_object):
        self._obj = reader_object
        _cached_readers.add(self)

    @property
    def cached_result(self):
        """Loads the cached result if available else None."""
        return self._result

    @cached_result.setter
    def cached_result(self, result):
        self._result = result

    def clear(self):
        """Clear the cache."""
        self._result = None

    def load(self):
        """Load the data from the cache if available."""
        if self.cached_result is None:
            self.cached_result = self._obj.load()

        return self.cached_result


def cache(cls: Type):
    """Class decorator to allow caching results from a reader's :meth:`load`
    method.

    """
    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        obj = cls(*args, **kwargs)
        return CachedReader(obj)

    return wrapper


def clear_all():
    """Clear all cached results."""
    for reader in _cached_readers:
        reader.clear()
