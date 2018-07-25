import functools
from pathlib import Path
from typing import Set  # noqa
from typing import Type


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

    def __str__(self):
        return "<{classname:s}({subject:s}, {experiment:s}, {session:d})".format(
            classname=self._obj.__class__.__name__,
            subject=(self._obj.subject or "???"),
            experiment=(self._obj.experiment or "???"),
            session=(self._obj.session if self._obj.session is not None else -1),
        ) + " object at {}>".format(hex(id(self._obj)))

    def __repr__(self):
        return str(self)

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


def cache(cache_type: str):
    """Class decorator to allow caching results from a reader's :meth:`load`
    method.

    Parameters
    ----------
    cache_type
        One of: "memory". What kind of caching to use.

    """
    if cache_type not in ["memory"]:
        raise ValueError("Invalid caching type specified: {}".format(cache_type))

    def decorator(cls: Type):
        @functools.wraps(cls)
        def wrapper(*args, **kwargs):
            obj = cls(*args, **kwargs)
            return CachedReader(obj)

        return wrapper

    return decorator


def clear_all():
    """Clear all cached results."""
    for reader in _cached_readers:
        reader.clear()
