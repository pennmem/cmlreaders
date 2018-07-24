from functools import wraps
from pathlib import Path
from typing import Callable, KeysView,  Optional

# Global flag indicating whether to enable caching
_caching_enabled = False


class Cache(object):
    """Singleton cache."""
    # Location to store cached results
    # FIXME: put in the proper place for app-specific data
    _cachedir = Path.home().joinpath(".cmlreaders", "cache")

    # singleton instance
    __instance = None  # type: Cache

    # cached function registry
    __funcs = {}

    def __init__(self):
        if Cache.__instance is not None:
            raise RuntimeError("Only one cache object is allowed; "
                               "use the instance classmethod instead")

    @classmethod
    def instance(cls) -> "Cache":
        """Return the singleton instance."""
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    @property
    def cached_names(self) -> KeysView:
        """Returns the names of registered functions."""
        return self.__funcs.keys()

    def register(self, func: "CachedFunction"):
        """Register a function with the cache."""
        self.__funcs[func.name] = func

    def clear(self, name: Optional[str] = None):
        """Clear the cache. If ``name`` is given, clear only the cached files
        associated with that function. Otherwise, clear the entire cache.

        """
        raise NotImplementedError


class CachedFunction(object):
    """Wraps functions that are cached with the :func:`cache` decorator."""
    def __init__(self, func: Callable):
        self._func = func

        cache = Cache.instance()

        if self.name not in cache.cached_names:
            cache.register(self)

    def __call__(self, *args, **kwargs):
        try:
            return self._load_cached_result(*args, **kwargs)
        except:
            return self._func(*args, **kwargs)

    def _load_cached_result(self, *args, **kwargs):
        """Load and return cached results."""
        raise NotImplementedError

    @property
    def name(self):
        return self._func.__name__

    def clear_cache(self):
        """Clears the cache affecting the wrapped function."""
        Cache.instance().clear(self.name)


def cache(func: Callable):
    """Cache on disk the results of the decorated function if caching has been
    enabled.

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _caching_enabled:
            return func(*args, **kwargs)

        if func.__name__ in CachedFunction._funcs:
            return CachedFunction._funcs[func.__name__](*args, **kwargs)

        return CachedFunction(func)(*args, **kwargs)

    return wrapper
